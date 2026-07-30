"""Aggregate native ShellBench Harbor job directories.

The postprocessor intentionally uses only the Python standard library so it can
run directly on a copied Harbor job directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence


CLASSIFICATIONS = (
    "infra",
    "agent_exit",
    "verifier_missing_reward",
    "clean_fail",
    "partial",
    "pass",
)

AGENT_EXIT_EXCEPTION_TYPES = {
    "AgentSafetyRefusalError",
    "AgentTimeoutError",
    "NonZeroAgentExitCodeError",
}

INFRA_EXCEPTION_TYPES = {
    "AgentSetupError",
    "AgentSetupTimeoutError",
    "DockerStartupError",
    "EnvironmentStartTimeoutError",
    "GatewaySetupError",
    "GatewayTimeoutError",
    "InvalidManifestError",
    "InvalidResultError",
    "MissingResultError",
    "VerifierJudgeInfraError",
    "VerifierRewardContractError",
    "VerifierTimeoutError",
}

INFRA_MESSAGE_PATTERNS = (
    "connection refused",
    "connection reset",
    "docker daemon",
    "docker startup",
    "failed to connect",
    "gateway",
    "host.docker.internal",
    "litellm",
    "model not found",
    "no healthy upstream",
    "proxy",
    "service unavailable",
    "timed out waiting for container",
    "upstream connect",
)

MISSING_REWARD_EXCEPTION_TYPES = {
    "RewardFileEmptyError",
    "RewardFileNotFoundError",
}

INVALID_REWARD_CONTRACT_EXCEPTION_TYPES = {
    "VerifierRewardContractError",
}

# Compatibility for manifests created before parity scopes were recorded.
# The pinned calibration validated only this exact native route and task count.
LEGACY_PARITY_VALIDATED_SCOPES = {
    ("codex", "gpt55", 20),
}

TIMING_FIELDS = (
    "environment_setup",
    "agent_setup",
    "agent_execution",
    "verifier",
)

PER_TASK_FIELDS = (
    "run_label",
    "pair_label",
    "repetition",
    "task_name",
    "task_path",
    "trial_name",
    "result_path",
    "classification",
    "execution_outcome",
    "execution_exit_code",
    "execution_reason",
    "reward",
    "exception_type",
    "exception_message",
    "exception_occurred_at",
    "started_at",
    "finished_at",
    "duration_sec",
    "environment_setup_started_at",
    "environment_setup_finished_at",
    "environment_setup_sec",
    "agent_setup_started_at",
    "agent_setup_finished_at",
    "agent_setup_sec",
    "agent_execution_started_at",
    "agent_execution_finished_at",
    "agent_execution_sec",
    "verifier_started_at",
    "verifier_finished_at",
    "verifier_sec",
    "n_input_tokens",
    "n_cache_tokens",
    "n_output_tokens",
    "cost_usd",
    "source",
    "trajectory_status",
    "runtime_model_name",
    "canonical_model_identity",
)

RUN_FIELDS = (
    "run_label",
    "pair_label",
    "repetition",
    "expected_task_count",
    "result_file_count",
    "valid_result_count",
    "completed_result_count",
    "coverage",
    "score",
    "exact_passes",
    "partials",
    "nonzero",
    "infra",
    "agent_exits",
    "clean_completed",
    "missing_reward",
    "verifier_contract_violations",
    "clean_coverage",
    "incomplete",
    "infra_dominated",
    "harness_wide_failure",
    "harness_wide_failure_signature",
    "run_accepted",
    "run_acceptance_reason",
    "canonical_model_identity",
    "trajectory_complete",
    "parity_validated",
    "parity_validation_status",
    "parity_validation_scope",
    "eligible",
    "exclusion_reason",
    "n_input_tokens",
    "n_cache_tokens",
    "n_output_tokens",
    "cost_usd",
)

REPAIR_SENSITIVITY_FIELDS = (
    "run_label",
    "pair_label",
    "repetition",
    "original_run_label",
    "expected_task_count",
    "original_task_count",
    "repaired_task_count",
    "unknown_source_task_count",
    "original_result_count",
    "repaired_result_count",
    "original_score",
    "repaired_score",
    "score_delta",
    "delta_status",
)

INFRA_FIELDS = (
    "run_label",
    "pair_label",
    "repetition",
    "task_name",
    "task_path",
    "trial_name",
    "result_path",
    "exception_type",
    "exception_message",
    "exception_occurred_at",
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(value, dict):
        return None, f"expected a JSON object, found {type(value).__name__}"
    return value, None


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = value
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(float(number)):
        return None
    return number


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None or int(number) != number:
        return None
    return int(number)


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _parse_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _duration_seconds(started_at: Any, finished_at: Any) -> float | None:
    started = _parse_datetime(started_at)
    finished = _parse_datetime(finished_at)
    if started is None or finished is None:
        return None
    try:
        return round((finished - started).total_seconds(), 6)
    except TypeError:
        return None


def _exception_fields(result: dict[str, Any]) -> tuple[str, str, str]:
    exception = result.get("exception_info")
    if not isinstance(exception, dict):
        return "", "", ""
    return (
        str(exception.get("exception_type") or ""),
        str(exception.get("exception_message") or ""),
        str(exception.get("occurred_at") or ""),
    )


def _reward(result: dict[str, Any]) -> int | float | None:
    return _number(_nested(result, "verifier_result", "rewards", "reward"))


def _is_agent_exit(exception_type: str) -> bool:
    return exception_type in AGENT_EXIT_EXCEPTION_TYPES or (
        exception_type.endswith("AgentExitCodeError")
    )


def _is_infra(exception_type: str, exception_message: str) -> bool:
    if exception_type in INFRA_EXCEPTION_TYPES:
        return True
    combined = f"{exception_type} {exception_message}".lower()
    return any(pattern in combined for pattern in INFRA_MESSAGE_PATTERNS)


def _classify(
    result: dict[str, Any],
    reward: int | float | None,
    execution_kind: str,
) -> str:
    exception_type, exception_message, _ = _exception_fields(result)
    if execution_kind in {"harness_error", "infra_error"}:
        return "infra"
    if execution_kind == "verifier_error":
        return (
            "verifier_missing_reward"
            if exception_type in MISSING_REWARD_EXCEPTION_TYPES
            else "infra"
        )
    if execution_kind == "agent_error":
        return "agent_exit"
    if exception_type in MISSING_REWARD_EXCEPTION_TYPES:
        return "verifier_missing_reward"
    if exception_type:
        if _is_infra(exception_type, exception_message):
            return "infra"
        return "agent_exit"
    if reward is None:
        return "verifier_missing_reward"
    if reward <= 0:
        return "clean_fail"
    if reward < 1:
        return "partial"
    return "pass"


def _execution_fields(
    result: dict[str, Any],
    *,
    harness: str,
) -> tuple[str, int | None, str]:
    outcome = result.get("execution_outcome")
    if isinstance(outcome, dict):
        kind = str(outcome.get("kind") or "")
        if kind and kind != "pending":
            exit_code = _number(outcome.get("exit_code"))
            return (
                kind,
                int(exit_code) if exit_code is not None else None,
                str(outcome.get("reason") or ""),
            )

    exception_type, exception_message, _ = _exception_fields(result)
    exit_match = re.search(r"\bcode\s+(\d+)\b", exception_message)
    exit_code = int(exit_match.group(1)) if exit_match else None
    if (
        harness == "openclaw"
        and exception_type == "NonZeroAgentExitCodeError"
        and exit_code in {70, 71}
    ):
        return "harness_error", exit_code, exception_message
    if exception_type in MISSING_REWARD_EXCEPTION_TYPES:
        return "verifier_error", exit_code, exception_message
    if exception_type in INVALID_REWARD_CONTRACT_EXCEPTION_TYPES:
        return "verifier_error", exit_code, exception_message
    if exception_type and _is_infra(exception_type, exception_message):
        return "infra_error", exit_code, exception_message
    if exception_type:
        return "agent_error", exit_code, exception_message
    return "clean", None, ""


def _scorecard_infra_error(trial_dir: Path) -> tuple[str, str] | None:
    scorecard_path = trial_dir / "verifier" / "scorecard.json"
    if not scorecard_path.is_file():
        return None
    scorecard, error = _read_json(scorecard_path)
    if scorecard is None or scorecard.get("status") != "infra_error":
        return None

    reasons = scorecard.get("judge_reasons")
    if isinstance(reasons, list):
        message = "; ".join(str(reason) for reason in reasons if reason)
    else:
        message = ""
    if not message:
        message = str(
            scorecard.get("reason")
            or error
            or "verifier scorecard reported infrastructure failure"
        )
    return "VerifierJudgeInfraError", message[:2000]


def _manifest_value(manifest: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = manifest.get(key)
        if value is not None and value != "":
            return value
    return None


def _compact_json(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def _scope_matches_value(actual: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_scope_matches_value(actual, item) for item in expected)
    if isinstance(actual, int) or isinstance(expected, int):
        return _integer(actual) == _integer(expected)
    return str(actual or "") == str(expected or "")


def _parity_validation(
    manifest: dict[str, Any],
    *,
    native_run: bool,
) -> tuple[bool, str, str]:
    if not native_run:
        return True, "not_required", ""

    validation = manifest.get("parity_validation")
    legacy_scope = manifest.get("parity_validation_scope")
    if isinstance(validation, dict):
        validated = validation.get("validated") is True
        scope = validation.get("scope")
        if scope is None:
            scope = {
                key: value
                for key, value in validation.items()
                if key not in {"validated", "evidence", "notes"}
            }
    elif isinstance(legacy_scope, dict):
        validated = manifest.get("parity_validated") is True
        scope = legacy_scope
    elif manifest.get("parity_validated") is True:
        legacy_scope = (
            str(manifest.get("harness") or ""),
            str(manifest.get("model_slug") or ""),
            _integer(manifest.get("expected_task_count")),
        )
        scope = {
            "expected_task_count": legacy_scope[2],
            "harness": legacy_scope[0],
            "model_slug": legacy_scope[1],
        }
        scope_text = _compact_json(scope)
        if legacy_scope in LEGACY_PARITY_VALIDATED_SCOPES:
            return True, "legacy_route_compatible", scope_text
        if legacy_scope[:2] == ("codex", "gpt55"):
            return False, "legacy_scope_mismatch", scope_text
        return False, "legacy_unscoped", ""
    else:
        return False, "not_validated", ""

    if not isinstance(scope, dict):
        return False, "scope_invalid", ""
    scope_text = _compact_json(scope)
    if not validated:
        return False, "not_validated", scope_text

    model_keys = ("model_slug", "model_id", "provider_model_id")
    if "harness" not in scope or not any(key in scope for key in model_keys):
        return False, "scope_incomplete", scope_text

    comparable = {
        "harness": manifest.get("harness"),
        "model_slug": manifest.get("model_slug"),
        "model_id": manifest.get("model_id"),
        "provider_model_id": manifest.get("provider_model_id"),
        "execution_mode": manifest.get("execution_mode"),
        "task_suite": manifest.get("task_suite"),
        "expected_task_count": manifest.get("expected_task_count"),
    }
    unsupported_keys = set(scope) - set(comparable)
    if unsupported_keys:
        return False, "scope_unsupported", scope_text
    for key, expected in scope.items():
        if not _scope_matches_value(comparable[key], expected):
            return False, "scope_mismatch", scope_text
    return True, "validated", scope_text


def _derive_repetition(run_label: str) -> int | None:
    match = re.search(r"(?:^|[-_])(?:rep(?:etition)?|repeat|r)[-_]?(\d+)$", run_label, re.I)
    return int(match.group(1)) if match else None


def _derive_pair_label(run_label: str) -> str:
    return re.sub(
        r"(?:[-_])(?:rep(?:etition)?|repeat|r)[-_]?\d+$",
        "",
        run_label,
        flags=re.I,
    )


def _pair_label(manifest: dict[str, Any], run_label: str) -> str:
    value = _manifest_value(
        manifest,
        "pair_label",
        "pair_name",
        "comparison_pair",
        "base_label",
        "pair",
    )
    if isinstance(value, dict):
        value = _manifest_value(value, "label", "name", "id")
    if value is not None:
        return str(value)
    harness = manifest.get("harness")
    model_slug = manifest.get("model_slug")
    if harness and model_slug:
        return f"{harness}-{model_slug}"
    return _derive_pair_label(run_label)


def _repetition(manifest: dict[str, Any], run_label: str) -> int | None:
    value = _manifest_value(
        manifest,
        "repetition",
        "repetition_index",
        "repeat",
        "repeat_index",
    )
    return _integer(value) if value is not None else _derive_repetition(run_label)


def _task_reference(item: Any) -> tuple[str, str] | None:
    if isinstance(item, str):
        path = item
        return Path(path).name or path, path
    if not isinstance(item, dict):
        return None
    path_value = _manifest_value(item, "path", "task_path")
    name_value = _manifest_value(item, "task_name", "name", "id")
    if isinstance(path_value, dict):
        path_value = path_value.get("path")
    path = str(path_value or "")
    name = str(name_value or (Path(path).name if path else ""))
    return (name, path) if name or path else None


def _expected_tasks(manifest: dict[str, Any]) -> list[tuple[str, str]]:
    for key in ("tasks", "expected_tasks", "task_paths"):
        value = manifest.get(key)
        if not isinstance(value, list):
            continue
        tasks = [task for item in value if (task := _task_reference(item)) is not None]
        if tasks:
            return tasks
    return []


def _expected_task_count(
    manifest: dict[str, Any],
    expected_tasks: Sequence[tuple[str, str]],
    discovered_count: int,
) -> int:
    value = _manifest_value(
        manifest,
        "expected_task_count",
        "task_count",
        "expected_trials",
        "n_tasks",
    )
    explicit = _integer(value)
    if explicit is not None and explicit >= 0:
        return explicit
    if expected_tasks:
        return len(expected_tasks)
    return discovered_count


def _task_identity(result: dict[str, Any], trial_dir: Path) -> tuple[str, str, str]:
    task_path = _nested(result, "task_id", "path")
    task_path = str(task_path or "")
    task_name = str(result.get("task_name") or "")
    if task_path:
        task_name = Path(task_path).name
    elif task_name:
        task_name = Path(task_name).name
    trial_name = str(result.get("trial_name") or trial_dir.name)
    return task_name or trial_name, task_path, trial_name


def _usage_value(agent_result: dict[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = _number(agent_result.get(key))
        if value is not None:
            return value
    return None


def _normalize_result(
    *,
    result: dict[str, Any],
    result_path: Path,
    run_label: str,
    pair_label: str,
    repetition: int | None,
    harness: str,
) -> dict[str, Any]:
    task_name, task_path, trial_name = _task_identity(result, result_path.parent)
    reward = _reward(result)
    exception_type, exception_message, exception_occurred_at = _exception_fields(result)
    scorecard_infra = _scorecard_infra_error(result_path.parent)
    if scorecard_infra is not None:
        exception_type, exception_message = scorecard_infra
    execution_kind, execution_exit_code, execution_reason = _execution_fields(
        result,
        harness=harness,
    )
    if scorecard_infra is not None:
        execution_kind = "infra_error"
        execution_reason = exception_message
    agent_result = result.get("agent_result")
    agent_result = agent_result if isinstance(agent_result, dict) else {}
    row: dict[str, Any] = {
        "run_label": run_label,
        "pair_label": pair_label,
        "repetition": repetition,
        "task_name": task_name,
        "task_path": task_path,
        "trial_name": trial_name,
        "result_path": str(result_path),
        "classification": (
            "infra"
            if scorecard_infra is not None
            else _classify(result, reward, execution_kind)
        ),
        "execution_outcome": execution_kind,
        "execution_exit_code": execution_exit_code,
        "execution_reason": execution_reason,
        "reward": reward,
        "exception_type": exception_type,
        "exception_message": exception_message,
        "exception_occurred_at": exception_occurred_at,
        "started_at": result.get("started_at") or "",
        "finished_at": result.get("finished_at") or "",
        "duration_sec": _duration_seconds(result.get("started_at"), result.get("finished_at")),
        "n_input_tokens": _usage_value(agent_result, "n_input_tokens", "input_tokens"),
        "n_cache_tokens": _usage_value(agent_result, "n_cache_tokens", "cache_tokens"),
        "n_output_tokens": _usage_value(agent_result, "n_output_tokens", "output_tokens"),
        "cost_usd": _usage_value(
            agent_result,
            "cost_usd",
            "estimated_cost_usd",
            "total_cost_usd",
        ),
        "source": str(result.get("source") or ""),
        "trajectory_status": str(agent_result.get("trajectory_status") or ""),
        "runtime_model_name": str(agent_result.get("runtime_model_name") or ""),
        "canonical_model_identity": agent_result.get("canonical_model_identity"),
        "_has_result_file": True,
        "_valid_result": True,
        "_completed_result": bool(result.get("finished_at")),
        "_scorable": True,
        "_reward_contract_valid": (
            exception_type not in INVALID_REWARD_CONTRACT_EXCEPTION_TYPES
        ),
    }
    for field in TIMING_FIELDS:
        timing = result.get(field)
        timing = timing if isinstance(timing, dict) else {}
        started_at = timing.get("started_at") or ""
        finished_at = timing.get("finished_at") or ""
        row[f"{field}_started_at"] = started_at
        row[f"{field}_finished_at"] = finished_at
        row[f"{field}_sec"] = _duration_seconds(started_at, finished_at)
    return row


def _infra_row(
    *,
    run_label: str,
    pair_label: str,
    repetition: int | None,
    task_name: str,
    task_path: str,
    trial_name: str,
    result_path: Path,
    exception_type: str,
    exception_message: str,
    has_result_file: bool,
    scorable: bool = True,
) -> dict[str, Any]:
    row = {field: "" for field in PER_TASK_FIELDS}
    row.update(
        {
            "run_label": run_label,
            "pair_label": pair_label,
            "repetition": repetition,
            "task_name": task_name,
            "task_path": task_path,
            "trial_name": trial_name,
            "result_path": str(result_path),
            "classification": "infra",
            "reward": None,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "_has_result_file": has_result_file,
            "_valid_result": False,
            "_completed_result": False,
            "_scorable": scorable,
        }
    )
    return row


def _exclude_row(
    row: dict[str, Any],
    *,
    exception_type: str,
    exception_message: str,
) -> None:
    row.update(
        {
            "classification": "infra",
            "reward": None,
            "exception_type": exception_type,
            "exception_message": exception_message,
            "_scorable": False,
        }
    )


def _matches_expected(row: dict[str, Any], task_name: str, task_path: str) -> bool:
    candidates = {
        str(row.get("task_name") or ""),
        str(row.get("task_path") or ""),
        Path(str(row.get("task_path") or "")).name,
    }
    return bool({task_name, task_path, Path(task_path).name if task_path else ""} & candidates)


def _load_run(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    run_label = run_dir.name
    manifest_path = run_dir / "run_manifest.json"
    manifest, manifest_error = _read_json(manifest_path)
    manifest = manifest or {}
    run_label = str(manifest.get("run_label") or run_label)
    pair_label = _pair_label(manifest, run_label)
    repetition = _repetition(manifest, run_label)
    harness = str(manifest.get("harness") or "")

    rows: list[dict[str, Any]] = []
    trial_dirs = sorted(path for path in run_dir.iterdir() if path.is_dir())
    for trial_dir in trial_dirs:
        result_path = trial_dir / "result.json"
        if not result_path.is_file():
            rows.append(
                _infra_row(
                    run_label=run_label,
                    pair_label=pair_label,
                    repetition=repetition,
                    task_name=trial_dir.name,
                    task_path="",
                    trial_name=trial_dir.name,
                    result_path=result_path,
                    exception_type="MissingResultError",
                    exception_message="trial directory has no result.json",
                    has_result_file=False,
                )
            )
            continue
        result, error = _read_json(result_path)
        if result is None:
            rows.append(
                _infra_row(
                    run_label=run_label,
                    pair_label=pair_label,
                    repetition=repetition,
                    task_name=trial_dir.name,
                    task_path="",
                    trial_name=trial_dir.name,
                    result_path=result_path,
                    exception_type="InvalidResultError",
                    exception_message=error or "invalid result.json",
                    has_result_file=True,
                )
            )
            continue
        rows.append(
            _normalize_result(
                result=result,
                result_path=result_path,
                run_label=run_label,
                pair_label=pair_label,
                repetition=repetition,
                harness=harness,
            )
        )

    expected_tasks = _expected_tasks(manifest)
    if expected_tasks:
        available = set(range(len(rows)))
        for task_name, task_path in expected_tasks:
            matches = [
                index
                for index in sorted(available)
                if _matches_expected(rows[index], task_name, task_path)
            ]
            if not matches:
                trial_name = task_name or Path(task_path).name
                rows.append(
                    _infra_row(
                        run_label=run_label,
                        pair_label=pair_label,
                        repetition=repetition,
                        task_name=task_name or trial_name,
                        task_path=task_path,
                        trial_name=trial_name,
                        result_path=run_dir / trial_name / "result.json",
                        exception_type="MissingResultError",
                        exception_message="expected task produced no result.json",
                        has_result_file=False,
                    )
                )
                continue
            selected = matches[0]
            available.remove(selected)
            rows[selected]["_scorable"] = True
            for duplicate in matches[1:]:
                available.remove(duplicate)
                _exclude_row(
                    rows[duplicate],
                    exception_type="DuplicateTaskResultError",
                    exception_message=(
                        f"multiple result.json files matched expected task {task_name}"
                    ),
                )
        for index in sorted(available):
            _exclude_row(
                rows[index],
                exception_type="UnexpectedTaskResultError",
                exception_message="result.json does not match any expected task",
            )

    expected_count = _expected_task_count(manifest, expected_tasks, len(rows))
    while sum(bool(row.get("_scorable")) for row in rows) < expected_count:
        missing_index = sum(bool(row.get("_scorable")) for row in rows) + 1
        rows.append(
            _infra_row(
                run_label=run_label,
                pair_label=pair_label,
                repetition=repetition,
                task_name=f"<missing-{missing_index}>",
                task_path="",
                trial_name=f"<missing-{missing_index}>",
                result_path=run_dir / f"<missing-{missing_index}>" / "result.json",
                exception_type="MissingResultError",
                exception_message="expected task produced no trial directory",
                has_result_file=False,
            )
        )

    if manifest_error:
        rows.append(
            _infra_row(
                run_label=run_label,
                pair_label=pair_label,
                repetition=repetition,
                task_name="<run-manifest>",
                task_path="",
                trial_name="<run-manifest>",
                result_path=manifest_path,
                exception_type="InvalidManifestError",
                exception_message=manifest_error,
                has_result_file=False,
                scorable=False,
            )
        )

    rows.sort(key=lambda row: (str(row["task_name"]), str(row["trial_name"])))
    summary = _summarize_run(
        run_label=run_label,
        pair_label=pair_label,
        repetition=repetition,
        expected_count=expected_count,
        rows=rows,
        manifest=manifest,
    )
    return summary, rows, manifest


def _sum_present(rows: Iterable[dict[str, Any]], key: str) -> int | float | None:
    values = [_number(row.get(key)) for row in rows]
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def _failure_signature(row: dict[str, Any]) -> str:
    exception_type = str(row.get("exception_type") or "")
    message = str(row.get("exception_message") or "").lower()
    normalized = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", message)
    normalized = re.sub(r"\b\d+\b", "<n>", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return f"{exception_type}: {normalized[:240]}"


def _harness_wide_failure(
    rows: Sequence[dict[str, Any]],
    expected_count: int,
) -> tuple[bool, str]:
    signatures: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("classification") != "infra":
            continue
        signature = _failure_signature(row)
        if signature:
            signatures[signature] += 1
    if not signatures:
        return False, ""
    signature, count = max(signatures.items(), key=lambda item: item[1])
    threshold = max(3, math.ceil(expected_count * 0.25))
    return count >= threshold, signature if count >= threshold else ""


def _summarize_run(
    *,
    run_label: str,
    pair_label: str,
    repetition: int | None,
    expected_count: int,
    rows: Sequence[dict[str, Any]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    scored_rows = [row for row in rows if row.get("_scorable")]
    classifications = [str(row["classification"]) for row in scored_rows]
    rewards = [
        (
            _number(row.get("reward"))
            if row.get("_reward_contract_valid", True)
            else None
        )
        for row in scored_rows
    ]
    all_classifications = [str(row["classification"]) for row in rows]
    execution_kinds = [
        str(row.get("execution_outcome") or "")
        for row in scored_rows
    ]
    result_file_count = sum(bool(row.get("_has_result_file")) for row in rows)
    valid_result_count = sum(bool(row.get("_valid_result")) for row in rows)
    completed_result_count = sum(bool(row.get("_completed_result")) for row in rows)
    exact_passes = sum(
        classification == "pass" and reward is not None and reward >= 1
        for classification, reward in zip(classifications, rewards)
    )
    partials = classifications.count("partial")
    infra = all_classifications.count("infra")
    agent_exits = classifications.count("agent_exit")
    missing_reward = classifications.count("verifier_missing_reward")
    verifier_contract_violations = sum(
        not row.get("_reward_contract_valid", True)
        for row in scored_rows
    )
    clean_completed = sum(
        classification in {"clean_fail", "partial", "pass"}
        for classification in classifications
    )
    nonzero = sum(reward is not None and reward > 0 for reward in rewards)
    reward_sum = sum(reward or 0 for reward in rewards)
    score = reward_sum / expected_count if expected_count else 0.0
    coverage = (
        min(completed_result_count / expected_count, 1.0) if expected_count else 0.0
    )
    clean_coverage = clean_completed / expected_count if expected_count else 0.0
    incomplete = (
        expected_count <= 0
        or len(scored_rows) != expected_count
        or result_file_count != expected_count
        or valid_result_count != expected_count
        or completed_result_count != expected_count
    )
    infra_dominated = expected_count > 0 and infra > expected_count / 2
    harness_wide_failure, harness_wide_failure_signature = _harness_wide_failure(
        rows,
        expected_count,
    )
    acceptance = manifest.get("execution_acceptance")
    acceptance = acceptance if isinstance(acceptance, dict) else {}
    accepted_value = acceptance.get("accepted")
    if isinstance(accepted_value, bool):
        run_accepted = accepted_value
    else:
        invalid_execution_kinds = {
            "harness_error",
            "infra_error",
            "verifier_error",
        }
        run_accepted = not (
            expected_count > 0
            and len(execution_kinds) == expected_count
            and all(kind in invalid_execution_kinds for kind in execution_kinds)
        )
    if verifier_contract_violations:
        run_accepted = False
    run_acceptance_reason = str(acceptance.get("reason") or "")
    if verifier_contract_violations:
        run_acceptance_reason = "verifier_reward_contract_violation"
    elif not run_accepted and not run_acceptance_reason:
        run_acceptance_reason = "all_trials_harness_or_infrastructure_errors"
    native_run = (
        manifest.get("runner") == "shellbench-native"
        or any(row.get("source") == "shellbench-native" for row in rows)
    )
    agent_attempt_rows = [
        row
        for row in scored_rows
        if row.get("_valid_result") and row.get("classification") != "infra"
    ]
    # Forced agent exits can retain raw logs without producing a final
    # structured trajectory. They remain scored failures, not run invalidators.
    trajectory_required_rows = [
        row
        for row in agent_attempt_rows
        if row.get("classification") != "agent_exit"
    ]
    identity_rows = [
        row
        for row in agent_attempt_rows
        if row.get("classification") != "agent_exit"
        or row.get("trajectory_status") == "real"
    ]
    canonical_model_identity = not native_run or (
        bool(identity_rows)
        and all(
            row.get("canonical_model_identity") is True
            for row in identity_rows
        )
    )
    trajectory_complete = not native_run or (
        manifest.get("trajectory_mode") == "real_harness_events"
        and all(
            row.get("trajectory_status") == "real"
            for row in trajectory_required_rows
        )
    )
    (
        parity_validated,
        parity_validation_status,
        parity_validation_scope,
    ) = _parity_validation(manifest, native_run=native_run)
    eligible = (
        not incomplete
        and run_accepted
        and not infra_dominated
        and not harness_wide_failure
        and canonical_model_identity
        and trajectory_complete
    )
    if incomplete:
        exclusion_reason = "incomplete"
    elif infra_dominated:
        exclusion_reason = "infra_dominated"
    elif not run_accepted:
        exclusion_reason = "run_rejected"
    elif harness_wide_failure:
        exclusion_reason = "harness_wide_failure"
    elif not canonical_model_identity:
        exclusion_reason = "canonical_model_identity_not_preserved"
    elif not trajectory_complete:
        exclusion_reason = "trajectory_unavailable"
    else:
        exclusion_reason = ""
    if manifest.get("leaderboard_eligible") is False:
        eligible = False
        exclusion_reason = (
            str(manifest.get("exclusion_reason") or "") or "explicitly_excluded"
        )

    return {
        "run_label": run_label,
        "pair_label": pair_label,
        "repetition": repetition,
        "expected_task_count": expected_count,
        "result_file_count": result_file_count,
        "valid_result_count": valid_result_count,
        "completed_result_count": completed_result_count,
        "coverage": coverage,
        "score": score,
        "exact_passes": exact_passes,
        "partials": partials,
        "nonzero": nonzero,
        "infra": infra,
        "agent_exits": agent_exits,
        "clean_completed": clean_completed,
        "missing_reward": missing_reward,
        "verifier_contract_violations": verifier_contract_violations,
        "clean_coverage": clean_coverage,
        "incomplete": incomplete,
        "infra_dominated": infra_dominated,
        "harness_wide_failure": harness_wide_failure,
        "harness_wide_failure_signature": harness_wide_failure_signature,
        "run_accepted": run_accepted,
        "run_acceptance_reason": run_acceptance_reason,
        "canonical_model_identity": canonical_model_identity,
        "trajectory_complete": trajectory_complete,
        "parity_validated": parity_validated,
        "parity_validation_status": parity_validation_status,
        "parity_validation_scope": parity_validation_scope,
        "eligible": eligible,
        "exclusion_reason": exclusion_reason,
        "n_input_tokens": _sum_present(scored_rows, "n_input_tokens"),
        "n_cache_tokens": _sum_present(scored_rows, "n_cache_tokens"),
        "n_output_tokens": _sum_present(scored_rows, "n_output_tokens"),
        "cost_usd": _sum_present(scored_rows, "cost_usd"),
    }


def _mean(values: Sequence[int | float]) -> float | None:
    return statistics.mean(values) if values else None


def _pair_summaries(runs: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        grouped[str(run["pair_label"])].append(run)

    summaries: list[dict[str, Any]] = []
    for pair_label, pair_runs in sorted(grouped.items()):
        ordered = sorted(
            pair_runs,
            key=lambda run: (
                run["repetition"] is None,
                run["repetition"] if run["repetition"] is not None else 0,
                run["run_label"],
            ),
        )
        eligible = [run for run in ordered if run["eligible"]]
        parity_validated_runs = [
            run for run in ordered if run["parity_validated"] is True
        ]
        scores = [float(run["score"]) for run in eligible]
        exact_passes = [int(run["exact_passes"]) for run in eligible]
        total_expected = sum(int(run["expected_task_count"]) for run in eligible)
        mean_score = _mean(scores)
        score_stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
        min_score = min(scores) if scores else None
        max_score = max(scores) if scores else None
        summaries.append(
            {
                "pair_label": pair_label,
                "total_repetitions": len(ordered),
                "eligible_repetitions": len(eligible),
                "excluded_repetitions": len(ordered) - len(eligible),
                "excluded_run_labels": [
                    str(run["run_label"]) for run in ordered if not run["eligible"]
                ],
                "parity_validated_repetitions": len(parity_validated_runs),
                "parity_unvalidated_run_labels": [
                    str(run["run_label"])
                    for run in ordered
                    if run["parity_validated"] is not True
                ],
                "mean_score": mean_score,
                "score_stdev": score_stdev,
                "min_score": min_score,
                "max_score": max_score,
                "mean": mean_score,
                "stdev": score_stdev,
                "min": min_score,
                "max": max_score,
                "mean_exact_passes": _mean(exact_passes),
                "pass_rate": (
                    sum(exact_passes) / total_expected if total_expected else None
                ),
                "clean_complete_repetitions": len(eligible),
                "mean_coverage": _mean([float(run["coverage"]) for run in eligible]),
                "mean_clean_completed": _mean(
                    [int(run["clean_completed"]) for run in eligible]
                ),
            }
        )
    return summaries


def _public_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def _repair_source_counts(
    manifest: dict[str, Any],
    expected_task_count: int,
) -> tuple[int, int, int] | None:
    overlay = manifest.get("repair_overlay")
    repair_names = manifest.get("repair_task_names")
    if not isinstance(overlay, dict):
        if not isinstance(repair_names, list) or not repair_names:
            return None
        repaired = len({str(name) for name in repair_names if name})
        return max(expected_task_count - repaired, 0), repaired, 0

    lineage = overlay.get("task_lineage")
    if isinstance(lineage, list):
        source_kinds = [
            str(item.get("source_kind") or "")
            for item in lineage
            if isinstance(item, dict)
        ]
        if source_kinds:
            original = source_kinds.count("canonical")
            repaired = source_kinds.count("repair")
            unknown = len(source_kinds) - original - repaired
            return original, repaired, unknown

    repair_names = overlay.get("repair_task_names", repair_names)
    if isinstance(repair_names, list):
        repaired = len({str(name) for name in repair_names if name})
        return max(expected_task_count - repaired, 0), repaired, 0
    return 0, 0, expected_task_count


def _repair_overlay_sensitivity(
    records: Sequence[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    summaries = {
        str(summary["run_label"]): summary
        for summary, _manifest in records
    }
    runs: list[dict[str, Any]] = []
    for summary, manifest in sorted(
        records,
        key=lambda record: str(record[0]["run_label"]),
    ):
        expected_task_count = int(summary["expected_task_count"])
        source_counts = _repair_source_counts(manifest, expected_task_count)
        if source_counts is None:
            continue
        original_tasks, repaired_tasks, unknown_tasks = source_counts
        overlay = manifest.get("repair_overlay")
        overlay = overlay if isinstance(overlay, dict) else {}
        original_run_label = str(
            overlay.get("rerun_of_canonical_run")
            or manifest.get("rerun_of_canonical_run")
            or ""
        )
        original_summary = summaries.get(original_run_label)
        original_score = (
            float(original_summary["score"])
            if original_summary is not None
            else None
        )
        repaired_score = float(summary["score"])
        if not original_run_label:
            delta_status = "original_label_unavailable"
        elif original_summary is None:
            delta_status = "original_unavailable"
        elif (
            int(original_summary["expected_task_count"])
            != expected_task_count
        ):
            delta_status = "task_count_mismatch"
            original_score = None
        else:
            delta_status = "available"
        runs.append(
            {
                "run_label": summary["run_label"],
                "pair_label": summary["pair_label"],
                "repetition": summary["repetition"],
                "original_run_label": original_run_label,
                "expected_task_count": expected_task_count,
                "original_task_count": original_tasks,
                "repaired_task_count": repaired_tasks,
                "unknown_source_task_count": unknown_tasks,
                "original_result_count": (
                    original_summary["result_file_count"]
                    if original_summary is not None
                    else None
                ),
                "repaired_result_count": summary["result_file_count"],
                "original_score": original_score,
                "repaired_score": repaired_score,
                "score_delta": (
                    repaired_score - original_score
                    if delta_status == "available" and original_score is not None
                    else None
                ),
                "delta_status": delta_status,
            }
        )
    return {
        "schema_version": 1,
        "overlay_run_count": len(runs),
        "runs": runs,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _format_metric(value: Any, digits: int = 4) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _markdown_label(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _write_leaderboard(path: Path, pairs: Sequence[dict[str, Any]]) -> None:
    ranked = sorted(
        (pair for pair in pairs if pair["eligible_repetitions"]),
        key=lambda pair: (
            -(float(pair["mean_score"]) if pair["mean_score"] is not None else -1.0),
            str(pair["pair_label"]),
        ),
    )
    lines = [
        "# Cleaned Native ShellBench Leaderboard",
        "",
        "Incomplete, rejected, infra-dominated, harness-wide failure, identity-invalid, "
        "and trajectory-incomplete repetitions are excluded. Harbor parity validation "
        "is reported separately and is not an eligibility gate.",
        "",
        "| Rank | Pair | Repetitions | Parity validated | Mean | Stdev | Min | Max | Mean exact passes | Pass rate | Clean complete |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, pair in enumerate(ranked, start=1):
        lines.append(
            "| {rank} | {label} | {eligible}/{total} | {parity}/{total} | {mean} | {stdev} | "
            "{minimum} | {maximum} | {passes} | {pass_rate} | {clean} |".format(
                rank=rank,
                label=_markdown_label(pair["pair_label"]),
                eligible=pair["eligible_repetitions"],
                total=pair["total_repetitions"],
                parity=pair["parity_validated_repetitions"],
                mean=_format_metric(pair["mean_score"]),
                stdev=_format_metric(pair["score_stdev"]),
                minimum=_format_metric(pair["min_score"]),
                maximum=_format_metric(pair["max_score"]),
                passes=_format_metric(pair["mean_exact_passes"], 2),
                pass_rate=_format_metric(pair["pass_rate"]),
                clean=pair["clean_complete_repetitions"],
            )
        )
    if not ranked:
        lines.append(
            "| - | No eligible repetitions | 0/0 | 0/0 | - | - | - | - | - | - | 0 |"
        )
    path.write_text("\n".join(lines) + "\n")


def aggregate(jobs_root: str | Path, summaries_dir: str | Path) -> dict[str, Any]:
    """Aggregate ``jobs/<run_label>/<trial>/result.json`` directories.

    Returns the same run and pair data written to ``aggregate_results.json``.
    """

    root = Path(jobs_root)
    jobs_dir = root / "jobs" if (root / "jobs").is_dir() else root
    output_dir = Path(summaries_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not jobs_dir.is_dir():
        raise FileNotFoundError(f"jobs directory does not exist: {jobs_dir}")

    run_summaries: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    run_records: list[tuple[dict[str, Any], dict[str, Any]]] = []
    run_dirs = sorted(
        path
        for path in jobs_dir.iterdir()
        if path.is_dir()
        and path.resolve() != output_dir.resolve()
        and (path / "run_manifest.json").is_file()
    )
    for run_dir in run_dirs:
        summary, rows, manifest = _load_run(run_dir)
        run_summaries.append(summary)
        task_rows.extend(rows)
        run_records.append((summary, manifest))

    run_summaries.sort(key=lambda run: str(run["run_label"]))
    task_rows.sort(
        key=lambda row: (
            str(row["run_label"]),
            str(row["task_name"]),
            str(row["trial_name"]),
        )
    )
    pair_summaries = _pair_summaries(run_summaries)
    repair_sensitivity = _repair_overlay_sensitivity(run_records)
    public_task_rows = _public_rows(task_rows)
    infra_rows = [row for row in public_task_rows if row["classification"] == "infra"]

    report = {
        "taxonomy": {
            "infra": "environment, setup, gateway, Docker, malformed, or missing result",
            "agent_exit": "agent timeout, refusal, nonzero exit, or other agent exception",
            "verifier_missing_reward": (
                "completed result with no reward, including missing reward files"
            ),
            "clean_fail": "clean completion with reward <= 0",
            "partial": "clean completion with 0 < reward < 1",
            "pass": "clean completion with reward >= 1",
        },
        "runs": run_summaries,
        "pairs": pair_summaries,
        "repair_overlay_sensitivity": repair_sensitivity,
    }

    _write_csv(output_dir / "aggregate_results.csv", RUN_FIELDS, run_summaries)
    (output_dir / "aggregate_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(output_dir / "per_task_results.csv", PER_TASK_FIELDS, public_task_rows)
    _write_csv(output_dir / "infra_failures.csv", INFRA_FIELDS, infra_rows)
    _write_csv(
        output_dir / "repair_overlay_sensitivity.csv",
        REPAIR_SENSITIVITY_FIELDS,
        repair_sensitivity["runs"],
    )
    (output_dir / "repair_overlay_sensitivity.json").write_text(
        json.dumps(repair_sensitivity, indent=2, sort_keys=True) + "\n"
    )
    _write_leaderboard(output_dir / "cleaned_leaderboard.md", pair_summaries)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate native ShellBench Harbor job directories."
    )
    parser.add_argument("jobs_root", type=Path, help="Root containing jobs/<run_label>")
    parser.add_argument(
        "summaries_dir",
        type=Path,
        nargs="?",
        help="Output directory (default: <jobs_root>/summaries)",
    )
    args = parser.parse_args(argv)
    summaries_dir = args.summaries_dir or args.jobs_root / "summaries"
    report = aggregate(args.jobs_root, summaries_dir)
    print(
        json.dumps(
            {
                "runs": len(report["runs"]),
                "pairs": len(report["pairs"]),
                "summaries_dir": str(summaries_dir),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
