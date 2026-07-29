"""Export task, turn, tool, usage, and model-identity research tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


TRACE_FIELDS = (
    "run_label",
    "harness",
    "harness_version",
    "installed_harness_version",
    "harness_version_status",
    "model_slug",
    "expected_model_id",
    "reasoning_effort",
    "judge_model_id",
    "judge_reasoning_effort",
    "judge_identity_status",
    "repetition",
    "task_name",
    "reward",
    "result_path",
    "trajectory_path",
    "toolchain_manifest_path",
    "proxy_log_path",
    "trajectory_status",
    "observed_model_ids",
    "model_identity_status",
    "turn_count",
    "tool_call_count",
    "n_input_tokens",
    "n_cache_tokens",
    "n_output_tokens",
    "cost_usd",
    "cost_provenance",
)

TURN_FIELDS = (
    "run_label",
    "harness",
    "model_slug",
    "expected_model_id",
    "reasoning_effort",
    "repetition",
    "task_name",
    "turn_index",
    "source",
    "timestamp",
    "message_chars",
    "reasoning_chars",
    "tool_call_count",
    "n_input_tokens",
    "n_cache_tokens",
    "n_output_tokens",
    "cost_usd",
    "cost_provenance",
    "trajectory_path",
)

TOOL_FIELDS = (
    "run_label",
    "harness",
    "model_slug",
    "expected_model_id",
    "reasoning_effort",
    "repetition",
    "task_name",
    "turn_index",
    "tool_index",
    "tool_call_id",
    "function_name",
    "arguments_json",
    "observation_chars",
    "observation_excerpt",
    "trajectory_path",
)

RUN_AUDIT_FIELDS = (
    "run_label",
    "harness",
    "harness_version",
    "installed_harness_version",
    "harness_version_status",
    "model_slug",
    "expected_model_id",
    "reasoning_effort",
    "judge_model_id",
    "judge_reasoning_effort",
    "judge_identity_status",
    "repetition",
    "expected_task_count",
    "result_count",
    "real_trace_count",
    "identity_match_count",
    "identity_mismatch_count",
    "identity_not_observed_count",
    "trace_missing_count",
    "model_identity_audit_passed",
    "toolchain_manifest_path",
    "proxy_log_path",
    "task_cost_exact_count",
    "task_cost_unavailable_count",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _number(value: Any) -> int | float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _first_number(mapping: dict[str, Any], keys: Iterable[str]) -> int | float | None:
    for key in keys:
        value = _number(mapping.get(key))
        if value is not None:
            return value
    return None


def _usage(
    agent_result: dict[str, Any],
    trajectory: dict[str, Any],
) -> tuple[int | float | None, int | float | None, int | float | None]:
    metrics = trajectory.get("final_metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return (
        _first_number(
            agent_result,
            ("n_input_tokens", "input_tokens", "prompt_tokens"),
        )
        or _first_number(metrics, ("total_prompt_tokens", "prompt_tokens")),
        _first_number(
            agent_result,
            ("n_cache_tokens", "cache_tokens", "cached_tokens"),
        )
        or _first_number(metrics, ("total_cached_tokens", "cached_tokens")),
        _first_number(
            agent_result,
            ("n_output_tokens", "output_tokens", "completion_tokens"),
        )
        or _first_number(metrics, ("total_completion_tokens", "completion_tokens")),
    )


def _cost(
    agent_result: dict[str, Any],
    trajectory: dict[str, Any],
) -> tuple[int | float | None, str]:
    cost = _first_number(
        agent_result,
        ("cost_usd", "total_cost_usd", "total_cost"),
    )
    if cost is not None:
        return cost, "exact_harness"
    metrics = trajectory.get("final_metrics")
    if isinstance(metrics, dict):
        cost = _first_number(metrics, ("total_cost_usd", "cost_usd", "total_cost"))
    if cost is not None:
        return cost, "exact_trace"
    return None, "unavailable_without_provider_spend_or_pricing_snapshot"


def _step_usage(step: dict[str, Any]) -> tuple[Any, Any, Any, Any, str]:
    metrics: dict[str, Any] = {}
    for key in ("usage", "token_usage", "metrics"):
        candidate = step.get(key)
        if isinstance(candidate, dict):
            metrics.update(candidate)
    input_tokens = _first_number(
        metrics,
        ("input_tokens", "prompt_tokens", "total_prompt_tokens"),
    )
    cache_tokens = _first_number(
        metrics,
        ("cache_read_input_tokens", "cached_tokens", "total_cached_tokens"),
    )
    output_tokens = _first_number(
        metrics,
        ("output_tokens", "completion_tokens", "total_completion_tokens"),
    )
    cost = _first_number(metrics, ("cost_usd", "total_cost_usd", "total_cost"))
    provenance = (
        "exact_trace_turn"
        if cost is not None
        else "unavailable_without_per_request_spend"
    )
    return input_tokens, cache_tokens, output_tokens, cost, provenance


def _task_name(result: dict[str, Any], result_path: Path) -> str:
    task_id = result.get("task_id")
    if isinstance(task_id, dict):
        path = task_id.get("path")
        if isinstance(path, str) and path:
            return Path(path).name
        name = task_id.get("name")
        if isinstance(name, str) and name:
            return name
    if isinstance(task_id, str) and task_id:
        return Path(task_id).name
    return result_path.parent.name


def _reward(result: dict[str, Any]) -> int | float | None:
    return _number(_nested(result, "verifier_result", "rewards", "reward"))


def _trajectory_path(result_path: Path, result: dict[str, Any]) -> Path:
    configured = _nested(result, "agent_result", "trajectory_path")
    if isinstance(configured, str) and configured:
        path = Path(configured)
        if path.is_file():
            return path
        relative = result_path.parent / path
        if relative.is_file():
            return relative
    return result_path.parent / "agent" / "trajectory.json"


def _observed_models(
    agent_result: dict[str, Any],
    trajectory: dict[str, Any],
) -> set[str]:
    observed: set[str] = set()
    runtime_model = agent_result.get("runtime_model_name")
    if isinstance(runtime_model, str) and runtime_model:
        observed.add(runtime_model)
    values = _nested(trajectory, "extra", "observed_models")
    if isinstance(values, list):
        observed.update(str(value) for value in values if value)
    model_name = _nested(trajectory, "agent", "model_name")
    if isinstance(model_name, str) and model_name:
        observed.add(model_name.rsplit("/", 1)[-1])
    return observed


def _identity_status(
    *,
    expected_model_id: str,
    observed_models: set[str],
    agent_result: dict[str, Any],
    trajectory_exists: bool,
) -> str:
    if not trajectory_exists:
        return "trace_missing"
    if not observed_models:
        return "not_observed"
    if observed_models == {expected_model_id} and (
        agent_result.get("canonical_model_identity") is not False
    ):
        return "match"
    return "mismatch"


def _observation(step: dict[str, Any], call_id: str) -> str:
    results = _nested(step, "observation", "results")
    if not isinstance(results, list):
        return ""
    matching = [
        item
        for item in results
        if isinstance(item, dict)
        and (not call_id or str(item.get("source_call_id") or "") == call_id)
    ]
    selected = matching or [item for item in results if isinstance(item, dict)]
    return "\n".join(str(item.get("content") or "") for item in selected)


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _job_directories(extracted_root: Path) -> dict[str, Path]:
    directories: dict[str, Path] = {}
    for manifest_path in extracted_root.rglob("run_manifest.json"):
        manifest = _read_json(manifest_path)
        label = str((manifest or {}).get("run_label") or "")
        if label:
            directories[label] = manifest_path.parent
    return directories


def _labeled_files(
    extracted_root: Path,
    filename: str,
    *,
    label_from_parent_prefix: str | None = None,
) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in extracted_root.rglob(filename):
        parent_name = path.parent.name
        if label_from_parent_prefix:
            if not parent_name.startswith(label_from_parent_prefix):
                continue
            label = parent_name.removeprefix(label_from_parent_prefix)
        else:
            label = parent_name
        if label:
            files[label] = path
    return files


def _installed_harness_version(
    toolchain: dict[str, Any],
    harness: str,
) -> str:
    if harness == "hermes":
        return str(toolchain.get("hermes_commit") or toolchain.get("hermes") or "")
    key = {
        "openclaw": "openclaw",
        "codex": "codex",
        "claude-code": "claude_code",
    }.get(harness, harness)
    return str(toolchain.get(key) or "")


def _version_status(requested: str, installed: str) -> str:
    if not installed:
        return "missing"
    if installed == requested or requested in installed:
        return "match"
    return "mismatch"


def export_research_tables(
    *,
    run_index_path: Path,
    extracted_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    run_index = _read_json(run_index_path)
    if run_index is None or not isinstance(run_index.get("runs"), list):
        raise ValueError(f"invalid run index: {run_index_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = _job_directories(extracted_root)
    toolchain_paths = _labeled_files(
        extracted_root,
        "toolchain_manifest.json",
        label_from_parent_prefix="shellbench_meta-",
    )
    proxy_logs = _labeled_files(extracted_root, "proxy.log")
    trace_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    tool_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for entry_value in run_index["runs"]:
        if not isinstance(entry_value, dict):
            continue
        entry = entry_value
        run_label = str(entry.get("run_label") or "")
        job_dir = jobs.get(run_label)
        toolchain_path = toolchain_paths.get(run_label)
        toolchain = _read_json(toolchain_path) if toolchain_path else {}
        toolchain = toolchain or {}
        harness = str(entry.get("harness") or "")
        requested_harness_version = str(entry.get("harness_version") or "")
        installed_harness_version = _installed_harness_version(toolchain, harness)
        harness_version_status = _version_status(
            requested_harness_version,
            installed_harness_version,
        )
        proxy_log_path = proxy_logs.get(run_label)
        counters: Counter[str] = Counter()
        if job_dir is not None:
            result_paths = sorted(
                path for path in job_dir.rglob("result.json") if path != job_dir / "result.json"
            )
        else:
            result_paths = []

        for result_path in result_paths:
            result = _read_json(result_path)
            if result is None:
                continue
            task_name = _task_name(result, result_path)
            agent_result = result.get("agent_result")
            if not isinstance(agent_result, dict):
                agent_result = {}
            trajectory_path = _trajectory_path(result_path, result)
            trajectory = _read_json(trajectory_path) or {}
            trajectory_exists = bool(trajectory)
            observed = _observed_models(agent_result, trajectory)
            expected_model_id = str(entry.get("model_id") or "")
            identity_status = _identity_status(
                expected_model_id=expected_model_id,
                observed_models=observed,
                agent_result=agent_result,
                trajectory_exists=trajectory_exists,
            )
            counters[identity_status] += 1
            if trajectory_exists:
                counters["real_trace"] += 1
            steps = trajectory.get("steps")
            if not isinstance(steps, list):
                steps = []
            input_tokens, cache_tokens, output_tokens = _usage(agent_result, trajectory)
            cost, cost_provenance = _cost(agent_result, trajectory)
            counters[
                "task_cost_exact"
                if cost is not None
                else "task_cost_unavailable"
            ] += 1
            task_tool_count = 0

            for turn_index, step_value in enumerate(steps):
                if not isinstance(step_value, dict):
                    continue
                step = step_value
                tool_calls = step.get("tool_calls")
                if not isinstance(tool_calls, list):
                    tool_calls = []
                task_tool_count += len(tool_calls)
                turn_usage = _step_usage(step)
                turn_rows.append(
                    {
                        "run_label": run_label,
                        "harness": entry.get("harness"),
                        "model_slug": entry.get("model_slug"),
                        "expected_model_id": expected_model_id,
                        "reasoning_effort": entry.get("reasoning_effort"),
                        "repetition": entry.get("repetition"),
                        "task_name": task_name,
                        "turn_index": turn_index,
                        "source": step.get("source"),
                        "timestamp": step.get("timestamp"),
                        "message_chars": len(str(step.get("message") or "")),
                        "reasoning_chars": len(str(step.get("reasoning_content") or "")),
                        "tool_call_count": len(tool_calls),
                        "n_input_tokens": turn_usage[0],
                        "n_cache_tokens": turn_usage[1],
                        "n_output_tokens": turn_usage[2],
                        "cost_usd": turn_usage[3],
                        "cost_provenance": turn_usage[4],
                        "trajectory_path": str(trajectory_path),
                    }
                )
                for tool_index, call_value in enumerate(tool_calls):
                    if not isinstance(call_value, dict):
                        continue
                    call_id = str(call_value.get("tool_call_id") or "")
                    observation = _observation(step, call_id)
                    arguments = call_value.get("arguments")
                    tool_rows.append(
                        {
                            "run_label": run_label,
                            "harness": entry.get("harness"),
                            "model_slug": entry.get("model_slug"),
                            "expected_model_id": expected_model_id,
                            "reasoning_effort": entry.get("reasoning_effort"),
                            "repetition": entry.get("repetition"),
                            "task_name": task_name,
                            "turn_index": turn_index,
                            "tool_index": tool_index,
                            "tool_call_id": call_id,
                            "function_name": call_value.get("function_name"),
                            "arguments_json": json.dumps(
                                arguments,
                                ensure_ascii=True,
                                sort_keys=True,
                            ),
                            "observation_chars": len(observation),
                            "observation_excerpt": observation[:500],
                            "trajectory_path": str(trajectory_path),
                        }
                    )

            trace_rows.append(
                {
                    "run_label": run_label,
                    "harness": harness,
                    "harness_version": requested_harness_version,
                    "installed_harness_version": installed_harness_version,
                    "harness_version_status": harness_version_status,
                    "model_slug": entry.get("model_slug"),
                    "expected_model_id": expected_model_id,
                    "reasoning_effort": entry.get("reasoning_effort"),
                    "judge_model_id": entry.get("judge_model_id"),
                    "judge_reasoning_effort": entry.get("judge_reasoning_effort"),
                    "judge_identity_status": (
                        "unverified_requires_proxy_request_evidence"
                    ),
                    "repetition": entry.get("repetition"),
                    "task_name": task_name,
                    "reward": _reward(result),
                    "result_path": str(result_path),
                    "trajectory_path": str(trajectory_path),
                    "toolchain_manifest_path": (
                        str(toolchain_path) if toolchain_path else ""
                    ),
                    "proxy_log_path": str(proxy_log_path) if proxy_log_path else "",
                    "trajectory_status": agent_result.get("trajectory_status"),
                    "observed_model_ids": json.dumps(sorted(observed)),
                    "model_identity_status": identity_status,
                    "turn_count": len(steps),
                    "tool_call_count": task_tool_count,
                    "n_input_tokens": input_tokens,
                    "n_cache_tokens": cache_tokens,
                    "n_output_tokens": output_tokens,
                    "cost_usd": cost,
                    "cost_provenance": cost_provenance,
                }
            )

        result_count = len(result_paths)
        passed = result_count > 0 and (
            counters["match"] == result_count
            and counters["real_trace"] == result_count
        )
        run_rows.append(
            {
                "run_label": run_label,
                "harness": harness,
                "harness_version": requested_harness_version,
                "installed_harness_version": installed_harness_version,
                "harness_version_status": harness_version_status,
                "model_slug": entry.get("model_slug"),
                "expected_model_id": entry.get("model_id"),
                "reasoning_effort": entry.get("reasoning_effort"),
                "judge_model_id": entry.get("judge_model_id"),
                "judge_reasoning_effort": entry.get("judge_reasoning_effort"),
                "judge_identity_status": (
                    "unverified_requires_proxy_request_evidence"
                ),
                "repetition": entry.get("repetition"),
                "expected_task_count": entry.get("expected_task_count"),
                "result_count": result_count,
                "real_trace_count": counters["real_trace"],
                "identity_match_count": counters["match"],
                "identity_mismatch_count": counters["mismatch"],
                "identity_not_observed_count": counters["not_observed"],
                "trace_missing_count": counters["trace_missing"],
                "model_identity_audit_passed": passed,
                "toolchain_manifest_path": (
                    str(toolchain_path) if toolchain_path else ""
                ),
                "proxy_log_path": str(proxy_log_path) if proxy_log_path else "",
                "task_cost_exact_count": counters["task_cost_exact"],
                "task_cost_unavailable_count": counters["task_cost_unavailable"],
            }
        )

    _write_csv(output_dir / "trace_inventory.csv", TRACE_FIELDS, trace_rows)
    _write_csv(output_dir / "turn_usage.csv", TURN_FIELDS, turn_rows)
    _write_csv(output_dir / "tool_calls.csv", TOOL_FIELDS, tool_rows)
    _write_csv(output_dir / "model_identity_audit.csv", RUN_AUDIT_FIELDS, run_rows)
    summary = {
        "run_count": len(run_rows),
        "task_result_count": len(trace_rows),
        "turn_count": len(turn_rows),
        "tool_call_count": len(tool_rows),
        "identity_audit_pass_count": sum(
            row["model_identity_audit_passed"] is True for row in run_rows
        ),
        "identity_audit_fail_count": sum(
            row["model_identity_audit_passed"] is not True for row in run_rows
        ),
        "exact_task_cost_count": sum(
            row["cost_provenance"] in {"exact_harness", "exact_trace"}
            for row in trace_rows
        ),
        "unavailable_task_cost_count": sum(
            row["cost_usd"] is None for row in trace_rows
        ),
        "outputs": {
            "trace_inventory": "trace_inventory.csv",
            "turn_usage": "turn_usage.csv",
            "tool_calls": "tool_calls.csv",
            "model_identity_audit": "model_identity_audit.csv",
        },
    }
    (output_dir / "research_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    summary = export_research_tables(
        run_index_path=args.run_index,
        extracted_root=args.extracted_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
