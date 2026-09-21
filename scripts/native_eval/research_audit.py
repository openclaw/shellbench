"""Export task, turn, tool, discovery, usage, and identity research tables."""

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
    "phase",
    "qualification_family",
    "leaderboard_eligible",
    "task_name",
    "reward",
    "result_path",
    "trajectory_path",
    "toolchain_manifest_path",
    "proxy_log_path",
    "runner_commit",
    "trajectory_status",
    "trace_fidelity",
    "observed_model_ids",
    "model_identity_status",
    "discovery_status",
    "discovery_event_count",
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

DISCOVERY_FIELDS = (
    "run_label",
    "harness",
    "model_slug",
    "expected_model_id",
    "reasoning_effort",
    "repetition",
    "task_name",
    "turn_index",
    "tool_call_id",
    "counter_scope",
    "operation",
    "count",
    "count_semantics",
    "query",
    "selected_id",
    "catalog_size",
    "source_breakdown_json",
    "success",
    "trace_fidelity",
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
    "phase",
    "qualification_family",
    "leaderboard_eligible",
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


def _json_values(text: str) -> list[Any]:
    stripped = text.strip()
    if not stripped:
        return []
    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass
    values: list[Any] = []
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _find_tool_search_telemetry(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        telemetry = value.get("telemetry")
        if isinstance(telemetry, dict) and any(
            key in telemetry for key in ("catalogSize", "searchCount", "describeCount", "callCount")
        ):
            return telemetry
        for child in value.values():
            found = _find_tool_search_telemetry(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_tool_search_telemetry(child)
            if found is not None:
                return found
    return None


def _openclaw_code_discovery_rows(
    *,
    base: dict[str, Any],
    observation: str,
) -> tuple[list[dict[str, Any]], bool]:
    telemetry = next(
        (
            found
            for value in _json_values(observation)
            if (found := _find_tool_search_telemetry(value)) is not None
        ),
        None,
    )
    if telemetry is None:
        return [], False
    sources = telemetry.get("sources")
    source_breakdown = sources if isinstance(sources, dict) else {}
    scope_value = telemetry.get("counterScope")
    counter_scope = scope_value.strip() if isinstance(scope_value, str) else ""
    rows: list[dict[str, Any]] = []
    for operation, key in (
        ("search", "searchCount"),
        ("describe", "describeCount"),
        ("call", "callCount"),
    ):
        count = _number(telemetry.get(key))
        if count is None or (count == 0 and not counter_scope):
            continue
        count_semantics = (
            "invalid_counter_scope"
            if count < 0 and not counter_scope
            else (
                "cumulative_scoped"
                if counter_scope
                else "cumulative_unscoped"
            )
        )
        rows.append(
            {
                **base,
                "counter_scope": counter_scope,
                "operation": operation,
                "count": count,
                "count_semantics": count_semantics,
                "query": "",
                "selected_id": "",
                "catalog_size": _number(telemetry.get("catalogSize")),
                "source_breakdown_json": json.dumps(
                    source_breakdown,
                    ensure_ascii=True,
                    sort_keys=True,
                ),
                "success": "",
            }
        )
    return rows, True


def _structured_discovery_row(
    *,
    base: dict[str, Any],
    function_name: str,
    arguments: Any,
) -> dict[str, Any] | None:
    operation = {
        "tool_search": "search",
        "tool_describe": "describe",
        "tool_call": "call",
    }.get(function_name)
    if operation is None:
        return None
    values = arguments if isinstance(arguments, dict) else {}
    return {
        **base,
        "counter_scope": "",
        "operation": operation,
        "count": 1,
        "count_semantics": "event",
        "query": values.get("query") if operation == "search" else "",
        "selected_id": (
            values.get("id") or values.get("toolId") or values.get("name")
            if operation in {"describe", "call"}
            else ""
        ),
        "catalog_size": "",
        "source_breakdown_json": "{}",
        "success": "",
    }


def _discovery_status(
    *,
    harness: str,
    openclaw_mode: str,
    tool_names: list[str],
    discovery_rows: list[dict[str, Any]],
    telemetry_observed: bool,
) -> str:
    if harness == "openclaw":
        if any(
            row.get("count_semantics") == "invalid_counter_scope"
            for row in discovery_rows
        ):
            return "invalid_counter_scope"
        if any(
            row.get("count_semantics") == "cumulative_unscoped"
            for row in discovery_rows
        ):
            return "observed_cumulative_unscoped"
        if discovery_rows and all(
            row.get("count_semantics") == "scope_marker"
            for row in discovery_rows
        ):
            return "supported_not_exercised"
        if discovery_rows:
            return "observed"
        if telemetry_observed:
            return "supported_not_exercised"
        if any(
            name
            in {
                "tool_search_code",
                "tool_search",
                "tool_describe",
                "tool_call",
            }
            for name in tool_names
        ):
            return "unobservable"
        return "supported_not_exercised" if openclaw_mode else "disabled"
    if harness == "codex":
        return "unobservable_native_stream"
    return "unsupported"


def _discovery_operation_count(rows: list[dict[str, Any]]) -> int:
    return sum(
        int(_number(row.get("count")) or 0)
        for row in rows
        if row.get("count_semantics")
        not in {
            "cumulative_scoped",
            "cumulative_unscoped",
            "invalid_counter_scope",
        }
    )


def _normalize_cumulative_discovery_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert scoped cumulative telemetry into deltas without guessing resets."""
    previous_counts: dict[tuple[str, str], int] = {}
    normalized_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    invalid_keys: set[tuple[str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for row in rows:
        semantics = row.get("count_semantics")
        if semantics != "cumulative_scoped":
            normalized.append(row)
            continue
        scope = str(row.get("counter_scope") or "")
        operation = str(row.get("operation") or "")
        count = int(_number(row.get("count")) or 0)
        key = (scope, operation)
        if key in invalid_keys:
            normalized.append({**row, "count_semantics": "invalid_counter_scope"})
            continue
        delta = count - previous_counts.get(key, 0)
        previous_counts[key] = count
        if delta < 0:
            invalid_keys.add(key)
            for prior_row in normalized_rows.get(key, []):
                prior_row["count_semantics"] = "invalid_counter_scope"
            invalid_row = {**row, "count_semantics": "invalid_counter_scope"}
            normalized.append(invalid_row)
        elif delta > 0:
            delta_row = {**row, "count": delta, "count_semantics": "delta"}
            normalized_rows.setdefault(key, []).append(delta_row)
            normalized.append(delta_row)
        else:
            marker_row = {
                **row,
                "count": 0,
                "count_semantics": "scope_marker",
            }
            normalized_rows.setdefault(key, []).append(marker_row)
            normalized.append(marker_row)
    return normalized


def _invalidate_cross_task_counter_scopes(
    rows: list[dict[str, Any]],
) -> None:
    owners_by_scope: dict[tuple[str, str], set[tuple[str, str]]] = {}
    rows_by_scope: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        scope = str(row.get("counter_scope") or "")
        if not scope:
            continue
        scope_key = (str(row.get("run_label") or ""), scope)
        owner = (
            str(row.get("task_name") or ""),
            str(row.get("trajectory_path") or ""),
        )
        owners_by_scope.setdefault(scope_key, set()).add(owner)
        rows_by_scope.setdefault(scope_key, []).append(row)
    for scope_key, owners in owners_by_scope.items():
        if len(owners) < 2:
            continue
        for row in rows_by_scope[scope_key]:
            row["count_semantics"] = "invalid_counter_scope"


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
    discovery_rows: list[dict[str, Any]] = []
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
        job_manifest = _read_json(job_dir / "run_manifest.json") if job_dir else {}
        job_manifest = job_manifest or {}
        runner_commit = str(job_manifest.get("runner_commit") or "")
        openclaw_mode = str(
            entry.get("openclaw_tool_search_mode")
            or job_manifest.get("openclaw_tool_search_mode")
            or ""
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
            trace_fidelity = str(_nested(trajectory, "extra", "trace_fidelity") or "")
            input_tokens, cache_tokens, output_tokens = _usage(agent_result, trajectory)
            cost, cost_provenance = _cost(agent_result, trajectory)
            counters[
                "task_cost_exact"
                if cost is not None
                else "task_cost_unavailable"
            ] += 1
            task_tool_count = 0
            task_tool_names: list[str] = []
            task_discovery_rows: list[dict[str, Any]] = []
            task_discovery_telemetry_observed = False

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
                    function_name = str(call_value.get("function_name") or "")
                    task_tool_names.append(function_name)
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
                            "function_name": function_name,
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
                    discovery_base = {
                        "run_label": run_label,
                        "harness": harness,
                        "model_slug": entry.get("model_slug"),
                        "expected_model_id": expected_model_id,
                        "reasoning_effort": entry.get("reasoning_effort"),
                        "repetition": entry.get("repetition"),
                        "task_name": task_name,
                        "turn_index": turn_index,
                        "tool_call_id": call_id,
                        "trace_fidelity": trace_fidelity,
                        "trajectory_path": str(trajectory_path),
                    }
                    if harness == "openclaw" and function_name == "tool_search_code":
                        code_rows, telemetry_observed = _openclaw_code_discovery_rows(
                            base=discovery_base,
                            observation=observation,
                        )
                        task_discovery_rows.extend(code_rows)
                        task_discovery_telemetry_observed |= telemetry_observed
                    elif harness == "openclaw":
                        discovery_row = _structured_discovery_row(
                            base=discovery_base,
                            function_name=function_name,
                            arguments=arguments,
                        )
                        if discovery_row is not None:
                            task_discovery_rows.append(discovery_row)

            if harness == "openclaw" and openclaw_mode == "code":
                task_discovery_rows = _normalize_cumulative_discovery_rows(
                    task_discovery_rows
                )
            discovery_status = _discovery_status(
                harness=harness,
                openclaw_mode=openclaw_mode,
                tool_names=task_tool_names,
                discovery_rows=task_discovery_rows,
                telemetry_observed=task_discovery_telemetry_observed,
            )
            task_discovery_operation_count = _discovery_operation_count(
                task_discovery_rows
            )
            discovery_rows.extend(task_discovery_rows)

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
                    "phase": entry.get("phase") or "full",
                    "qualification_family": entry.get("qualification_family"),
                    "leaderboard_eligible": entry.get("leaderboard_eligible"),
                    "task_name": task_name,
                    "reward": _reward(result),
                    "result_path": str(result_path),
                    "trajectory_path": str(trajectory_path),
                    "toolchain_manifest_path": (
                        str(toolchain_path) if toolchain_path else ""
                    ),
                    "proxy_log_path": str(proxy_log_path) if proxy_log_path else "",
                    "runner_commit": runner_commit,
                    "trajectory_status": agent_result.get("trajectory_status"),
                    "trace_fidelity": trace_fidelity,
                    "observed_model_ids": json.dumps(sorted(observed)),
                    "model_identity_status": identity_status,
                    "discovery_status": discovery_status,
                    "discovery_event_count": task_discovery_operation_count,
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
                "phase": entry.get("phase") or "full",
                "qualification_family": entry.get("qualification_family"),
                "leaderboard_eligible": entry.get("leaderboard_eligible"),
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

    _invalidate_cross_task_counter_scopes(discovery_rows)
    discovery_rows_by_trace: dict[
        tuple[str, str, str], list[dict[str, Any]]
    ] = {}
    for row in discovery_rows:
        key = (
            str(row.get("run_label") or ""),
            str(row.get("task_name") or ""),
            str(row.get("trajectory_path") or ""),
        )
        discovery_rows_by_trace.setdefault(key, []).append(row)
    for trace_row in trace_rows:
        key = (
            str(trace_row.get("run_label") or ""),
            str(trace_row.get("task_name") or ""),
            str(trace_row.get("trajectory_path") or ""),
        )
        task_rows = discovery_rows_by_trace.get(key, [])
        if not task_rows:
            continue
        trace_row["discovery_status"] = _discovery_status(
            harness=str(trace_row.get("harness") or ""),
            openclaw_mode=str(trace_row.get("openclaw_tool_search_mode") or ""),
            tool_names=[],
            discovery_rows=task_rows,
            telemetry_observed=True,
        )
        trace_row["discovery_event_count"] = _discovery_operation_count(task_rows)

    _write_csv(output_dir / "trace_inventory.csv", TRACE_FIELDS, trace_rows)
    _write_csv(output_dir / "turn_usage.csv", TURN_FIELDS, turn_rows)
    _write_csv(output_dir / "tool_calls.csv", TOOL_FIELDS, tool_rows)
    _write_csv(output_dir / "discovery_events.csv", DISCOVERY_FIELDS, discovery_rows)
    _write_csv(output_dir / "model_identity_audit.csv", RUN_AUDIT_FIELDS, run_rows)
    summary = {
        "run_count": len(run_rows),
        "task_result_count": len(trace_rows),
        "turn_count": len(turn_rows),
        "tool_call_count": len(tool_rows),
        "discovery_event_count": _discovery_operation_count(discovery_rows),
        "identity_audit_pass_count": sum(
            row["model_identity_audit_passed"] is True for row in run_rows
        ),
        "identity_audit_fail_count": sum(
            row["model_identity_audit_passed"] is not True for row in run_rows
        ),
        "r0_run_count": sum(row["phase"] == "r0" for row in run_rows),
        "scoring_run_count": sum(
            row["leaderboard_eligible"] is not False for row in run_rows
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
            "discovery_events": "discovery_events.csv",
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
