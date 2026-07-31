from __future__ import annotations

import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.native_eval.models import RunSpec


OpenClawSessionTrace = tuple[str, Path, int, list[dict[str, Any]]]
_OPENCLAW_FATAL_EXPORT_WARNINGS = {
    "cyclic-session-branch",
    "incomplete-session-branch",
}
_OPENCLAW_CODE_MODE_HIDDEN_TOOLS = {
    "tool_call",
    "tool_describe",
    "tool_search",
    "tool_search_code",
}


def load_openclaw_envelope(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    decoder = json.JSONDecoder()
    for start in range(len(text) - 1, -1, -1):
        if text[start] != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[start:])
        except (json.JSONDecodeError, ValueError):
            continue
        if (
            isinstance(value, dict)
            and isinstance(value.get("payloads"), list)
            and isinstance(value.get("meta"), dict)
        ):
            return value
    return None


def write_openclaw_trajectory(
    instruction: str,
    run: RunSpec,
    agent_dir: Path,
) -> dict[str, Any]:
    log_path = agent_dir / "openclaw.txt"
    session_path = _openclaw_root_session_path(agent_dir)
    envelope = load_openclaw_envelope(log_path)
    meta = envelope.get("meta") if envelope else {}
    if not isinstance(meta, dict):
        meta = {}
    agent_meta = meta.get("agentMeta")
    if not isinstance(agent_meta, dict):
        agent_meta = {}
    envelope_models = {
        value
        for value in (
            agent_meta.get("model"),
            (meta.get("executionTrace") or {}).get("winnerModel")
            if isinstance(meta.get("executionTrace"), dict)
            else None,
        )
        if isinstance(value, str) and value
    }
    session_tree, session_tree_validation = _openclaw_session_tree(
        agent_dir,
        session_path,
    )
    export_metadata = _openclaw_export_metadata(session_path)
    root_records = session_tree[0][3] if session_tree else None
    session_models = _openclaw_session_models(session_path, records=root_records)
    export_model = export_metadata.get("export_model")
    if isinstance(export_model, str) and export_model:
        session_models.add(export_model)
    log_models = _openclaw_log_models(log_path)
    child_models = {
        result["resolved_model"]
        for _, path, _, records in session_tree
        for result in _openclaw_spawn_results(path, records=records)
        if result.get("resolved_model")
    } | {
        model
        for _, path, _, records in session_tree[1:]
        for model in _openclaw_session_models(path, records=records)
    }
    parent_models = {
        _normalize_observed_model(value, run) for value in session_models | envelope_models
    }
    normalized_log_models = {_normalize_observed_model(value, run) for value in log_models}
    normalized_child_models = {_normalize_observed_model(value, run) for value in child_models}
    observed_models = parent_models | normalized_log_models | normalized_child_models
    runtime_model_name = (
        next(iter(parent_models))
        if len(parent_models) == 1
        else run.model_id
        if run.model_id in parent_models
        else None
    )
    canonical_model_identity = bool(observed_models) and observed_models == {run.model_id}
    terminal_event_seen = _openclaw_session_terminal(
        session_path,
        records=root_records,
    )
    if export_metadata:
        terminal_event_seen = export_metadata.get("export_terminal_event_seen") is True
    elif not terminal_event_seen and envelope is not None:
        terminal_event_seen = _openclaw_envelope_terminal(envelope)
    visible_tools = export_metadata.get("export_visible_tools")
    provider_transcript_used = (
        export_metadata.get("export_branch_used") is True
        or export_metadata.get("export_snapshot_used") is True
    )
    tool_mode_observed = (
        export_metadata.get("export_provider_visible_tools_recorded") is True
        and isinstance(visible_tools, list)
        and {"exec", "wait"} <= set(visible_tools)
        and not _OPENCLAW_CODE_MODE_HIDDEN_TOOLS & set(visible_tools)
        and export_metadata.get("export_snapshot_available") is True
        and provider_transcript_used
        if run.openclaw_tool_mode == "code" and export_metadata
        else True
    )
    snapshot_complete = (
        export_metadata.get("export_snapshot_outcome") != "unresolved_tool_call"
        and export_metadata.get("export_snapshot_pending_tool_call_count", 0) == 0
        if export_metadata
        else True
    )
    provider_transcript_complete = (
        export_metadata.get("export_provider_outcome") != "unresolved_tool_call"
        and export_metadata.get("export_provider_pending_tool_call_count", 0) == 0
        if export_metadata.get("export_branch_used") is True
        else snapshot_complete
    )
    if (
        export_metadata.get("export_valid") is False
        or (
            export_metadata
            and export_metadata.get("export_terminal_status") != "success"
        )
        # The provider branch is authoritative when present. Code-mode snapshots can
        # lag on hidden nested calls that are not part of the public trajectory.
        or not provider_transcript_complete
        or not tool_mode_observed
        or session_tree_validation["session_tree_complete"] is not True
    ):
        return _unavailable(
            session_path,
            runtime_model_name=runtime_model_name,
            canonical_model_identity=canonical_model_identity,
            observed_models=observed_models,
            extra_validation={
                "terminal_event_seen": terminal_event_seen,
                "snapshot_complete": snapshot_complete,
                "provider_transcript_complete": provider_transcript_complete,
                "tool_mode_observed": tool_mode_observed,
                "parent_models": sorted(parent_models),
                "child_models": sorted(normalized_child_models),
                "log_models": sorted(normalized_log_models),
                **export_metadata,
                **session_tree_validation,
            },
        )

    steps = (
        _openclaw_session_tree_steps(
            session_tree,
            instruction=instruction,
            run=run,
            failed_session_keys=set(session_tree_validation["session_tree_failed_session_keys"]),
        )
        if len(session_tree) > 1
        else _openclaw_session_steps(
            session_path,
            instruction=instruction,
            model_name=f"{run.provider}/{run.model_id}",
            records=root_records,
        )
    )
    trace_fidelity = "session_tree" if len(session_tree) > 1 else "session"
    source_path = session_path
    if not steps:
        if envelope is None:
            return _unavailable(
                session_path,
                runtime_model_name=runtime_model_name,
                canonical_model_identity=canonical_model_identity,
                observed_models=observed_models,
            )
        steps = _openclaw_envelope_steps(
            envelope,
            instruction=instruction,
            model_name=f"{run.provider}/{run.model_id}",
        )
        trace_fidelity = "envelope"
        source_path = log_path
    if len(steps) < 2 or not terminal_event_seen:
        return _unavailable(
            source_path,
            runtime_model_name=runtime_model_name,
            canonical_model_identity=canonical_model_identity,
            observed_models=observed_models,
            extra_validation={
                "terminal_event_seen": terminal_event_seen,
                "parent_models": sorted(parent_models),
                "child_models": sorted(normalized_child_models),
                "log_models": sorted(normalized_log_models),
                "tool_mode_observed": tool_mode_observed,
                **export_metadata,
            },
        )

    usage = (
        _openclaw_session_tree_usage(session_tree)
        if export_metadata and session_tree
        else agent_meta.get("usage")
    )
    if not isinstance(usage, dict):
        usage = _openclaw_session_tree_usage(session_tree) if session_tree else None
    if not isinstance(usage, dict):
        usage = export_metadata.get("export_usage")
    if not isinstance(usage, dict):
        usage = _openclaw_session_usage(session_path, records=root_records)
    input_tokens = _int(usage.get("input"))
    output_tokens = _int(usage.get("output"))
    cache_read = _int(usage.get("cacheRead"))
    cache_write = _int(usage.get("cacheWrite"))
    prompt_tokens = input_tokens + cache_read + cache_write
    session_id = str(
        agent_meta.get("sessionId") or _openclaw_session_id(session_path) or uuid.uuid4()
    )
    model_name = runtime_model_name or run.model_id
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "agent": {
            "name": run.harness,
            "version": run.harness_version,
            "model_name": f"{run.provider}/{model_name}",
        },
        "steps": steps,
        "final_metrics": _without_none(
            {
                "total_prompt_tokens": prompt_tokens or None,
                "total_completion_tokens": output_tokens or None,
                "total_cached_tokens": cache_read or None,
                "total_steps": len(steps),
            }
        ),
        "extra": {
            "native_raw_trace_file": source_path.name,
            "trace_fidelity": trace_fidelity,
            "observed_models": sorted(observed_models),
            "parent_models": sorted(parent_models),
            "child_models": sorted(normalized_child_models),
            "log_models": sorted(normalized_log_models),
            "stop_reason": _nested_string(meta, "completion", "stopReason"),
            "aborted": meta.get("aborted"),
            "terminal_event_seen": terminal_event_seen,
            "tool_mode_observed": tool_mode_observed,
            "snapshot_complete": snapshot_complete,
            "provider_transcript_complete": provider_transcript_complete,
            "cache_write_tokens": cache_write or None,
            **export_metadata,
            **session_tree_validation,
        },
    }
    _atomic_write_json(agent_dir / "trajectory.json", trajectory)
    return {
        "trajectory_status": "real",
        "trajectory_source": str(source_path),
        "trajectory_event_count": len(steps),
        "runtime_model_name": runtime_model_name,
        "canonical_model_identity": canonical_model_identity,
        "trajectory_validation": {
            "trace_fidelity": trace_fidelity,
            "observed_models": sorted(observed_models),
            "parent_models": sorted(parent_models),
            "child_models": sorted(normalized_child_models),
            "log_models": sorted(normalized_log_models),
            "session_id": session_id,
            "terminal_event_seen": terminal_event_seen,
            "tool_mode_observed": tool_mode_observed,
            "snapshot_complete": snapshot_complete,
            "provider_transcript_complete": provider_transcript_complete,
            **export_metadata,
            **session_tree_validation,
        },
    }


def write_hermes_trajectory(
    instruction: str,
    run: RunSpec,
    agent_dir: Path,
) -> dict[str, Any]:
    source_path = agent_dir / "hermes-session.jsonl"
    sessions = _load_hermes_sessions(source_path)
    if not sessions:
        return _unavailable(source_path)
    session = next(
        (
            item
            for item in reversed(sessions)
            if item.get("model") == run.model_id and item.get("messages")
        ),
        sessions[-1],
    )
    parent_models = {
        _normalize_observed_model(value, run)
        for value in [session.get("model"), *_hermes_message_models(session)]
        if isinstance(value, str) and value
    }
    observed_models = {
        _normalize_observed_model(value, run)
        for item in sessions
        for value in [item.get("model"), *_hermes_message_models(item)]
        if isinstance(value, str) and value
    }
    runtime_model_name = next(iter(parent_models)) if len(parent_models) == 1 else None
    canonical_model_identity = bool(observed_models) and observed_models == {run.model_id}
    validation = _validate_hermes_session(session, instruction)
    steps = _hermes_steps(
        session,
        instruction,
        model_name=f"{run.provider}/{run.model_id}",
    )
    if len(steps) < 2 or validation["terminal_event_seen"] is not True:
        return _unavailable(
            source_path,
            runtime_model_name=runtime_model_name,
            canonical_model_identity=canonical_model_identity,
            observed_models=observed_models,
            extra_validation=validation,
        )

    session_id = str(session.get("id") or session.get("session_id") or uuid.uuid4())
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "agent": {
            "name": run.harness,
            "version": run.harness_version,
            "model_name": f"{run.provider}/{run.model_id}",
        },
        "steps": steps,
        "final_metrics": _without_none(
            {
                "total_prompt_tokens": _int(session.get("input_tokens")) or None,
                "total_completion_tokens": _int(session.get("output_tokens")) or None,
                "total_cached_tokens": (_int(session.get("cache_read_tokens")) or None),
                "total_steps": len(steps),
            }
        ),
        "extra": {
            "native_raw_trace_file": source_path.name,
            "trace_fidelity": "session",
            "observed_models": sorted(observed_models),
            "exported_session_count": len(sessions),
            "reasoning_tokens": _int(session.get("reasoning_tokens")) or None,
            "cache_write_tokens": _int(session.get("cache_write_tokens")) or None,
            "api_call_count": _int(session.get("api_call_count")) or None,
            "tool_call_count": _int(session.get("tool_call_count")) or None,
        },
    }
    _atomic_write_json(agent_dir / "trajectory.json", trajectory)
    return {
        "trajectory_status": "real",
        "trajectory_source": str(source_path),
        "trajectory_event_count": len(steps),
        "runtime_model_name": runtime_model_name,
        "canonical_model_identity": canonical_model_identity,
        "trajectory_validation": {
            "trace_fidelity": "session",
            "observed_models": sorted(observed_models),
            "session_id": session_id,
            **validation,
        },
    }


def write_claude_code_trajectory(
    instruction: str,
    run: RunSpec,
    agent_dir: Path,
) -> dict[str, Any]:
    source_path = agent_dir / "claude-code.txt"
    events = _load_json_lines(source_path)
    observed_models: set[str] = set()
    assistant_models: set[str] = set()
    terminal_event_seen = False
    session_id = str(uuid.uuid4())
    steps: list[dict[str, Any]] = [{"step_id": 1, "source": "user", "message": instruction}]
    pending_calls: dict[str, dict[str, Any]] = {}
    result_event: dict[str, Any] | None = None

    for event in events:
        event_type = str(event.get("type") or "")
        if event_type == "system":
            if event.get("session_id"):
                session_id = str(event["session_id"])
            _add_model(observed_models, event.get("model"), run)
            continue
        if event_type == "assistant":
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            _add_model(observed_models, message.get("model"), run)
            _add_model(assistant_models, message.get("model"), run)
            text, reasoning, calls = _claude_content(message.get("content"))
            if not (text or reasoning or calls):
                continue
            step = _without_none(
                {
                    "step_id": len(steps) + 1,
                    "source": "agent",
                    "message": text or "[tool call]",
                    "model_name": f"{run.provider}/{run.model_id}",
                    "reasoning_content": reasoning or None,
                    "tool_calls": calls or None,
                    "llm_call_count": 1,
                }
            )
            steps.append(step)
            for call in calls:
                pending_calls[call["tool_call_id"]] = step
            continue
        if event_type == "user":
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            for result in _claude_tool_results(message.get("content")):
                step = pending_calls.get(str(result.get("source_call_id") or ""))
                if step is None:
                    continue
                observation = step.setdefault("observation", {"results": []})
                observation["results"].append(result)
            continue
        if event_type == "result":
            result_event = event
            if event.get("session_id"):
                session_id = str(event["session_id"])
            model_usage = event.get("modelUsage")
            if isinstance(model_usage, dict):
                for model_name, details in model_usage.items():
                    _add_model(observed_models, model_name, run)
                    if isinstance(details, dict):
                        _add_model(
                            observed_models,
                            details.get("canonicalModel"),
                            run,
                        )
            terminal_event_seen = (
                event.get("is_error") is not True
                and str(event.get("subtype") or "").lower() == "success"
                and str(event.get("terminal_reason") or "completed").lower() == "completed"
            )
            result_text = event.get("result")
            if (
                isinstance(result_text, str)
                and result_text.strip()
                and not any(
                    step.get("source") == "agent" and step.get("message") == result_text.strip()
                    for step in steps
                )
            ):
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "source": "agent",
                        "message": result_text.strip(),
                        "model_name": f"{run.provider}/{run.model_id}",
                        "llm_call_count": 1,
                    }
                )

    runtime_model_name = (
        next(iter(assistant_models))
        if len(assistant_models) == 1
        else run.model_id
        if run.model_id in assistant_models
        else None
    )
    canonical_model_identity = bool(observed_models) and observed_models == {run.model_id}
    validation = {
        "terminal_event_seen": terminal_event_seen,
        "observed_models": sorted(observed_models),
        "assistant_models": sorted(assistant_models),
    }
    if len(steps) < 2 or not terminal_event_seen or runtime_model_name is None:
        return _unavailable(
            source_path,
            runtime_model_name=runtime_model_name,
            canonical_model_identity=canonical_model_identity,
            observed_models=observed_models,
            extra_validation=validation,
        )

    usage = result_event.get("usage") if isinstance(result_event, dict) else {}
    if not isinstance(usage, dict):
        usage = {}
    final_metrics = _without_none(
        {
            "total_prompt_tokens": _int(usage.get("input_tokens")) or None,
            "total_completion_tokens": _int(usage.get("output_tokens")) or None,
            "total_cached_tokens": _int(usage.get("cache_read_input_tokens")) or None,
            "total_cost_usd": (
                result_event.get("total_cost_usd")
                if isinstance(result_event, dict)
                and isinstance(result_event.get("total_cost_usd"), (int, float))
                else None
            ),
            "total_steps": len(steps),
        }
    )
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": session_id,
        "agent": {
            "name": run.harness,
            "version": run.harness_version,
            "model_name": f"{run.provider}/{runtime_model_name}",
        },
        "steps": steps,
        "final_metrics": final_metrics,
        "extra": {
            "native_raw_trace_file": source_path.name,
            "native_raw_event_count": len(events),
            **validation,
        },
    }
    _atomic_write_json(agent_dir / "trajectory.json", trajectory)
    return {
        "trajectory_status": "real",
        "trajectory_source": str(source_path),
        "trajectory_event_count": len(events),
        "runtime_model_name": runtime_model_name,
        "canonical_model_identity": canonical_model_identity,
        "trajectory_validation": validation,
    }


def _openclaw_envelope_steps(
    envelope: dict[str, Any],
    *,
    instruction: str,
    model_name: str,
) -> list[dict[str, Any]]:
    meta = envelope.get("meta")
    if not isinstance(meta, dict):
        meta = {}
    payloads = envelope.get("payloads")
    if not isinstance(payloads, list):
        payloads = []
    visible: list[str] = []
    reasoning: list[str] = []
    for item in payloads:
        if not isinstance(item, dict):
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        (reasoning if item.get("isReasoning") is True else visible).append(text.strip())
    assistant_text = "\n\n".join(visible)
    if not assistant_text and isinstance(meta.get("finalAssistantVisibleText"), str):
        assistant_text = meta["finalAssistantVisibleText"].strip()

    agent_meta = meta.get("agentMeta")
    usage = agent_meta.get("usage") if isinstance(agent_meta, dict) else None
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = _int(usage.get("input"))
    output_tokens = _int(usage.get("output"))
    cache_read = _int(usage.get("cacheRead"))
    cache_write = _int(usage.get("cacheWrite"))
    metrics = _without_none(
        {
            "prompt_tokens": input_tokens + cache_read + cache_write or None,
            "completion_tokens": output_tokens or None,
            "cached_tokens": cache_read or None,
            "extra": {"cache_write_tokens": cache_write} if cache_write else None,
        }
    )
    agent_step = _without_none(
        {
            "step_id": 2,
            "source": "agent",
            "message": assistant_text or "(no assistant text in OpenClaw output)",
            "model_name": model_name,
            "reasoning_content": "\n\n".join(reasoning) or None,
            "metrics": metrics or None,
            "llm_call_count": 1,
        }
    )
    return [
        {"step_id": 1, "source": "user", "message": instruction},
        agent_step,
    ]


def _openclaw_session_steps(
    path: Path,
    *,
    instruction: str,
    model_name: str,
    session_key: str | None = None,
    session_depth: int = 0,
    deduplicated_handoff_sources: set[str] | None = None,
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    session_records = records if records is not None else _openclaw_session_records(path)
    for record in session_records:
        if record.get("type") != "message":
            continue
        message = record.get("message")
        if isinstance(message, dict) and message.get("role") in {
            "user",
            "assistant",
            "toolResult",
        }:
            rows.append((record, message))
    if not rows:
        return []

    steps: list[dict[str, Any]] = []
    first_user = True
    index = 0
    while index < len(rows):
        record, message = rows[index]
        timestamp = record.get("timestamp")
        timestamp = timestamp if isinstance(timestamp, str) else None
        role = message.get("role")
        if role == "user":
            handoff_source = _openclaw_subagent_announce_source(message)
            if (
                handoff_source
                and deduplicated_handoff_sources
                and handoff_source in deduplicated_handoff_sources
            ):
                index += 1
                continue
            body = _content_text(message.get("content"))
            if handoff_source:
                steps.append(
                    _with_openclaw_session_provenance(
                        _without_none(
                            {
                                "step_id": len(steps) + 1,
                                "source": "agent",
                                "message": body or "(empty subagent handoff)",
                                "timestamp": timestamp,
                                "extra": {
                                    "openclaw_event": "subagent_announce",
                                    "openclaw_source_session_key": handoff_source,
                                },
                            }
                        ),
                        session_key=session_key,
                        session_depth=session_depth,
                    )
                )
                index += 1
                continue
            text = instruction.strip() if first_user and instruction.strip() else body
            first_user = False
            steps.append(
                _with_openclaw_session_provenance(
                    _without_none(
                        {
                            "step_id": len(steps) + 1,
                            "source": "user",
                            "message": text or "(empty user message)",
                            "timestamp": timestamp,
                        }
                    ),
                    session_key=session_key,
                    session_depth=session_depth,
                )
            )
            index += 1
            continue
        if role != "assistant":
            index += 1
            continue

        text, tool_calls = _openclaw_assistant_content(message.get("content"))
        error = message.get("errorMessage")
        if text.strip():
            agent_text = text.strip()
        elif isinstance(error, str) and error.strip():
            agent_text = f"(error) {error.strip()}"
        else:
            agent_text = "(no assistant text)"

        pending = {call["tool_call_id"] for call in tool_calls if call.get("tool_call_id")}
        observations: list[dict[str, Any]] = []
        cursor = index + 1
        while cursor < len(rows) and rows[cursor][1].get("role") == "toolResult":
            tool_result = rows[cursor][1]
            call_id = str(tool_result.get("toolCallId") or "")
            if call_id not in pending:
                break
            details = tool_result.get("details")
            content = details.get("aggregated") if isinstance(details, dict) else None
            if not isinstance(content, str) or not content.strip():
                content = _content_text(tool_result.get("content"))
            observations.append(
                _without_none(
                    {
                        "source_call_id": call_id or None,
                        "content": content or None,
                    }
                )
            )
            pending.discard(call_id)
            cursor += 1
            if not pending:
                break

        usage = message.get("usage")
        metrics = _openclaw_usage_metrics(usage)
        steps.append(
            _with_openclaw_session_provenance(
                _without_none(
                    {
                        "step_id": len(steps) + 1,
                        "source": "agent",
                        "message": agent_text,
                        "timestamp": timestamp,
                        "model_name": model_name,
                        "tool_calls": tool_calls or None,
                        "observation": ({"results": observations} if observations else None),
                        "metrics": metrics,
                        "llm_call_count": 1,
                    }
                ),
                session_key=session_key,
                session_depth=session_depth,
            )
        )
        index = cursor
    return steps if len(steps) >= 2 else []


def _openclaw_session_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if path.name == "session-branch.json":
        bundle = _openclaw_export_bundle(path)
        if bundle is None:
            return []
        _, branch, events = bundle
        # The exported branch is the provider-visible transcript. The runtime
        # snapshot also contains Code Mode's nested bridge calls so the agent
        # can resume them, but those are not separate model tool calls.
        entries = branch.get("entries")
        records = (
            [entry for entry in entries if isinstance(entry, dict)]
            if isinstance(entries, list)
            else []
        )
        if not records:
            records = _openclaw_export_snapshot_records(events)
        header = branch.get("header")
        return [header, *records] if isinstance(header, dict) else records
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _openclaw_export_bundle(
    branch_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]] | None:
    manifest = _load_json_object(branch_path.parent / "manifest.json")
    branch = _load_json_object(branch_path)
    if (
        manifest.get("traceSchema") != "openclaw-trajectory"
        or manifest.get("schemaVersion") != 1
        or not isinstance(manifest.get("traceId"), str)
        or not isinstance(manifest.get("sessionId"), str)
        or not isinstance(manifest.get("sessionKey"), str)
        or not isinstance(manifest.get("eventCount"), int)
        or not isinstance(manifest.get("runtimeEventCount"), int)
        or not isinstance(manifest.get("transcriptEventCount"), int)
        or not isinstance(manifest.get("sourceFiles"), dict)
        or not isinstance(branch.get("entries"), list)
    ):
        return None
    warnings = manifest.get("warnings")
    if isinstance(warnings, list) and any(
        isinstance(warning, dict)
        and warning.get("code") in _OPENCLAW_FATAL_EXPORT_WARNINGS
        for warning in warnings
    ):
        return None
    events: list[dict[str, Any]] = []
    events_path = branch_path.parent / "events.jsonl"
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                return None
            if (
                not isinstance(event, dict)
                or event.get("traceSchema") != "openclaw-trajectory"
                or event.get("schemaVersion") != 1
                or event.get("traceId") != manifest["traceId"]
                or event.get("sessionId") != manifest["sessionId"]
                or event.get("sessionKey") != manifest["sessionKey"]
            ):
                return None
            events.append(event)
    if len(events) != manifest["eventCount"]:
        return None
    if (
        sum(event.get("source") == "runtime" for event in events)
        != manifest["runtimeEventCount"]
        or sum(event.get("source") == "transcript" for event in events)
        != manifest["transcriptEventCount"]
    ):
        return None
    return manifest, branch, events


def _openclaw_export_runtime_turn(
    events: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    runtime_events = [event for event in events if event.get("source") == "runtime"]
    terminal = next(
        (event for event in reversed(runtime_events) if event.get("type") == "session.ended"),
        None,
    )
    run_id = terminal.get("runId") if isinstance(terminal, dict) else None

    def matching(event: dict[str, Any], event_type: str) -> bool:
        if event.get("type") != event_type:
            return False
        return not isinstance(run_id, str) or event.get("runId") == run_id

    completion = next(
        (event for event in reversed(runtime_events) if matching(event, "model.completed")),
        None,
    )
    context = next(
        (event for event in reversed(runtime_events) if matching(event, "context.compiled")),
        None,
    )
    return terminal, completion, context


def _openclaw_export_snapshot_records(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    _, completion, _ = _openclaw_export_runtime_turn(events)
    data = completion.get("data") if isinstance(completion, dict) else None
    if not isinstance(data, dict) or data.get("truncated") is True:
        return []
    snapshot = data.get("messagesSnapshot")
    if not isinstance(snapshot, list):
        return []
    records: list[dict[str, Any]] = []
    for index, message in enumerate(snapshot):
        if not isinstance(message, dict):
            return []
        timestamp = message.get("timestamp")
        if isinstance(timestamp, (int, float)):
            timestamp = (
                datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        records.append(
            {
                "type": "message",
                "id": f"runtime-message-{index + 1}",
                "timestamp": timestamp,
                "message": message,
            }
        )
    roles = {
        record["message"].get("role")
        for record in records
        if isinstance(record.get("message"), dict)
        and isinstance(record["message"].get("role"), str)
    }
    if not {"user", "assistant"} <= roles:
        return []
    return records


def _openclaw_export_snapshot_diagnostics(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    pending: set[str] = set()
    tool_call_count = 0
    tool_result_count = 0
    tool_error_count = 0
    terminal_outcome = "unknown"
    for record in records:
        message = record.get("message")
        if record.get("type") != "message" or not isinstance(message, dict):
            continue
        role = message.get("role")
        if role == "assistant":
            text, tools = _openclaw_assistant_content(message.get("content"))
            for tool in tools:
                call_id = tool.get("tool_call_id")
                if isinstance(call_id, str) and call_id:
                    pending.add(call_id)
                tool_call_count += 1
            terminal_assistant_text = (
                bool(text.strip())
                and not tools
                and str(message.get("stopReason") or "").lower() in {"end_turn", "stop"}
            )
            terminal_outcome = (
                "assistant_text"
                if terminal_assistant_text
                else "unresolved_tool_call"
                if tools
                else "unknown"
            )
        elif role == "toolResult":
            tool_result_count += 1
            call_id = message.get("toolCallId")
            if isinstance(call_id, str) and call_id:
                pending.discard(call_id)
            is_error = message.get("isError") is True
            if is_error:
                tool_error_count += 1
            terminal_outcome = "tool_error" if is_error else "resolved_tool_result"
        elif role in {"user"}:
            terminal_outcome = "unknown"
    outcome = "unresolved_tool_call" if pending else terminal_outcome
    return {
        "export_snapshot_outcome": outcome,
        "export_snapshot_tool_call_count": tool_call_count,
        "export_snapshot_tool_result_count": tool_result_count,
        "export_snapshot_tool_error_count": tool_error_count,
        "export_snapshot_pending_tool_call_count": len(pending),
    }


def _openclaw_export_metadata(path: Path) -> dict[str, Any]:
    if path.name != "session-branch.json":
        return {}
    bundle = _openclaw_export_bundle(path)
    if bundle is None:
        return {"export_valid": False}
    manifest, branch, events = bundle
    terminal, completion, context = _openclaw_export_runtime_turn(events)
    completion_data = completion.get("data") if isinstance(completion, dict) else None
    context_data = context.get("data") if isinstance(context, dict) else None
    snapshot_recorded = (
        isinstance(completion_data, dict) and "messagesSnapshot" in completion_data
    )
    snapshot_records = _openclaw_export_snapshot_records(events)
    snapshot_diagnostics = _openclaw_export_snapshot_diagnostics(snapshot_records)
    entries = branch.get("entries")
    branch_records = (
        [entry for entry in entries if isinstance(entry, dict)]
        if isinstance(entries, list)
        else []
    )
    branch_diagnostics = {
        key.replace("export_snapshot_", "export_provider_"): value
        for key, value in _openclaw_export_snapshot_diagnostics(branch_records).items()
    }
    provider_visible_tools_recorded = (
        isinstance(context_data, dict) and "providerVisibleTools" in context_data
    )
    visible_tools = None
    if isinstance(context_data, dict):
        visible_tools = (
            context_data.get("providerVisibleTools")
            if "providerVisibleTools" in context_data
            else context_data.get("tools")
        )
    visible_tool_names = (
        sorted(
            {
                tool["name"]
                for tool in visible_tools
                if isinstance(tool, dict) and isinstance(tool.get("name"), str)
            }
        )
        if isinstance(visible_tools, list)
        else []
    )
    return {
        "export_valid": True,
        "export_event_count": len(events),
        "export_runtime_event_count": manifest["runtimeEventCount"],
        "export_transcript_event_count": manifest["transcriptEventCount"],
        "export_terminal_event_seen": terminal is not None,
        "export_terminal_status": (
            terminal.get("data", {}).get("status")
            if isinstance(terminal, dict) and isinstance(terminal.get("data"), dict)
            else None
        ),
        "export_snapshot_recorded": snapshot_recorded,
        "export_snapshot_available": bool(snapshot_records),
        "export_snapshot_used": bool(snapshot_records) and not bool(branch_records),
        **snapshot_diagnostics,
        "export_branch_used": bool(branch_records),
        **branch_diagnostics,
        "export_provider_visible_tools_recorded": provider_visible_tools_recorded,
        "export_visible_tools": visible_tool_names,
        "export_model": completion.get("modelId") if isinstance(completion, dict) else None,
        "export_usage": (
            completion_data.get("usage") if isinstance(completion_data, dict) else None
        ),
    }


def _openclaw_root_session_path(agent_dir: Path) -> Path:
    legacy = agent_dir / "openclaw.session.jsonl"
    if legacy.is_file():
        return legacy
    entries, ambiguous_keys = _load_openclaw_session_index(agent_dir / "openclaw.sessions")
    if "agent:main:main" in ambiguous_keys:
        return legacy
    indexed = entries.get("agent:main:main")
    if indexed is None:
        return legacy
    entry, store_dir = indexed
    return _resolve_archived_openclaw_session_path(store_dir, entry) or legacy


_OPENCLAW_CANONICAL_SESSION_ENTRY_TYPES = {
    "message",
    "thinking_level_change",
    "model_change",
    "compaction",
    "reset",
    "branch_summary",
    "custom",
    "custom_message",
    "label",
    "session_info",
}
_OPENCLAW_FAILURE_STATUSES = {
    "cancelled",
    "deleted",
    "error",
    "failed",
    "killed",
    "reset",
    "timeout",
}


def _openclaw_active_session_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select OpenClaw's active parent-linked transcript branch."""

    def nonempty(value: Any) -> str | None:
        return value.strip() if isinstance(value, str) and value.strip() else None

    def canonical(record: dict[str, Any]) -> bool:
        return record.get("type") in _OPENCLAW_CANONICAL_SESSION_ENTRY_TYPES

    def parse(
        record: dict[str, Any],
        *,
        fallback_parent: str | None,
    ) -> dict[str, Any] | None:
        if record.get("type") == "session":
            return None
        explicit = "parentId" in record
        if not explicit and not canonical(record):
            return None
        record_id = nonempty(record.get("id"))
        if not record_id:
            return None
        if explicit:
            raw_parent = record.get("parentId")
            parent_id = None if raw_parent is None else nonempty(raw_parent)
            if raw_parent is not None and parent_id is None:
                return None
        else:
            parent_id = fallback_parent
        if record.get("type") == "leaf":
            raw_target = record.get("targetId")
            target_id = None if raw_target is None else nonempty(raw_target)
            if raw_target is not None and target_id is None:
                return None
            raw_append_parent = record.get("appendParentId", raw_target)
            append_parent_id = None if raw_append_parent is None else nonempty(raw_append_parent)
            if raw_append_parent is not None and append_parent_id is None:
                return None
            append_mode = record.get("appendMode")
            if append_mode not in {None, "side"}:
                return None
            return {
                "id": record_id,
                "parent_id": target_id,
                "leaf_id": target_id,
                "append_parent_id": append_parent_id,
                "is_leaf": True,
                "explicit": True,
            }
        append_mode = record.get("appendMode")
        return {
            "id": record_id,
            "parent_id": parent_id,
            "leaf_id": (record_id if canonical(record) and append_mode != "side" else ...),
            "append_parent_id": record_id,
            "is_leaf": False,
            "explicit": explicit,
            "append_mode": append_mode,
        }

    def resolve_parent(
        parent_id: str | None,
        by_id: dict[str, dict[str, Any]],
    ) -> str | None:
        seen: set[str] = set()
        current = parent_id
        while current is not None:
            if current in seen:
                return current
            seen.add(current)
            parent = by_id.get(current)
            if parent is None or not parent["is_leaf"]:
                return current
            current = parent["parent_id"]
        return None

    nodes: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    leaf_id: str | None = None
    append_parent_id: str | None = None
    has_explicit_leaf_update = False
    invalid_leaf_ids: set[str] = set()
    for index, record in enumerate(records):
        node = parse(record, fallback_parent=leaf_id)
        if node is None:
            continue
        if node["is_leaf"]:
            references = (node["leaf_id"], node["append_parent_id"])
            invalid = any(
                reference is not None and (reference not in by_id or reference in invalid_leaf_ids)
                for reference in references
            )
            if invalid:
                invalid_leaf_ids.add(node["id"])
                node["leaf_id"] = ...
                node["append_parent_id"] = append_parent_id
                node["parent_id"] = (
                    record.get("parentId") if isinstance(record.get("parentId"), str) else None
                )
        elif canonical(record):
            parent_id = node["parent_id"]
            if (
                node["explicit"]
                and parent_id is not None
                and parent_id not in by_id
                and leaf_id is not None
            ):
                parent_id = leaf_id
            elif (
                node["explicit"]
                and node.get("append_mode") != "side"
                and parent_id == append_parent_id
                and leaf_id != append_parent_id
            ):
                parent_id = leaf_id
            node["parent_id"] = resolve_parent(parent_id, by_id)
        node["record"] = record
        node["index"] = index
        nodes.append(node)
        by_id[node["id"]] = node
        append_parent_id = node["append_parent_id"]
        if node["leaf_id"] is not ...:
            leaf_id = node["leaf_id"]
            if node["explicit"]:
                has_explicit_leaf_update = True

    # Older synthetic/plain transcripts have no parent links and remain linear.
    if not has_explicit_leaf_update:
        return records
    if leaf_id is None:
        return []

    active: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: str | None = leaf_id
    while current is not None:
        if current in seen:
            return []
        seen.add(current)
        node = by_id.get(current)
        if node is None:
            break
        if not node["is_leaf"]:
            active.append(node["record"])
        current = node["parent_id"]
    active.reverse()
    return active


def _openclaw_child_session_records(
    parent_path: Path,
    child_path: Path,
    entry: dict[str, Any],
) -> list[dict[str, Any]]:
    child_records = _openclaw_session_records(child_path)
    header = child_records[0] if child_records else None
    forked = (
        entry.get("forkedFromParent") is True
        or bool(entry.get("forkSource"))
        or (isinstance(header, dict) and bool(header.get("parentSession")))
    )
    if not forked or len(child_records) < 2:
        return child_records

    parent_identities = {
        _openclaw_record_identity(record) for record in _openclaw_session_records(parent_path)
    }
    first_child_record = 1
    while (
        first_child_record < len(child_records)
        and _openclaw_record_identity(child_records[first_child_record]) in parent_identities
    ):
        first_child_record += 1
    return [child_records[0], *child_records[first_child_record:]]


def _openclaw_record_identity(record: dict[str, Any]) -> tuple[str, str]:
    record_id = record.get("id")
    if isinstance(record_id, (str, int)) and str(record_id):
        return "id", str(record_id)
    return (
        "json",
        json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


def _openclaw_session_tree(
    agent_dir: Path,
    root_path: Path,
) -> tuple[list[OpenClawSessionTrace], dict[str, Any]]:
    empty_validation = {
        "session_tree_session_count": 0,
        "session_tree_descendant_count": 0,
        "session_tree_accepted_spawn_count": 0,
        "session_tree_reused_spawn_count": 0,
        "session_tree_ignored_entry_count": 0,
        "session_tree_missing_transcript_count": 0,
        "session_tree_missing_timestamp_count": 0,
        "session_tree_ambiguous_timestamp_count": 0,
        "session_tree_session_keys": [],
        "session_tree_failed_session_keys": [],
        "session_tree_nonterminal_session_keys": [],
        "session_tree_errors": [],
        "session_tree_complete": True,
    }
    if not root_path.is_file():
        return [], empty_validation

    archive = agent_dir / "openclaw.sessions"
    entries, ambiguous_keys = _load_openclaw_session_index(archive)
    audit = _load_openclaw_session_audit(archive)
    root_session_id = _openclaw_session_id(root_path)
    root_key = next(
        (
            key
            for key, (entry, _) in entries.items()
            if root_session_id
            and isinstance(entry.get("sessionId"), str)
            and entry["sessionId"] == root_session_id
        ),
        None,
    )
    if root_key is None and "agent:main:main" in entries:
        root_key = "agent:main:main"
    if root_key is None:
        root_key = "agent:main:main"

    root_records = _openclaw_active_session_records(_openclaw_session_records(root_path))
    traces: list[OpenClawSessionTrace] = [(root_key, root_path, 0, root_records)]
    depth_by_key = {root_key: 0}
    parent_by_key: dict[str, str | None] = {root_key: None}
    entry_by_key: dict[str, dict[str, Any]] = {}
    pending = [(root_key, root_path, root_records)]
    errors: list[str] = []
    missing_transcripts = 0
    accepted_spawn_count = 0
    reused_spawn_count = 0
    seen_paths = {root_path.resolve()}
    while pending:
        parent_key, parent_path, parent_records = pending.pop(0)
        transcript_children = [
            spawn["child_session_key"]
            for spawn in _openclaw_spawn_results(
                parent_path,
                records=parent_records,
            )
            if spawn.get("child_session_key")
        ]
        audited_children = sorted(
            key
            for key, event in audit.items()
            if event.get("spawnedBy") == parent_key and key not in transcript_children
        )
        for child_key in [*transcript_children, *audited_children]:
            accepted_spawn_count += 1
            ancestor: str | None = parent_key
            while ancestor is not None and ancestor != child_key:
                ancestor = parent_by_key.get(ancestor)
            if ancestor == child_key:
                errors.append(f"cycle:{parent_key}->{child_key}")
                continue
            if child_key in depth_by_key:
                if parent_by_key.get(child_key) != parent_key:
                    errors.append(f"duplicate-session-reference:{child_key}")
                else:
                    reused_spawn_count += 1
                continue
            indexed = entries.get(child_key)
            if indexed is None:
                deleted, deleted_error = _resolve_deleted_openclaw_session(
                    audit=audit,
                    audit_root=archive / "audit",
                    child_key=child_key,
                    parent_key=parent_key,
                    seen_paths=seen_paths,
                )
                if deleted_error:
                    errors.append(deleted_error)
                    missing_transcripts += 1
                    continue
                if deleted is None:
                    errors.append(f"missing-session-entry:{child_key}")
                    missing_transcripts += 1
                    continue
                entry, path = deleted
            else:
                entry, store_dir = indexed
                entry = {**entry, **audit.get(child_key, {})}
                path = _resolve_archived_openclaw_session_path(store_dir, entry)
                if path is None:
                    deleted, deleted_error = _resolve_deleted_openclaw_session(
                        audit=audit,
                        audit_root=archive / "audit",
                        child_key=child_key,
                        parent_key=parent_key,
                        seen_paths=seen_paths,
                    )
                    if deleted_error:
                        errors.append(deleted_error)
                        missing_transcripts += 1
                        continue
                    if deleted is not None:
                        entry, path = deleted
            if child_key in ambiguous_keys:
                errors.append(f"ambiguous-session-entry:{child_key}")
                continue
            spawned_by = entry.get("spawnedBy")
            if isinstance(spawned_by, str) and spawned_by.strip() and spawned_by != parent_key:
                errors.append(f"spawn-lineage-mismatch:{child_key}:{spawned_by}!={parent_key}")
                continue
            if path is None:
                errors.append(f"missing-session-transcript:{child_key}")
                missing_transcripts += 1
                continue
            resolved = path.resolve()
            if resolved in seen_paths:
                errors.append(f"duplicate-session-transcript:{child_key}")
                continue
            seen_paths.add(resolved)
            depth_by_key[child_key] = depth_by_key[parent_key] + 1
            parent_by_key[child_key] = parent_key
            entry_by_key[child_key] = entry
            child_records = _openclaw_active_session_records(
                _openclaw_child_session_records(
                    parent_path,
                    path,
                    entry,
                )
            )
            traces.append((child_key, path, depth_by_key[child_key], child_records))
            pending.append((child_key, path, child_records))

    nonterminal_keys = [
        key
        for key, path, _, records in traces[1:]
        if not _openclaw_indexed_session_terminal(
            entry_by_key[key],
            path,
            records=records,
        )
    ]
    failed_keys = [
        key
        for key, entry in entry_by_key.items()
        if str(entry.get("status") or "").lower() in _OPENCLAW_FAILURE_STATUSES
    ]
    errors.extend(f"nonterminal-session:{key}" for key in nonterminal_keys)
    missing_timestamps = 0
    ambiguous_timestamps = 0
    if not errors and len(traces) > 1:
        order_errors, missing_timestamps, ambiguous_timestamps = (
            _openclaw_session_tree_order_errors(
                traces,
                failed_session_keys=set(failed_keys),
            )
        )
        errors.extend(order_errors)

    return traces, {
        "session_tree_session_count": len(traces),
        "session_tree_descendant_count": max(0, len(traces) - 1),
        "session_tree_accepted_spawn_count": accepted_spawn_count,
        "session_tree_reused_spawn_count": reused_spawn_count,
        "session_tree_ignored_entry_count": max(0, len(entries) - len(depth_by_key)),
        "session_tree_missing_transcript_count": missing_transcripts,
        "session_tree_missing_timestamp_count": missing_timestamps,
        "session_tree_ambiguous_timestamp_count": ambiguous_timestamps,
        "session_tree_session_keys": [key for key, _, _, _ in traces],
        "session_tree_failed_session_keys": failed_keys,
        "session_tree_nonterminal_session_keys": nonterminal_keys,
        "session_tree_errors": errors,
        "session_tree_complete": not errors,
    }


def _openclaw_session_tree_order_errors(
    session_tree: list[OpenClawSessionTrace],
    *,
    failed_session_keys: set[str] | None = None,
) -> tuple[list[str], int, int]:
    missing: list[str] = []
    timestamp_sessions: dict[float, str] = {}
    ambiguous: set[tuple[str, str, float]] = set()
    deduplicated_handoff_sources = _openclaw_terminal_child_session_keys(
        session_tree,
        failed_session_keys=failed_session_keys,
    )
    for session_key, _, _, records in session_tree:
        for record_index, record in enumerate(records):
            message = record.get("message")
            if (
                record.get("type") != "message"
                or not isinstance(message, dict)
                or message.get("role") not in {"user", "assistant"}
            ):
                continue
            handoff_source = _openclaw_subagent_announce_source(message)
            if handoff_source in deduplicated_handoff_sources:
                continue
            timestamp = _openclaw_timestamp_value(record.get("timestamp"))
            if timestamp is None:
                missing.append(f"missing-step-timestamp:{session_key}:{record_index}")
                continue
            other_session = timestamp_sessions.setdefault(timestamp, session_key)
            if other_session != session_key:
                first, second = sorted((other_session, session_key))
                ambiguous.add((first, second, timestamp))
    errors = [
        *missing,
        *[
            f"ambiguous-step-timestamp:{first}:{second}:{timestamp}"
            for first, second, timestamp in sorted(ambiguous)
        ],
    ]
    return errors, len(missing), len(ambiguous)


def _load_openclaw_session_index(
    archive: Path,
) -> tuple[dict[str, tuple[dict[str, Any], Path]], set[str]]:
    entries: dict[str, tuple[dict[str, Any], Path]] = {}
    ambiguous_keys: set[str] = set()
    store_paths = sorted(archive.rglob("sessions.json")) if archive.is_dir() else []
    for store_path in store_paths:
        store = _load_json_object(store_path)
        for key, value in store.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            if key in entries:
                ambiguous_keys.add(key)
                continue
            entries[key] = (value, store_path.parent)
    manifest_paths = sorted(archive.rglob("manifest.json")) if archive.is_dir() else []
    for manifest_path in manifest_paths:
        manifest = _load_json_object(manifest_path)
        key = manifest.get("sessionKey")
        session_id = manifest.get("sessionId")
        branch_path = manifest_path.parent / "session-branch.json"
        if (
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(session_id, str)
            or not session_id.strip()
            or not branch_path.is_file()
        ):
            continue
        key = key.strip()
        if key in entries:
            ambiguous_keys.add(key)
            continue
        entries[key] = (
            {
                "sessionId": session_id.strip(),
                "sessionFile": branch_path.name,
            },
            manifest_path.parent,
        )
    return entries, ambiguous_keys


def _load_openclaw_session_audit(archive: Path) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    for event in _openclaw_session_records(archive / "audit" / "sessions.jsonl"):
        key = event.get("sessionKey")
        if not isinstance(key, str) or not key.strip():
            continue
        audit.setdefault(key.strip(), {}).update(event)
    return audit


def _resolve_deleted_openclaw_session(
    *,
    audit: dict[str, dict[str, Any]],
    audit_root: Path,
    child_key: str,
    parent_key: str,
    seen_paths: set[Path],
) -> tuple[tuple[dict[str, Any], Path] | None, str | None]:
    audited = audit.get(child_key)
    if audited is not None:
        spawned_by = audited.get("spawnedBy")
        if isinstance(spawned_by, str) and spawned_by.strip() and spawned_by != parent_key:
            return None, (f"spawn-lineage-mismatch:{child_key}:{spawned_by}!={parent_key}")
        transcript = audited.get("auditTranscript")
        if isinstance(transcript, str) and transcript.strip():
            resolved_root = audit_root.resolve()
            path = (audit_root / transcript).resolve()
            if path.is_relative_to(resolved_root) and path.is_file():
                if path in seen_paths:
                    return None, f"duplicate-session-transcript:{child_key}"
                return (audited, path), None
            return None, f"missing-session-transcript:{child_key}"

    return None, None


def _resolve_archived_openclaw_session_path(
    store_dir: Path,
    entry: dict[str, Any],
) -> Path | None:
    candidates: list[Path] = []
    session_file = entry.get("sessionFile")
    if isinstance(session_file, str) and session_file.strip():
        source = Path(session_file)
        if not source.is_absolute():
            candidates.append(store_dir / source)
        else:
            session_parts = [index for index, part in enumerate(source.parts) if part == "sessions"]
            if session_parts and session_parts[-1] + 1 < len(source.parts):
                candidates.append(store_dir / Path(*source.parts[session_parts[-1] + 1 :]))
        candidates.append(store_dir / source.name)
    session_id = entry.get("sessionId")
    if isinstance(session_id, str) and session_id.strip():
        candidates.append(store_dir / f"{session_id}.jsonl")
    resolved_store = store_dir.resolve()
    return next(
        (
            candidate
            for candidate in candidates
            if candidate.resolve().is_relative_to(resolved_store) and candidate.is_file()
        ),
        None,
    )


def _openclaw_session_tree_steps(
    session_tree: list[OpenClawSessionTrace],
    *,
    instruction: str,
    run: RunSpec,
    failed_session_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    ordered: list[tuple[tuple[int, float, str], int, int, int, dict[str, Any]]] = []
    initial_user: dict[str, Any] | None = None
    deduplicated_handoff_sources = _openclaw_terminal_child_session_keys(
        session_tree,
        failed_session_keys=failed_session_keys,
    )
    for session_index, (session_key, path, depth, records) in enumerate(session_tree):
        models = {
            _normalize_observed_model(model, run)
            for model in _openclaw_session_models(path, records=records)
        }
        session_model = next(iter(models)) if len(models) == 1 else run.model_id
        session_steps = _openclaw_session_steps(
            path,
            instruction=instruction if session_index == 0 else "",
            model_name=f"{run.provider}/{session_model}",
            session_key=session_key,
            session_depth=depth,
            deduplicated_handoff_sources=deduplicated_handoff_sources,
            records=records,
        )
        if session_index == 0 and session_steps and session_steps[0].get("source") == "user":
            initial_user = session_steps.pop(0)
        for local_index, step in enumerate(session_steps):
            ordered.append(
                (
                    _openclaw_timestamp_order(step.get("timestamp")),
                    depth,
                    session_index,
                    local_index,
                    step,
                )
            )

    merged = ([initial_user] if initial_user is not None else []) + [
        item[-1] for item in sorted(ordered, key=lambda item: item[:-1])
    ]
    for step_id, step in enumerate(merged, start=1):
        step["step_id"] = step_id
    return merged


def _openclaw_timestamp_order(value: Any) -> tuple[int, float, str]:
    timestamp = _openclaw_timestamp_value(value)
    if timestamp is not None:
        return 0, timestamp, ""
    if not isinstance(value, str) or not value.strip():
        return 1, 0.0, ""
    return 1, 0.0, value


def _openclaw_timestamp_value(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


def _with_openclaw_session_provenance(
    step: dict[str, Any],
    *,
    session_key: str | None,
    session_depth: int,
) -> dict[str, Any]:
    if not session_key:
        return step
    extra = step.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    step["extra"] = {
        **extra,
        "openclaw_session_key": session_key,
        "openclaw_session_depth": session_depth,
    }
    return step


def _openclaw_subagent_announce_source(message: dict[str, Any]) -> str | None:
    provenance = message.get("provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("kind") != "inter_session"
        or provenance.get("sourceTool") != "subagent_announce"
    ):
        return None
    source = provenance.get("sourceSessionKey")
    return source if isinstance(source, str) and source.strip() else None


def _openclaw_terminal_child_session_keys(
    session_tree: list[OpenClawSessionTrace],
    *,
    failed_session_keys: set[str] | None = None,
) -> set[str]:
    # Only a successful child response supersedes its parent announcement.
    # Failure announcements carry the terminal outcome missing from the child log.
    return {
        key
        for key, path, _, records in session_tree[1:]
        if key not in (failed_session_keys or set())
        and _openclaw_session_terminal(path, records=records)
    }


def _openclaw_session_tree_usage(
    session_tree: list[OpenClawSessionTrace],
) -> dict[str, int] | None:
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    observed = False
    for _, path, _, records in session_tree:
        observed = observed or any(
            isinstance(record.get("message"), dict)
            and record["message"].get("role") == "assistant"
            and isinstance(record["message"].get("usage"), dict)
            for record in records
        )
        usage = _openclaw_session_usage(path, records=records)
        for key in totals:
            totals[key] += usage[key]
    return totals if observed else None


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _openclaw_session_models(
    path: Path,
    *,
    records: list[dict[str, Any]] | None = None,
) -> set[str]:
    models: set[str] = set()
    session_records = records if records is not None else _openclaw_session_records(path)
    for record in session_records:
        if record.get("type") == "model_change":
            model_id = record.get("modelId")
            if isinstance(model_id, str) and model_id:
                models.add(model_id)
        message = record.get("message")
        if (
            record.get("type") == "message"
            and isinstance(message, dict)
            and message.get("role") == "assistant"
            and isinstance(message.get("model"), str)
            and message["model"]
        ):
            models.add(message["model"])
    return models


def _openclaw_spawn_results(
    path: Path,
    *,
    records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    pending: dict[str, str | None] = {}
    results: list[dict[str, Any]] = []
    session_records = records if records is not None else _openclaw_session_records(path)
    for record in session_records:
        if record.get("type") != "message":
            continue
        message = record.get("message")
        if not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            _, calls = _openclaw_assistant_content(message.get("content"))
            for call in calls:
                if call.get("function_name") != "sessions_spawn" or not call.get("tool_call_id"):
                    continue
                arguments = call.get("arguments")
                task = (
                    arguments.get("task")
                    if isinstance(arguments, dict) and isinstance(arguments.get("task"), str)
                    else None
                )
                pending[call["tool_call_id"]] = task
            continue
        if message.get("role") != "toolResult":
            continue
        call_id = str(message.get("toolCallId") or "")
        if call_id not in pending:
            continue
        task = pending.pop(call_id)
        payload = _openclaw_tool_result_object(message)
        if str(payload.get("status") or "").lower() != "accepted":
            continue
        child_key = payload.get("childSessionKey")
        if not isinstance(child_key, str) or not child_key.strip():
            continue
        resolved_model = payload.get("resolvedModel")
        results.append(
            _without_none(
                {
                    "child_session_key": child_key.strip(),
                    "task": task,
                    "resolved_model": (
                        resolved_model.strip()
                        if isinstance(resolved_model, str) and resolved_model.strip()
                        else None
                    ),
                }
            )
        )
    return results


def _openclaw_tool_result_object(message: dict[str, Any]) -> dict[str, Any]:
    details = message.get("details")
    if isinstance(details, dict):
        return details
    text = _content_text(message.get("content")).strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return {}
    return value if isinstance(value, dict) else {}


def _openclaw_log_models(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(
        re.findall(
            r"\[model-fetch\].*?\bmodel=([^\s]+)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
    )


def _openclaw_session_terminal(
    path: Path,
    *,
    records: list[dict[str, Any]] | None = None,
) -> bool:
    session_records = records if records is not None else _openclaw_session_records(path)
    active_records = _openclaw_active_session_records(session_records)
    for record in reversed(active_records):
        message = record.get("message")
        if record.get("type") != "message" or not isinstance(message, dict):
            continue
        if message.get("role") != "assistant":
            return False
        text, tools = _openclaw_assistant_content(message.get("content"))
        stop_reason = str(message.get("stopReason") or "").lower()
        return (
            bool(text.strip())
            and not tools
            and stop_reason
            in {
                "end_turn",
                "stop",
            }
        )
    return False


def _openclaw_indexed_session_terminal(
    entry: dict[str, Any],
    path: Path,
    *,
    records: list[dict[str, Any]] | None = None,
) -> bool:
    transcript_terminal = _openclaw_session_terminal(path, records=records)
    status = str(entry.get("status") or "").lower()
    if status in _OPENCLAW_FAILURE_STATUSES:
        return True
    # Index writes can lag transcript archival. Require a terminal transcript for
    # normal completion, but keep explicit failure states that may have no final turn.
    return transcript_terminal


def _openclaw_envelope_terminal(envelope: dict[str, Any]) -> bool:
    meta = envelope.get("meta")
    if not isinstance(meta, dict):
        return False
    liveness = str(meta.get("livenessState") or "").lower()
    if meta.get("yielded") is True or liveness in {
        "active",
        "paused",
        "running",
        "waiting",
    }:
        return False
    payloads = envelope.get("payloads")
    visible = (
        any(
            isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and item["text"].strip()
            and item.get("isReasoning") is not True
            for item in payloads
        )
        if isinstance(payloads, list)
        else False
    )
    return visible or meta.get("aborted") is True


def _openclaw_session_id(path: Path) -> str | None:
    for record in _openclaw_session_records(path):
        if record.get("type") == "session" and isinstance(record.get("id"), str):
            return record["id"]
    return path.stem if path.is_file() else None


def _openclaw_session_usage(
    path: Path,
    *,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    session_records = records if records is not None else _openclaw_session_records(path)
    for record in session_records:
        message = record.get("message")
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        usage = message.get("usage")
        if not isinstance(usage, dict):
            continue
        for key in totals:
            totals[key] += _int(usage.get(key))
    return totals


def _openclaw_assistant_content(
    content: Any,
) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(content, list):
        return "", []
    text: list[str] = []
    tools: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            text.append(part["text"])
        elif part.get("type") == "toolCall" and isinstance(part.get("name"), str):
            tools.append(
                {
                    "tool_call_id": str(part.get("id") or ""),
                    "function_name": part["name"],
                    "arguments": _arguments(part.get("arguments")),
                }
            )
    return "".join(text), tools


def _openclaw_usage_metrics(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    input_tokens = _int(value.get("input"))
    output_tokens = _int(value.get("output"))
    cache_read = _int(value.get("cacheRead"))
    cache_write = _int(value.get("cacheWrite"))
    metrics = _without_none(
        {
            "prompt_tokens": input_tokens + cache_read + cache_write or None,
            "completion_tokens": output_tokens or None,
            "cached_tokens": cache_read or None,
            "extra": {"cache_write_tokens": cache_write} if cache_write else None,
        }
    )
    return metrics or None


def _load_hermes_sessions(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    sessions: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("messages"), list):
            sessions.append(value)
    return sessions


def _hermes_steps(
    session: dict[str, Any],
    instruction: str,
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    messages = [item for item in session.get("messages", []) if isinstance(item, dict)]
    steps: list[dict[str, Any]] = []
    first_user = True
    index = 0
    while index < len(messages):
        message = messages[index]
        role = message.get("role")
        if role == "user":
            content = _content_text(message.get("content"))
            if first_user and instruction.strip():
                content = instruction.strip()
            first_user = False
            if content:
                steps.append(
                    {
                        "step_id": len(steps) + 1,
                        "source": "user",
                        "message": content,
                    }
                )
        elif role == "assistant":
            content = _content_text(message.get("content"))
            tool_calls = _hermes_tool_calls(message.get("tool_calls"))
            observations: list[dict[str, Any]] = []
            cursor = index + 1
            while cursor < len(messages) and messages[cursor].get("role") == "tool":
                tool_message = messages[cursor]
                observations.append(
                    _without_none(
                        {
                            "source_call_id": tool_message.get("tool_call_id"),
                            "content": _content_text(tool_message.get("content")) or None,
                        }
                    )
                )
                cursor += 1
            if content or tool_calls:
                steps.append(
                    _without_none(
                        {
                            "step_id": len(steps) + 1,
                            "source": "agent",
                            "message": content or "[tool call]",
                            "model_name": model_name,
                            "reasoning_content": (
                                message.get("reasoning_content")
                                if isinstance(message.get("reasoning_content"), str)
                                else None
                            ),
                            "tool_calls": tool_calls or None,
                            "observation": ({"results": observations} if observations else None),
                            "llm_call_count": 1,
                        }
                    )
                )
            index = cursor - 1
        index += 1
    return steps


def _validate_hermes_session(
    session: dict[str, Any],
    instruction: str,
) -> dict[str, Any]:
    messages = [item for item in session.get("messages", []) if isinstance(item, dict)]
    message_ids = [item.get("id") for item in messages]
    unique_message_ids = all(value is not None for value in message_ids) and len(
        message_ids
    ) == len(set(message_ids))
    timestamps = [
        float(item["timestamp"])
        for item in messages
        if isinstance(item.get("timestamp"), (int, float))
    ]
    timestamps_ordered = timestamps == sorted(timestamps)
    flattened_tool_calls = sum(
        len(item.get("tool_calls") or [])
        for item in messages
        if isinstance(item.get("tool_calls"), list)
    )
    tool_call_ids = {
        str(call.get("id"))
        for item in messages
        if isinstance(item.get("tool_calls"), list)
        for call in item["tool_calls"]
        if isinstance(call, dict) and call.get("id")
    }
    tool_result_ids = {
        str(item.get("tool_call_id"))
        for item in messages
        if item.get("role") == "tool" and item.get("tool_call_id")
    }
    first_user = next(
        (item for item in messages if item.get("role") == "user"),
        None,
    )
    instruction_matches = first_user is not None and _normalize_text(
        _content_text(first_user.get("content"))
    ) == _normalize_text(instruction)
    final_message = messages[-1] if messages else {}
    terminal_event_seen = (
        final_message.get("role") == "assistant"
        and bool(_content_text(final_message.get("content")).strip())
        and final_message.get("finish_reason") == "stop"
        and tool_call_ids <= tool_result_ids
    )
    return {
        "message_count_matches": session.get("message_count") == len(messages),
        "tool_call_count_matches": (session.get("tool_call_count") == flattened_tool_calls),
        "message_ids_unique": unique_message_ids,
        "timestamps_ordered": timestamps_ordered,
        "instruction_matches": instruction_matches,
        "unmatched_tool_call_ids": sorted(tool_call_ids - tool_result_ids),
        "orphan_tool_result_ids": sorted(tool_result_ids - tool_call_ids),
        "terminal_event_seen": terminal_event_seen,
    }


def _hermes_tool_calls(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    calls: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        calls.append(
            {
                "tool_call_id": str(item.get("id") or uuid.uuid4().hex[:8]),
                "function_name": str(function.get("name") or "unknown"),
                "arguments": _arguments(function.get("arguments")),
            }
        )
    return calls


def _hermes_message_models(session: dict[str, Any]) -> list[str]:
    models: list[str] = []
    for message in session.get("messages", []):
        if isinstance(message, dict) and isinstance(message.get("model"), str):
            models.append(message["model"])
    return models


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return " ".join(
        str(part.get("text") or part.get("content") or "")
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text") or part.get("content"), str)
    )


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    return events


def _claude_content(
    content: Any,
) -> tuple[str, str, list[dict[str, Any]]]:
    if not isinstance(content, list):
        return "", "", []
    text: list[str] = []
    reasoning: list[str] = []
    calls: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "")
        if part_type == "text" and isinstance(part.get("text"), str):
            text.append(part["text"])
        elif part_type == "thinking" and isinstance(part.get("thinking"), str):
            reasoning.append(part["thinking"])
        elif part_type == "tool_use" and isinstance(part.get("name"), str):
            calls.append(
                {
                    "tool_call_id": str(part.get("id") or uuid.uuid4().hex[:8]),
                    "function_name": part["name"],
                    "arguments": _arguments(part.get("input")),
                }
            )
    return "\n".join(text), "\n".join(reasoning), calls


def _claude_tool_results(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return []
    results: list[dict[str, Any]] = []
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "tool_result":
            continue
        results.append(
            _without_none(
                {
                    "source_call_id": str(part.get("tool_use_id") or ""),
                    "content": _content_text(part.get("content")) or None,
                    "is_error": part.get("is_error"),
                }
            )
        )
    return results


def _add_model(models: set[str], value: Any, run: RunSpec) -> None:
    if isinstance(value, str) and value:
        models.add(_normalize_observed_model(value, run))


def _normalize_observed_model(value: str, run: RunSpec) -> str:
    if value == run.proxy_model_name:
        return run.model_id
    for prefix in ("anthropic/", "openai/"):
        if value.startswith(prefix):
            return value[len(prefix) :]
    return value


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"raw": value}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _unavailable(
    source_path: Path,
    *,
    runtime_model_name: str | None = None,
    canonical_model_identity: bool = False,
    observed_models: set[str] | None = None,
    extra_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validation = {
        "observed_models": sorted(observed_models or set()),
    }
    if extra_validation:
        validation.update(extra_validation)
    return {
        "trajectory_status": "unavailable",
        "trajectory_source": str(source_path),
        "trajectory_event_count": 0,
        "runtime_model_name": runtime_model_name,
        "canonical_model_identity": canonical_model_identity,
        "trajectory_validation": validation,
    }


def _nested_string(value: dict[str, Any], *keys: str) -> str | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current if isinstance(current, str) else None


def _without_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0


def _normalize_text(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.strip().splitlines())


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
