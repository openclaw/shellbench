from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

from scripts.native_eval.models import RunSpec


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
    session_path = agent_dir / "openclaw.session.jsonl"
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
    session_models = _openclaw_session_models(session_path)
    log_models = _openclaw_log_models(log_path)
    observed_models = session_models or envelope_models
    if not observed_models and run.model_id in log_models:
        observed_models = {run.model_id}
    runtime_model_name = (
        next(iter(observed_models)) if len(observed_models) == 1 else None
    )
    canonical_model_identity = observed_models == {run.model_id}

    steps = _openclaw_session_steps(
        session_path,
        instruction=instruction,
        model_name=f"{run.provider}/{run.model_id}",
    )
    trace_fidelity = "session"
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
    if len(steps) < 2:
        return _unavailable(
            source_path,
            runtime_model_name=runtime_model_name,
            canonical_model_identity=canonical_model_identity,
            observed_models=observed_models,
        )

    usage = agent_meta.get("usage")
    if not isinstance(usage, dict):
        usage = _openclaw_session_usage(session_path)
    input_tokens = _int(usage.get("input"))
    output_tokens = _int(usage.get("output"))
    cache_read = _int(usage.get("cacheRead"))
    session_id = str(
        agent_meta.get("sessionId")
        or _openclaw_session_id(session_path)
        or uuid.uuid4()
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
                "total_prompt_tokens": input_tokens + cache_read or None,
                "total_completion_tokens": output_tokens or None,
                "total_cached_tokens": cache_read or None,
                "total_steps": len(steps),
            }
        ),
        "extra": {
            "native_raw_trace_file": source_path.name,
            "trace_fidelity": trace_fidelity,
            "observed_models": sorted(observed_models),
            "log_models": sorted(log_models),
            "stop_reason": _nested_string(meta, "completion", "stopReason"),
            "aborted": meta.get("aborted"),
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
            "log_models": sorted(log_models),
            "session_id": session_id,
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
    observed_models = {
        value
        for value in [session.get("model"), *_hermes_message_models(session)]
        if isinstance(value, str) and value
    }
    runtime_model_name = (
        next(iter(observed_models)) if len(observed_models) == 1 else None
    )
    canonical_model_identity = observed_models == {run.model_id}
    validation = _validate_hermes_session(session, instruction)
    steps = _hermes_steps(
        session,
        instruction,
        model_name=f"{run.provider}/{run.model_id}",
    )
    if len(steps) < 2:
        return _unavailable(
            source_path,
            runtime_model_name=runtime_model_name,
            canonical_model_identity=canonical_model_identity,
            observed_models=observed_models,
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
                "total_cached_tokens": (
                    _int(session.get("cache_read_tokens")) or None
                ),
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
            "prompt_tokens": input_tokens + cache_read or None,
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
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict) or record.get("type") != "message":
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
            body = _content_text(message.get("content"))
            text = instruction.strip() if first_user and instruction.strip() else body
            first_user = False
            steps.append(
                _without_none(
                    {
                        "step_id": len(steps) + 1,
                        "source": "user",
                        "message": text or "(empty user message)",
                        "timestamp": timestamp,
                    }
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

        pending = {
            call["tool_call_id"] for call in tool_calls if call.get("tool_call_id")
        }
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
            _without_none(
                {
                    "step_id": len(steps) + 1,
                    "source": "agent",
                    "message": agent_text,
                    "timestamp": timestamp,
                    "model_name": model_name,
                    "tool_calls": tool_calls or None,
                    "observation": (
                        {"results": observations} if observations else None
                    ),
                    "metrics": metrics,
                    "llm_call_count": 1,
                }
            )
        )
        index = cursor
    return steps if len(steps) >= 2 else []


def _openclaw_session_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _openclaw_session_models(path: Path) -> set[str]:
    models: set[str] = set()
    for record in _openclaw_session_records(path):
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


def _openclaw_log_models(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(
        re.findall(
            r"\[model-fetch\].*?\bmodel=([^\s]+)",
            path.read_text(encoding="utf-8", errors="replace"),
        )
    )


def _openclaw_session_id(path: Path) -> str | None:
    for record in _openclaw_session_records(path):
        if record.get("type") == "session" and isinstance(record.get("id"), str):
            return record["id"]
    return path.stem if path.is_file() else None


def _openclaw_session_usage(path: Path) -> dict[str, int]:
    totals = {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
    for record in _openclaw_session_records(path):
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
            "prompt_tokens": input_tokens + cache_read or None,
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
    messages = [
        item for item in session.get("messages", []) if isinstance(item, dict)
    ]
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
                            "content": _content_text(tool_message.get("content"))
                            or None,
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
                                if isinstance(
                                    message.get("reasoning_content"), str
                                )
                                else None
                            ),
                            "tool_calls": tool_calls or None,
                            "observation": (
                                {"results": observations} if observations else None
                            ),
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
    messages = [
        item for item in session.get("messages", []) if isinstance(item, dict)
    ]
    message_ids = [item.get("id") for item in messages]
    unique_message_ids = all(
        value is not None for value in message_ids
    ) and len(message_ids) == len(set(message_ids))
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
        "tool_call_count_matches": (
            session.get("tool_call_count") == flattened_tool_calls
        ),
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
        if isinstance(part, dict)
        and isinstance(part.get("text") or part.get("content"), str)
    )


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
) -> dict[str, Any]:
    return {
        "trajectory_status": "unavailable",
        "trajectory_source": str(source_path),
        "trajectory_event_count": 0,
        "runtime_model_name": runtime_model_name,
        "canonical_model_identity": canonical_model_identity,
        "trajectory_validation": {
            "observed_models": sorted(observed_models or set()),
        },
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
