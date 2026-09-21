"""Capture completed native OpenClaw sessions, including delegated continuations.

The OpenClaw trajectory snapshot survives child-session cleanup. Read it on the
host; never instruct the actor to poll or send a synthetic continuation prompt.
Relay reconciliation remains necessary before these transcripts can be graded.
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
import io
from pathlib import Path
import sqlite3

import zstandard

from clawbench.client import _correlate_transcript, _parse_single_message
from clawbench.schemas import Transcript


def _session_file(path: Path) -> Path:
    transcript_path = path.with_name(path.name.removesuffix(".trajectory.jsonl") + ".jsonl")
    if not transcript_path.exists():
        archived = list(transcript_path.parent.glob(transcript_path.name + ".deleted.*"))
        if len(archived) > 1:
            raise ValueError("Native transcript recovery has ambiguous archived sessions")
        if archived:
            transcript_path = archived[0]
    return transcript_path


def _untruncated_messages(
    path: Path, captured_at: str, *, require_final: bool = True
) -> list[dict]:
    """Recover a linear native transcript when the trajectory reports truncation.

    Branches and compactions need their own replay semantics, so leave those
    unknown. The caller still requires native termination and relay agreement.
    """
    transcript_path = _session_file(path)
    if transcript_path.is_symlink():
        raise ValueError("Native transcript recovery cannot follow a symlink")
    rows = [json.loads(line) for line in transcript_path.read_text().splitlines()]
    return _linear_messages(
        rows, path.name.removesuffix(".trajectory.jsonl"), captured_at, require_final=require_final
    )


def _linear_messages(
    rows: list[dict], session_id: str, captured_at: str, *, require_final: bool = True
) -> list[dict]:
    if not rows or rows[0].get("type") != "session":
        raise ValueError("Native transcript recovery lacks a session header")
    if rows[0].get("id") != session_id:
        raise ValueError("Native transcript recovery has a different session identity")
    previous = None
    seen = set()
    messages = []
    for row in rows[1:]:
        if not isinstance(row.get("timestamp"), str):
            raise ValueError("Native transcript recovery lacks an entry timestamp")
        if row["timestamp"] > captured_at:
            break
        ident = row.get("id")
        if not isinstance(ident, str) or not ident or ident in seen:
            raise ValueError("Native transcript recovery has invalid entry identities")
        if row.get("parentId") != previous:
            raise ValueError("Native transcript recovery requires an unbranched history")
        if row.get("type") not in {
            "message",
            "model_change",
            "thinking_level_change",
            "custom",
            "custom_message",
        }:
            raise ValueError("Native transcript recovery has unsupported history entries")
        previous = ident
        seen.add(ident)
        if row["type"] == "message":
            message = row.get("message")
            if not isinstance(message, dict):
                raise ValueError("Native transcript recovery has an invalid message")
            messages.append(message)
        elif row["type"] == "custom_message":
            # Preserve native context/handoff text as runtime data, without
            # relabeling it as a user instruction or a provider response.
            messages.append({"role": "custom", "content": row.get("content", "")})
    assistants = [message for message in messages if message.get("role") == "assistant"]
    last_assistant = assistants[-1] if assistants else {}
    yielded = any(
        isinstance(block, dict)
        and block.get("type") == "toolCall"
        and block.get("name") == "sessions_yield"
        for block in last_assistant.get("content", [])
    )
    if require_final and last_assistant.get("stopReason") != "stop" and not yielded:
        raise ValueError("Native transcript recovery lacks a final assistant response")
    # A native yield is a valid terminal turn, but not a finished user task.
    # Preserve it so read_tree can report pending until the native continuation.
    return messages


def native_sources(state: Path) -> list[dict]:
    """Read native trajectory and transcript storage from both runtime formats."""
    sources = []
    for path in sorted(state.glob("agents/*/sessions/*.trajectory.jsonl")):
        events = []
        partial = False
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                partial = True
        sources.append({"path": path, "events": events, "entries": None, "partial": partial})
    for path in sorted(state.glob("agents/*/agent/openclaw-agent.sqlite")):
        if path.is_symlink():
            raise ValueError("Native database capture cannot follow a symlink")
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as database:
            # Trajectories and transcript entries must come from one consistent
            # read snapshot while the Gateway may still be writing.
            database.execute("BEGIN")
            ids = database.execute(
                "SELECT DISTINCT session_id FROM trajectory_runtime_events"
            ).fetchall()
            for (session_id,) in ids:
                if not isinstance(session_id, str) or Path(session_id).name != session_id:
                    raise ValueError("Invalid native database session identity")
                events = [
                    json.loads(row[0])
                    for row in database.execute(
                        "SELECT event_json FROM trajectory_runtime_events WHERE session_id=? ORDER BY seq",
                        (session_id,),
                    )
                ]
                entries = [
                    json.loads(row[0])
                    for row in database.execute(
                        "SELECT event_json FROM transcript_events WHERE session_id=? ORDER BY seq",
                        (session_id,),
                    )
                ]
                sources.append(
                    {
                        "path": path.parent / f"{session_id}.trajectory.jsonl",
                        "events": events,
                        "entries": entries,
                        "partial": False,
                        "database": path,
                    }
                )
            has_archives = database.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='session_transcript_archives'"
            ).fetchone()
            if has_archives:
                for ident, key, reason, encoding, blob, digest, created in database.execute(
                    "SELECT session_id,session_key,reason,encoding,archive_blob,archive_sha256,created_at "
                    "FROM session_transcript_archives WHERE reason='deleted'"
                ):
                    if not ident or Path(ident).name != ident or not isinstance(key, str):
                        raise ValueError("Invalid native archive session identity")
                    if encoding != "zstd" or hashlib.sha256(blob).hexdigest() != digest:
                        raise ValueError("Native transcript archive encoding or checksum mismatch")
                    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(blob)) as reader:
                        decoded = reader.read(64 * 1024 * 1024 + 1)
                    if len(decoded) > 64 * 1024 * 1024:
                        raise ValueError("Native transcript archive exceeds capture limit")
                    sources.append(
                        {
                            "path": path.parent / f"{ident}.trajectory.jsonl",
                            "events": [],
                            "entries": [json.loads(line) for line in decoded.splitlines() if line],
                            "partial": False,
                            "database": path,
                            "archive": {
                                "session_key": key,
                                "reason": reason,
                                "sha256": digest,
                                "archived_at": datetime.fromtimestamp(created / 1000, timezone.utc)
                                .isoformat(timespec="milliseconds")
                                .replace("+00:00", "Z"),
                            },
                        }
                    )
    return sources


def _source_messages(source: dict, captured_at: str, *, require_final: bool = True) -> list[dict]:
    if source["entries"] is not None:
        return _linear_messages(
            source["entries"],
            source["path"].name.removesuffix(".trajectory.jsonl"),
            captured_at,
            require_final=require_final,
        )
    return _untruncated_messages(source["path"], captured_at, require_final=require_final)


def _empty_abort_placeholder(raw: dict, aborts: list[dict]) -> bool:
    usage = raw.get("usage")
    timestamp = raw.get("timestamp")
    return (
        raw.get("content") == []
        and raw.get("stopReason") == "aborted"
        and raw.get("errorMessage") == "Request was aborted."
        and isinstance(usage, dict)
        and all(
            usage.get(key) == 0
            for key in ("input", "output", "cacheRead", "cacheWrite", "totalTokens")
        )
        and isinstance(timestamp, (int, float))
        and any(entry["started_ms"] <= timestamp <= entry["ended_ms"] for entry in aborts)
    )


def accepted_children(transcript: dict) -> list[str]:
    children = []
    for message in transcript.get("messages", []):
        for call in message.get("tool_calls", []):
            if call["name"] != "sessions_spawn":
                continue
            try:
                result = json.loads(call.get("output", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(result, dict) and result.get("status") == "accepted":
                key = result.get("childSessionKey")
                if not isinstance(key, str) or not key:
                    raise ValueError("Accepted delegation has no native child session key")
                children.append(key)
    return list(dict.fromkeys(children))


def read_tree(state: Path, root_key: str, *, allow_interrupted: bool = False) -> dict:
    """Return pending until every child and the parent's native continuation end."""
    sessions: dict[str, dict] = {}
    incomplete = []
    sources = native_sources(state)
    for source in sources:
        path = source["path"]
        if source["partial"]:
            incomplete.append(str(path.relative_to(state)))
        for event in source["events"]:
            key = event.get("sessionKey")
            if not isinstance(key, str):
                continue
            record = sessions.setdefault(
                key, {"active": set(), "raw": None, "ended": "", "starts": {}, "aborts": []}
            )
            run = event.get("runId")
            if event.get("type") == "session.started":
                record["active"].add(run)
                record["starts"][run] = datetime.fromisoformat(event["ts"]).timestamp() * 1000
            elif event.get("type") == "model.completed":
                data = event.get("data", {})
                snapshot = data.get("messagesSnapshot")
                # Never keep an older completed snapshot when a later one is
                # truncated or unavailable.
                record["raw"] = None
                record["recovery_path"] = (
                    path
                    if data.get("truncated") is True
                    and data.get("reason") == "trajectory-event-size-limit"
                    else None
                )
                record["captured_at"] = event["ts"]
                record["path"] = path
                record["source"] = source
                if isinstance(snapshot, list):
                    record["raw"] = snapshot
            elif event.get("type") == "session.ended":
                record["active"].discard(run)
                record["ended"] = event["ts"]
                record["status"] = event.get("data", {}).get("status")
                record["terminal_aborted"] = event.get("data", {}).get("aborted") is True
                if event.get("data", {}).get("aborted") is True and run in record["starts"]:
                    record["aborts"].append(
                        {
                            "run_id": run,
                            "started_ms": record["starts"][run],
                            "ended_ms": datetime.fromisoformat(event["ts"]).timestamp() * 1000,
                        }
                    )

    archived: dict[str, list[dict]] = {}
    for source in sources:
        if source.get("archive"):
            archived.setdefault(source["archive"]["session_key"], []).append(source)
    for key, candidates in archived.items():
        if key in sessions:
            continue  # An old archive cannot stand in for a live/newer history.
        if len(candidates) != 1:
            raise ValueError("Native transcript archive has ambiguous generations")
        source = candidates[0]
        raw = _source_messages(source, source["archive"]["archived_at"])
        assistants = [message for message in raw if message.get("role") == "assistant"]
        if not assistants or assistants[-1].get("stopReason") != "stop":
            continue
        # Deleted native archive + complete linear final stop establishes a
        # retained completed transcript. Do not invent a session.ended event.
        # Independent relay reconciliation is still required by the runner.
        captured_at = source["entries"][-1]["timestamp"]
        sessions[key] = {
            "active": set(),
            "raw": raw,
            "ended": captured_at,
            "captured_at": captured_at,
            "starts": {},
            "aborts": [],
            "status": "success",
            "source": source,
            "path": source["path"],
            "archive": source["archive"],
        }

    queue = [root_key]
    seen_keys = set()
    seen_responses: dict[str, str] = {}
    captured = []
    pending = [f"incomplete-record:{name}" for name in incomplete]
    latest_child_end = ""
    root_capture_time = ""
    while queue:
        key = queue.pop(0)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        record = sessions.get(key)
        require_final = not (
            allow_interrupted
            and record
            and record.get("terminal_aborted")
            and record.get("status") == "interrupted"
            and record["aborts"]
        )
        if (
            record
            and record.get("recovery_path")
            and not record["active"]
            and record["ended"] >= record["captured_at"]
        ):
            record["raw"] = _source_messages(
                record["source"], record["captured_at"], require_final=require_final
            )
        if not record or record["raw"] is None or record["active"] or not record["ended"]:
            pending.append(key)
            continue
        raw_path = _session_file(record["path"]) if record["source"]["entries"] is None else None
        capture_source = "trajectory_snapshot"
        if record["source"]["entries"] is not None or (raw_path is not None and raw_path.exists()):
            try:
                # Native trajectory logging can redact values inside tool
                # arguments. A complete linear session preserves the real call.
                raw_messages = _source_messages(
                    record["source"], record["captured_at"], require_final=require_final
                )
            except ValueError:
                # A full native snapshot still handles histories that need
                # branch/compaction semantics. Independent relay reconciliation
                # will reject any lossy snapshot; never guess missing actions.
                if record.get("recovery_path"):
                    raise
            else:
                record["raw"] = raw_messages
                capture_source = (
                    "native_sqlite_archive"
                    if record.get("archive")
                    else "native_sqlite"
                    if record["source"]["entries"] is not None
                    else "native_session"
                )
        messages = []
        lifecycle = []
        provider_response_ids = []
        for raw in record["raw"]:
            if not isinstance(raw, dict):
                raise ValueError("Native snapshot contains truncated or invalid messages")
            # A fork can include the parent's already observed assistant output.
            # Count each provider response once, retaining the child's new work.
            if raw.get("role") == "assistant":
                response_id = raw.get("responseId")
                if not isinstance(response_id, str) or not response_id:
                    if _empty_abort_placeholder(raw, record["aborts"]):
                        lifecycle.append({"message": raw, "native_aborts": record["aborts"]})
                        continue
                    raise ValueError("Native assistant snapshot lacks provider response ID")
                signature = json.dumps(raw, sort_keys=True)
                if response_id in seen_responses:
                    if seen_responses[response_id] != signature:
                        raise ValueError("Conflicting native records for one provider response")
                    continue
                seen_responses[response_id] = signature
                provider_response_ids.append(response_id)
            parsed = _parse_single_message(raw)
            if parsed is not None:
                messages.append(parsed)
        transcript = _correlate_transcript(Transcript(messages=messages))
        transcript.stop_reason = "complete" if record.get("status") == "success" else "error"
        queue.extend(accepted_children(transcript.model_dump()))
        assistants = transcript.assistant_messages
        if assistants and any(c.name == "sessions_yield" for c in assistants[-1].tool_calls):
            pending.append(key)
        if key == root_key:
            root_capture_time = record["captured_at"]
        else:
            latest_child_end = max(latest_child_end, record["ended"])
        captured.append(
            {
                "session_key": key,
                "transcript": transcript.model_dump(),
                "capture_source": capture_source,
                "runtime_lifecycle": lifecycle,
                "provider_response_ids": provider_response_ids,
                **(
                    {
                        "native_archive": record["archive"],
                        "terminal_basis": "deleted archive with linear native final stop",
                    }
                    if record.get("archive")
                    else {}
                ),
            }
        )
    # Child completion delivery can resume a previously finished parent. Wait
    # for that native turn too, instead of stopping the container in the gap.
    if latest_child_end > root_capture_time:
        pending.append(root_key)
    return {"complete": not pending, "pending": sorted(set(pending)), "sessions": captured}
