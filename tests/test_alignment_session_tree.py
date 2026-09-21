"""Native delegation lifecycle regression: no synthetic follow-up or early stop."""

import json
import sqlite3
import hashlib

import zstandard

import pytest

from clawbench.alignment.session_tree import read_tree


def assistant(response_id, text="", calls=()):
    return {
        "role": "assistant",
        "responseId": response_id,
        "content": [
            *([{"type": "text", "text": text}] if text else []),
            *[
                {"type": "toolCall", "id": ident, "name": name, "arguments": args}
                for ident, name, args in calls
            ],
        ],
    }


def result(ident, output):
    return {
        "role": "toolResult",
        "toolCallId": ident,
        "content": [{"type": "text", "text": json.dumps(output)}],
        "isError": False,
    }


SPAWN = assistant("root-1", calls=[("spawn-1", "sessions_spawn", {"task": "verify"})])
ACCEPTED = result("spawn-1", {"status": "accepted", "childSessionKey": "child"})
YIELD = assistant("root-2", calls=[("yield-1", "sessions_yield", {})])
CHILD = assistant("child-1", "Verified", [("read-1", "read", {"path": "source"})])
FINAL = assistant("root-3", "Completed and independently verified")


def trace(tmp_path, key, messages, start, end=None, *, filename=None):
    folder = tmp_path / "agents/a/sessions"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{filename or key}.trajectory.jsonl"

    def event(kind, ts, data):
        return {
            "type": kind,
            "ts": f"2026-09-21T00:00:{ts:02d}.000Z",
            "runId": f"{key}-{start}",
            "sessionKey": key,
            "data": data,
        }

    records = [event("session.started", start, {})]
    if messages is not None:
        records.append(event("model.completed", end or start, {"messagesSnapshot": messages}))
    if end is not None:
        records.append(event("session.ended", end, {"status": "success"}))
    with path.open("a") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")
    return path


def test_yield_is_pending_even_if_children_already_finished(tmp_path):
    trace(tmp_path, "root", [SPAWN, ACCEPTED, YIELD], 1, 3)
    trace(tmp_path, "child", [CHILD], 2, 4)
    tree = read_tree(tmp_path, "root")
    assert not tree["complete"] and tree["pending"] == ["root"]


def test_native_parent_resume_and_cleaned_child_are_captured(tmp_path):
    trace(tmp_path, "root", [SPAWN, ACCEPTED, YIELD], 1, 3)
    trace(tmp_path, "child", [CHILD], 2, 4)
    trace(tmp_path, "root", [SPAWN, ACCEPTED, YIELD, FINAL], 5, 6)
    tree = read_tree(tmp_path, "root")
    assert tree["complete"]
    assert [s["session_key"] for s in tree["sessions"]] == ["root", "child"]
    assert tree["sessions"][0]["transcript"]["messages"][-1]["text"] == FINAL["content"][0]["text"]
    assert tree["sessions"][1]["transcript"]["messages"][0]["tool_calls"][0]["name"] == "read"


def test_parent_final_before_child_completion_waits_for_auto_announce(tmp_path):
    trace(tmp_path, "root", [SPAWN, ACCEPTED, FINAL], 1, 3)
    trace(tmp_path, "child", [CHILD], 2, 4)
    assert read_tree(tmp_path, "root")["pending"] == ["root"]


def test_missing_or_still_running_child_is_pending(tmp_path):
    trace(tmp_path, "root", [SPAWN, ACCEPTED, FINAL], 1, 6)
    assert read_tree(tmp_path, "root")["pending"] == ["child"]
    trace(tmp_path, "child", None, 2)
    assert read_tree(tmp_path, "root")["pending"] == ["child"]


def test_fork_context_does_not_double_count_parent_model_actions(tmp_path):
    trace(tmp_path, "root", [SPAWN, ACCEPTED, YIELD, FINAL], 1, 6)
    trace(tmp_path, "child", [SPAWN, ACCEPTED, CHILD], 2, 4)
    tree = read_tree(tmp_path, "root")
    assert tree["complete"]
    child = tree["sessions"][1]["transcript"]
    calls = [c["name"] for m in child["messages"] for c in m["tool_calls"]]
    assert calls == ["read"]
    assert tree["sessions"][0]["provider_response_ids"] == ["root-1", "root-2", "root-3"]
    assert tree["sessions"][1]["provider_response_ids"] == ["child-1"]


def test_partial_new_record_prevents_accepting_old_terminal_snapshot(tmp_path):
    path = trace(tmp_path, "root", [FINAL], 1, 3)
    with path.open("a") as file:
        file.write('{"type":"session.started"')
    assert not read_tree(tmp_path, "root")["complete"]


def test_active_parent_continuation_is_not_a_finished_tree(tmp_path):
    trace(tmp_path, "root", [SPAWN, ACCEPTED, FINAL], 1, 3)
    trace(tmp_path, "child", [CHILD], 2, 4)
    trace(tmp_path, "root", None, 5)
    assert not read_tree(tmp_path, "root")["complete"]


def truncated_child(tmp_path, *, branch=False, entry_type="message"):
    trace(tmp_path, "root", [SPAWN, ACCEPTED, FINAL], 1, 6)
    path = trace(tmp_path, "child", [CHILD], 2, 4)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["data"] = {
        "truncated": True,
        "reason": "trajectory-event-size-limit",
        "originalBytes": 450024,
        "limitBytes": 262144,
    }
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    raw = path.with_name("child.jsonl")
    messages = [
        {"type": "session", "id": "child"},
        {
            "type": "message",
            "id": "request",
            "parentId": None,
            "timestamp": "2026-09-21T00:00:02.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": "verify"}]},
        },
        {
            "type": entry_type,
            "id": "reply",
            "parentId": None if branch else "request",
            "timestamp": "2026-09-21T00:00:03.000Z",
            "message": {**assistant("child-1", "Verified"), "stopReason": "stop"},
        },
    ]
    raw.write_text("".join(json.dumps(row) + "\n" for row in messages))
    return raw


def test_truncated_native_snapshot_recovers_completed_linear_child(tmp_path):
    truncated_child(tmp_path)
    tree = read_tree(tmp_path, "root")
    assert tree["complete"]
    assert len(tree["sessions"]) == 2
    assert tree["sessions"][1]["transcript"]["messages"][-1]["text"] == "Verified"


def test_truncated_recovery_does_not_guess_branch_semantics(tmp_path):
    truncated_child(tmp_path, branch=True)
    with pytest.raises(ValueError, match="unbranched"):
        read_tree(tmp_path, "root")


def test_truncated_recovery_does_not_silently_ignore_compaction(tmp_path):
    truncated_child(tmp_path, entry_type="compaction")
    with pytest.raises(ValueError, match="unsupported"):
        read_tree(tmp_path, "root")


def test_truncated_recovery_rejects_different_session_identity(tmp_path):
    raw = truncated_child(tmp_path)
    raw.write_text(raw.read_text().replace('"id": "child"', '"id": "other"'))
    with pytest.raises(ValueError, match="session identity"):
        read_tree(tmp_path, "root")


def test_later_truncation_cannot_reuse_older_full_snapshot(tmp_path):
    trace(tmp_path, "root", [FINAL], 1, 3)
    path = trace(tmp_path, "root", [FINAL], 4, 5)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-2]["data"] = {"truncated": True, "reason": "unrecognized"}
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert not read_tree(tmp_path, "root")["complete"]


def test_archived_child_transcript_survives_native_cleanup(tmp_path):
    raw = truncated_child(tmp_path)
    raw.rename(raw.with_name(raw.name + ".deleted.2026-09-21T00-00-05.000Z"))
    tree = read_tree(tmp_path, "root")
    assert tree["complete"]
    assert tree["sessions"][1]["capture_source"] == "native_session"


def test_native_session_restores_arguments_redacted_in_logging(tmp_path):
    raw = truncated_child(tmp_path)
    path = raw.with_name("child.trajectory.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["data"] = {"messagesSnapshot": [assistant("child-1", "<redacted>")]}
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    tree = read_tree(tmp_path, "root")
    assert tree["sessions"][1]["transcript"]["messages"][-1]["text"] == "Verified"


@pytest.mark.parametrize("valid_abort", [True, False])
def test_empty_runtime_abort_is_separate_from_provider_output(tmp_path, valid_abort):
    trace(tmp_path, "root", [FINAL], 1, 3)
    path = tmp_path / "agents/a/sessions/root.trajectory.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["data"].update(status="error", aborted=valid_abort)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    placeholder = {
        "role": "assistant",
        "content": [],
        "stopReason": "aborted",
        "errorMessage": "Request was aborted.",
        "usage": {key: 0 for key in ["input", "output", "cacheRead", "cacheWrite", "totalTokens"]},
        "timestamp": 1789948802000,
    }
    trace(tmp_path, "root", [placeholder, FINAL], 4, 6)
    if not valid_abort:
        with pytest.raises(ValueError, match="provider response ID"):
            read_tree(tmp_path, "root")
        return
    tree = read_tree(tmp_path, "root")
    assert tree["complete"]
    session = tree["sessions"][0]
    assert len(session["runtime_lifecycle"]) == 1
    assert len(session["transcript"]["messages"]) == 1


def test_providerless_content_is_never_discarded_as_abort_metadata(tmp_path):
    trace(
        tmp_path,
        "root",
        [{"role": "assistant", "content": [{"type": "text", "text": "Do something"}]}],
        1,
        3,
    )
    with pytest.raises(ValueError, match="provider response ID"):
        read_tree(tmp_path, "root")


def sqlite_trace(tmp_path, *, interrupted=False, native_abort=True):
    """Actual native storage shape, with deliberately lossy event logging."""
    raw = truncated_child(tmp_path)
    path = raw.with_name("child.trajectory.jsonl")
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[1]["data"] = {"messagesSnapshot": ["[Truncated]"]}
    entries = [json.loads(line) for line in raw.read_text().splitlines()]
    if interrupted:
        entries[-1]["message"] = {**CHILD, "stopReason": "toolUse"}
        events[-1]["data"] = {"status": "interrupted", "aborted": native_abort}
    target = tmp_path / "agents/a/agent/openclaw-agent.sqlite"
    target.parent.mkdir()
    with sqlite3.connect(target) as database:
        for table, rows in [("trajectory_runtime_events", events), ("transcript_events", entries)]:
            database.execute(f"CREATE TABLE {table} (session_id TEXT, seq INT, event_json TEXT)")
            database.executemany(
                f"INSERT INTO {table} VALUES (?, ?, ?)",
                [("child", i, json.dumps(row)) for i, row in enumerate(rows)],
            )
    path.unlink()
    raw.unlink()
    return target


def test_sqlite_child_retains_complete_native_messages(tmp_path):
    database = sqlite_trace(tmp_path)
    before = database.read_bytes()
    tree = read_tree(tmp_path, "root")
    assert tree["complete"]
    child = tree["sessions"][1]
    assert child["capture_source"] == "native_sqlite"
    assert child["transcript"]["messages"][-1]["text"] == "Verified"
    assert database.read_bytes() == before


def test_interrupted_sqlite_capture_requires_explicit_mode_and_native_proof(tmp_path):
    sqlite_trace(tmp_path, interrupted=True)
    with pytest.raises(ValueError, match="truncated or invalid"):
        read_tree(tmp_path, "root")
    tree = read_tree(tmp_path, "root", allow_interrupted=True)
    assert tree["complete"]
    child = tree["sessions"][1]["transcript"]
    assert child["stop_reason"] == "error"
    assert child["messages"][-1]["tool_calls"][0]["name"] == "read"


def test_interrupted_sqlite_status_alone_does_not_authorize_recovery(tmp_path):
    sqlite_trace(tmp_path, interrupted=True, native_abort=False)
    with pytest.raises(ValueError, match="truncated or invalid"):
        read_tree(tmp_path, "root", allow_interrupted=True)


def test_sqlite_unsafe_identity_is_rejected(tmp_path):
    path = sqlite_trace(tmp_path)
    with sqlite3.connect(path) as database:
        database.execute("UPDATE trajectory_runtime_events SET session_id='../other'")
    with pytest.raises(ValueError, match="session identity"):
        read_tree(tmp_path, "root")


def test_sqlite_yield_with_runtime_context_waits_for_native_continuation(tmp_path):
    path = sqlite_trace(tmp_path)
    with sqlite3.connect(path) as database:
        rows = database.execute(
            "SELECT seq,event_json FROM transcript_events WHERE session_id='child' ORDER BY seq"
        ).fetchall()
        final = json.loads(rows[-1][1])
        final["message"] = {**YIELD, "stopReason": "toolUse"}
        database.execute(
            "UPDATE transcript_events SET event_json=? WHERE seq=?",
            (json.dumps(final), rows[-1][0]),
        )
        context = {
            "type": "custom_message",
            "id": "handoff",
            "parentId": final["id"],
            "timestamp": "2026-09-21T00:00:03.500Z",
            "content": "Native pending child context",
        }
        database.execute(
            "INSERT INTO transcript_events VALUES ('child', ?, ?)",
            (rows[-1][0] + 1, json.dumps(context)),
        )
    tree = read_tree(tmp_path, "root")
    assert not tree["complete"] and tree["pending"] == ["child"]
    child = tree["sessions"][1]["transcript"]
    assert child["messages"][-1]["role"] == "custom"
    assert child["messages"][-1]["text"] == "Native pending child context"


def archived_sqlite_child(tmp_path, change=None):
    path = sqlite_trace(tmp_path)
    with sqlite3.connect(path) as database:
        rows = [
            json.loads(r[0])
            for r in database.execute(
                "SELECT event_json FROM transcript_events WHERE session_id='child' ORDER BY seq"
            )
        ]
        if change == "identity":
            rows[0]["id"] = "different"
        elif change == "branch":
            rows[-1]["parentId"] = "missing"
        elif change == "yield":
            rows[-1]["message"] = {**YIELD, "stopReason": "toolUse"}
        blob = zstandard.ZstdCompressor().compress(
            ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
        )
        database.execute(
            "CREATE TABLE session_transcript_archives (session_id TEXT,session_key TEXT,reason TEXT,encoding TEXT,archive_blob BLOB,archive_sha256 TEXT,created_at INTEGER)"
        )
        values = (
            "child",
            "child",
            "deleted",
            "zstd",
            blob,
            hashlib.sha256(blob).hexdigest(),
            1789948805000,
        )
        database.execute("INSERT INTO session_transcript_archives VALUES (?,?,?,?,?,?,?)", values)
        if change == "ambiguous":
            database.execute(
                "INSERT INTO session_transcript_archives VALUES (?,?,?,?,?,?,?)", values
            )
        if change == "checksum":
            database.execute("UPDATE session_transcript_archives SET archive_sha256='wrong'")
        if change == "encoding":
            database.execute("UPDATE session_transcript_archives SET encoding='unknown'")
        database.execute("DELETE FROM transcript_events WHERE session_id='child'")
        database.execute("DELETE FROM trajectory_runtime_events WHERE session_id='child'")
    return path


def test_deleted_sqlite_archive_is_captured_without_inventing_a_runtime_event(tmp_path):
    path = archived_sqlite_child(tmp_path)
    before = path.read_bytes()
    tree = read_tree(tmp_path, "root")
    assert tree["complete"]
    child = tree["sessions"][1]
    assert child["capture_source"] == "native_sqlite_archive"
    assert child["terminal_basis"] == "deleted archive with linear native final stop"
    assert child["transcript"]["messages"][-1]["text"] == "Verified"
    assert path.read_bytes() == before


@pytest.mark.parametrize("change", ["identity", "branch", "ambiguous", "checksum", "encoding"])
def test_invalid_or_ambiguous_archives_fail_closed(tmp_path, change):
    archived_sqlite_child(tmp_path, change)
    with pytest.raises(ValueError):
        read_tree(tmp_path, "root")


def test_archived_yield_is_not_a_completed_child(tmp_path):
    archived_sqlite_child(tmp_path, "yield")
    assert read_tree(tmp_path, "root")["pending"] == ["child"]


def test_old_archive_cannot_replace_a_new_running_child(tmp_path):
    archived_sqlite_child(tmp_path)
    trace(tmp_path, "child", None, 7)
    assert read_tree(tmp_path, "root")["pending"] == ["child"]
