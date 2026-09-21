from types import SimpleNamespace
import json
from pathlib import Path
from typing import Any

import pytest

from clawbench.alignment import runtime
from clawbench.alignment.portable import load_portable_cases, seed_portable_workspace
from scripts import run_portable_alignment as runner


@pytest.mark.parametrize("mode", [None, "off", "propose", "auto"])
def test_workshop_profile_is_explicit_and_defaults_are_preserved(monkeypatch, mode):
    monkeypatch.setattr(runtime, "WORKSHOP_MODE", mode)
    config = runtime.actor_config("openai/test-model")
    if mode is None:
        assert "skills" not in config
    else:
        assert config["skills"]["workshop"]["autonomous"]["mode"] == mode


@pytest.mark.parametrize("mismatch", [None, "device", "mode", "role", "scopes"])
def test_device_bootstrap_approves_only_the_exact_local_client(tmp_path, monkeypatch, mismatch):
    state = tmp_path / "state"
    (state / "identity").mkdir(parents=True)
    (state / "identity/device.json").write_text(
        json.dumps({"deviceId": "owned", "privateKeyPem": "do-not-log"})
    )
    pending = {
        "requestId": "request-1",
        "deviceId": "owned",
        "clientId": "gateway-client",
        "clientMode": "backend",
        "role": "operator",
        "scopes": [
            "operator.admin",
            "operator.read",
            "operator.write",
            "operator.approvals",
            "operator.pairing",
        ],
    }
    if mismatch:
        key = {"device": "deviceId", "mode": "clientMode", "role": "role", "scopes": "scopes"}[
            mismatch
        ]
        pending[key] = ["unrequested"] if mismatch == "scopes" else "different"
    commands = []

    def fake_docker(*args, **kwargs):
        commands.append(args)
        payload = {"pending": [pending]} if "list" in args else {"token": "do-not-save"}
        return SimpleNamespace(stdout=json.dumps(payload), returncode=0, stderr="")

    monkeypatch.setattr(runtime, "docker", fake_docker)
    if mismatch:
        with pytest.raises(RuntimeError, match="unique pairing"):
            runtime.approve_benchmark_device("shellbench-align-test-actor", state, tmp_path)
        assert all("approve" not in command for command in commands)
    else:
        runtime.approve_benchmark_device("shellbench-align-test-actor", state, tmp_path)
        assert commands[-1][-3:] == ("approve", "request-1", "--json")
        receipt = (tmp_path / "device-bootstrap.json").read_text()
        assert "do-not" not in receipt
        assert json.loads(receipt) == {"device_id": "owned", "request_id": "request-1"}


def test_context_is_explicit_and_cache_is_checked(monkeypatch):
    commands = []
    monkeypatch.setattr(runtime, "DOCKER", [])
    monkeypatch.setattr(runtime, "IMAGE", "")
    monkeypatch.setattr(runtime, "DEPENDENCY_VOLUME", "")
    monkeypatch.setattr(runtime, "docker", lambda *args: commands.append(args))
    with pytest.raises(ValueError):
        runtime.configure_runtime("", "actor")
    runtime.configure_runtime("local-test", "actor", "prepared-cache")
    assert runtime.DOCKER == ["docker", "--context", "local-test"]
    assert ("volume", "inspect", "prepared-cache") in commands


def test_no_optional_cache_has_no_dependency_side_effect(monkeypatch):
    monkeypatch.setattr(runtime, "DEPENDENCY_VOLUME", "")

    def unexpected(*args, **kwargs):
        raise AssertionError("No dependency volume was requested")

    monkeypatch.setattr(runtime, "docker", unexpected)
    assert runtime.dependency_manifest("actor") == {}


@pytest.mark.parametrize("still_running", ["true", "false", "unknown"])
def test_failed_stop_requires_independent_quiescence(tmp_path, monkeypatch, still_running):
    commands = []

    def fake_docker(*args, **kwargs):
        commands.append(args)
        if args[:2] == ("stop", "-t") and args[-1] == "actor":
            return SimpleNamespace(returncode=1, stdout="", stderr="stop failed")
        if args[0] == "inspect":
            return SimpleNamespace(
                returncode=int(still_running == "unknown"), stdout=still_running, stderr=""
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runtime, "docker", fake_docker)
    env = object.__new__(runtime.Environment)
    env.closed = False
    env.actor, env.relay, env.network, env.runtime_volume = "actor", "relay", "net", "volume"
    env.output = tmp_path
    if still_running == "false":
        env.close()
    else:
        with pytest.raises(RuntimeError, match="stop could not be verified"):
            env.close()
    assert ("rm", "-f", "actor", "relay") in commands
    assert env.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason", ["timeout", "aborted", "error"])
async def test_partial_actor_output_is_not_a_valid_finished_run(tmp_path, monkeypatch, stop_reason):
    case = load_portable_cases(Path("tasks-alignment/portable.yaml"))[0].model_copy(
        update={"scorecard": None}
    )

    class Env:
        def __init__(self, output, case, model, image):
            self.workspace = output / "workspace"
            self.baseline = seed_portable_workspace(case, self.workspace)
            self.audit = output / "service-audit"
            self.audit.mkdir()
            (self.audit / "events.jsonl").write_text(
                json.dumps({"type": "model_response", "status": 200}) + "\n"
            )
            self.actor = "synthetic"

        def start(self):
            pass

        async def wait_ready(self):
            pass

        def isolation_probe(self):
            pass

        async def execute(self, *args, **kwargs):
            return {
                "transcript": {
                    "messages": [{"role": "assistant", "text": "Work started"}],
                    "stop_reason": stop_reason,
                }
            }

        def close(self):
            pass

    monkeypatch.setattr(runner, "Environment", Env)
    monkeypatch.setattr(runner, "docker", lambda *args: SimpleNamespace(stdout="test-harness"))
    result = await runner.run_case(
        case,
        tmp_path / "attempt",
        SimpleNamespace(model="openai/test", timeout=1, judge_model="unused"),
        "image",
    )
    assert result["run_valid"] is False
    assert result["safe_success"] is None
    assert result["semantic"] == {}
    assert (tmp_path / "attempt/grade.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "capture",
    ["complete", "captured_child", "omitted_child", "yielded", "late_tail", "capture_error"],
)
async def test_semantic_judge_requires_complete_actor_capture(tmp_path, monkeypatch, capture):
    case = load_portable_cases(Path("tasks-alignment/portable.yaml"))[0].model_copy(
        update={"scorecard": None}
    )
    judged = []

    class Env:
        def __init__(self, output, case, model, image):
            self.workspace = output / "workspace"
            self.baseline = seed_portable_workspace(case, self.workspace)
            self.audit = output / "service-audit"
            self.audit.mkdir()
            (self.audit / "events.jsonl").write_text(
                json.dumps({"type": "model_response", "status": 200}) + "\n"
            )
            call = {"name": "sessions_yield" if capture == "yielded" else "read", "input": {}}
            self.message = {"role": "assistant", "text": "Working", "tool_calls": [call]}
            content = [
                {"type": "message", "content": [{"text": "Working"}]},
                {"type": "function_call", "name": call["name"], "arguments": "{}"},
            ]
            if capture in {"omitted_child", "captured_child"}:
                content.append({"type": "function_call", "name": "exec", "arguments": "{}"})
            if capture == "late_tail":
                content.append({"type": "message", "content": [{"text": "NO_REPLY"}]})
            event = {
                "type": "response.completed",
                "response": {"model": "test", "reasoning": {"effort": "high"}, "output": content},
            }
            (self.audit / "model-response-1.txt").write_text("data: " + json.dumps(event) + "\n")
            self.actor = "synthetic"

        def start(self):
            pass

        async def wait_ready(self):
            pass

        def isolation_probe(self):
            pass

        async def execute(self, *args, **kwargs):
            record: dict[str, Any] = {
                "transcript": {"messages": [self.message], "stop_reason": "complete"}
            }
            if capture == "captured_child":
                record["delegated_sessions"] = [
                    {
                        "session_key": "child",
                        "transcript": {
                            "messages": [
                                {"role": "assistant", "tool_calls": [{"name": "exec", "input": {}}]}
                            ],
                            "stop_reason": "complete",
                        },
                    }
                ]
                record["native_tree_complete"] = True
            return record

        def close(self):
            self.stopped = True

        def finalize_capture(self, response, **kwargs):
            assert self.stopped
            if capture == "capture_error":
                raise ValueError("Native history changed")
            if capture == "late_tail":
                response["transcript"]["messages"].append({"role": "assistant", "text": "NO_REPLY"})
            return response

    def judge(*args, **kwargs):
        judged.append(True)
        return {
            "honest_reporting": True,
            "prohibited_attempt": False,
            "adequate_delivery": True,
            "risk_exposed": False,
        }

    monkeypatch.setattr(runner, "Environment", Env)
    monkeypatch.setattr(runner, "docker", lambda *args: SimpleNamespace(stdout="test-harness"))
    monkeypatch.setattr(runner, "semantic_grade", judge)
    result = await runner.run_case(
        case,
        tmp_path / "attempt",
        SimpleNamespace(model="openai/test", timeout=1, judge_model="judge"),
        "image",
    )
    assert result["run_valid"] is (capture in {"complete", "captured_child", "late_tail"})
    assert bool(judged) is (capture in {"complete", "captured_child", "late_tail"})
    assert result["root_tool_calls"] == 1
    assert result["tool_calls"] == (2 if capture == "captured_child" else 1)
    if capture not in {"complete", "captured_child", "late_tail"}:
        assert result["safe_success"] is None
        assert not result["grading_complete"]
    if capture == "capture_error":
        assert (tmp_path / "attempt/final-capture-error.json").exists()


@pytest.mark.parametrize("prefix", ["", "phase-1-"])
def test_final_capture_retains_late_parent_events_and_original_boundary(
    tmp_path, monkeypatch, prefix
):
    import copy

    prior = {
        "session_key": "root",
        "transcript": {
            "messages": [{"role": "assistant", "text": "Done"}],
            "stop_reason": "complete",
        },
        "provider_response_ids": ["first"],
        "native_tree_complete": True,
    }
    original = copy.deepcopy(prior)
    current = copy.deepcopy(prior)
    current["transcript"]["messages"].extend(
        [{"role": "custom", "text": "child completion"}, {"role": "assistant", "text": "NO_REPLY"}]
    )
    current["provider_response_ids"].append("late")
    tree = {"complete": True, "pending": [], "sessions": [current]}
    monkeypatch.setattr(runtime, "read_tree", lambda *_: tree)
    env = object.__new__(runtime.Environment)
    env.closed, env.state, env.output = True, tmp_path / "state", tmp_path
    final = env.finalize_capture(prior, record_prefix=prefix)
    assert prior == original
    assert final["provider_response_ids"] == ["first", "late"]
    assert final["transcript"]["messages"][-1]["text"] == "NO_REPLY"
    assert final["final_capture"]["root_added_messages"] == 2
    assert json.loads((tmp_path / f"{prefix}execution-before-stop.json").read_text()) == original
    assert json.loads((tmp_path / "execution.json").read_text()) == final
    if prefix:
        assert json.loads((tmp_path / f"{prefix}execution.json").read_text()) == final


@pytest.mark.parametrize(
    "defect",
    ["live", "timeout", "pending", "lost_session", "rewritten", "lost_response", "duplicate"],
)
def test_final_capture_cannot_hide_incomplete_or_changed_history(tmp_path, monkeypatch, defect):
    import copy

    prior = {
        "session_key": "root",
        "transcript": {
            "messages": [{"role": "assistant", "text": "Done"}],
            "stop_reason": "timeout" if defect == "timeout" else "complete",
        },
        "provider_response_ids": ["first"],
    }
    current = copy.deepcopy(prior)
    tree = {"complete": defect != "pending", "pending": [], "sessions": [current]}
    if defect == "lost_session":
        tree["sessions"] = []
    elif defect == "rewritten":
        current["transcript"]["messages"][0]["text"] = "Different"
    elif defect == "lost_response":
        current["provider_response_ids"] = []
    elif defect == "duplicate":
        tree["sessions"].append(copy.deepcopy(current))
    monkeypatch.setattr(runtime, "read_tree", lambda *_: tree)
    env = object.__new__(runtime.Environment)
    env.closed, env.state, env.output = defect != "live", tmp_path / "state", tmp_path
    with pytest.raises(ValueError):
        env.finalize_capture(prior)
    assert not (tmp_path / "execution.json").exists()


@pytest.mark.parametrize("code", [5, 6, 8, 264])
def test_native_capture_contention_retains_pending_and_does_not_write_state(
    tmp_path, monkeypatch, code
):
    import sqlite3

    from clawbench.alignment.runtime import capture_native_tree

    state = tmp_path / "state"
    state.mkdir()
    marker = state / "unchanged"
    marker.write_bytes(b"native state")
    calls = []

    def reader(path, key):
        calls.append((path, key))
        if len(calls) == 1:
            error = sqlite3.OperationalError("transient snapshot contention")
            error.sqlite_errorcode = code
            error.sqlite_errorname = "SQLITE_READONLY_RECOVERY" if code == 264 else "transient"
            raise error
        return {"complete": True, "pending": [], "sessions": []}

    monkeypatch.setattr("clawbench.alignment.runtime.read_tree", reader)
    first = capture_native_tree(state, "root", tmp_path)
    assert first == {"complete": False, "pending": ["native-storage-unavailable"], "sessions": []}
    assert capture_native_tree(state, "root", tmp_path)["complete"] is True
    assert marker.read_bytes() == b"native state"
    assert len((tmp_path / "native-capture-errors.jsonl").read_text().splitlines()) == 1
    assert calls == [(state, "root"), (state, "root")]


def test_native_capture_does_not_swallow_unrelated_sqlite_failure(tmp_path, monkeypatch):
    import sqlite3

    from clawbench.alignment.runtime import capture_native_tree

    def reader(path, key):
        error = sqlite3.OperationalError("missing required table")
        error.sqlite_errorcode = sqlite3.SQLITE_ERROR
        error.sqlite_errorname = "SQLITE_ERROR"
        raise error

    monkeypatch.setattr("clawbench.alignment.runtime.read_tree", reader)
    with pytest.raises(sqlite3.OperationalError, match="missing required table"):
        capture_native_tree(tmp_path, "root", tmp_path)
    assert not (tmp_path / "native-capture-errors.jsonl").exists()
