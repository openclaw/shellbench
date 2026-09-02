import asyncio
import json
import os
import signal
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from clawbench.queue import JobQueue
from clawbench.worker import GATEWAY_PORT, GATEWAY_PORT_SPACING, EvalWorker, JobProgressTracker, ParallelLane


class DummyTask:
    def __init__(
        self,
        task_id: str,
        tier: str,
        family: str,
        phases: int = 1,
        capabilities: list[str] | None = None,
    ) -> None:
        self.id = task_id
        self.tier = SimpleNamespace(value=tier)
        self.family = SimpleNamespace(value=family)
        self._phases = phases
        self.capabilities = [SimpleNamespace(value=value) for value in (capabilities or [])]

    def normalized_phases(self):
        return [object()] * self._phases


def test_configure_browser_runtime_sets_benchmark_safe_openclaw_config(monkeypatch):
    worker = EvalWorker(JobQueue())
    state_dir = Path("/tmp/test-openclaw-config-basic")
    if state_dir.exists():
        import shutil

        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "openclaw.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state_dir))

    worker._configure_browser_runtime(["node", "/openclaw/dist/cli.js"], {"HOME": "/tmp/home"})

    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "agents": {"defaults": {"skipBootstrap": True}},
        "browser": {"headless": True, "noSandbox": True},
        "tools": {"exec": {"host": "gateway", "security": "full", "ask": "off"}},
        "approvals": {"exec": {"enabled": False}},
    }
    approvals = json.loads((state_dir / "exec-approvals.json").read_text(encoding="utf-8"))
    assert approvals["socket"]["path"] == str(state_dir / "exec-approvals.sock")
    assert approvals["defaults"] == {"security": "full", "ask": "off", "askFallback": "full"}
    assert approvals["agents"]["*"] == {"security": "full", "ask": "off", "askFallback": "full"}


def test_write_eval_exec_approvals_replaces_gateway_defaults(tmp_path: Path):
    approvals_path = tmp_path / "exec-approvals.json"
    approvals_path.write_text(
        json.dumps({"socket": {"path": "/home/node/.openclaw/exec-approvals.sock"}}),
        encoding="utf-8",
    )

    EvalWorker._write_eval_exec_approvals(tmp_path)

    approvals = json.loads(approvals_path.read_text(encoding="utf-8"))
    assert approvals["socket"]["path"] == str(tmp_path / "exec-approvals.sock")
    assert approvals["defaults"] == {"security": "full", "ask": "off", "askFallback": "full"}


def test_configure_browser_runtime_pins_subagents_to_active_model(monkeypatch):
    worker = EvalWorker(JobQueue())
    worker.set_active_model("openai-codex/gpt-5.4")
    state_dir = Path("/tmp/test-openclaw-config-model")
    if state_dir.exists():
        import shutil

        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "openclaw.json"
    config_path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state_dir))

    worker._configure_browser_runtime(["node", "/openclaw/dist/cli.js"], {"HOME": "/tmp/home"})

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["agents"]["defaults"] == {
        "skipBootstrap": True,
        "model": {"primary": "openai-codex/gpt-5.4"},
        "subagents": {"model": {"primary": "openai-codex/gpt-5.4"}},
    }
    assert data["browser"] == {"headless": True, "noSandbox": True}
    assert data["tools"] == {"exec": {"host": "gateway", "security": "full", "ask": "off"}}
    assert data["approvals"] == {"exec": {"enabled": False}}
    assert data["models"]["providers"]["openai-codex"]["auth"] == "api-key"


def test_configure_browser_runtime_sets_requested_agent_runtime(monkeypatch):
    worker = EvalWorker(JobQueue())
    worker.set_active_model("openai/gpt-5.5")
    state_dir = Path("/tmp/test-openclaw-config-runtime")
    if state_dir.exists():
        import shutil

        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "models": {
                            "openai/gpt-5.5": {"agentRuntime": {"id": "codex"}}
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state_dir))
    monkeypatch.setenv("CLAWBENCH_OPENCLAW_AGENT_RUNTIME", "codex")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    worker._configure_browser_runtime(["node", "/openclaw/dist/cli.js"], {"HOME": "/tmp/home"})

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["agents"]["defaults"]["agentRuntime"]["id"] == "codex"
    models = data["agents"]["defaults"].get("models", {})
    assert "agentRuntime" not in models.get("openai/gpt-5.5", {})
    provider = data["models"]["providers"]["openai"]
    assert provider["apiKey"] == "OPENAI_API_KEY"
    assert provider["api"] == "openai-responses"
    assert provider["auth"] == "api-key"
    assert any(item.get("id") == "gpt-5.5" for item in provider["models"])
    codex_provider = data["models"]["providers"]["openai-codex"]
    assert codex_provider["apiKey"] == "OPENAI_API_KEY"
    assert codex_provider["auth"] == "api-key"
    assert any(item.get("id") == "gpt-5.5" for item in codex_provider["models"])
    auth_profile = data["auth"]["profiles"]["openai-codex:clawbench-env"]
    assert auth_profile["provider"] == "openai-codex"
    assert auth_profile["mode"] == "api_key"
    assert data["auth"]["order"]["openai-codex"][0] == "openai-codex:clawbench-env"
    for agent_name in ("main", "dev"):
        store = json.loads(
            (state_dir / "agents" / agent_name / "agent" / "auth-profiles.json").read_text(
                encoding="utf-8"
            )
        )
        credential = store["profiles"]["openai-codex:clawbench-env"]
        assert credential == {
            "type": "api_key",
            "provider": "openai-codex",
            "keyRef": {"source": "env", "provider": "default", "id": "OPENAI_API_KEY"},
            "key": "test-openai-key",
        }


def test_configure_browser_runtime_strips_source_agent_runtime_when_unset(monkeypatch):
    worker = EvalWorker(JobQueue())
    worker.set_active_model("anthropic/claude-sonnet-4-6")
    state_dir = Path("/tmp/test-openclaw-config-runtime-strip")
    if state_dir.exists():
        import shutil

        shutil.rmtree(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    config_path = state_dir / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "agentRuntime": {"id": "codex"},
                        "models": {
                            "openai/gpt-5.5": {"agentRuntime": {"id": "codex"}},
                            "anthropic/claude-sonnet-4-6": {
                                "agentRuntime": {"id": "codex"}
                            },
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state_dir))
    monkeypatch.delenv("CLAWBENCH_OPENCLAW_AGENT_RUNTIME", raising=False)
    monkeypatch.delenv("OPENCLAW_AGENT_RUNTIME", raising=False)

    worker._configure_browser_runtime(["node", "/openclaw/dist/cli.js"], {"HOME": "/tmp/home"})

    data = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = data["agents"]["defaults"]
    assert "agentRuntime" not in defaults
    assert "agentRuntime" not in defaults["models"]["openai/gpt-5.5"]
    assert "agentRuntime" not in defaults["models"]["anthropic/claude-sonnet-4-6"]
    assert defaults["model"]["primary"] == "anthropic/claude-sonnet-4-6"


def test_sanitize_lane_state_removes_schema_incompatible_whatsapp_config(tmp_path):
    worker = EvalWorker(JobQueue())
    worker.set_active_model("openai-codex/gpt-5.5")
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "channels": {
                    "telegram": {"enabled": True, "streaming": "partial"},
                    "discord": {"enabled": True, "streaming": {"mode": "partial"}},
                    "whatsapp": {
                        "enabled": True,
                        "accounts": {"default": {"name": "WhatsApp", "enabled": True}},
                        "groupAllowFrom": ["*"],
                    },
                },
                "plugins": {
                    "allow": ["openai", "whatsapp", "marxbiotech-git-tools"],
                    "entries": {
                        "whatsapp": {"enabled": True},
                        "marxbiotech-git-tools": {"enabled": True},
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    worker._sanitize_lane_state_dir(tmp_path)

    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert "whatsapp" not in data["channels"]
    assert data["channels"]["telegram"]["enabled"] is False
    assert data["channels"]["telegram"]["streaming"] == {"mode": "off"}
    assert data["channels"]["discord"]["enabled"] is False
    assert data["channels"]["discord"]["streaming"] == {"mode": "off"}
    assert data["plugins"]["allow"] == ["openai"]
    assert "whatsapp" not in data["plugins"]["entries"]
    assert "marxbiotech-git-tools" not in data["plugins"]["entries"]
    assert data["agents"]["defaults"]["model"]["primary"] == "openai-codex/gpt-5.5"
    assert (tmp_path / "exec-approvals.json").exists()


def test_sanitize_lane_state_strips_codex_runtime_when_runtime_unset(tmp_path, monkeypatch):
    worker = EvalWorker(JobQueue())
    worker.set_active_model("anthropic/claude-opus-4-7")
    monkeypatch.delenv("CLAWBENCH_OPENCLAW_AGENT_RUNTIME", raising=False)
    monkeypatch.delenv("OPENCLAW_AGENT_RUNTIME", raising=False)
    config_path = tmp_path / "openclaw.json"
    config_path.write_text(
        json.dumps(
            {
                "agents": {
                    "defaults": {
                        "agentRuntime": {"id": "codex"},
                        "models": {
                            "anthropic/claude-opus-4-7": {
                                "agentRuntime": {"id": "codex"}
                            }
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    worker._sanitize_lane_state_dir(tmp_path)

    data = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = data["agents"]["defaults"]
    assert "agentRuntime" not in defaults
    assert "agentRuntime" not in defaults["models"]["anthropic/claude-opus-4-7"]
    assert defaults["model"]["primary"] == "anthropic/claude-opus-4-7"


def test_seed_lane_codex_home_copies_auth_and_config(tmp_path, monkeypatch):
    worker = EvalWorker(JobQueue())
    source = tmp_path / "codex-source"
    source.mkdir()
    (source / "auth.json").write_text('{"token":"test"}', encoding="utf-8")
    (source / "config.toml").write_text("model = 'gpt-5.5'\n", encoding="utf-8")
    lane_home = tmp_path / "lane-home"
    monkeypatch.setenv("CODEX_CONFIG_SOURCE", str(source))

    worker._seed_lane_codex_home(lane_home)

    assert (lane_home / ".codex" / "auth.json").read_text(encoding="utf-8") == '{"token":"test"}'
    assert (lane_home / ".codex" / "config.toml").read_text(encoding="utf-8") == "model = 'gpt-5.5'\n"


@pytest.mark.asyncio
async def test_prepare_benchmark_run_restarts_gateway_on_task_boundary(monkeypatch):
    worker = EvalWorker(JobQueue())
    calls: list[str] = []

    def fake_stop_gateway() -> None:
        calls.append("stop")

    async def fake_ensure_gateway() -> None:
        calls.append("ensure")

    monkeypatch.setattr(worker, "_stop_gateway", fake_stop_gateway)
    monkeypatch.setattr(worker, "_ensure_gateway", fake_ensure_gateway)

    task = DummyTask("t1-bugfix-discount", "tier1", "coding")

    await worker._prepare_benchmark_run(task, 0)
    await worker._prepare_benchmark_run(task, 1)
    await worker._prepare_benchmark_run(DummyTask("t1-refactor-csv-loader", "tier1", "coding"), 0)

    assert calls == ["stop", "ensure"]


@pytest.mark.asyncio
async def test_prepare_benchmark_run_restarts_each_run_for_automation(monkeypatch):
    worker = EvalWorker(JobQueue())
    calls: list[str] = []

    def fake_stop_gateway() -> None:
        calls.append("stop")

    async def fake_ensure_gateway() -> None:
        calls.append("ensure")

    monkeypatch.setattr(worker, "_stop_gateway", fake_stop_gateway)
    monkeypatch.setattr(worker, "_ensure_gateway", fake_ensure_gateway)

    task = DummyTask(
        "t3-monitoring-automation",
        "tier3",
        "tools",
        capabilities=["automation"],
    )

    await worker._prepare_benchmark_run(task, 0)
    await worker._prepare_benchmark_run(task, 1)

    assert calls == ["stop", "ensure"]


def test_plan_parallel_lanes_serializes_browser_tasks():
    worker = EvalWorker(JobQueue())
    tasks = [
        DummyTask("t1", "tier1", "coding"),
        DummyTask("t2", "tier4", "browser"),
        DummyTask("t3", "tier3", "repo"),
        DummyTask("t4", "tier2", "browser"),
        DummyTask("t5", "tier5", "multi_tool", phases=2),
    ]

    lanes = worker._plan_parallel_lanes(tasks, requested_parallel_lanes=3)

    assert len(lanes) == 3
    browser_lanes = [lane for lane in lanes if lane.browser_lane]
    assert len(browser_lanes) == 1
    assert [task.id for task in browser_lanes[0].tasks] == ["t2", "t4"]
    assert all(
        task.family.value != "browser"
        for lane in lanes
        if not lane.browser_lane
        for task in lane.tasks
    )


def test_plan_parallel_lanes_can_distribute_browser_tasks(monkeypatch):
    monkeypatch.setenv("CLAWBENCH_SERIALIZE_BROWSER_LANES", "0")
    worker = EvalWorker(JobQueue())
    tasks = [
        DummyTask("t1", "tier1", "coding"),
        DummyTask("t2", "tier2", "browser"),
        DummyTask("t3", "tier3", "browser"),
    ]

    lanes = worker._plan_parallel_lanes(tasks, requested_parallel_lanes=3)

    assert len(lanes) == 3
    assert all(len(lane.tasks) == 1 for lane in lanes)
    assert sorted(task.id for lane in lanes for task in lane.tasks) == ["t1", "t2", "t3"]


def test_materialize_lane_runtime_spaces_ports_and_copies_auth(tmp_path: Path, monkeypatch):
    source_state = tmp_path / "source-state"
    auth_path = source_state / "agents" / "main" / "agent"
    auth_path.mkdir(parents=True, exist_ok=True)
    (auth_path / "auth-profiles.json").write_text('{"default": "ok"}', encoding="utf-8")
    (source_state / "openclaw.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(source_state))

    worker = EvalWorker(JobQueue())
    lane0 = ParallelLane(index=0, tasks=[DummyTask("t1", "tier1", "coding")])
    lane1 = ParallelLane(index=1, tasks=[DummyTask("t2", "tier2", "browser")], browser_lane=True)

    job_root = tmp_path / "job-root"
    worker._materialize_lane_runtime(lane0, job_root)
    worker._materialize_lane_runtime(lane1, job_root)

    assert lane0.port == GATEWAY_PORT
    assert lane1.port == GATEWAY_PORT + GATEWAY_PORT_SPACING
    assert lane1.state_dir is not None
    assert (lane1.state_dir / "agents" / "main" / "agent" / "auth-profiles.json").exists()
    lane_cfg = json.loads((lane1.state_dir / "openclaw.json").read_text(encoding="utf-8"))
    assert lane_cfg["tools"]["exec"] == {"host": "gateway", "security": "full", "ask": "off"}
    assert lane_cfg["approvals"]["exec"] == {"enabled": False}
    lane_approvals = json.loads((lane1.state_dir / "exec-approvals.json").read_text(encoding="utf-8"))
    assert lane_approvals["socket"]["path"] == str(lane1.state_dir / "exec-approvals.sock")
    assert lane_approvals["defaults"] == {"security": "full", "ask": "off", "askFallback": "full"}


def test_job_progress_tracker_drops_finished_parallel_lane():
    tracker = JobProgressTracker(total_tasks=20, runs_per_task=3, requested_parallel_lanes=2)

    tracker.mark_lane(0, "t4-delegation-repair", 0, stage="running")
    tracker.mark_lane(1, "t4-browser-research-and-code", 1, stage="running")
    snapshot = tracker.clear_lane(0)

    assert snapshot == {
        "current_task_id": "t4-browser-research-and-code",
        "current_run_index": 2,
        "current_run_total": 3,
        "progress_message": "L2 running t4-browser-research-and-code (run 2/3)",
    }


@pytest.mark.asyncio
async def test_run_job_heartbeat_flushes_latest_progress_snapshot(monkeypatch):
    queue = JobQueue()
    worker = EvalWorker(queue)
    calls: list[tuple[str, dict[str, object]]] = []

    async def fake_update_progress(job_id: str, **kwargs) -> None:
        calls.append((job_id, kwargs))

    monkeypatch.setattr(queue, "update_progress", fake_update_progress)

    tracker = JobProgressTracker(total_tasks=20, runs_per_task=3, requested_parallel_lanes=1)
    tracker.mark_serial("t1-bugfix-discount", 1, stage="running")
    stop_event = asyncio.Event()
    stop_event.set()

    await worker._run_job_heartbeat("job-1", tracker, stop_event)

    assert calls == [
        (
            "job-1",
            {
                "current_task_id": "t1-bugfix-discount",
                "current_run_index": 2,
                "current_run_total": 3,
                "progress_message": "Running t1-bugfix-discount (run 2/3)",
            },
        )
    ]


def test_run_lane_prepare_hook_kills_hung_hook(tmp_path: Path, monkeypatch):
    hook = tmp_path / "hung-hook"
    hook.write_text(
        '#!/bin/sh\necho $$ > "$HOME/hook.pid"\nexec sleep 1000\n',
        encoding="utf-8",
    )
    hook.chmod(0o755)

    state_dir = tmp_path / "lane" / "state"
    state_dir.mkdir(parents=True)
    (state_dir.parent / "home").mkdir(parents=True)
    pid_path = state_dir.parent / "home" / "hook.pid"

    monkeypatch.setenv("CLAWBENCH_LANE_PREPARE_CMD", str(hook))
    monkeypatch.setenv("CLAWBENCH_LANE_PREPARE_TIMEOUT_SECONDS", "1")

    worker = EvalWorker(JobQueue())
    lane = ParallelLane(index=0, tasks=[DummyTask("t1", "tier1", "coding")])
    lane.state_dir = state_dir
    lane.port = GATEWAY_PORT

    outcome: list[BaseException | None] = []

    def run() -> None:
        try:
            worker._run_lane_prepare_hook(lane)
            outcome.append(None)
        except BaseException as exc:
            outcome.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    try:
        thread.start()
        thread.join(5)
        assert not thread.is_alive(), "prepare hook hung without timeout"
        assert outcome
        exc = outcome[0]
        assert exc is not None
        assert isinstance(exc, (subprocess.TimeoutExpired, RuntimeError))
        if isinstance(exc, RuntimeError):
            assert "timed out" in str(exc).lower()
        assert pid_path.is_file()
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if pid_path.is_file():
            try:
                os.kill(int(pid_path.read_text(encoding="utf-8").strip()), signal.SIGKILL)
            except (ValueError, ProcessLookupError, OSError):
                pass
