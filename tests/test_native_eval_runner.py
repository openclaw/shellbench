from __future__ import annotations

import asyncio
import json
import tarfile
from argparse import Namespace
from pathlib import Path

from scripts.native_eval.checkpoint_loop import (
    count_result_json,
    next_checkpoint_sequence,
)
from scripts.native_eval.harnesses import build_harness_command
from scripts.native_eval.models import (
    HARNESSES,
    MODELS,
    RunSpec,
    build_matrix_plan,
)
from scripts.native_eval import plan as native_plan
from scripts.native_eval.proxy import write_proxy_config
from scripts.native_eval.run_job import _git_commit, _run_manifest, build_run_spec
from scripts.native_eval.runtime import (
    DockerTaskEnvironment,
    collect_agent_metrics,
    read_reward,
    write_agent_trajectory,
)
from scripts.native_eval import runtime as native_runtime
from scripts.native_eval.tasks import TaskSpec, validate_suite


def test_matrix_plan_contains_only_requested_models_and_harnesses() -> None:
    plan = build_matrix_plan(116, run_date="20260727")

    assert len(plan) == 72
    assert len({run.run_label for run in plan}) == 72
    assert {run.harness for run in plan} == {harness.name for harness in HARNESSES}
    assert {run.model_slug for run in plan} == {model.slug for model in MODELS}
    assert {run.repetition for run in plan} == {1, 2, 3}


def test_run_index_records_agent_and_judge_reasoning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(native_plan, "validate_suite", lambda _root: [object()])

    entries = native_plan.write_run_index(
        tasks_root=tmp_path,
        output=tmp_path / "run-index.json",
        public_tasks_commit="tasks-commit",
        run_date="20260728",
        reasoning_effort="high",
        judge_reasoning_effort="high",
    )

    assert len(entries) == 72
    assert {entry["reasoning_effort"] for entry in entries} == {"high"}
    assert {entry["judge_reasoning_effort"] for entry in entries} == {"high"}


def test_task_loader_accepts_rich_manifest_and_compose(tmp_path: Path) -> None:
    task_dir = tmp_path / "browser-task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text("use the browser", encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\n",
        encoding="utf-8",
    )
    (task_dir / "environment" / "docker-compose.yaml").write_text(
        "services:\n  main:\n    build: .\n",
        encoding="utf-8",
    )
    (task_dir / "tests" / "test.sh").write_text(
        "echo 1 >/logs/verifier/reward.txt\n",
        encoding="utf-8",
    )
    (task_dir / "solution" / "solve.sh").write_text("true\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        """
[task]
name = "computer-use/browser-task"

[agent]
timeout_sec = 3600

[verifier]
timeout_sec = 300

[environment]
build_timeout_sec = 1800

[[environment.mcp_servers]]
name = "computer"
transport = "streamable-http"
url = "http://computer-mcp:8000/mcp"

[environment.env]
NO_PROXY = "computer-mcp"
""".strip(),
        encoding="utf-8",
    )

    task = TaskSpec.load(task_dir)

    assert task.title == "computer-use/browser-task"
    assert task.compose_file is not None
    assert task.agent_timeout_sec == 3600
    assert task.mcp_servers[0].url == "http://computer-mcp:8000/mcp"
    assert task.environment_env == {"NO_PROXY": "computer-mcp"}


def test_validate_suite_reports_missing_required_file(tmp_path: Path) -> None:
    task_dir = tmp_path / "broken"
    task_dir.mkdir()

    try:
        validate_suite(tmp_path)
    except ValueError as exc:
        assert "missing task.toml" in str(exc)
        assert "missing tests/test.sh" in str(exc)
    else:
        raise AssertionError("expected suite validation to fail")


def test_proxy_config_pins_only_requested_upstreams(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "high")
    path = tmp_path / "config.json"
    write_proxy_config(path)
    config = json.loads(path.read_text(encoding="utf-8"))

    assert [item["model_name"] for item in config["model_list"]] == [
        model.provider_model_id for model in MODELS
    ]
    assert [item["litellm_params"]["model"] for item in config["model_list"]] == [
        f"{model.provider}/{model.provider_model_id}" for model in MODELS
    ]
    serialized = path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in serialized
    assert "ANTHROPIC_API_KEY" in serialized
    assert "sk-" not in serialized
    assert all(
        item["litellm_params"].get("reasoning_effort") == "high"
        for item in config["model_list"]
        if item["model_info"]["provider"] == "openai"
    )


def test_run_manifest_records_native_audit_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_EXECUTION_MODE", "native")
    monkeypatch.setenv("SHELLBENCH_HARBOR_REFERENCE_COMMIT", "harbor-commit")
    monkeypatch.setenv("SHELLBENCH_JUDGE_MODEL_ID", "gpt-5.5")
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "high")
    monkeypatch.setenv("SHELLBENCH_JUDGE_REASONING_EFFORT", "high")
    run = RunSpec(
        run_label="openclaw-gpt55-full-2-r1-20260727",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=2,
        run_date="20260727",
    )

    manifest = _run_manifest(
        run,
        public_tasks_commit="tasks-commit",
        task_suite_path="combined tasks/tasks",
        concurrency=16,
        started_at="2026-07-27T00:00:00Z",
        tasks_root=tmp_path,
        tasks=[],
    )

    assert manifest["execution_mode"] == "native"
    assert manifest["harbor_reference_commit"] == "harbor-commit"
    assert manifest["judge_model_id"] == "gpt-5.5"
    assert manifest["trajectory_mode"] == "unsupported"
    assert manifest["canonical_model_identity"] is None
    assert manifest["provider_model_id"] == "gpt-5.5"
    assert manifest["reasoning_effort"] == "high"
    assert manifest["judge_reasoning_effort"] == "high"


def test_run_spec_preserves_explicit_planned_identity() -> None:
    run = build_run_spec(
        Namespace(
            run_label="codex-planned",
            harness="codex",
            harness_version="planned-version",
            model_slug="gpt55",
            model_id="planned-model-id",
            model_provider="planned-provider",
            proxy_model_name="planned-proxy-name",
            repetition=1,
            expected_task_count=20,
            run_date="20260727",
        )
    )

    assert run.harness_version == "planned-version"
    assert run.model_id == "planned-model-id"
    assert run.provider == "planned-provider"
    assert run.proxy_model_name == "planned-proxy-name"


def test_harness_commands_preserve_canonical_model_identity() -> None:
    for harness in HARNESSES:
        run = RunSpec(
            run_label=f"{harness.name}-test",
            harness=harness.name,
            harness_version=harness.version,
            model_slug="opus5",
            model_id="claude-opus-5",
            provider="anthropic",
            proxy_model_name="sb-opus5",
            repetition=1,
            expected_task_count=116,
            run_date="20260727",
        )
        command = build_harness_command(
            run,
            proxy_url="http://host.docker.internal:4000",
            proxy_key="local-proxy-key",
            mcp_servers=(),
        )

        assert "claude-opus-5" in command.run_command or (
            harness.name == "claude-code"
            and command.env["ANTHROPIC_MODEL"] == "claude-opus-5"
        )
        assert "sb-opus5" not in command.run_command
        assert "OPENROUTER_API_KEY" not in command.env
        assert 'exit "$status"' in command.run_command
        if harness.name == "openclaw":
            assert "ended with stopReason=" in command.run_command
            assert "--thinking off" in command.run_command


def test_hermes_uses_named_local_proxy_provider() -> None:
    run = RunSpec(
        run_label="hermes-test",
        harness="hermes",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=116,
        run_date="20260727",
    )

    command = build_harness_command(
        run,
        proxy_url="http://host.docker.internal:4000",
        proxy_key="local-proxy-key",
        mcp_servers=(),
    )

    assert "--provider custom:shellbench" in command.run_command
    assert "--provider openai" not in command.run_command
    assert "Unknown provider" in command.run_command
    assert command.env == {"SHELLBENCH_PROXY_KEY": "local-proxy-key"}
    assert "custom:shellbench" in command.setup_command
    assert "http://host.docker.internal:4000/v1" in command.setup_command
    assert '--session-id "$session_id" --yes --redact' in command.cleanup_command


def test_workdir_falls_back_to_existing_container_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commands: list[list[str]] = []

    async def fake_capture(command: list[str]) -> str:
        commands.append(command)
        return "" if command[1] == "inspect" else "/workspace"

    monkeypatch.setattr(native_runtime, "capture_process", fake_capture)
    environment = DockerTaskEnvironment(
        task=object(),  # type: ignore[arg-type]
        trial_dir=tmp_path,
        container_name="trial",
        project_name="trial",
        toolchain_root=tmp_path,
        container_id="container-id",
    )

    asyncio.run(environment._discover_workdir())

    assert environment.workdir == "/workspace"
    assert commands[1][-1] == (
        "if [ -d /app ]; then printf /app; "
        "elif [ -d /workspace ]; then printf /workspace; else pwd; fi"
    )


def test_claude_code_selects_canonical_model_explicitly() -> None:
    run = RunSpec(
        run_label="claude-code-test",
        harness="claude-code",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=116,
        run_date="20260727",
    )

    command = build_harness_command(
        run,
        proxy_url="http://host.docker.internal:4000",
        proxy_key="local-proxy-key",
        mcp_servers=(),
    )

    assert "--model gpt-5.5" in command.run_command
    assert command.env["ANTHROPIC_MODEL"] == "gpt-5.5"


def test_codex_trajectory_uses_real_stream_events(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    events = [
        {"type": "thread.started", "thread_id": "thread-123"},
        {
            "type": "item.completed",
            "item": {"id": "reason-1", "type": "reasoning", "text": "inspect files"},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "call-1",
                "type": "command_execution",
                "command": "ls -la",
                "aggregated_output": "total 4",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "message-1", "type": "agent_message", "text": "done"},
        },
        {
            "type": "item.started",
            "item": {
                "id": "todo-1",
                "type": "todo_list",
                "items": [{"text": "inspect", "completed": True}],
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "file-1",
                "type": "file_change",
                "changes": [{"path": "/app/output/result.txt", "kind": "add"}],
                "status": "completed",
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 20,
                "output_tokens": 30,
            },
        },
    ]
    (agent_dir / "codex.txt").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    session_dir = agent_dir / "sessions"
    session_dir.mkdir()
    (session_dir / "rollout.jsonl").write_text(
        json.dumps(
            {
                "type": "turn_context",
                "payload": {"model": "gpt-5.5"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="codex-gpt55-calibration",
        harness="codex",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=1,
        run_date="20260727",
    )
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    task = TaskSpec(
        name="task",
        title="task",
        path=task_dir,
        instruction="do the task",
        raw_config={},
        checksum="abc",
        dockerfile=task_dir / "Dockerfile",
        build_context=task_dir,
        compose_file=None,
        verifier_command="bash /tests/test.sh",
        agent_timeout_sec=900,
        verifier_timeout_sec=300,
        build_timeout_sec=1800,
        mcp_servers=(),
        environment_env={},
        verifier_env={},
    )

    metadata = write_agent_trajectory(task, run, agent_dir)
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert metadata["runtime_model_name"] == "gpt-5.5"
    assert metadata["canonical_model_identity"] is True
    assert trajectory["session_id"] == "thread-123"
    assert len(trajectory["steps"]) == 6
    assert trajectory["steps"][2]["tool_calls"][0]["function_name"] == "shell"
    assert trajectory["steps"][2]["observation"]["results"][0]["content"] == "total 4"
    assert trajectory["steps"][4]["tool_calls"][0]["function_name"] == "todo_list"
    assert trajectory["steps"][5]["tool_calls"][0]["function_name"] == "file_change"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 100


def test_codex_trajectory_rejects_truncated_or_mismatched_stream(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    session_dir = agent_dir / "sessions"
    session_dir.mkdir(parents=True)
    (agent_dir / "codex.txt").write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {
                            "id": "message-1",
                            "type": "agent_message",
                            "text": "unfinished",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (session_dir / "rollout.jsonl").write_text(
        json.dumps(
            {
                "type": "turn_context",
                "payload": {"model": "gpt-5.6-sol"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="codex-gpt55-calibration",
        harness="codex",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="gpt-5.5",
        repetition=1,
        expected_task_count=1,
        run_date="20260727",
    )
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    task = TaskSpec(
        name="task",
        title="task",
        path=task_dir,
        instruction="do the task",
        raw_config={},
        checksum="abc",
        dockerfile=task_dir / "Dockerfile",
        build_context=task_dir,
        compose_file=None,
        verifier_command="bash /tests/test.sh",
        agent_timeout_sec=900,
        verifier_timeout_sec=300,
        build_timeout_sec=1800,
        mcp_servers=(),
        environment_env={},
        verifier_env={},
    )

    metadata = write_agent_trajectory(task, run, agent_dir)

    assert metadata["trajectory_status"] == "unavailable"
    assert metadata["runtime_model_name"] == "gpt-5.6-sol"
    assert metadata["canonical_model_identity"] is False
    assert metadata["trajectory_validation"]["terminal_event_seen"] is False


def test_non_codex_trajectory_is_explicitly_unranked(tmp_path: Path) -> None:
    run = RunSpec(
        run_label="openclaw-gpt55",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="gpt-5.5",
        repetition=1,
        expected_task_count=1,
        run_date="20260727",
    )
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    task = TaskSpec(
        name="task",
        title="task",
        path=task_dir,
        instruction="do the task",
        raw_config={},
        checksum="abc",
        dockerfile=task_dir / "Dockerfile",
        build_context=task_dir,
        compose_file=None,
        verifier_command="bash /tests/test.sh",
        agent_timeout_sec=900,
        verifier_timeout_sec=300,
        build_timeout_sec=1800,
        mcp_servers=(),
        environment_env={},
        verifier_env={},
    )

    metadata = write_agent_trajectory(task, run, tmp_path / "agent")

    assert metadata["trajectory_status"] == "unsupported"
    assert metadata["runtime_model_name"] is None
    assert metadata["canonical_model_identity"] is False


def test_codex_metrics_include_cached_input_tokens(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "codex.txt").write_text(
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 212_605,
                    "cached_input_tokens": 186_240,
                    "output_tokens": 4_605,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = collect_agent_metrics("codex", agent_dir)

    assert metrics["n_input_tokens"] == 212_605
    assert metrics["n_cache_tokens"] == 186_240
    assert metrics["n_output_tokens"] == 4_605


def test_claude_metrics_include_top_level_cost(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "claude-code.txt").write_text(
        json.dumps(
            {
                "type": "result",
                "total_cost_usd": 2.23827,
                "usage": {
                    "input_tokens": 377_989,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 13_933,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = collect_agent_metrics("claude-code", agent_dir)

    assert metrics["n_input_tokens"] == 377_989
    assert metrics["n_output_tokens"] == 13_933
    assert metrics["cost_usd"] == 2.23827


def test_runner_commit_can_be_supplied_without_git(monkeypatch) -> None:
    monkeypatch.setenv("SHELLBENCH_RUNNER_COMMIT", "runner-commit")

    assert _git_commit() == "runner-commit"


def test_reward_json_takes_precedence_over_text(tmp_path: Path) -> None:
    (tmp_path / "reward.txt").write_text("0\n", encoding="utf-8")
    (tmp_path / "reward.json").write_text(
        json.dumps({"reward": 0.75, "safety": 1}),
        encoding="utf-8",
    )

    assert read_reward(tmp_path) == {"reward": 0.75, "safety": 1}


def test_checkpoint_helpers_count_only_trials_and_never_reuse_sequence(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    run_label = "openclaw-gpt55-full-116-r1-20260727"
    (raw_dir / f"{run_label}-checkpoint-0001-artifacts.tar.gz").touch()
    (raw_dir / f"{run_label}-checkpoint-0003-artifacts.tar.gz").touch()
    archive_root = tmp_path / "archive"
    trial_dir = archive_root / "results" / "jobs" / run_label / "task__trial"
    trial_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text("{}")
    (trial_dir.parent / "result.json").write_text("{}")
    archive = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(archive_root, arcname=".")

    assert next_checkpoint_sequence(raw_dir, run_label) == 4
    assert count_result_json(archive) == 1
