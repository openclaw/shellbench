from __future__ import annotations

import json
import tarfile
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
from scripts.native_eval.proxy import write_proxy_config
from scripts.native_eval.run_job import _git_commit
from scripts.native_eval.runtime import collect_agent_metrics, read_reward
from scripts.native_eval.tasks import TaskSpec, validate_suite


def test_matrix_plan_contains_only_requested_models_and_harnesses() -> None:
    plan = build_matrix_plan(116, run_date="20260727")

    assert len(plan) == 72
    assert len({run.run_label for run in plan}) == 72
    assert {run.harness for run in plan} == {harness.name for harness in HARNESSES}
    assert {run.model_slug for run in plan} == {model.slug for model in MODELS}
    assert {run.repetition for run in plan} == {1, 2, 3}


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


def test_proxy_config_pins_only_requested_upstreams(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    write_proxy_config(path)
    config = json.loads(path.read_text(encoding="utf-8"))

    assert [item["model_name"] for item in config["model_list"]] == [
        model.proxy_model_name for model in MODELS
    ]
    assert [item["litellm_params"]["model"] for item in config["model_list"]] == [
        f"{model.provider}/{model.provider_model_id}" for model in MODELS
    ]
    serialized = path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in serialized
    assert "ANTHROPIC_API_KEY" in serialized
    assert "sk-" not in serialized


def test_harness_commands_use_proxy_alias_not_upstream_id() -> None:
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

        assert "sb-opus5" in command.run_command or (
            harness.name == "claude-code"
            and command.env["ANTHROPIC_MODEL"] == "sb-opus5"
        )
        assert "claude-opus-5" not in command.run_command
        assert "OPENROUTER_API_KEY" not in command.env
        assert 'exit "$status"' in command.run_command


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


def test_claude_code_selects_proxy_alias_explicitly() -> None:
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

    assert "--model sb-gpt55" in command.run_command
    assert command.env["ANTHROPIC_MODEL"] == "sb-gpt55"


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
