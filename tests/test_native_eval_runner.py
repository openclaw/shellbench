from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tarfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.native_eval.checkpoint_loop import (
    count_result_json,
    next_checkpoint_sequence,
)
from scripts.native_eval.harnesses import (
    _OPENCLAW_CHILD_EXPORTS_READY,
    _OPENCLAW_EXPORT_READY,
    build_harness_command,
)
from scripts.native_eval.harness_trajectories import (
    _openclaw_session_terminal,
    _openclaw_session_tree,
    _openclaw_session_tree_steps,
    _resolve_archived_openclaw_session_path,
)
from scripts.native_eval.models import (
    HARNESSES,
    MODELS,
    RunSpec,
    build_matrix_plan,
    trajectory_mode_for_harness,
)
from scripts.native_eval import plan as native_plan
from scripts.native_eval.proxy import JUDGE_PROXY_MODEL_NAME, write_proxy_config
from scripts.native_eval.run_job import (
    _execution_acceptance,
    _git_commit,
    _run_manifest,
    build_run_spec,
    parse_args as parse_run_job_args,
)
from scripts.native_eval.runtime import (
    DockerTaskEnvironment,
    NonZeroAgentExitCodeError,
    VerifierRewardContractError,
    build_judge_env,
    collect_agent_metrics,
    execution_outcome,
    load_reward_contract,
    read_reward,
    validate_reward_contract,
    write_agent_trajectory,
)
from scripts.native_eval import runtime as native_runtime
from scripts.native_eval.tasks import TaskSpec, validate_suite


def test_matrix_plan_contains_only_requested_models_and_harnesses() -> None:
    plan = build_matrix_plan(
        116,
        run_date="20260727",
        reasoning_effort="high",
    )

    assert len(plan) == 96
    assert len({run.run_label for run in plan}) == 96
    assert {run.harness for run in plan} == {harness.name for harness in HARNESSES}
    assert {run.model_slug for run in plan} == {model.slug for model in MODELS}
    assert {run.repetition for run in plan} == {1, 2, 3}
    assert {run.reasoning_effort for run in plan} == {"high"}


def test_openclaw_terminal_evidence_exit_is_a_harness_error() -> None:
    outcome = execution_outcome(
        harness="openclaw",
        exception=NonZeroAgentExitCodeError("Agent exited with code 71"),
        agent_exit_code=71,
    )

    assert outcome == {
        "kind": "harness_error",
        "exit_code": 71,
        "reason": "terminal_session_evidence_unavailable",
    }


def test_all_harness_errors_reject_run_but_agent_errors_do_not() -> None:
    rejected = _execution_acceptance(
        [
            {"execution_outcome": {"kind": "harness_error"}},
            {"execution_outcome": {"kind": "infra_error"}},
        ],
        2,
    )
    accepted = _execution_acceptance(
        [
            {"execution_outcome": {"kind": "clean"}},
            {"execution_outcome": {"kind": "agent_error"}},
        ],
        2,
    )

    assert rejected == {
        "accepted": False,
        "reason": "all_trials_harness_or_infrastructure_errors",
        "outcome_counts": {"harness_error": 1, "infra_error": 1},
    }
    assert accepted == {
        "accepted": True,
        "reason": None,
        "outcome_counts": {"agent_error": 1, "clean": 1},
    }


def test_all_native_harnesses_emit_real_trajectories() -> None:
    assert {
        harness.name: trajectory_mode_for_harness(harness.name) for harness in HARNESSES
    } == {
        "openclaw": "real_harness_events",
        "hermes": "real_harness_events",
        "codex": "real_harness_events",
        "claude-code": "real_harness_events",
    }


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

    assert len(entries) == 96
    assert {entry["reasoning_effort"] for entry in entries} == {"high"}
    assert {entry["judge_model_id"] for entry in entries} == {"gpt-5.6-sol"}
    assert {entry["judge_reasoning_effort"] for entry in entries} == {"high"}
    assert all("-high-full-" in str(entry["run_label"]) for entry in entries)


def test_run_index_supports_six_repetitions_and_filters(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(native_plan, "validate_suite", lambda _root: [object()])

    entries = native_plan.write_run_index(
        tasks_root=tmp_path,
        output=tmp_path / "run-index.json",
        public_tasks_commit="tasks-commit",
        run_date="20260729",
        reasoning_effort="medium",
        judge_reasoning_effort="high",
        repetitions=6,
        harness_names=["openclaw", "hermes"],
        model_slugs=["gpt56-sol"],
    )

    assert len(entries) == 12
    assert {entry["harness"] for entry in entries} == {"openclaw", "hermes"}
    assert {entry["model_slug"] for entry in entries} == {"gpt56-sol"}
    assert {entry["repetition"] for entry in entries} == set(range(1, 7))
    assert len({entry["run_label"] for entry in entries}) == 12
    assert all("-medium-full-" in str(entry["run_label"]) for entry in entries)


def test_run_index_builds_non_scoring_ten_task_r0(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = [SimpleNamespace(name=f"task-{index}") for index in range(12)]
    monkeypatch.setattr(native_plan, "validate_suite", lambda _root: tasks)
    selected = [task.name for task in tasks[:10]]

    entries = native_plan.write_run_index(
        tasks_root=tmp_path,
        output=tmp_path / "run-index.json",
        public_tasks_commit="tasks-commit",
        run_date="20260729",
        reasoning_effort="high",
        judge_reasoning_effort="high",
        harness_names=["openclaw"],
        model_slugs=["gpt56-sol"],
        phase="r0",
        task_names=selected,
        qualification_family="gpt-5.6",
    )

    assert len(entries) == 1
    entry = entries[0]
    assert entry["repetition"] == 0
    assert entry["expected_task_count"] == 10
    assert entry["task_names"] == selected
    assert entry["phase"] == "r0"
    assert entry["qualification_family"] == "gpt-5.6"
    assert entry["leaderboard_eligible"] is False
    assert entry["exclusion_reason"] == "r0_non_scoring_qualification"
    assert entry["run_label"] == (
        "openclaw-gpt56-sol-high-smoke-10-r0-20260729"
    )
    index = json.loads((tmp_path / "run-index.json").read_text())
    assert index["expected_task_count"] == 12
    assert index["planned_repetitions"] == [0]


def test_plan_cli_accepts_repetition_count_and_filters() -> None:
    args = native_plan.parse_args(
        [
            "--tasks-root",
            "tasks",
            "--output",
            "run-index.json",
            "--public-tasks-commit",
            "abc",
            "--run-date",
            "20260729",
            "--reasoning-effort",
            "low",
            "--repetitions",
            "6",
            "--harness",
            "openclaw",
            "--model",
            "gpt56-sol",
        ]
    )

    assert args.repetitions == 6
    assert args.harness_names == ["openclaw"]
    assert args.model_slugs == ["gpt56-sol"]


def test_plan_cli_accepts_r0_qualification_inputs() -> None:
    task_args = [item for index in range(10) for item in ("--task", f"task-{index}")]
    args = native_plan.parse_args(
        [
            "--tasks-root",
            "tasks",
            "--output",
            "run-index.json",
            "--public-tasks-commit",
            "abc",
            "--run-date",
            "20260729",
            "--reasoning-effort",
            "high",
            "--phase",
            "r0",
            "--qualification-family",
            "gpt-5.6",
            "--harness",
            "openclaw",
            "--model",
            "gpt56-sol",
            *task_args,
        ]
    )

    assert args.phase == "r0"
    assert args.qualification_family == "gpt-5.6"
    assert args.task_names == [f"task-{index}" for index in range(10)]


def test_run_job_cli_accepts_repetition_six() -> None:
    args = parse_run_job_args(
        [
            "--tasks-root",
            "tasks",
            "--jobs-dir",
            "jobs",
            "--run-label",
            "openclaw-gpt56-sol-high-full-115-r6-20260729",
            "--harness",
            "openclaw",
            "--model-slug",
            "gpt56-sol",
            "--repetition",
            "6",
            "--expected-task-count",
            "115",
            "--public-tasks-commit",
            "abc",
            "--task-suite-path",
            "combined tasks/tasks",
            "--run-date",
            "20260729",
        ]
    )

    assert args.repetition == 6


def test_run_job_cli_accepts_r0() -> None:
    args = parse_run_job_args(
        [
            "--tasks-root",
            "tasks",
            "--jobs-dir",
            "jobs",
            "--run-label",
            "openclaw-gpt56-sol-high-smoke-10-r0-20260729",
            "--harness",
            "openclaw",
            "--model-slug",
            "gpt56-sol",
            "--repetition",
            "0",
            "--expected-task-count",
            "10",
            "--public-tasks-commit",
            "abc",
            "--task-suite-path",
            "combined tasks/tasks",
            "--run-date",
            "20260729",
        ]
    )

    assert args.repetition == 0


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

    assert [item["model_name"] for item in config["model_list"][:-1]] == [
        model.provider_model_id for model in MODELS
    ]
    assert [item["litellm_params"]["model"] for item in config["model_list"][:-1]] == [
        f"{model.provider}/{model.provider_model_id}" for model in MODELS
    ]
    assert config["model_list"][-1]["model_name"] == JUDGE_PROXY_MODEL_NAME
    serialized = path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" in serialized
    assert "ANTHROPIC_API_KEY" in serialized
    assert "sk-" not in serialized
    assert all(
        item["litellm_params"].get("reasoning_effort") == "high"
        for item in config["model_list"]
        if item["model_info"]["provider"] == "openai"
    )


def test_judge_env_uses_dedicated_proxy_alias() -> None:
    judge_env = build_judge_env("http://proxy:4000/", "proxy-key")

    assert judge_env["AGENT_JUDGE_MODEL"] == JUDGE_PROXY_MODEL_NAME
    assert judge_env["LLM_JUDGE_MODEL"] == JUDGE_PROXY_MODEL_NAME
    assert judge_env["AGENT_JUDGE_API_URL"] == ("http://proxy:4000/v1/chat/completions")


def test_run_manifest_records_native_audit_metadata(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("SHELLBENCH_PARITY_VALIDATED", raising=False)
    monkeypatch.delenv("SHELLBENCH_PARITY_VALIDATION_JSON", raising=False)
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
        openclaw_package={
            "source_kind": "npm_tarball",
            "package_name": "openclaw",
            "package_version": "2026.7.29-candidate.1",
            "sha256": "candidate-sha",
            "artifact_filename": "openclaw-candidate.tgz",
        },
    )

    assert manifest["execution_mode"] == "native"
    assert manifest["harbor_reference_commit"] == "harbor-commit"
    assert manifest["judge_model_id"] == "gpt-5.5"
    assert manifest["trajectory_mode"] == "real_harness_events"
    assert manifest["canonical_model_identity"] is None
    assert manifest["provider_model_id"] == "gpt-5.5"
    assert manifest["reasoning_effort"] == "high"
    assert manifest["judge_reasoning_effort"] == "high"
    assert manifest["phase"] == "full"
    assert manifest["qualification_family"] is None
    assert manifest["leaderboard_eligible"] is None
    assert manifest["exclusion_reason"] is None
    assert manifest["repair_mode"] is False
    assert manifest["rerun_of_canonical_run"] is None
    assert manifest["repair_task_names"] == []
    assert manifest["parity_validated"] is False
    assert manifest["parity_validation"] is None
    assert manifest["legacy_parity_validated_claim"] is False
    assert manifest["openclaw_tool_mode"] is None
    assert manifest["openclaw_package"]["sha256"] == "candidate-sha"


def test_run_manifest_excludes_r0_from_leaderboard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_RUN_PHASE", "r0")
    monkeypatch.setenv("SHELLBENCH_QUALIFICATION_FAMILY", "gpt-5.6")
    monkeypatch.setenv("SHELLBENCH_LEADERBOARD_ELIGIBLE", "false")
    monkeypatch.setenv(
        "SHELLBENCH_EXCLUSION_REASON",
        "r0_non_scoring_qualification",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-sol-high-smoke-10-r0-20260729",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=0,
        expected_task_count=10,
        run_date="20260729",
    )

    manifest = _run_manifest(
        run,
        public_tasks_commit="tasks-commit",
        task_suite_path="combined tasks/tasks",
        concurrency=2,
        started_at="2026-07-29T00:00:00Z",
        tasks_root=tmp_path,
        tasks=[],
    )

    assert manifest["phase"] == "r0"
    assert manifest["qualification_family"] == "gpt-5.6"
    assert manifest["leaderboard_eligible"] is False
    assert manifest["exclusion_reason"] == "r0_non_scoring_qualification"


def test_run_manifest_scopes_parity_to_matching_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    validation = {
        "validated": True,
        "scope": {"harness": "codex", "model_slug": "gpt55"},
        "evidence": {"task_count": 20},
    }
    monkeypatch.setenv(
        "SHELLBENCH_PARITY_VALIDATION_JSON",
        json.dumps(validation),
    )
    monkeypatch.setenv("SHELLBENCH_PARITY_VALIDATED", "true")
    codex_run = RunSpec(
        run_label="codex-gpt55-cal20-native-r1",
        harness="codex",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=20,
        run_date="20260727",
    )
    hermes_run = RunSpec(
        run_label="hermes-gpt55-cal20-native-r1",
        harness="hermes",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=20,
        run_date="20260727",
    )

    codex_manifest = _run_manifest(
        codex_run,
        public_tasks_commit="tasks-commit",
        task_suite_path="combined tasks/tasks",
        concurrency=4,
        started_at="2026-07-27T00:00:00Z",
        tasks_root=tmp_path,
        tasks=[],
    )
    hermes_manifest = _run_manifest(
        hermes_run,
        public_tasks_commit="tasks-commit",
        task_suite_path="combined tasks/tasks",
        concurrency=4,
        started_at="2026-07-27T00:00:00Z",
        tasks_root=tmp_path,
        tasks=[],
    )

    assert codex_manifest["parity_validated"] is True
    assert codex_manifest["parity_validation"] == validation
    assert codex_manifest["legacy_parity_validated_claim"] is True
    assert hermes_manifest["parity_validated"] is False
    assert hermes_manifest["parity_validation"] == validation


def test_legacy_global_parity_flag_does_not_publish_validated(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_PARITY_VALIDATED", "true")
    monkeypatch.delenv("SHELLBENCH_PARITY_VALIDATION_JSON", raising=False)
    run = RunSpec(
        run_label="openclaw-gpt55-full-116-r1",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=116,
        run_date="20260728",
    )

    manifest = _run_manifest(
        run,
        public_tasks_commit="tasks-commit",
        task_suite_path="combined tasks/tasks",
        concurrency=16,
        started_at="2026-07-28T00:00:00Z",
        tasks_root=tmp_path,
        tasks=[],
    )

    assert manifest["parity_validated"] is False
    assert manifest["parity_validation"] is None
    assert manifest["legacy_parity_validated_claim"] is True


def test_run_manifest_records_targeted_repair_lineage(tmp_path: Path) -> None:
    run = RunSpec(
        run_label="openclaw-gpt55-low-full-116-r1-parent-targetedfix1",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=2,
        run_date="20260728",
    )
    tasks = [
        SimpleNamespace(name="task-a", path=tmp_path / "task-a", checksum="a"),
        SimpleNamespace(name="task-b", path=tmp_path / "task-b", checksum="b"),
    ]

    manifest = _run_manifest(
        run,
        public_tasks_commit="tasks-commit",
        task_suite_path="combined tasks/tasks",
        concurrency=2,
        started_at="2026-07-28T00:00:00Z",
        tasks_root=tmp_path,
        tasks=tasks,
        rerun_of_canonical_run="openclaw-gpt55-low-full-116-r1-parent",
    )

    assert manifest["repair_mode"] is True
    assert manifest["rerun_of_canonical_run"] == ("openclaw-gpt55-low-full-116-r1-parent")
    assert manifest["repair_task_names"] == ["task-a", "task-b"]


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


def test_run_spec_preserves_planned_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "high")

    run = build_run_spec(
        Namespace(
            run_label="openclaw-reasoning-high",
            harness="openclaw",
            harness_version="planned-version",
            model_slug="gpt55",
            model_id="gpt-5.5",
            model_provider="openai",
            proxy_model_name="gpt-5.5",
            repetition=1,
            expected_task_count=3,
            run_date="20260730",
        )
    )

    assert run.reasoning_effort == "high"


def test_run_spec_rejects_invalid_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "extreme")

    with pytest.raises(ValueError, match="reasoning_effort must be"):
        build_run_spec(
            Namespace(
                run_label="openclaw-reasoning-invalid",
                harness="openclaw",
                harness_version="planned-version",
                model_slug="gpt55",
                model_id="gpt-5.5",
                model_provider="openai",
                proxy_model_name="gpt-5.5",
                repetition=1,
                expected_task_count=3,
                run_date="20260730",
            )
        )


def test_run_spec_normalizes_empty_openclaw_tool_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_OPENCLAW_TOOL_MODE", "")

    run = build_run_spec(
        Namespace(
            run_label="openclaw-tool-search-off",
            harness="openclaw",
            harness_version="planned-version",
            model_slug="gpt55",
            model_id="gpt-5.5",
            model_provider="openai",
            proxy_model_name="gpt-5.5",
            repetition=1,
            expected_task_count=3,
            run_date="20260729",
        )
    )

    assert run.openclaw_tool_mode is None


def test_run_spec_rejects_retired_openclaw_tool_search_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_OPENCLAW_TOOL_SEARCH_MODE", "code")

    with pytest.raises(
        ValueError,
        match="SHELLBENCH_OPENCLAW_TOOL_SEARCH_MODE is retired",
    ):
        build_run_spec(
            Namespace(
                run_label="openclaw-tool-search-code",
                harness="openclaw",
                harness_version="planned-version",
                model_slug="gpt55",
                model_id="gpt-5.5",
                model_provider="openai",
                proxy_model_name="gpt-5.5",
                repetition=1,
                expected_task_count=3,
                run_date="20260729",
            )
        )


def test_run_spec_accepts_empty_retired_openclaw_tool_search_mode(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_OPENCLAW_TOOL_SEARCH_MODE", "")

    run = build_run_spec(
        Namespace(
            run_label="openclaw-tool-search-off",
            harness="openclaw",
            harness_version="planned-version",
            model_slug="gpt55",
            model_id="gpt-5.5",
            model_provider="openai",
            proxy_model_name="gpt-5.5",
            repetition=1,
            expected_task_count=3,
            run_date="20260730",
        )
    )

    assert run.openclaw_tool_mode is None


def test_run_spec_rejects_invalid_openclaw_tool_mode(monkeypatch) -> None:
    monkeypatch.setenv("SHELLBENCH_OPENCLAW_TOOL_MODE", "cod")

    with pytest.raises(ValueError, match="must be one of"):
        build_run_spec(
            Namespace(
                run_label="openclaw-tool-mode-invalid",
                harness="openclaw",
                harness_version="planned-version",
                model_slug="gpt55",
                model_id="gpt-5.5",
                model_provider="openai",
                proxy_model_name="gpt-5.5",
                repetition=1,
                expected_task_count=3,
                run_date="20260730",
            )
        )


def test_run_spec_rejects_openclaw_tool_mode_for_other_harness(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SHELLBENCH_OPENCLAW_TOOL_MODE", "code")

    with pytest.raises(ValueError, match="requires the OpenClaw harness"):
        build_run_spec(
            Namespace(
                run_label="codex-tool-mode-invalid",
                harness="codex",
                harness_version="planned-version",
                model_slug="gpt55",
                model_id="gpt-5.5",
                model_provider="openai",
                proxy_model_name="gpt-5.5",
                repetition=1,
                expected_task_count=3,
                run_date="20260730",
            )
        )


def test_run_spec_accepts_codex_tool_mode(monkeypatch) -> None:
    monkeypatch.setenv("SHELLBENCH_CODEX_TOOL_MODE", "code")

    run = build_run_spec(
        Namespace(
            run_label="codex-code-mode",
            harness="codex",
            harness_version="planned-version",
            model_slug="gpt55",
            model_id="gpt-5.5",
            model_provider="openai",
            proxy_model_name="gpt-5.5",
            repetition=1,
            expected_task_count=3,
            run_date="20260730",
        )
    )

    assert run.codex_tool_mode == "code"


def test_run_spec_rejects_codex_tool_mode_for_other_harness(monkeypatch) -> None:
    monkeypatch.setenv("SHELLBENCH_CODEX_TOOL_MODE", "code")

    with pytest.raises(ValueError, match="requires the Codex harness"):
        build_run_spec(
            Namespace(
                run_label="openclaw-codex-mode-invalid",
                harness="openclaw",
                harness_version="planned-version",
                model_slug="gpt55",
                model_id="gpt-5.5",
                model_provider="openai",
                proxy_model_name="gpt-5.5",
                repetition=1,
                expected_task_count=3,
                run_date="20260730",
            )
        )


def test_run_spec_rejects_invalid_codex_tool_mode(monkeypatch) -> None:
    monkeypatch.setenv("SHELLBENCH_CODEX_TOOL_MODE", "directory")

    with pytest.raises(ValueError, match="must be one of"):
        build_run_spec(
            Namespace(
                run_label="codex-tool-mode-invalid",
                harness="codex",
                harness_version="planned-version",
                model_slug="gpt55",
                model_id="gpt-5.5",
                model_provider="openai",
                proxy_model_name="gpt-5.5",
                repetition=1,
                expected_task_count=3,
                run_date="20260730",
            )
        )


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
            reasoning_effort="high",
        )
        command = build_harness_command(
            run,
            proxy_url="http://host.docker.internal:4000",
            proxy_key="local-proxy-key",
            mcp_servers=(),
        )

        assert "claude-opus-5" in command.run_command or (
            harness.name == "claude-code" and command.env["ANTHROPIC_MODEL"] == "claude-opus-5"
        )
        assert "sb-opus5" not in command.run_command
        assert "OPENROUTER_API_KEY" not in command.env
        assert 'exit "$status"' in command.run_command
        if harness.name == "openclaw":
            assert "python3 -c" in command.run_command
            assert "--thinking high" in command.run_command
            assert "openclaw gateway --port 18789" in command.run_command
            assert "127.0.0.1:18789/readyz" in command.run_command
            assert "openclaw agent --local" not in command.run_command
            assert "kill -KILL" not in command.run_command
            assert "openclaw sessions export-trajectory" in command.run_command
            assert '--session-key "agent:main:main"' in command.run_command
            assert "OPENCLAW_GATEWAY_TOKEN" in command.env
            assert len(command.env["OPENCLAW_GATEWAY_TOKEN"]) >= 32
            assert "seq 1 10" in command.run_command
            assert "status=71" in command.run_command
            assert command.setup_command.startswith("set -eu;")
            assert "setup --baseline --workspace ." in command.setup_command
            assert "--skip-bootstrap" not in command.setup_command
            assert "rm -f AGENTS.md BOOTSTRAP.md HEARTBEAT.md" in (command.setup_command)
            assert '"skipBootstrap":true' in command.setup_command
            assert '"primary":"openai/claude-opus-5"' in command.setup_command
            assert '"thinkingDefault":"high"' in command.setup_command
            assert (
                '"subagents":{"model":"openai/claude-opus-5","thinking":"high"}'
                in command.setup_command
            )
            assert '"agentRuntime":{"id":"openclaw"}' in command.setup_command
            assert '"shellbench-audit":{"enabled":true}' in command.setup_command
            assert "openclaw.plugin.json" in command.setup_command
            assert '"activation":{"onCapabilities":["hook"]}' in command.setup_command
            assert '"configSchema":{"type":"object"' in command.setup_command
            assert 'api.on("subagent_spawned"' in command.setup_command
            assert 'api.on("subagent_progress"' in command.setup_command
            assert 'api.on("subagent_ended"' in command.setup_command
            assert "event.phase !==" in command.setup_command
            assert "execFile(" in command.setup_command
            assert '"export-trajectory"' in command.setup_command
            assert "exportsByRun" in command.setup_command
            assert "zstdDecompressSync" not in command.setup_command
            assert "shellbench-openclaw-child-exports.txt" in command.run_command
            assert "audit_ready=0" in command.run_command
            assert "ShellBench audit plugin did not start" in command.run_command
            assert "exit 72" in command.run_command
            assert 'kill -0 "$gateway_pid"' in command.run_command
            assert "child_wait" in command.run_command
            assert "did not settle within 300s" in command.run_command
            assert "item.chmod(0o755 if item.is_dir() else 0o644)" in (command.cleanup_command)
            assert "archive/'audit'" in command.cleanup_command
            assert "archive/'exports'" in command.cleanup_command
            assert "sessions.json" not in command.cleanup_command
        if harness.name == "hermes":
            assert '"delegation":{"max_iterations":50' in command.setup_command
            assert '"provider":"custom:shellbench"' in command.setup_command
            assert '"model":"claude-opus-5"' in command.setup_command
        if harness.name == "codex":
            assert "2>/logs/agent/codex-stderr.txt" in command.run_command
            assert "cat /logs/agent/codex-stderr.txt >&2" in command.run_command


def test_openclaw_harness_uses_direct_tools_by_default() -> None:
    run = RunSpec(
        run_label="openclaw-tool-search-off",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=4,
        run_date="20260729",
    )

    command = build_harness_command(
        run,
        proxy_url="http://host.docker.internal:4000",
        proxy_key="local-proxy-key",
        mcp_servers=(),
    )

    assert '"codeMode":false' in command.setup_command
    assert '"toolSearch":false' in command.setup_command
    assert '"skills":[]' in command.setup_command
    assert '"allow":["group:runtime","group:fs","bundle-mcp"]' in command.setup_command
    assert '"allow":["openai","shellbench-audit"]' in command.setup_command
    assert '"slots":{"memory":"none"}' in command.setup_command
    assert 'export OPENCLAW_STATE_DIR="$HOME/.openclaw"' in command.setup_command
    assert 'export OPENCLAW_STATE_DIR="$HOME/.openclaw"' in command.run_command
    assert '"deny":["message","computer"]' not in command.setup_command


def test_openclaw_harness_configures_code_mode() -> None:
    run = RunSpec(
        run_label="openclaw-tool-search-code",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=4,
        run_date="20260729",
        openclaw_tool_mode="code",
    )

    command = build_harness_command(
        run,
        proxy_url="http://host.docker.internal:4000",
        proxy_key="local-proxy-key",
        mcp_servers=(),
        agent_timeout_sec=3600,
    )

    assert '"codeMode":true' in command.setup_command
    assert '"toolSearch":false' in command.setup_command
    assert '"timeoutSeconds":3600' in command.setup_command
    assert '"reasoning":true' in command.setup_command
    assert '"input":["text","image"]' in command.setup_command


def test_openclaw_harness_configures_tool_directory_mode() -> None:
    run = RunSpec(
        run_label="openclaw-tool-directory",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=4,
        run_date="20260729",
        openclaw_tool_mode="directory",
    )

    command = build_harness_command(
        run,
        proxy_url="http://host.docker.internal:4000",
        proxy_key="local-proxy-key",
        mcp_servers=(),
    )

    assert '"codeMode":false' in command.setup_command
    assert '"toolSearch":{"enabled":true,"mode":"directory"}' in command.setup_command


def test_codex_harness_configures_code_mode() -> None:
    run = RunSpec(
        run_label="codex-code-mode",
        harness="codex",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=4,
        run_date="20260730",
        codex_tool_mode="code",
    )

    command = build_harness_command(
        run,
        proxy_url="http://host.docker.internal:4000",
        proxy_key="local-proxy-key",
        mcp_servers=(),
    )

    assert "--enable code_mode_only" in command.run_command
    assert command.setup_command.startswith("set -eu;")
    assert (
        "cp /opt/shellbench-native/codex-models-code_mode_only.json "
        "/tmp/shellbench-codex/models.json"
    ) in command.setup_command
    assert "curl " not in command.setup_command
    assert "jq " not in command.setup_command
    assert 'model_catalog_json = "/tmp/shellbench-codex/models.json"' in command.setup_command


def test_codex_harness_uses_direct_tools_by_default() -> None:
    run = RunSpec(
        run_label="codex-direct-mode",
        harness="codex",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=4,
        run_date="20260730",
    )

    command = build_harness_command(
        run,
        proxy_url="http://host.docker.internal:4000",
        proxy_key="local-proxy-key",
        mcp_servers=(),
    )

    assert "--disable code_mode_only --disable code_mode" in command.run_command
    assert (
        "cp /opt/shellbench-native/codex-models-direct.json "
        "/tmp/shellbench-codex/models.json"
    ) in command.setup_command


def test_openclaw_child_exports_wait_for_every_spawned_session(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "sessions.jsonl"
    child = "agent:main:subagent:child"
    nested = "agent:main:subagent:nested"

    def run_probe() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_CHILD_EXPORTS_READY,
                str(audit),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    assert run_probe().returncode == 1
    audit.write_text(
        json.dumps({"type": "audit_ready"}) + "\n",
        encoding="utf-8",
    )
    assert run_probe().returncode == 0
    with audit.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "subagent_spawned",
                    "sessionKey": child,
                    "runId": "run-child",
                }
            )
            + "\n"
        )
    assert run_probe().returncode == 1

    with audit.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "subagent_spawned",
                    "sessionKey": nested,
                    "runId": "run-nested",
                }
            )
            + "\n"
        )
        handle.write(
            json.dumps(
            {
                "type": "subagent_exported",
                "sessionKey": child,
                "runId": "run-child",
                "exportOk": True,
                "exportOutput": "shellbench-child-one",
            }
        )
            + "\n"
        )
    assert run_probe().returncode == 1

    with audit.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "subagent_exported",
                    "sessionKey": nested,
                    "runId": "run-nested",
                    "exportOk": True,
                    "exportOutput": "shellbench-child-two",
                }
            )
            + "\n"
        )
    completed = run_probe()
    assert completed.returncode == 0
    assert completed.stdout.splitlines() == [
        "shellbench-child-one",
        "shellbench-child-two",
    ]

    with audit.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "type": "subagent_exported",
                    "sessionKey": nested,
                    "runId": "run-nested",
                    "exportOk": False,
                    "exportOutput": "shellbench-child-two",
                }
            )
            + "\n"
        )
    failed = run_probe()
    assert failed.returncode == 2
    assert nested in failed.stderr


def test_openclaw_terminal_rejects_an_unanswered_latest_user_turn(
    tmp_path: Path,
) -> None:
    records = [
        {"type": "session", "id": "root"},
        {
            "type": "message",
            "message": {"role": "user", "content": "first"},
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "first answer"}],
                "stopReason": "stop",
            },
        },
        {
            "type": "message",
            "message": {"role": "user", "content": "second"},
        },
    ]
    session = tmp_path / "session.jsonl"
    session.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert _openclaw_session_terminal(session) is False


def test_openclaw_archived_session_path_preserves_safe_subdirectories(
    tmp_path: Path,
) -> None:
    store = tmp_path / "sessions"
    nested = store / "nested" / "child.jsonl"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}\n", encoding="utf-8")

    assert (
        _resolve_archived_openclaw_session_path(
            store,
            {"sessionFile": "nested/child.jsonl"},
        )
        == nested
    )
    assert (
        _resolve_archived_openclaw_session_path(
            store,
            {"sessionFile": "/tmp/source/sessions/nested/child.jsonl"},
        )
        == nested
    )
    assert (
        _resolve_archived_openclaw_session_path(
            store,
            {"sessionFile": "../outside.jsonl"},
        )
        is None
    )


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
    assert "--session-id" not in command.cleanup_command
    assert command.cleanup_command.endswith(
        "hermes sessions export /logs/agent/hermes-session.jsonl --yes --redact"
    )


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
        "\n".join(
            [
                json.dumps(events[0]),
                (
                    "2026-07-28T07:10:46.280064Z ERROR "
                    "codex_core::tools::router: apply_patch verification failed"
                ),
                *(json.dumps(event) for event in events[1:]),
            ]
        )
        + "\n",
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
    assert metadata["trajectory_validation"]["diagnostic_lines"] == 1
    assert metadata["trajectory_validation"]["malformed_event_lines"] == 0
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


def test_codex_trajectory_rejects_unknown_interleaved_output(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    session_dir = agent_dir / "sessions"
    session_dir.mkdir(parents=True)
    events = [
        {"type": "thread.started", "thread_id": "thread-123"},
        "not a recognized codex diagnostic",
        {
            "type": "item.completed",
            "item": {"id": "message-1", "type": "agent_message", "text": "done"},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ]
    (agent_dir / "codex.txt").write_text(
        "\n".join(event if isinstance(event, str) else json.dumps(event) for event in events)
        + "\n",
        encoding="utf-8",
    )
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
        proxy_model_name="gpt-5.5",
        repetition=1,
        expected_task_count=1,
        run_date="20260727",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )

    assert metadata["trajectory_status"] == "unavailable"
    assert metadata["trajectory_validation"]["diagnostic_lines"] == 0
    assert metadata["trajectory_validation"]["malformed_event_lines"] == 1


def test_codex_trajectory_retries_transient_malformed_stream(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent_dir = tmp_path / "agent"
    session_dir = agent_dir / "sessions"
    session_dir.mkdir(parents=True)
    events = [
        {"type": "thread.started", "thread_id": "thread-123"},
        {
            "type": "item.completed",
            "item": {"id": "message-1", "type": "agent_message", "text": "done"},
        },
        {"type": "turn.completed", "usage": {"input_tokens": 1}},
    ]
    (agent_dir / "codex.txt").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
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
    original_loader = native_runtime._load_codex_stream
    calls = 0

    def transient_loader(path: Path):
        nonlocal calls
        calls += 1
        loaded = original_loader(path)
        if calls == 1:
            return (*loaded[:3], 1)
        return loaded

    monkeypatch.setattr(native_runtime, "_load_codex_stream", transient_loader)
    monkeypatch.setattr(native_runtime.time, "sleep", lambda _seconds: None)
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

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )

    assert metadata["trajectory_status"] == "real"
    assert metadata["trajectory_validation"]["malformed_event_lines"] == 0
    assert metadata["trajectory_validation"]["stream_read_attempts"] == 2


def test_codex_trajectory_falls_back_to_complete_session_stream(
    tmp_path: Path,
    monkeypatch,
) -> None:
    agent_dir = tmp_path / "agent"
    session_dir = agent_dir / "sessions"
    session_dir.mkdir(parents=True)
    (agent_dir / "codex.txt").write_text(
        "\n".join(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-123"}),
                '{"type":"item.completed","item":{"id":"broken"',
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 1}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    session_events = [
        {
            "type": "session_meta",
            "payload": {"session_id": "session-123"},
        },
        {
            "type": "turn_context",
            "payload": {"model": "gpt-5.5"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "inspect"}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call",
                "call_id": "call-1",
                "name": "shell",
                "input": '{"command":"pwd"}',
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "custom_tool_call_output",
                "call_id": "call-1",
                "output": "/app",
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {"type": "task_complete"},
        },
    ]
    (session_dir / "rollout.jsonl").write_text(
        "\n".join(json.dumps(event) for event in session_events) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(native_runtime.time, "sleep", lambda _seconds: None)
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

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert metadata["trajectory_validation"]["session_fallback"] is True
    assert metadata["trajectory_validation"]["malformed_event_lines"] == 0
    assert trajectory["session_id"] == "session-123"
    assert trajectory["steps"][1]["reasoning_content"] == "inspect"
    assert trajectory["steps"][2]["tool_calls"][0]["function_name"] == "shell"
    assert trajectory["steps"][2]["observation"]["results"][0]["content"] == "/app"
    assert trajectory["steps"][3]["message"] == "done"


def test_claude_code_stream_converts_to_real_trajectory(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "claude-session-123",
            "model": "sb-gpt55",
        },
        {
            "type": "assistant",
            "message": {
                "model": "gpt-5.5",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "call-1",
                        "name": "Bash",
                        "input": {"command": "pwd"},
                    }
                ],
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call-1",
                        "content": "/app",
                        "is_error": False,
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "model": "gpt-5.5",
                "content": [{"type": "text", "text": "done"}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "terminal_reason": "completed",
            "is_error": False,
            "session_id": "claude-session-123",
            "result": "done",
            "usage": {"input_tokens": 10, "output_tokens": 3},
            "modelUsage": {"sb-gpt55": {"canonicalModel": "sb-gpt55"}},
        },
    ]
    (agent_dir / "claude-code.txt").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="claude-code-gpt55",
        harness="claude-code",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=1,
        run_date="20260727",
    )
    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert metadata["runtime_model_name"] == "gpt-5.5"
    assert metadata["canonical_model_identity"] is True
    assert metadata["trajectory_validation"]["terminal_event_seen"] is True
    assert trajectory["session_id"] == "claude-session-123"
    assert trajectory["steps"][1]["tool_calls"][0]["function_name"] == "Bash"
    assert trajectory["steps"][1]["observation"]["results"][0]["content"] == "/app"
    assert trajectory["steps"][-1]["message"] == "done"


def test_claude_code_child_model_mismatch_invalidates_identity(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "claude-session-123",
            "model": "sb-gpt55",
        },
        {
            "type": "assistant",
            "message": {
                "model": "gpt-5.5",
                "content": [{"type": "text", "text": "done"}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "terminal_reason": "completed",
            "is_error": False,
            "session_id": "claude-session-123",
            "result": "done",
            "modelUsage": {
                "sb-gpt55": {"canonicalModel": "sb-gpt55"},
                "gpt-5.6-sol": {"canonicalModel": "gpt-5.6-sol"},
            },
        },
    ]
    (agent_dir / "claude-code.txt").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="claude-code-gpt55",
        harness="claude-code",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
        repetition=1,
        expected_task_count=1,
        run_date="20260727",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )

    assert metadata["trajectory_status"] == "real"
    assert metadata["runtime_model_name"] == "gpt-5.5"
    assert metadata["canonical_model_identity"] is False
    assert metadata["trajectory_validation"]["observed_models"] == [
        "gpt-5.5",
        "gpt-5.6-sol",
    ]


def test_openclaw_mixed_log_converts_harbor_envelope_to_atif(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    envelope = {
        "payloads": [{"text": "done", "mediaUrl": None}],
        "meta": {
            "agentMeta": {
                "sessionId": "session-123",
                "model": "gpt-5.6-luna",
                "usage": {
                    "input": 10,
                    "output": 20,
                    "cacheRead": 30,
                    "cacheWrite": 4,
                },
            },
            "executionTrace": {
                "winnerProvider": "openai",
                "winnerModel": "gpt-5.6-luna",
            },
            "completion": {"stopReason": "stop"},
            "aborted": False,
        },
    }
    (agent_dir / "openclaw.txt").write_text(
        "debug preamble\n"
        + json.dumps(envelope, indent=2)
        + "\n[agents/agent-command] ended with stopReason=stop\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-luna",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-luna",
        model_id="gpt-5.6-luna",
        provider="openai",
        proxy_model_name="gpt-5.6-luna",
        repetition=1,
        expected_task_count=1,
        run_date="20260728",
    )
    task = _trajectory_task(tmp_path, "do the task")

    metadata = write_agent_trajectory(task, run, agent_dir)
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())
    metrics = collect_agent_metrics("openclaw", agent_dir)

    assert metadata["trajectory_status"] == "real"
    assert metadata["canonical_model_identity"] is True
    assert metadata["trajectory_validation"]["trace_fidelity"] == "envelope"
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["session_id"] == "session-123"
    assert trajectory["agent"]["model_name"] == "openai/gpt-5.6-luna"
    assert trajectory["steps"][1]["message"] == "done"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 44
    assert trajectory["extra"]["cache_write_tokens"] == 4
    assert metrics["n_input_tokens"] == 44
    assert metrics["n_cache_tokens"] == 30
    assert metrics["n_output_tokens"] == 20
    assert metrics["metadata"]["cache_write_tokens"] == 4
    assert metrics["metadata"]["input_token_semantics"] == "total_prompt_including_cache"


def test_openclaw_session_without_envelope_converts_to_atif(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    records = [
        {
            "type": "session",
            "id": "session-only-123",
            "timestamp": "2026-07-28T00:00:00Z",
            "cwd": "/app",
        },
        {
            "type": "model_change",
            "id": "model-1",
            "parentId": None,
            "timestamp": "2026-07-28T00:00:01Z",
            "provider": "openai",
            "modelId": "gpt-5.6-terra",
        },
        {
            "type": "message",
            "id": "user-1",
            "parentId": "model-1",
            "timestamp": "2026-07-28T00:00:02Z",
            "message": {"role": "user", "content": "do the task"},
        },
        {
            "type": "message",
            "id": "assistant-1",
            "parentId": "user-1",
            "timestamp": "2026-07-28T00:00:03Z",
            "message": {
                "role": "assistant",
                "provider": "openai",
                "model": "gpt-5.6-terra",
                "content": [{"type": "text", "text": "done"}],
                "usage": {
                    "input": 12,
                    "output": 7,
                    "cacheRead": 30,
                    "cacheWrite": 2,
                },
                "stopReason": "stop",
            },
        },
    ]
    (agent_dir / "openclaw.session.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (agent_dir / "openclaw.txt").write_text(
        "[provider-transport-fetch] [model-fetch] response "
        "provider=openai api=openai-responses model=gpt-5.6-terra status=200\n"
        "run ended before JSON envelope\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-terra",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-terra",
        model_id="gpt-5.6-terra",
        provider="openai",
        proxy_model_name="gpt-5.6-terra",
        repetition=1,
        expected_task_count=1,
        run_date="20260728",
    )
    task = _trajectory_task(tmp_path, "do the task")

    metadata = write_agent_trajectory(task, run, agent_dir)
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert metadata["canonical_model_identity"] is True
    assert metadata["trajectory_validation"]["trace_fidelity"] == "session"
    assert metadata["trajectory_validation"]["log_models"] == ["gpt-5.6-terra"]
    assert trajectory["session_id"] == "session-only-123"
    assert trajectory["agent"]["model_name"] == "openai/gpt-5.6-terra"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 44
    assert trajectory["final_metrics"]["total_completion_tokens"] == 7
    assert trajectory["final_metrics"]["total_cached_tokens"] == 30
    assert trajectory["extra"]["cache_write_tokens"] == 2

    records[-1]["message"]["usage"] = {"input": 1}
    (agent_dir / "openclaw.session.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    envelope = {
        "payloads": [{"text": "done"}],
        "meta": {
            "agentMeta": {
                "sessionId": "session-only-123",
                "model": "gpt-5.6-terra",
                "usage": {"input": 80, "output": 9, "cacheRead": 20},
            },
            "executionTrace": {
                "winnerProvider": "openai",
                "winnerModel": "gpt-5.6-terra",
            },
            "completion": {"stopReason": "stop"},
            "aborted": False,
        },
    }
    (agent_dir / "openclaw.txt").write_text(
        json.dumps(envelope),
        encoding="utf-8",
    )

    write_agent_trajectory(task, run, agent_dir)
    fallback_trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert fallback_trajectory["final_metrics"]["total_prompt_tokens"] == 100
    assert fallback_trajectory["final_metrics"]["total_completion_tokens"] == 9
    assert fallback_trajectory["final_metrics"]["total_cached_tokens"] == 20


def test_openclaw_exported_trajectory_bundle_converts_to_atif(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    bundle = agent_dir / "openclaw.sessions" / "exports" / "shellbench-root"
    bundle.mkdir(parents=True)
    user = {"role": "user", "content": "do the task", "timestamp": 1_753_833_602_000}
    exec_call = {
        "role": "assistant",
        "model": "gpt-5.5",
        "timestamp": 1_753_833_603_000,
        "content": [
            {
                "type": "toolCall",
                "id": "call-exec",
                "name": "exec",
                "arguments": {"code": "await tools.shell({cmd: 'pwd'})"},
            }
        ],
        "usage": {"input": 12, "output": 3, "cacheRead": 5},
        "stopReason": "toolUse",
    }
    exec_result = {
        "role": "toolResult",
        "toolCallId": "call-exec",
        "timestamp": 1_753_833_604_000,
        "content": [{"type": "text", "text": "code completed"}],
    }
    nested_call = {
        "role": "assistant",
        "timestamp": 1_753_833_604_100,
        "content": [
            {
                "type": "toolCall",
                "id": "tool_search_code:call-exec:shell:1",
                "name": "shell",
                "arguments": {"cmd": "pwd"},
            }
        ],
        "stopReason": "toolUse",
    }
    nested_result = {
        "role": "toolResult",
        "toolCallId": "tool_search_code:call-exec:shell:1",
        "timestamp": 1_753_833_604_200,
        "content": [{"type": "text", "text": "/app"}],
    }
    final = {
        "role": "assistant",
        "model": "gpt-5.5",
        "timestamp": 1_753_833_605_000,
        "content": [{"type": "text", "text": "done"}],
        "usage": {"input": 4, "output": 2, "cacheRead": 1},
        "stopReason": "stop",
    }
    persisted_messages = [user, exec_call, exec_result, final]
    snapshot = [user, exec_call, exec_result, nested_call, nested_result, final]
    records = [
        {
            "type": "message",
            "id": f"entry-{index}",
            "timestamp": f"2026-07-30T00:00:0{index}Z",
            "message": message,
        }
        for index, message in enumerate(persisted_messages, start=1)
    ]
    runtime_events = [
        {
            "traceSchema": "openclaw-trajectory",
            "schemaVersion": 1,
            "traceId": "session-export-123",
            "source": "runtime",
            "type": "context.compiled",
            "ts": "2026-07-30T00:00:01Z",
            "seq": 1,
            "sourceSeq": 1,
            "sessionId": "session-export-123",
            "sessionKey": "agent:main:main",
            "runId": "run-1",
            "data": {
                "tools": [{"name": "shell"}, {"name": "read"}],
                "providerVisibleTools": [
                    {"name": "exec"},
                    {"name": "image"},
                    {"name": "sessions_yield"},
                    {"name": "wait"},
                ],
            },
        },
        {
            "traceSchema": "openclaw-trajectory",
            "schemaVersion": 1,
            "traceId": "session-export-123",
            "source": "runtime",
            "type": "model.completed",
            "ts": "2026-07-30T00:00:03Z",
            "seq": 2,
            "sourceSeq": 2,
            "sessionId": "session-export-123",
            "sessionKey": "agent:main:main",
            "runId": "run-1",
            "provider": "openai",
            "modelId": "gpt-5.5",
            "data": {"usage": {"input": 12, "output": 3, "cacheRead": 5}},
        },
        {
            "traceSchema": "openclaw-trajectory",
            "schemaVersion": 1,
            "traceId": "session-export-123",
            "source": "runtime",
            "type": "model.completed",
            "ts": "2026-07-30T00:00:05Z",
            "seq": 3,
            "sourceSeq": 3,
            "sessionId": "session-export-123",
            "sessionKey": "agent:main:main",
            "runId": "run-1",
            "provider": "openai",
            "modelId": "gpt-5.5",
            "data": {
                "usage": {"input": 4, "output": 2, "cacheRead": 1},
                "messagesSnapshot": snapshot,
            },
        },
        {
            "traceSchema": "openclaw-trajectory",
            "schemaVersion": 1,
            "traceId": "session-export-123",
            "source": "runtime",
            "type": "session.ended",
            "ts": "2026-07-30T00:00:06Z",
            "seq": 4,
            "sourceSeq": 4,
            "sessionId": "session-export-123",
            "sessionKey": "agent:main:main",
            "runId": "run-1",
            "data": {"status": "success"},
        },
    ]
    transcript_events = [
        {
            "traceSchema": "openclaw-trajectory",
            "schemaVersion": 1,
            "traceId": "session-export-123",
            "source": "transcript",
            "type": "assistant.message",
            "ts": record["timestamp"],
            "seq": 4 + index,
            "sourceSeq": index,
            "sessionId": "session-export-123",
            "sessionKey": "agent:main:main",
            "entryId": record["id"],
            "data": {"message": record["message"]},
        }
        for index, record in enumerate(records, start=1)
    ]
    events = [*runtime_events, *transcript_events]
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "traceSchema": "openclaw-trajectory",
                "schemaVersion": 1,
                "generatedAt": "2026-07-30T00:00:07Z",
                "traceId": "session-export-123",
                "sessionId": "session-export-123",
                "sessionKey": "agent:main:main",
                "workspaceDir": "$WORKSPACE_DIR",
                "leafId": "entry-4",
                "eventCount": len(events),
                "runtimeEventCount": len(runtime_events),
                "transcriptEventCount": len(transcript_events),
                "sourceFiles": {"session": "$OPENCLAW_STATE/session"},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "session-branch.json").write_text(
        json.dumps(
            {
                "header": {
                    "type": "session",
                    "id": "session-export-123",
                    "timestamp": "2026-07-30T00:00:00Z",
                    "cwd": "/app",
                },
                "leafId": "entry-4",
                "entries": records,
            }
        ),
        encoding="utf-8",
    )
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (agent_dir / "openclaw.txt").write_text(
        "[provider-transport-fetch] [model-fetch] response "
        "provider=openai api=openai-responses model=gpt-5.5 status=200\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt55-code",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="gpt-5.5",
        repetition=1,
        expected_task_count=1,
        run_date="20260730",
        openclaw_tool_mode="code",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert metadata["trajectory_validation"]["trace_fidelity"] == "session"
    assert metadata["trajectory_validation"]["session_tree_session_count"] == 1
    assert metadata["trajectory_validation"]["export_snapshot_recorded"] is True
    assert metadata["trajectory_validation"]["export_snapshot_used"] is False
    assert metadata["trajectory_validation"]["export_branch_used"] is True
    assert metadata["trajectory_validation"]["export_snapshot_outcome"] == "assistant_text"
    assert metadata["trajectory_validation"]["export_snapshot_tool_call_count"] == 2
    assert metadata["trajectory_validation"]["export_snapshot_tool_result_count"] == 2
    assert metadata["trajectory_validation"]["export_snapshot_tool_error_count"] == 0
    assert metadata["trajectory_validation"]["export_provider_tool_call_count"] == 1
    assert metadata["trajectory_validation"]["export_provider_tool_result_count"] == 1
    assert (
        metadata["trajectory_validation"]["export_snapshot_pending_tool_call_count"]
        == 0
    )
    assert metadata["trajectory_validation"]["export_provider_visible_tools_recorded"] is True
    assert metadata["trajectory_validation"]["export_visible_tools"] == [
        "exec",
        "image",
        "sessions_yield",
        "wait",
    ]
    assert metadata["trajectory_validation"]["tool_mode_observed"] is True
    assert trajectory["session_id"] == "session-export-123"
    assert len(trajectory["steps"]) == 3
    assert trajectory["steps"][1]["tool_calls"][0]["function_name"] == "exec"
    assert trajectory["steps"][1]["observation"]["results"][0]["content"] == "code completed"
    assert all(
        call["function_name"] != "shell"
        for step in trajectory["steps"]
        for call in step.get("tool_calls", [])
    )
    assert trajectory["steps"][-1]["message"] == "done"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 22
    assert trajectory["final_metrics"]["total_completion_tokens"] == 5
    assert trajectory["final_metrics"]["total_cached_tokens"] == 6
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "direct",
                "root",
            ],
            check=False,
        ).returncode
        == 0
    )
    latest_completion = runtime_events[2]
    terminal_event = runtime_events[3]
    bookkeeping = {"role": "system", "content": "runtime bookkeeping"}
    latest_completion["data"]["messagesSnapshot"] = [*snapshot, bookkeeping]
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    bookkeeping_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert bookkeeping_metadata["trajectory_status"] == "real"
    assert (
        bookkeeping_metadata["trajectory_validation"]["export_snapshot_outcome"]
        == "assistant_text"
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 0
    )
    failed_exec_result = {**exec_result, "isError": True}
    latest_completion["data"]["messagesSnapshot"] = [
        user,
        exec_call,
        failed_exec_result,
        bookkeeping,
    ]
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    failed_tool_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert failed_tool_metadata["trajectory_status"] == "real"
    assert (
        failed_tool_metadata["trajectory_validation"]["export_snapshot_outcome"]
        == "tool_error"
    )
    assert (
        failed_tool_metadata["trajectory_validation"]["export_snapshot_tool_error_count"]
        == 1
    )
    for tool_mode in ("direct", "code"):
        assert (
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _OPENCLAW_EXPORT_READY,
                    str(bundle),
                    tool_mode,
                    "root",
                ],
                check=False,
            ).returncode
            == 0
        )
    latest_completion["data"]["messagesSnapshot"] = [
        user,
        exec_call,
        failed_exec_result,
        final,
        bookkeeping,
    ]
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    recovered_tool_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert recovered_tool_metadata["trajectory_status"] == "real"
    assert (
        recovered_tool_metadata["trajectory_validation"]["export_snapshot_outcome"]
        == "assistant_text"
    )
    assert (
        recovered_tool_metadata["trajectory_validation"]["export_snapshot_tool_error_count"]
        == 1
    )
    latest_completion["data"]["messagesSnapshot"] = [user, exec_call, bookkeeping]
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    unresolved_tool_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert unresolved_tool_metadata["trajectory_status"] == "real"
    assert (
        unresolved_tool_metadata["trajectory_validation"]["export_snapshot_outcome"]
        == "unresolved_tool_call"
    )
    assert (
        unresolved_tool_metadata["trajectory_validation"][
            "export_snapshot_pending_tool_call_count"
        ]
        == 1
    )
    assert (
        unresolved_tool_metadata["trajectory_validation"]["snapshot_complete"] is False
    )
    assert (
        unresolved_tool_metadata["trajectory_validation"]["provider_transcript_complete"]
        is True
    )
    branch_payload = json.loads((bundle / "session-branch.json").read_text())
    branch_payload["entries"] = records[:2]
    branch_payload["leafId"] = records[1]["id"]
    (bundle / "session-branch.json").write_text(
        json.dumps(branch_payload),
        encoding="utf-8",
    )
    incomplete_provider_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert incomplete_provider_metadata["trajectory_status"] == "unavailable"
    assert (
        incomplete_provider_metadata["trajectory_validation"][
            "provider_transcript_complete"
        ]
        is False
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 1
    )
    branch_payload["entries"] = records
    branch_payload["leafId"] = records[-1]["id"]
    (bundle / "session-branch.json").write_text(
        json.dumps(branch_payload),
        encoding="utf-8",
    )
    latest_completion["data"]["messagesSnapshot"] = snapshot
    terminal_event["data"] = {"status": "error"}
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    failed_root_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert failed_root_metadata["trajectory_status"] == "unavailable"
    assert failed_root_metadata["trajectory_validation"]["export_terminal_status"] == "error"
    terminal_event["data"] = {"status": "success"}
    latest_completion["data"]["messagesSnapshot"] = []
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 0
    )
    latest_completion["data"]["messagesSnapshot"] = snapshot
    latest_completion["data"].pop("messagesSnapshot")
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    incomplete_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert incomplete_metadata["trajectory_status"] == "real"
    assert incomplete_metadata["trajectory_validation"]["export_snapshot_used"] is False
    assert incomplete_metadata["trajectory_validation"]["tool_mode_observed"] is True
    assert (
        incomplete_metadata["trajectory_validation"]["provider_transcript_complete"]
        is True
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 0
    )
    latest_completion["data"]["messagesSnapshot"] = snapshot
    runtime_events[0]["data"]["providerVisibleTools"] = [{"name": "exec"}]
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 1
    )
    runtime_events[0]["data"]["tools"] = [{"name": "exec"}, {"name": "wait"}]
    runtime_events[0]["data"]["providerVisibleTools"] = []
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 1
    )
    empty_visible_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert empty_visible_metadata["trajectory_status"] == "unavailable"
    assert (
        empty_visible_metadata["trajectory_validation"][
            "export_provider_visible_tools_recorded"
        ]
        is True
    )
    assert empty_visible_metadata["trajectory_validation"]["export_visible_tools"] == []
    assert empty_visible_metadata["trajectory_validation"]["tool_mode_observed"] is False
    runtime_events[0]["data"]["tools"] = [{"name": "shell"}, {"name": "read"}]
    runtime_events[0]["data"].pop("providerVisibleTools")
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 1
    )
    missing_visible_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert missing_visible_metadata["trajectory_status"] == "unavailable"
    assert (
        missing_visible_metadata["trajectory_validation"][
            "export_provider_visible_tools_recorded"
        ]
        is False
    )
    runtime_events[0]["data"]["providerVisibleTools"] = [
        {"name": "exec"},
        {"name": "tool_search"},
        {"name": "wait"},
    ]
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 1
    )
    visible_search_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert visible_search_metadata["trajectory_status"] == "unavailable"
    assert visible_search_metadata["trajectory_validation"]["tool_mode_observed"] is False
    runtime_events[0]["data"]["providerVisibleTools"] = [
        {"name": "exec"},
        {"name": "wait"},
    ]
    latest_completion["traceId"] = "stale-trace"
    (bundle / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 1
    )
    invalid_trace_metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )
    assert invalid_trace_metadata["trajectory_status"] == "unavailable"
    assert invalid_trace_metadata["trajectory_validation"]["export_valid"] is False
    latest_completion["traceId"] = "session-export-123"
    terminal_only_events = [
        {
            **event,
            "data": {"status": "error"},
        }
        for event in events
        if event["type"] == "session.ended"
    ]
    manifest = json.loads((bundle / "manifest.json").read_text())
    manifest["eventCount"] = len(terminal_only_events) + len(transcript_events)
    manifest["runtimeEventCount"] = len(terminal_only_events)
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (bundle / "events.jsonl").write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in [*terminal_only_events, *transcript_events]
        ),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "child",
            ],
            check=False,
        ).returncode
        == 0
    )
    terminal_only_events[0]["data"] = {"status": "success"}
    (bundle / "events.jsonl").write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in [*terminal_only_events, *transcript_events]
        ),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "child",
            ],
            check=False,
        ).returncode
        == 1
    )
    terminal_only_events[0]["data"] = {"status": "error"}
    (bundle / "events.jsonl").write_text(
        "".join(
            json.dumps(event) + "\n"
            for event in [*terminal_only_events, *transcript_events]
        ),
        encoding="utf-8",
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-c",
                _OPENCLAW_EXPORT_READY,
                str(bundle),
                "code",
                "root",
            ],
            check=False,
        ).returncode
        == 1
    )


def test_openclaw_transport_model_mismatch_invalidates_identity(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    records = [
        {"type": "session", "id": "session-123"},
        {"type": "model_change", "modelId": "gpt-5.6-sol"},
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00Z",
            "message": {"role": "user", "content": "do the task"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:01Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "done"}],
                "stopReason": "stop",
            },
        },
    ]
    (agent_dir / "openclaw.session.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (agent_dir / "openclaw.txt").write_text(
        "[provider-transport-fetch] [model-fetch] response "
        "provider=openai api=openai-responses model=gpt-5.5 status=200\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-sol",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=1,
        run_date="20260729",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )

    assert metadata["trajectory_status"] == "real"
    assert metadata["canonical_model_identity"] is False
    assert metadata["trajectory_validation"]["observed_models"] == [
        "gpt-5.5",
        "gpt-5.6-sol",
    ]
    assert metadata["trajectory_validation"]["log_models"] == ["gpt-5.5"]


def test_openclaw_child_model_mismatch_invalidates_identity(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    archive = agent_dir / "openclaw.sessions"
    archive.mkdir(parents=True)
    child_key = "agent:main:subagent:model-mismatch"
    records = [
        {
            "type": "session",
            "id": "session-123",
        },
        {
            "type": "model_change",
            "modelId": "gpt-5.6-sol",
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00Z",
            "message": {"role": "user", "content": "do the task"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:01Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "spawn-1",
                        "name": "sessions_spawn",
                        "arguments": {"task": "inspect"},
                    }
                ],
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:02Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "spawn-1",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "accepted",
                                "childSessionKey": child_key,
                                "resolvedModel": "openai/gpt-5.5",
                            }
                        ),
                    }
                ],
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:05Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "done"}],
                "stopReason": "stop",
            },
        },
    ]
    child_records = [
        {
            "type": "session",
            "id": "child-session-123",
        },
        {
            "type": "model_change",
            "modelId": "gpt-5.5",
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:03Z",
            "message": {"role": "user", "content": "[Subagent Task] inspect"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:04Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.5",
                "content": [{"type": "text", "text": "child done"}],
                "stopReason": "stop",
            },
        },
    ]
    (agent_dir / "openclaw.session.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (archive / "child-session-123.jsonl").write_text(
        "\n".join(json.dumps(record) for record in child_records) + "\n",
        encoding="utf-8",
    )
    (archive / "sessions.json").write_text(
        json.dumps(
            {
                "agent:main:main": {"sessionId": "session-123"},
                child_key: {
                    "sessionId": "child-session-123",
                    "spawnedBy": "agent:main:main",
                },
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "openclaw.txt").write_text(
        "[provider-transport-fetch] [model-fetch] response "
        "provider=openai api=openai-responses model=gpt-5.6-sol status=200\n"
        "[provider-transport-fetch] [model-fetch] response "
        "provider=openai api=openai-responses model=gpt-5.5 status=200\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-sol",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=1,
        run_date="20260728",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )

    assert metadata["trajectory_status"] == "real"
    assert metadata["trajectory_validation"]["trace_fidelity"] == "session_tree"
    assert metadata["runtime_model_name"] == "gpt-5.6-sol"
    assert metadata["canonical_model_identity"] is False
    assert metadata["trajectory_validation"]["observed_models"] == [
        "gpt-5.5",
        "gpt-5.6-sol",
    ]
    assert metadata["trajectory_validation"]["child_models"] == ["gpt-5.5"]


def test_openclaw_session_tree_includes_descendant_tools_and_ignores_stale_sessions(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    archive = agent_dir / "openclaw.sessions"
    archive.mkdir(parents=True)
    root_key = "agent:main:main"
    child_key = "agent:main:subagent:worker"
    stale_key = "agent:main:subagent:stale"
    root_records = [
        {
            "type": "session",
            "id": "root-session",
            "timestamp": "2026-07-29T08:00:00Z",
        },
        {
            "type": "model_change",
            "modelId": "gpt-5.6-sol",
            "timestamp": "2026-07-29T08:00:00.100Z",
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:01Z",
            "message": {"role": "user", "content": "fix both files"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:02Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "spawn-1",
                        "name": "sessions_spawn",
                        "arguments": {"task": "fix notifications.py"},
                    }
                ],
                "usage": {
                    "input": 10,
                    "output": 2,
                    "cacheRead": 3,
                },
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:02.100Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "spawn-1",
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "status": "accepted",
                                "childSessionKey": child_key,
                                "resolvedModel": "openai/gpt-5.6-sol",
                            }
                        ),
                    }
                ],
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:06Z",
            "message": {
                "role": "user",
                "content": "A completed subagent task is ready: child fixed notifications.py",
                "provenance": {
                    "kind": "inter_session",
                    "sourceSessionKey": child_key,
                    "sourceTool": "subagent_announce",
                },
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:07Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "integrated and verified"}],
                "usage": {"input": 4, "output": 2},
                "stopReason": "stop",
            },
        },
    ]
    child_records = [
        {
            "type": "session",
            "id": "child-session",
            "timestamp": "2026-07-29T08:00:02.200Z",
            "parentSession": "/tmp/openclaw/root-session.jsonl",
        },
        root_records[2],
        root_records[3],
        {
            "type": "model_change",
            "modelId": "gpt-5.6-sol",
            "timestamp": "2026-07-29T08:00:02.300Z",
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:03Z",
            "message": {
                "role": "user",
                "content": (
                    "[Subagent Context] You are running as a subagent (depth 2/2)."
                    "\n\n[Subagent Task]\nfix notifications.py"
                ),
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:04Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "read-1",
                        "name": "read",
                        "arguments": {"path": "notifications.py"},
                    }
                ],
                "usage": {
                    "input": 5,
                    "output": 1,
                    "cacheRead": 2,
                },
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:04.100Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "read-1",
                "content": [{"type": "text", "text": "broken code"}],
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:05Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "edit-1",
                        "name": "edit",
                        "arguments": {"path": "notifications.py"},
                    }
                ],
                "usage": {"input": 6, "output": 1},
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:05.100Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "edit-1",
                "content": [{"type": "text", "text": "updated"}],
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:05.500Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "child fixed notifications.py"}],
                "usage": {"input": 2, "output": 2},
                "stopReason": "stop",
            },
        },
    ]
    stale_records = [
        {
            "type": "session",
            "id": "stale-session",
            "timestamp": "2026-07-29T07:00:00Z",
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T07:00:01Z",
            "message": {"role": "user", "content": "old task"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T07:00:02Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "stale-1",
                        "name": "exec",
                        "arguments": {"command": "old-command"},
                    }
                ],
                "stopReason": "toolUse",
            },
        },
    ]
    for path, records in (
        (agent_dir / "openclaw.session.jsonl", root_records),
        (archive / "root-session.jsonl", root_records),
        (archive / "child-session.jsonl", child_records),
        (archive / "stale-session.jsonl", stale_records),
    ):
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    (archive / "sessions.json").write_text(
        json.dumps(
            {
                root_key: {
                    "sessionId": "root-session",
                    "sessionFile": "/tmp/openclaw/root-session.jsonl",
                },
                child_key: {
                    "sessionId": "child-session",
                    "sessionFile": "/tmp/openclaw/child-session.jsonl",
                    "spawnedBy": root_key,
                    "forkedFromParent": True,
                },
                stale_key: {
                    "sessionId": "stale-session",
                    "sessionFile": "/tmp/openclaw/stale-session.jsonl",
                    "spawnedBy": root_key,
                },
            }
        ),
        encoding="utf-8",
    )
    (agent_dir / "openclaw.txt").write_text(
        "[provider-transport-fetch] [model-fetch] response "
        "provider=openai api=openai-responses model=gpt-5.6-sol status=200\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-sol-tree",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=1,
        run_date="20260729",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "fix both files"),
        run,
        agent_dir,
    )
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())
    function_names = [
        call["function_name"] for step in trajectory["steps"] for call in step.get("tool_calls", [])
    ]
    messages = [step["message"] for step in trajectory["steps"]]

    assert metadata["trajectory_status"] == "real"
    assert metadata["canonical_model_identity"] is True
    assert metadata["trajectory_validation"]["trace_fidelity"] == "session_tree"
    assert metadata["trajectory_validation"]["session_tree_session_count"] == 2
    assert metadata["trajectory_validation"]["session_tree_descendant_count"] == 1
    assert metadata["trajectory_validation"]["session_tree_accepted_spawn_count"] == 1
    assert metadata["trajectory_validation"]["session_tree_reused_spawn_count"] == 0
    assert metadata["trajectory_validation"]["session_tree_ignored_entry_count"] == 1
    assert metadata["trajectory_validation"]["session_tree_session_keys"] == [
        root_key,
        child_key,
    ]
    assert function_names == ["sessions_spawn", "read", "edit"]
    assert messages.count("child fixed notifications.py") == 1
    assert messages.count("fix both files") == 1
    assert "old task" not in messages
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 32
    assert trajectory["final_metrics"]["total_completion_tokens"] == 8
    assert trajectory["final_metrics"]["total_cached_tokens"] == 5
    assert trajectory["steps"][2]["extra"]["openclaw_session_key"] == child_key


def test_openclaw_session_tree_preserves_reused_child_tasks_once(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    archive = agent_dir / "openclaw.sessions"
    archive.mkdir(parents=True)
    root_key = "agent:main:main"
    child_key = "agent:main:subagent:worker"
    root_records = [
        {"type": "session", "id": "root-session"},
        {"type": "model_change", "modelId": "gpt-5.6-sol"},
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.000Z",
            "message": {"role": "user", "content": "delegate twice"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.100Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "spawn-1",
                        "name": "sessions_spawn",
                        "arguments": {"task": "first task"},
                    },
                    {
                        "type": "toolCall",
                        "id": "spawn-2",
                        "name": "sessions_spawn",
                        "arguments": {"task": "second task"},
                    },
                ],
                "usage": {"input": 10, "output": 2, "cacheRead": 2},
                "stopReason": "toolUse",
            },
        },
        *[
            {
                "type": "message",
                "timestamp": f"2026-07-29T08:00:00.{index}00Z",
                "message": {
                    "role": "toolResult",
                    "toolCallId": call_id,
                    "content": json.dumps(
                        {
                            "status": "accepted",
                            "childSessionKey": child_key,
                            "resolvedModel": "openai/gpt-5.6-sol",
                        }
                    ),
                },
            }
            for index, call_id in enumerate(("spawn-1", "spawn-2"), start=2)
        ],
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.900Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "both tasks complete"}],
                "usage": {"input": 4, "output": 1},
                "stopReason": "stop",
            },
        },
    ]
    child_records = [
        {"type": "session", "id": "child-session"},
        {"type": "model_change", "modelId": "gpt-5.6-sol"},
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.400Z",
            "message": {"role": "user", "content": "first task"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.500Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "exec-1",
                        "name": "exec",
                        "arguments": {"command": "inspect"},
                    }
                ],
                "usage": {"input": 5, "output": 1, "cacheRead": 1},
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.550Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "exec-1",
                "content": "inspected",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.600Z",
            "message": {"role": "user", "content": "second task"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.700Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "edit-1",
                        "name": "edit",
                        "arguments": {"path": "target.py"},
                    }
                ],
                "usage": {"input": 7, "output": 2},
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.750Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "edit-1",
                "content": "edited",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00.800Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "child finished both"}],
                "usage": {"input": 3, "output": 1},
                "stopReason": "stop",
            },
        },
    ]
    for path, records in (
        (agent_dir / "openclaw.session.jsonl", root_records),
        (archive / "child-session.jsonl", child_records),
    ):
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
    (archive / "sessions.json").write_text(
        json.dumps(
            {
                root_key: {"sessionId": "root-session"},
                child_key: {
                    "sessionId": "child-session",
                    "spawnedBy": root_key,
                },
            }
        ),
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-sol-reuse",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=1,
        run_date="20260729",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "delegate twice"),
        run,
        agent_dir,
    )
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())
    function_names = [
        call["function_name"] for step in trajectory["steps"] for call in step.get("tool_calls", [])
    ]
    messages = [step["message"] for step in trajectory["steps"]]

    assert metadata["trajectory_status"] == "real"
    assert metadata["trajectory_validation"]["session_tree_accepted_spawn_count"] == 2
    assert metadata["trajectory_validation"]["session_tree_reused_spawn_count"] == 1
    assert metadata["trajectory_validation"]["session_tree_session_count"] == 2
    assert function_names == ["sessions_spawn", "sessions_spawn", "exec", "edit"]
    assert messages.count("first task") == 1
    assert messages.count("second task") == 1
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 32
    assert trajectory["final_metrics"]["total_completion_tokens"] == 7
    assert trajectory["final_metrics"]["total_cached_tokens"] == 3

    child_records[2].pop("timestamp")
    (archive / "child-session.jsonl").write_text(
        "\n".join(json.dumps(record) for record in child_records) + "\n",
        encoding="utf-8",
    )
    _, missing_timestamp_validation = _openclaw_session_tree(
        agent_dir,
        agent_dir / "openclaw.session.jsonl",
    )

    assert missing_timestamp_validation["session_tree_complete"] is False
    assert missing_timestamp_validation["session_tree_missing_timestamp_count"] == 1

    child_records[2]["timestamp"] = "2026-07-29T08:00:00.900Z"
    (archive / "child-session.jsonl").write_text(
        "\n".join(json.dumps(record) for record in child_records) + "\n",
        encoding="utf-8",
    )
    _, ambiguous_timestamp_validation = _openclaw_session_tree(
        agent_dir,
        agent_dir / "openclaw.session.jsonl",
    )

    assert ambiguous_timestamp_validation["session_tree_complete"] is False
    assert ambiguous_timestamp_validation["session_tree_ambiguous_timestamp_count"] == 1


def test_openclaw_session_tree_rejects_missing_accepted_child(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    archive = agent_dir / "openclaw.sessions"
    archive.mkdir(parents=True)
    child_key = "agent:main:subagent:missing"
    records = [
        {"type": "session", "id": "root-session"},
        {"type": "model_change", "modelId": "gpt-5.6-sol"},
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00Z",
            "message": {"role": "user", "content": "delegate the task"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:01Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "spawn-1",
                        "name": "sessions_spawn",
                        "arguments": {"task": "do work"},
                    }
                ],
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:02Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "spawn-1",
                "content": json.dumps(
                    {
                        "status": "accepted",
                        "childSessionKey": child_key,
                        "resolvedModel": "openai/gpt-5.6-sol",
                    }
                ),
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:05Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "done"}],
                "stopReason": "stop",
            },
        },
    ]
    (agent_dir / "openclaw.session.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (archive / "sessions.json").write_text(
        json.dumps(
            {
                "agent:main:main": {
                    "sessionId": "root-session",
                    "sessionFile": "/tmp/openclaw/root-session.jsonl",
                }
            }
        ),
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="openclaw-gpt56-sol-missing-child",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=1,
        run_date="20260729",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "delegate the task"),
        run,
        agent_dir,
    )

    assert metadata["trajectory_status"] == "unavailable"
    assert metadata["trajectory_validation"]["session_tree_complete"] is False
    assert metadata["trajectory_validation"]["session_tree_missing_transcript_count"] == 1
    assert metadata["trajectory_validation"]["session_tree_errors"] == [
        f"missing-session-entry:{child_key}"
    ]

    child_records = [
        {"type": "session", "id": "deleted-child-session"},
        {"type": "model_change", "modelId": "gpt-5.6-sol"},
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:02.500Z",
            "message": {"role": "user", "content": "inherited parent prompt"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:03Z",
            "message": {
                "role": "user",
                "content": (
                    "[Subagent Context] You are running as a subagent "
                    "(depth 1/2).\n\n[Subagent Task]\n\ndo work\n\n"
                    "Begin. Execute the assigned task to completion."
                ),
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:04Z",
            "message": {
                "role": "assistant",
                "model": "gpt-5.6-sol",
                "content": [{"type": "text", "text": "child done"}],
                "stopReason": "stop",
            },
        },
    ]
    audit_dir = archive / "audit"
    audit_transcripts = audit_dir / "transcripts"
    audit_transcripts.mkdir(parents=True)
    audited_child = audit_transcripts / "deleted-child-session.jsonl"
    audited_child.write_text(
        "\n".join(json.dumps(record) for record in child_records) + "\n",
        encoding="utf-8",
    )
    (audit_dir / "sessions.jsonl").write_text(
        json.dumps(
            {
                "type": "session_end",
                "sessionKey": child_key,
                "sessionId": "deleted-child-session",
                "spawnedBy": "agent:main:main",
                "status": "done",
                "auditTranscript": "transcripts/deleted-child-session.jsonl",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (archive / "sessions.json").write_text(
        json.dumps(
            {
                "agent:main:main": {"sessionId": "root-session"},
                child_key: {
                    "sessionId": "deleted-child-session",
                    "sessionFile": "missing-child.jsonl",
                    "spawnedBy": "agent:main:main",
                    "status": "done",
                },
            }
        ),
        encoding="utf-8",
    )

    audited_tree, audited_validation = _openclaw_session_tree(
        agent_dir,
        agent_dir / "openclaw.session.jsonl",
    )

    assert [key for key, _, _, _ in audited_tree] == [
        "agent:main:main",
        child_key,
    ]
    assert audited_validation["session_tree_complete"] is True

    (audit_dir / "sessions.jsonl").unlink()
    audited_child.unlink()
    (archive / "sessions.json").write_text(
        json.dumps({"agent:main:main": {"sessionId": "root-session"}}),
        encoding="utf-8",
    )
    deleted_path = archive / "deleted-child-session.jsonl.deleted.2026-07-29T08-00-04.000Z"
    deleted_path.write_text(
        "\n".join(json.dumps(record) for record in child_records) + "\n",
        encoding="utf-8",
    )

    unaudited_tree, unaudited_validation = _openclaw_session_tree(
        agent_dir,
        agent_dir / "openclaw.session.jsonl",
    )

    assert [key for key, _, _, _ in unaudited_tree] == ["agent:main:main"]
    assert unaudited_validation["session_tree_complete"] is False
    assert unaudited_validation["session_tree_errors"] == [f"missing-session-entry:{child_key}"]


def test_openclaw_session_tree_uses_only_the_active_transcript_branch(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    archive = agent_dir / "openclaw.sessions"
    archive.mkdir(parents=True)
    root_path = agent_dir / "openclaw.session.jsonl"
    stale_child_key = "agent:main:subagent:stale"
    records = [
        {"type": "session", "id": "root-session"},
        {
            "type": "message",
            "id": "user-root",
            "parentId": None,
            "message": {"role": "user", "content": "delegate"},
        },
        {
            "type": "message",
            "id": "assistant-stale",
            "parentId": "user-root",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "spawn-stale",
                        "name": "sessions_spawn",
                        "arguments": {"task": "stale branch"},
                    }
                ],
                "stopReason": "toolUse",
                "usage": {"input": 1000, "output": 100},
            },
        },
        {
            "type": "message",
            "id": "result-stale",
            "parentId": "assistant-stale",
            "message": {
                "role": "toolResult",
                "toolCallId": "spawn-stale",
                "content": json.dumps(
                    {
                        "status": "accepted",
                        "childSessionKey": stale_child_key,
                    }
                ),
            },
        },
        {
            "type": "message",
            "id": "user-active",
            "parentId": "user-root",
            "message": {"role": "user", "content": "do it locally instead"},
        },
        {
            "type": "message",
            "id": "assistant-active",
            "parentId": "user-active",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "stopReason": "stop",
                "usage": {"input": 10, "output": 2},
            },
        },
    ]
    root_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )
    (archive / "sessions.json").write_text(
        json.dumps({"agent:main:main": {"sessionId": "root-session"}}),
        encoding="utf-8",
    )

    tree, validation = _openclaw_session_tree(agent_dir, root_path)

    assert validation["session_tree_complete"] is True
    assert validation["session_tree_accepted_spawn_count"] == 0
    assert validation["session_tree_session_count"] == 1
    assert [record.get("id") for record in tree[0][3]] == [
        "user-root",
        "user-active",
        "assistant-active",
    ]

    run = RunSpec(
        run_label="openclaw-gpt55-active-branch",
        harness="openclaw",
        harness_version="2026.7.1-2",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="gpt-5.5",
        repetition=1,
        expected_task_count=1,
        run_date="20260729",
    )
    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "delegate"),
        run,
        agent_dir,
    )
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert [
        call["function_name"] for step in trajectory["steps"] for call in step.get("tool_calls", [])
    ] == []
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 10
    assert trajectory["final_metrics"]["total_completion_tokens"] == 2


def test_openclaw_session_tree_recurses_through_grandchildren(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    archive = agent_dir / "openclaw.sessions"
    archive.mkdir(parents=True)
    root_key = "agent:main:main"
    child_key = "agent:main:subagent:child"
    grandchild_key = "agent:main:subagent:grandchild"

    def write_session(
        path: Path,
        session_id: str,
        child_session_key: str | None = None,
    ) -> None:
        timeline = {
            "root-session": ("00", "01", "02", "09"),
            "child-session": ("03", "04", "05", "08"),
            "grandchild-session": ("06", "07"),
        }[session_id]
        records: list[dict[str, object]] = [
            {"type": "session", "id": session_id},
            {"type": "model_change", "modelId": "gpt-5.6-sol"},
            {
                "type": "message",
                "timestamp": f"2026-07-29T08:00:{timeline[0]}Z",
                "message": {"role": "user", "content": f"task for {session_id}"},
            },
        ]
        if child_session_key:
            records.extend(
                [
                    {
                        "type": "message",
                        "timestamp": f"2026-07-29T08:00:{timeline[1]}Z",
                        "message": {
                            "role": "assistant",
                            "model": "gpt-5.6-sol",
                            "content": [
                                {
                                    "type": "toolCall",
                                    "id": f"spawn-{session_id}",
                                    "name": "sessions_spawn",
                                    "arguments": {"task": "delegate"},
                                }
                            ],
                            "stopReason": "toolUse",
                        },
                    },
                    {
                        "type": "message",
                        "timestamp": f"2026-07-29T08:00:{timeline[2]}Z",
                        "message": {
                            "role": "toolResult",
                            "toolCallId": f"spawn-{session_id}",
                            "content": json.dumps(
                                {
                                    "status": "accepted",
                                    "childSessionKey": child_session_key,
                                    "resolvedModel": "openai/gpt-5.6-sol",
                                }
                            ),
                        },
                    },
                ]
            )
        records.append(
            {
                "type": "message",
                "timestamp": f"2026-07-29T08:00:{timeline[-1]}Z",
                "message": {
                    "role": "assistant",
                    "model": "gpt-5.6-sol",
                    "content": [{"type": "text", "text": f"done {session_id}"}],
                    "stopReason": "stop",
                },
            }
        )
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    root_path = agent_dir / "openclaw.session.jsonl"
    write_session(root_path, "root-session", child_key)
    write_session(archive / "child-session.jsonl", "child-session", grandchild_key)
    write_session(archive / "grandchild-session.jsonl", "grandchild-session")
    (archive / "sessions.json").write_text(
        json.dumps(
            {
                root_key: {"sessionId": "root-session"},
                child_key: {
                    "sessionId": "child-session",
                    "spawnedBy": root_key,
                },
                grandchild_key: {
                    "sessionId": "grandchild-session",
                    "spawnedBy": child_key,
                },
            }
        ),
        encoding="utf-8",
    )

    tree, validation = _openclaw_session_tree(agent_dir, root_path)

    assert [(key, depth) for key, _, depth, _ in tree] == [
        (root_key, 0),
        (child_key, 1),
        (grandchild_key, 2),
    ]
    assert validation["session_tree_accepted_spawn_count"] == 2
    assert validation["session_tree_complete"] is True


def test_openclaw_session_tree_reconciles_persisted_child_status(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    archive = agent_dir / "openclaw.sessions"
    archive.mkdir(parents=True)
    root_key = "agent:main:main"
    child_key = "agent:main:subagent:child"
    root_path = agent_dir / "openclaw.session.jsonl"
    child_path = archive / "child-session.jsonl"
    root_records = [
        {"type": "session", "id": "root-session"},
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:00Z",
            "message": {"role": "user", "content": "delegate"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:01Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "spawn-1",
                        "name": "sessions_spawn",
                        "arguments": {"task": "inspect"},
                    }
                ],
                "stopReason": "toolUse",
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:02Z",
            "message": {
                "role": "toolResult",
                "toolCallId": "spawn-1",
                "content": json.dumps(
                    {
                        "status": "accepted",
                        "childSessionKey": child_key,
                    }
                ),
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:04.500Z",
            "message": {
                "role": "user",
                "content": "The child failed while running false.",
                "provenance": {
                    "kind": "inter_session",
                    "sourceSessionKey": child_key,
                    "sourceTool": "subagent_announce",
                },
            },
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:05Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "stopReason": "stop",
            },
        },
    ]
    child_records = [
        {"type": "session", "id": "child-session"},
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:03Z",
            "message": {"role": "user", "content": "inspect"},
        },
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:04Z",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "id": "exec-1",
                        "name": "exec",
                        "arguments": {"command": "false"},
                    }
                ],
                "stopReason": "toolUse",
            },
        },
    ]
    root_path.write_text(
        "\n".join(json.dumps(record) for record in root_records) + "\n",
        encoding="utf-8",
    )
    child_path.write_text(
        "\n".join(json.dumps(record) for record in child_records) + "\n",
        encoding="utf-8",
    )

    def write_index(status: str) -> None:
        (archive / "sessions.json").write_text(
            json.dumps(
                {
                    root_key: {"sessionId": "root-session"},
                    child_key: {
                        "sessionId": "child-session",
                        "spawnedBy": root_key,
                        "status": status,
                    },
                }
            ),
            encoding="utf-8",
        )

    run = RunSpec(
        run_label="openclaw-gpt55-status",
        harness="openclaw",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="gpt-5.5",
        repetition=1,
        expected_task_count=1,
        run_date="20260729",
    )

    for terminal_status in (
        "cancelled",
        "deleted",
        "error",
        "failed",
        "killed",
        "reset",
        "timeout",
    ):
        write_index(terminal_status)
        _, terminal_validation = _openclaw_session_tree(agent_dir, root_path)
        assert terminal_validation["session_tree_complete"] is True
        assert terminal_validation["session_tree_nonterminal_session_keys"] == []

    write_index("failed")
    failed_tree, failed_validation = _openclaw_session_tree(agent_dir, root_path)
    failed_steps = _openclaw_session_tree_steps(
        failed_tree,
        instruction="delegate",
        run=run,
    )

    assert failed_validation["session_tree_complete"] is True
    assert failed_validation["session_tree_nonterminal_session_keys"] == []
    assert [
        step["message"]
        for step in failed_steps
        if step.get("extra", {}).get("openclaw_event") == "subagent_announce"
    ] == ["The child failed while running false."]

    child_records.append(
        {
            "type": "message",
            "timestamp": "2026-07-29T08:00:04.250Z",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "late final text"}],
                "stopReason": "stop",
            },
        }
    )
    child_path.write_text(
        "\n".join(json.dumps(record) for record in child_records) + "\n",
        encoding="utf-8",
    )
    write_index("running")
    running_tree, running_validation = _openclaw_session_tree(agent_dir, root_path)
    running_steps = _openclaw_session_tree_steps(
        running_tree,
        instruction="delegate",
        run=run,
    )

    assert running_validation["session_tree_complete"] is True
    assert running_validation["session_tree_nonterminal_session_keys"] == []
    assert not any(
        step.get("extra", {}).get("openclaw_event") == "subagent_announce" for step in running_steps
    )

    write_index("failed")
    failed_final_tree, failed_final_validation = _openclaw_session_tree(
        agent_dir,
        root_path,
    )
    failed_final_steps = _openclaw_session_tree_steps(
        failed_final_tree,
        instruction="delegate",
        run=run,
        failed_session_keys=set(failed_final_validation["session_tree_failed_session_keys"]),
    )

    assert [
        step["message"]
        for step in failed_final_steps
        if step.get("extra", {}).get("openclaw_event") == "subagent_announce"
    ] == ["The child failed while running false."]

    child_path.write_text(
        "\n".join(json.dumps(record) for record in child_records[:-1]) + "\n",
        encoding="utf-8",
    )
    write_index("done")
    _, stale_done_validation = _openclaw_session_tree(agent_dir, root_path)

    assert stale_done_validation["session_tree_complete"] is False
    assert stale_done_validation["session_tree_nonterminal_session_keys"] == [child_key]


def test_hermes_session_converts_parallel_tools_to_atif(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    session = {
        "id": "hermes-session-123",
        "model": "gpt-5.6-sol",
        "message_count": 4,
        "tool_call_count": 1,
        "input_tokens": 100,
        "output_tokens": 25,
        "cache_read_tokens": 75,
        "reasoning_tokens": 8,
        "api_call_count": 2,
        "messages": [
            {
                "id": 1,
                "role": "user",
                "content": "do the task",
                "timestamp": 1.0,
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "",
                "timestamp": 2.0,
                "finish_reason": "tool_calls",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path":"/app/input.txt"}',
                        },
                    }
                ],
            },
            {
                "id": 3,
                "role": "tool",
                "content": "input",
                "tool_call_id": "call-1",
                "tool_name": "read_file",
                "timestamp": 3.0,
            },
            {
                "id": 4,
                "role": "assistant",
                "content": "done",
                "timestamp": 4.0,
                "finish_reason": "stop",
            },
        ],
    }
    (agent_dir / "hermes-session.jsonl").write_text(
        json.dumps(session) + "\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="hermes-gpt56-sol",
        harness="hermes",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=1,
        run_date="20260728",
    )
    task = _trajectory_task(tmp_path, "do the task")

    metadata = write_agent_trajectory(task, run, agent_dir)
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert metadata["canonical_model_identity"] is True
    assert metadata["trajectory_validation"]["terminal_event_seen"] is True
    assert metadata["trajectory_validation"]["instruction_matches"] is True
    assert len(trajectory["steps"]) == 3
    assert trajectory["steps"][1]["tool_calls"][0]["function_name"] == "read_file"
    assert trajectory["steps"][1]["observation"]["results"][0]["content"] == "input"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 100
    assert trajectory["final_metrics"]["total_cached_tokens"] == 75
    assert trajectory["final_metrics"]["total_completion_tokens"] == 25


def test_hermes_session_preserves_unicode_line_separator(tmp_path: Path) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    session = {
        "id": "hermes-session-unicode",
        "model": "gpt-5.5",
        "message_count": 2,
        "tool_call_count": 0,
        "messages": [
            {
                "id": 1,
                "role": "user",
                "content": "do the task",
                "timestamp": 1.0,
            },
            {
                "id": 2,
                "role": "assistant",
                "content": "line one\u2028line two",
                "timestamp": 2.0,
                "finish_reason": "stop",
            },
        ],
    }
    (agent_dir / "hermes-session.jsonl").write_text(
        json.dumps(session, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="hermes-gpt55",
        harness="hermes",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="gpt-5.5",
        repetition=1,
        expected_task_count=1,
        run_date="20260728",
    )
    task = _trajectory_task(tmp_path, "do the task")

    metadata = write_agent_trajectory(task, run, agent_dir)
    trajectory = json.loads((agent_dir / "trajectory.json").read_text())

    assert metadata["trajectory_status"] == "real"
    assert metadata["canonical_model_identity"] is True
    assert trajectory["steps"][-1]["message"] == "line one\u2028line two"


def test_hermes_child_session_model_mismatch_invalidates_identity(
    tmp_path: Path,
) -> None:
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    parent = {
        "id": "parent",
        "model": "gpt-5.6-sol",
        "message_count": 2,
        "tool_call_count": 0,
        "messages": [
            {"id": 1, "role": "user", "content": "do the task", "timestamp": 1.0},
            {
                "id": 2,
                "role": "assistant",
                "content": "done",
                "model": "gpt-5.6-sol",
                "finish_reason": "stop",
                "timestamp": 2.0,
            },
        ],
    }
    child = {
        "id": "child",
        "model": "gpt-5.5",
        "message_count": 2,
        "tool_call_count": 0,
        "messages": [
            {"id": 3, "role": "user", "content": "inspect", "timestamp": 1.0},
            {
                "id": 4,
                "role": "assistant",
                "content": "child done",
                "model": "gpt-5.5",
                "finish_reason": "stop",
                "timestamp": 2.0,
            },
        ],
    }
    (agent_dir / "hermes-session.jsonl").write_text(
        json.dumps(child) + "\n" + json.dumps(parent) + "\n",
        encoding="utf-8",
    )
    run = RunSpec(
        run_label="hermes-gpt56-sol",
        harness="hermes",
        harness_version="test",
        model_slug="gpt56-sol",
        model_id="gpt-5.6-sol",
        provider="openai",
        proxy_model_name="gpt-5.6-sol",
        repetition=1,
        expected_task_count=1,
        run_date="20260728",
    )

    metadata = write_agent_trajectory(
        _trajectory_task(tmp_path, "do the task"),
        run,
        agent_dir,
    )

    assert metadata["trajectory_status"] == "real"
    assert metadata["runtime_model_name"] == "gpt-5.6-sol"
    assert metadata["canonical_model_identity"] is False
    assert metadata["trajectory_validation"]["observed_models"] == [
        "gpt-5.5",
        "gpt-5.6-sol",
    ]


def _trajectory_task(tmp_path: Path, instruction: str) -> TaskSpec:
    task_dir = tmp_path / "task"
    task_dir.mkdir(exist_ok=True)
    return TaskSpec(
        name="task",
        title="task",
        path=task_dir,
        instruction=instruction,
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


def test_all_or_nothing_reward_contract_rejects_partial_reward(
    tmp_path: Path,
) -> None:
    rubrics_dir = tmp_path / "tests"
    rubrics_dir.mkdir()
    (rubrics_dir / "rubrics.json").write_text(
        json.dumps(
            {
                "reward": {
                    "scoring": "all_or_nothing",
                    "pass_reward": 1.0,
                    "fail_reward": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )

    contract = load_reward_contract(tmp_path)

    assert contract == {
        "source": "tests/rubrics.json",
        "scoring": "all_or_nothing",
        "pass_reward": 1.0,
        "fail_reward": 0.0,
    }
    validate_reward_contract(contract, {"reward": 0})
    validate_reward_contract(contract, {"reward": 1})
    with pytest.raises(
        VerifierRewardContractError,
        match="all_or_nothing contract",
    ):
        validate_reward_contract(contract, {"reward": 0.5})


def test_all_or_nothing_reward_contract_requires_numeric_endpoints(
    tmp_path: Path,
) -> None:
    rubrics_dir = tmp_path / "tests"
    rubrics_dir.mkdir()
    (rubrics_dir / "rubrics.json").write_text(
        json.dumps({"reward": {"scoring": "all_or_nothing"}}),
        encoding="utf-8",
    )

    with pytest.raises(
        VerifierRewardContractError,
        match="requires numeric pass_reward",
    ):
        load_reward_contract(tmp_path)


def test_all_or_nothing_reward_contract_preserves_exact_integer_equality() -> None:
    contract = {
        "scoring": "all_or_nothing",
        "pass_reward": 9_007_199_254_740_992,
        "fail_reward": 0,
    }

    with pytest.raises(VerifierRewardContractError):
        validate_reward_contract(
            contract,
            {"reward": 9_007_199_254_740_993},
        )


def test_malformed_reward_contract_is_classified_as_contract_error(
    tmp_path: Path,
) -> None:
    rubrics_dir = tmp_path / "tests"
    rubrics_dir.mkdir()
    (rubrics_dir / "rubrics.json").write_text("{", encoding="utf-8")

    with pytest.raises(
        VerifierRewardContractError,
        match="Unable to read reward contract",
    ):
        load_reward_contract(tmp_path)


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
