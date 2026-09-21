from pathlib import Path

import pytest

from clawbench.client import GatewayConfig
from clawbench.harness import BenchmarkHarness
from clawbench.schemas import CompletionResult, JudgeResult, TaskRunResult
from clawbench.tasks import load_all_tasks


class FakeGatewayClient:
    def __init__(self) -> None:
        self.create_agent_calls: list[tuple[str, str]] = []

    async def create_agent(self, *, name: str, workspace: str) -> str:
        self.create_agent_calls.append((name, workspace))
        return "agent-test-123"


@pytest.mark.asyncio
async def test_run_agent_uses_staged_run_workspace(tmp_path: Path):
    task = next(task for task in load_all_tasks() if task.id == "t1-bugfix-discount")
    harness = BenchmarkHarness(
        gateway_config=GatewayConfig(), model="test-model", randomize_order=False
    )
    workspace = tmp_path / "run-workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    client = FakeGatewayClient()

    agent_id = await harness._create_run_agent(
        client,  # type: ignore[arg-type]
        task=task,
        workspace=workspace,
        run_index=2,
    )

    assert agent_id == "agent-test-123"
    assert client.create_agent_calls == [(client.create_agent_calls[0][0], str(workspace))]
    assert task.id in client.create_agent_calls[0][0]


@pytest.mark.asyncio
async def test_run_workspace_is_sibling_of_openclaw_workspace_and_cleaned_up(
    tmp_path: Path,
    monkeypatch,
):
    task = next(task for task in load_all_tasks() if task.id == "t1-bugfix-discount")
    state_dir = tmp_path / "state"
    created_workspaces: list[Path] = []

    class DoneSimulator:
        is_done = True

        def __init__(self, *args, **kwargs) -> None:
            pass

    class RunGatewayClient:
        def __init__(self, config) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def create_agent(self, *, name: str, workspace: str) -> str:
            created_workspaces.append(Path(workspace))
            return "agent-test-123"

        async def create_session(self, *, model: str, agent_id: str, label: str) -> str:
            return "session-test-123"

        async def subscribe(self, session_key: str) -> None:
            pass

        async def delete_session(self, session_key: str) -> None:
            pass

        async def delete_agent(self, agent_id: str, *, delete_files: bool) -> None:
            pass

    def fake_setup_workspace(self, current_task, workspace: Path) -> None:
        workspace.joinpath("marker.txt").write_text("created", encoding="utf-8")

    async def fake_start_background_services(services, *, workspace, repo_root, runtime_values):
        return [], runtime_values

    async def fake_score_task_run(**kwargs):
        return TaskRunResult(
            task_id=task.id,
            tier=task.tier.value,
            family=task.family.value,
            run_index=0,
            run_score=1.0,
        )

    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state_dir))
    monkeypatch.setenv("CLAWBENCH_RUN_CACHE_DIR", "")
    monkeypatch.delenv("CLAWBENCH_KEEP_WORKSPACES", raising=False)
    monkeypatch.setattr(BenchmarkHarness, "_setup_workspace", fake_setup_workspace)
    monkeypatch.setattr("clawbench.harness.GatewayClient", RunGatewayClient)
    monkeypatch.setattr("clawbench.harness.UserSimulator", DoneSimulator)
    monkeypatch.setattr(
        "clawbench.harness.start_background_services", fake_start_background_services
    )
    monkeypatch.setattr("clawbench.harness.score_task_run", fake_score_task_run)

    harness = BenchmarkHarness(
        gateway_config=GatewayConfig(), model="test-model", randomize_order=False
    )

    result = await harness._run_single(task, run_index=0)

    assert result.run_score == 1.0
    assert len(created_workspaces) == 1
    workspace = created_workspaces[0]
    assert workspace.parent == state_dir / "workspace-clawbench" / task.id
    assert not workspace.is_relative_to(state_dir / "workspace")
    assert not workspace.exists()


@pytest.mark.asyncio
async def test_prepare_run_hook_executes_before_each_run(monkeypatch):
    task = next(task for task in load_all_tasks() if task.id == "t1-bugfix-discount")
    calls: list[tuple[str, int]] = []

    async def prepare_run(current_task, run_index: int) -> None:
        calls.append((current_task.id, run_index))

    async def fake_run_single(self, current_task, run_index: int):
        from clawbench.schemas import TaskRunResult

        return TaskRunResult(
            task_id=current_task.id,
            tier=current_task.tier.value,
            family=current_task.family.value,
            run_index=run_index,
            run_score=1.0,
        )

    monkeypatch.setattr("clawbench.harness.load_all_tasks", lambda **_: [task])
    monkeypatch.setattr(BenchmarkHarness, "_run_single", fake_run_single)

    harness = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test-model",
        task_ids=[task.id],
        runs_per_task=2,
        randomize_order=False,
        prepare_run=prepare_run,
    )

    await harness.run()

    assert calls == [(task.id, 0), (task.id, 1)]


def test_aggregate_reports_advisory_judge_metrics():
    task = next(
        task for task in load_all_tasks() if task.id == "t5-hallucination-resistant-evidence"
    )
    harness = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test-model",
        judge_model="judge-model",
        task_ids=[task.id],
        randomize_order=False,
    )
    runs = [
        TaskRunResult(
            task_id=task.id,
            tier=task.tier.value,
            family=task.family.value,
            run_index=0,
            run_score=0.9,
            completion_result=CompletionResult(total_assertions=1, passed_assertions=1, score=1.0),
            judge_result=JudgeResult(
                enabled=True, model="judge-model", score=0.9, confidence=0.7, passed=True
            ),
        ),
        TaskRunResult(
            task_id=task.id,
            tier=task.tier.value,
            family=task.family.value,
            run_index=1,
            run_score=0.6,
            completion_result=CompletionResult(total_assertions=1, passed_assertions=1, score=1.0),
            judge_result=JudgeResult(
                enabled=True, model="judge-model", score=0.5, confidence=0.9, passed=False
            ),
        ),
    ]

    result = harness._aggregate([task], {task.id: runs})
    task_result = result.task_results[0]

    assert result.judge_model == "judge-model"
    assert result.overall_judge_score == pytest.approx(0.7)
    assert result.overall_judge_confidence == pytest.approx(0.8)
    assert result.overall_judge_pass_rate == pytest.approx(0.5)
    assert result.judge_task_coverage == 1.0
    assert task_result.mean_judge_score == pytest.approx(0.7)
    assert task_result.mean_judge_confidence == pytest.approx(0.8)
    assert task_result.judge_pass_rate == pytest.approx(0.5)
    assert task_result.judged_runs == 2


def test_compose_result_from_task_stats_supports_parallel_environment_metadata():
    task = next(task for task in load_all_tasks() if task.id == "t1-bugfix-discount")
    harness = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test-model",
        task_ids=[task.id],
        randomize_order=False,
        print_report=False,
        quiet=True,
    )
    runs = [
        TaskRunResult(
            task_id=task.id,
            tier=task.tier.value,
            family=task.family.value,
            run_index=0,
            run_score=0.9,
            completion_result=CompletionResult(total_assertions=1, passed_assertions=1, score=1.0),
        ),
        TaskRunResult(
            task_id=task.id,
            tier=task.tier.value,
            family=task.family.value,
            run_index=1,
            run_score=0.7,
            completion_result=CompletionResult(total_assertions=1, passed_assertions=1, score=1.0),
        ),
    ]

    base_result = harness._aggregate([task], {task.id: runs})
    merged_result = harness.compose_result_from_task_stats(
        base_result.task_results,
        tasks=[task],
        environment_extra={
            "parallel_lanes": 2,
            "requested_parallel_lanes": 3,
            "browser_tasks_serialized": False,
        },
        print_report=False,
    )

    assert merged_result.overall_score == pytest.approx(base_result.overall_score)
    assert merged_result.overall_completion == pytest.approx(base_result.overall_completion)
    assert merged_result.environment["parallel_lanes"] == 2
    assert merged_result.environment["requested_parallel_lanes"] == 3
    assert merged_result.environment["browser_tasks_serialized"] is False


def test_run_cache_path_includes_scoring_inputs(tmp_path: Path):
    task = next(task for task in load_all_tasks() if task.id == "t1-bugfix-discount")
    base = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test/model",
        task_ids=[task.id],
        prompt_variant="clear",
        judge_model="judge-a",
        randomize_order=False,
    )
    same = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test/model",
        task_ids=[task.id],
        prompt_variant="clear",
        judge_model="judge-a",
        randomize_order=False,
    )
    different_judge = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test/model",
        task_ids=[task.id],
        prompt_variant="clear",
        judge_model="judge-b",
        randomize_order=False,
    )
    different_judge_gate = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test/model",
        task_ids=[task.id],
        prompt_variant="clear",
        judge_model="judge-a",
        judge_affects_score=True,
        randomize_order=False,
    )
    different_prompt = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test/model",
        task_ids=[task.id],
        prompt_variant="ambiguous",
        judge_model="judge-a",
        randomize_order=False,
    )

    base_path = base._run_cache_path(tmp_path, task, 0)

    assert "v4-" in str(base_path)
    assert base_path == same._run_cache_path(tmp_path, task, 0)
    assert base_path != different_judge._run_cache_path(tmp_path, task, 0)
    assert base_path != different_judge_gate._run_cache_path(tmp_path, task, 0)
    assert base_path != different_prompt._run_cache_path(tmp_path, task, 0)


@pytest.mark.asyncio
async def test_run_records_adapter_surface(monkeypatch):
    task = next(task for task in load_all_tasks() if task.id == "t1-bugfix-discount")

    async def fake_run_single(self, current_task, run_index: int):
        return TaskRunResult(
            task_id=current_task.id,
            tier=current_task.tier.value,
            family=current_task.family.value,
            run_index=run_index,
            run_score=1.0,
            completion_result=CompletionResult(total_assertions=1, passed_assertions=1, score=1.0),
        )

    monkeypatch.setattr("clawbench.harness.load_all_tasks", lambda **_: [task])
    monkeypatch.setattr(BenchmarkHarness, "_run_single", fake_run_single)

    harness = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test-model",
        adapter="openclaw",
        runs_per_task=1,
        randomize_order=False,
        print_report=False,
        quiet=True,
    )

    result = await harness.run()

    assert result.environment["adapter"] == "openclaw"
    assert "hermes" in result.environment["known_adapters"]


@pytest.mark.asyncio
async def test_run_rejects_registered_but_unwired_adapter(monkeypatch):
    task = next(task for task in load_all_tasks() if task.id == "t1-bugfix-discount")
    monkeypatch.setattr("clawbench.harness.load_all_tasks", lambda **_: [task])

    harness = BenchmarkHarness(
        gateway_config=GatewayConfig(),
        model="test-model",
        adapter="hermes",
        runs_per_task=1,
        randomize_order=False,
        print_report=False,
        quiet=True,
    )

    with pytest.raises(ValueError, match="not yet wired"):
        await harness.run()
