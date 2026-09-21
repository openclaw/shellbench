from pathlib import Path

import pytest

from clawbench.environment import _memory_visible_in_transcript, _verify_memory
from clawbench.environment_files import memory_visible_in_transcript, verify_memory_fallback
from clawbench.schemas import MemoryState, ToolCall, Transcript, TranscriptMessage
from clawbench.trajectory import classify_tool_call, evaluate_trajectory
from clawbench.schemas import TrajectoryExpectations


def memory_trace(
    name: str = "memory_store", success: bool | None = True, error: str = ""
) -> Transcript:
    return Transcript(
        messages=[
            TranscriptMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name=name,
                        family="memory",
                        success=success,
                        error=error,
                        input={"text": "Beta rollout regions: US, EU. Retry budget: 3."},
                        output="Stored memory",
                    )
                ],
            )
        ]
    )


@pytest.mark.parametrize("matcher", [memory_visible_in_transcript, _memory_visible_in_transcript])
def test_real_regex_memory_pattern_matches_acknowledged_write(matcher):
    spec = MemoryState(key_pattern="(?i)beta.*region|region.*beta", value_contains=["us", "eu"])
    assert matcher(spec, memory_trace())
    assert not matcher(spec, memory_trace(success=False, error="write rejected"))
    assert not matcher(spec, memory_trace(success=None))
    assert not matcher(spec, memory_trace(name="memory_search"))
    assert not matcher(spec, memory_trace(name="memory_recall"))
    assert not matcher(spec, memory_trace(name="memory_delete"))


def test_workspace_memory_regex_and_missing_values(tmp_path: Path):
    (tmp_path / "MEMORY.md").write_text("Beta rollout regions: US, EU. Retry budget: 3.")
    spec = MemoryState(key_pattern="(?i)beta.*region|region.*beta", value_contains=["us", "eu"])
    assert verify_memory_fallback(spec, tmp_path)[0]
    assert not verify_memory_fallback(
        spec.model_copy(update={"value_contains": ["APAC"]}), tmp_path
    )[0]
    assert not verify_memory_fallback(spec.model_copy(update={"exists": False}), tmp_path)[0]


@pytest.mark.asyncio
async def test_legacy_gateway_fallback_uses_regex():
    class Client:
        async def _rpc(self, method, params):
            raise RuntimeError("unknown method: memory.search")

        async def get_agent_file(self, agent, filename):
            return {
                "file": {
                    "content": "Beta rollout regions: US, EU" if filename == "MEMORY.md" else ""
                }
            }

    result = await _verify_memory(
        MemoryState(key_pattern="(?i)beta.*region|region.*beta", value_contains=["us", "eu"]),
        Client(),  # type: ignore[arg-type]  # Only the RPC/file-read surface is exercised.
        "test-session",
        agent_id="test-agent",
    )
    assert result[0]


def test_native_spawn_counts_but_stop_and_status_do_not_count_as_delegation():
    assert classify_tool_call(ToolCall(name="sessions_spawn")) == ("delegate", False)
    for action in ("kill", "stop", "cancel", "list", "status", "log"):
        call = ToolCall(name="subagents", input={"action": action}, success=True)
        assert classify_tool_call(call)[0] != "delegate"
        trajectory = evaluate_trajectory(
            Transcript(messages=[TranscriptMessage(role="assistant", tool_calls=[call])]),
            TrajectoryExpectations(required_families=["delegate"]),
        )
        assert trajectory.required_families_missing == ["delegate"]


@pytest.mark.parametrize("success", [None, False, True])
def test_unknown_or_failed_spawn_does_not_earn_successful_delegation_credit(success):
    result = evaluate_trajectory(
        Transcript(
            messages=[
                TranscriptMessage(
                    role="assistant",
                    tool_calls=[ToolCall(name="sessions_spawn", success=success)],
                )
            ]
        ),
        TrajectoryExpectations(min_successful_delegations=1),
    )
    assert (result.tool_fit_score > 0) is (success is True)


@pytest.mark.parametrize("module_name", ["clawbench.environment", "clawbench.environment_files"])
async def test_verifier_process_start_timeout_is_a_failed_check(module_name, tmp_path, monkeypatch):
    import importlib

    from clawbench.schemas import ExecutionCheck

    module = importlib.import_module(module_name)

    async def timeout(*args, **kwargs):
        raise TimeoutError("process startup timed out")

    monkeypatch.setattr(module.asyncio, "create_subprocess_exec", timeout)
    result = await module.run_execution_check(
        ExecutionCheck(name="startup", command="python3 --version", shell=False, timeout_seconds=1),
        workspace=tmp_path,
        runtime_values={},
    )
    assert result.passed is False
    assert result.exit_code == -1
    assert result.reason == "Timed out after 1s"
