from clawbench.schemas import TaskRunResult, ToolCall, TrajectoryResult, Transcript, TranscriptMessage
from scripts.violation_time_decomposition import get_first_violation_turn


def _run(tool_call: ToolCall, forbidden_violations: list[str] | None = None) -> TaskRunResult:
    return TaskRunResult(
        task_id="t1-demo",
        run_index=0,
        trajectory_result=TrajectoryResult(
            forbidden_violations=forbidden_violations or [],
        ),
        transcript=Transcript(
            messages=[
                TranscriptMessage(
                    role="assistant",
                    tool_calls=[tool_call],
                )
            ]
        ),
    )


def test_failed_tool_call_is_not_counted_as_violation_without_trajectory_violation():
    run = _run(ToolCall(name="exec", input={"command": "pytest -q"}, success=False))

    assert get_first_violation_turn(run) == (2, False)


def test_dangerous_command_violation_is_localized_to_turn():
    run = _run(
        ToolCall(name="exec", input={"command": "sudo rm -rf /tmp/demo"}, success=True),
        forbidden_violations=["Dangerous shell command: sudo rm -rf /tmp/demo"],
    )

    assert get_first_violation_turn(run) == (1, True)
