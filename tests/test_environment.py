from pathlib import Path

import pytest

from clawbench.environment import (
    _evaluate_execution_result,
    run_execution_check,
    verify_completion,
)
from clawbench.schemas import (
    CompletionSpec,
    CronState,
    ExecutionCheck,
    FileState,
    GatewayAssertion,
    MemoryState,
    SessionState,
    ToolCall,
    Transcript,
    TranscriptMessage,
)


class MemoryFallbackClient:
    async def _rpc(self, method: str, params=None):  # noqa: ANN001
        if method == "memory.search":
            raise RuntimeError("unknown method: memory.search")
        raise AssertionError(f"Unexpected RPC: {method} {params}")

    async def get_agent_file(self, agent_id: str, name: str):  # noqa: ARG002
        if name == "MEMORY.md":
            return {
                "file": {
                    "content": "beta rollout regions: us, eu; retry budget: 3\n",
                }
            }
        return {"file": {"content": ""}}


class CompletionClient:
    async def _rpc(self, method: str, params=None):  # noqa: ANN001
        if method == "sessions.resolve":
            return {"payload": {"model": "anthropic/claude-sonnet-4-6"}}
        if method == "cron.list":
            return {"payload": {"jobs": [{"description": "nightly cleanup"}]}}
        if method == "tools.inventory":
            return {
                "payload": {
                    "groups": [
                        {
                            "tools": [
                                {
                                    "id": "browser",
                                    "status": "available",
                                }
                            ]
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected RPC: {method} {params}")


def test_text_assertions_normalize_line_endings_but_not_content(tmp_path: Path):
    spec = ExecutionCheck(
        name="text-assertions",
        command="unused",
        stdout_contains=["out-first\nout-second"],
        stdout_not_contains=["forbidden\ntext"],
        stderr_contains=["err-first\nerr-second"],
        stdout_matches=r"^out-first\nout-second$",
        stderr_matches=r"^err-first\nerr-second$",
    )

    passed, _ = _evaluate_execution_result(
        spec,
        tmp_path,
        {},
        0,
        "out-first\r\nout-second",
        "err-first\r\nerr-second",
    )
    excluded, _ = _evaluate_execution_result(
        ExecutionCheck(
            name="excluded-text",
            command="unused",
            stdout_not_contains=["out-first\nout-second"],
        ),
        tmp_path,
        {},
        0,
        "out-first\r\nout-second",
        "",
    )
    different, _ = _evaluate_execution_result(
        ExecutionCheck(
            name="different-text",
            command="unused",
            stdout_matches=r"^out-first\nwrong$",
        ),
        tmp_path,
        {},
        0,
        "out-first\r\nout-second",
        "",
    )

    assert passed is True
    assert excluded is False
    assert different is False


def test_expected_stdout_normalizes_line_endings_but_not_content(tmp_path: Path):
    spec = ExecutionCheck(
        name="stdout",
        command="unused",
        expected_stdout="first\nsecond\nthird",
    )

    passed, _ = _evaluate_execution_result(
        spec, tmp_path, {}, 0, "first\r\nsecond\rthird", ""
    )
    different, _ = _evaluate_execution_result(
        spec, tmp_path, {}, 0, "first\r\nsecond\rwrong", ""
    )

    assert passed is True
    assert different is False


def test_expected_stdout_file_normalizes_line_endings_but_not_content(tmp_path: Path):
    (tmp_path / "expected.txt").write_bytes(b"first\nsecond\nthird\n")
    spec = ExecutionCheck(
        name="stdout-file",
        command="unused",
        expected_stdout_file="expected.txt",
    )

    passed, _ = _evaluate_execution_result(
        spec, tmp_path, {}, 0, "first\rsecond\r\nthird", ""
    )
    different, _ = _evaluate_execution_result(
        spec, tmp_path, {}, 0, "first\rsecond\r\nwrong", ""
    )

    assert passed is True
    assert different is False


@pytest.mark.asyncio
async def test_memory_completion_falls_back_to_agent_memory_files(tmp_path: Path):
    completion = CompletionSpec(
        memory=[
            MemoryState(
                key_pattern="beta rollout regions",
                value_contains=["us", "eu", "3"],
            )
        ]
    )

    result = await verify_completion(
        completion,
        workspace=tmp_path,
        client=MemoryFallbackClient(),  # type: ignore[arg-type]
        session_key="session-test",
        agent_id="agent-test",
        runtime_values={},
    )

    assert result.score == 1.0


@pytest.mark.asyncio
async def test_verify_completion_scores_mixed_successful_assertions(tmp_path: Path):
    report = tmp_path / "report.txt"
    report.write_text("status: green\nowner: benchmark\n", encoding="utf-8")
    completion = CompletionSpec(
        files=[
            FileState(
                path="report.txt",
                content_contains=["green"],
                content_not_contains=["red"],
                content_matches=r"owner:\s+benchmark",
                min_size_bytes=10,
            )
        ],
        session=SessionState(model_should_be="claude-sonnet"),
        cron=[CronState(description_contains="cleanup")],
        gateway_assertions=[
            GatewayAssertion(
                method="tools.inventory",
                assert_path="$.groups[0].tools[0].id",
                assert_equals="browser",
            ),
            GatewayAssertion(
                method="tools.inventory",
                assert_path="$.groups[0].tools[0].status",
                assert_contains="avail",
            ),
        ],
    )

    result = await verify_completion(
        completion,
        workspace=tmp_path,
        client=CompletionClient(),  # type: ignore[arg-type]
        session_key="session-test",
        runtime_values={},
    )

    assert result.total_assertions == 5
    assert result.passed_assertions == 5
    assert result.failed_assertions == []
    assert result.score == 1.0


@pytest.mark.asyncio
async def test_file_completion_rejects_paths_outside_workspace(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    completion = CompletionSpec(files=[FileState(path="../outside.txt")])

    result = await verify_completion(
        completion,
        workspace=tmp_path,
        client=MemoryFallbackClient(),  # type: ignore[arg-type]
        session_key="session-test",
        runtime_values={},
    )

    assert result.score == 0.0
    assert "escapes workspace" in result.failed_assertions[0]


@pytest.mark.asyncio
async def test_execution_check_supports_cwd_env_and_expected_json_file(tmp_path: Path):
    expected = tmp_path / "expected.json"
    expected.write_text('{"status": "ok"}', encoding="utf-8")
    workdir = tmp_path / "subdir"
    workdir.mkdir()

    result = await run_execution_check(
        ExecutionCheck(
            name="json-check",
            command='python -c "import json, os; print(json.dumps({\'status\': os.environ[\'CHECK_STATUS\']}))"',
            cwd="subdir",
            env={"CHECK_STATUS": "ok"},
            expected_json_file="expected.json",
        ),
        workspace=tmp_path,
        runtime_values={},
    )

    assert result.passed is True
    assert result.reason == "OK"


@pytest.mark.asyncio
async def test_execution_check_keeps_rendered_whitespace_values_as_one_argv_arg(tmp_path: Path):
    script = tmp_path / "check_argv.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    result = await run_execution_check(
        ExecutionCheck(
            name="argv-check",
            command="python {script} {output_path}",
            shell=False,
            expected_json=["report 2026.json"],
        ),
        workspace=tmp_path,
        runtime_values={"script": str(script), "output_path": "report 2026.json"},
    )

    assert result.passed is True
    assert result.reason == "OK"


@pytest.mark.asyncio
async def test_execution_check_rejects_cwd_outside_workspace(tmp_path: Path):
    result = await run_execution_check(
        ExecutionCheck(
            name="unsafe-cwd",
            command="true",
            cwd="../outside",
        ),
        workspace=tmp_path,
        runtime_values={},
    )

    assert result.passed is False
    assert "escapes workspace" in result.reason


@pytest.mark.asyncio
async def test_execution_check_rejects_expected_file_outside_workspace(tmp_path: Path):
    result = await run_execution_check(
        ExecutionCheck(
            name="unsafe-expected",
            command="printf secret",
            expected_stdout_file="../outside.txt",
        ),
        workspace=tmp_path,
        runtime_values={},
    )

    assert result.passed is False
    assert "escapes workspace" in result.reason


@pytest.mark.asyncio
async def test_memory_completion_falls_back_to_transcript_when_memory_rpc_is_unavailable(tmp_path: Path):
    completion = CompletionSpec(
        memory=[
            MemoryState(
                key_pattern="beta rollout regions",
                value_contains=["us", "eu", "3"],
            )
        ]
    )
    transcript = Transcript(
        messages=[
            TranscriptMessage(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="write",
                        family="edit",
                        input={
                            "path": "memory/notes.md",
                            "content": "beta rollout regions: us, eu; retry budget: 3\n",
                        },
                        success=True,
                    )
                ],
            )
        ]
    )

    result = await verify_completion(
        completion,
        workspace=tmp_path,
        client=MemoryFallbackClient(),  # type: ignore[arg-type]
        session_key="session-test",
        agent_id="agent-test",
        runtime_values={},
        transcript=transcript,
    )

    assert result.score == 1.0
