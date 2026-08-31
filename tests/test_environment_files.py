from pathlib import Path

import pytest

from clawbench.environment_files import (
    evaluate_execution_result,
    run_execution_check,
    verify_file_state,
)
from clawbench.schemas import ExecutionCheck, FileState


def test_expected_stdout_normalizes_line_endings_but_not_content(tmp_path: Path):
    spec = ExecutionCheck(
        name="stdout",
        command="unused",
        expected_stdout="first\nsecond\nthird",
    )

    passed, _ = evaluate_execution_result(
        spec, tmp_path, {}, 0, "first\r\nsecond\rthird", ""
    )
    different, _ = evaluate_execution_result(
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

    passed, _ = evaluate_execution_result(
        spec, tmp_path, {}, 0, "first\rsecond\r\nthird", ""
    )
    different, _ = evaluate_execution_result(
        spec, tmp_path, {}, 0, "first\rsecond\r\nwrong", ""
    )

    assert passed is True
    assert different is False


def test_verify_file_state_rejects_paths_outside_workspace(tmp_path: Path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")

    ok, reason = verify_file_state(
        FileState(path="../outside.txt"),
        workspace=tmp_path,
        runtime_values={},
    )

    assert ok is False
    assert "escapes workspace" in reason


@pytest.mark.asyncio
async def test_execution_check_supports_cwd_env_and_expected_json_file(tmp_path: Path):
    expected = tmp_path / "expected.json"
    expected.write_text('{"status": "ok"}', encoding="utf-8")
    workdir = tmp_path / "subdir"
    workdir.mkdir()

    result = await run_execution_check(
        ExecutionCheck(
            name="json-check",
            command=(
                "python -c \"import json, os; "
                "print(json.dumps({'status': os.environ['CHECK_STATUS']}))\""
            ),
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
async def test_execution_check_rejects_expected_stdout_file_outside_workspace(
    tmp_path: Path,
):
    result = await run_execution_check(
        ExecutionCheck(
            name="unsafe-expected-stdout",
            command="printf secret",
            expected_stdout_file="../outside.txt",
        ),
        workspace=tmp_path,
        runtime_values={},
    )

    assert result.passed is False
    assert "escapes workspace" in result.reason


@pytest.mark.asyncio
async def test_execution_check_rejects_expected_json_file_outside_workspace(
    tmp_path: Path,
):
    result = await run_execution_check(
        ExecutionCheck(
            name="unsafe-expected-json",
            command="printf '{}'",
            expected_json_file="../outside.json",
        ),
        workspace=tmp_path,
        runtime_values={},
    )

    assert result.passed is False
    assert "escapes workspace" in result.reason
