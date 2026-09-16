import asyncio
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from clawbench.environment import run_execution_check as run_gateway_execution_check
from clawbench.environment_files import run_execution_check as run_file_execution_check
from clawbench.schemas import ExecutionCheck


RUNNERS = [
    pytest.param(run_gateway_execution_check, id="gateway"),
    pytest.param(run_file_execution_check, id="files"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("run_execution_check", RUNNERS)
async def test_shell_execution_check_quotes_unquoted_runtime_values(
    tmp_path: Path,
    run_execution_check,
):
    output = tmp_path / "report 2026.json"
    output.write_text("success\n", encoding="utf-8")

    result = await run_execution_check(
        ExecutionCheck(
            name="shell-path-check",
            command="cat {output_path}",
            stdout_contains=["success"],
        ),
        workspace=tmp_path,
        runtime_values={"output_path": output.name},
    )

    assert result.passed is True
    assert result.reason == "OK"


@pytest.mark.asyncio
@pytest.mark.parametrize("run_execution_check", RUNNERS)
async def test_shell_execution_check_treats_metacharacters_as_data(
    tmp_path: Path,
    run_execution_check,
):
    marker = tmp_path / "injected_marker"

    result = await run_execution_check(
        ExecutionCheck(
            name="shell-metachar-check",
            command="printf '%s' {title}",
            expected_stdout="safe; touch injected_marker",
        ),
        workspace=tmp_path,
        runtime_values={"title": "safe; touch injected_marker"},
    )

    assert result.passed is True
    assert marker.exists() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("run_execution_check", RUNNERS)
async def test_shell_execution_check_raw_placeholder_allows_shell_fragments(
    tmp_path: Path,
    run_execution_check,
):
    script = tmp_path / "check_argv.py"
    script.write_text(
        "import json, sys\n"
        "print(json.dumps(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    result = await run_execution_check(
        ExecutionCheck(
            name="raw-shell-fragment-check",
            command="{python_exe} {script} {extra_args:raw}",
            expected_json=["one", "two"],
        ),
        workspace=tmp_path,
        runtime_values={
            "python_exe": sys.executable,
            "script": str(script),
            "extra_args": "one two",
        },
    )

    assert result.passed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("run_execution_check", RUNNERS)
async def test_shell_execution_check_preserves_double_quoted_placeholders(
    tmp_path: Path,
    run_execution_check,
):
    output = tmp_path / "report $HOME.json"
    output.write_text("success\n", encoding="utf-8")

    result = await run_execution_check(
        ExecutionCheck(
            name="double-quoted-shell-path-check",
            command='cat "{output_path}"',
            stdout_contains=["success"],
        ),
        workspace=tmp_path,
        runtime_values={"output_path": output.name},
    )

    assert result.passed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("run_execution_check", RUNNERS)
async def test_shell_execution_check_preserves_single_quoted_placeholders(
    tmp_path: Path,
    run_execution_check,
):
    output = tmp_path / "report '26.json"
    output.write_text("success\n", encoding="utf-8")

    result = await run_execution_check(
        ExecutionCheck(
            name="single-quoted-shell-path-check",
            command="cat '{output_path}'",
            stdout_contains=["success"],
        ),
        workspace=tmp_path,
        runtime_values={"output_path": output.name},
    )

    assert result.passed is True


def _write_pipe_holder(path: Path, *, leader_exits: bool) -> None:
    path.write_text(
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n"
        "\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(120)']\n"
        ")\n"
        "Path(sys.argv[1]).write_text(str(child.pid), encoding='utf-8')\n"
        + ("" if leader_exits else "time.sleep(120)\n"),
        encoding="utf-8",
    )


def _force_kill_pid(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            check=False,
            capture_output=True,
        )
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass


@pytest.mark.asyncio
@pytest.mark.parametrize("run_execution_check", RUNNERS)
@pytest.mark.parametrize("leader_exits", [False, True])
async def test_execution_check_timeout_reaps_shell_child_process_group(
    tmp_path: Path,
    run_execution_check,
    leader_exits: bool,
):
    _write_pipe_holder(tmp_path / "hold_pipe.py", leader_exits=leader_exits)
    pid_file = tmp_path / "child.pid"
    try:
        try:
            result = await asyncio.wait_for(
                run_execution_check(
                    ExecutionCheck(
                        name="timeout-reap",
                        command="python hold_pipe.py child.pid",
                        timeout_seconds=1,
                    ),
                    workspace=tmp_path,
                    runtime_values={},
                ),
                timeout=8,
            )
        except TimeoutError:
            pytest.fail(
                "run_execution_check hung after timeout_seconds; "
                "child still held stdout/stderr"
            )

        assert result.passed is False
        assert result.exit_code == -1
        assert result.reason == "Timed out after 1s"
        assert pid_file.exists()
        child_pid = int(pid_file.read_text(encoding="utf-8").strip())
        if sys.platform != "win32":
            child_state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(child_pid)],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            assert not child_state or child_state.startswith("Z"), child_state
    finally:
        if pid_file.exists():
            _force_kill_pid(int(pid_file.read_text(encoding="utf-8").strip()))


@pytest.mark.asyncio
@pytest.mark.parametrize("wait_hangs_after_kill", [False, True])
async def test_windows_tree_cleanup_is_async_and_bounded(monkeypatch, wait_hangs_after_kill):
    from clawbench import environment_files

    class Process:
        pid = 123
        returncode = None
        killed = False

        def kill(self):
            self.killed = True

        async def wait(self):
            if self.killed and not wait_hangs_after_kill:
                self.returncode = 1
                return 1
            await asyncio.Event().wait()

    killer = Process()
    original = Process()

    async def spawn(*args, **kwargs):
        assert args == ("taskkill", "/F", "/T", "/PID", "123")
        return killer

    monkeypatch.setattr(environment_files, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(environment_files.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(environment_files, "EXECUTION_CLEANUP_TIMEOUT_SECONDS", 0.02)
    cleanup = asyncio.create_task(environment_files._kill_execution_pgroup(original))
    await asyncio.sleep(0)
    assert not cleanup.done()
    await asyncio.wait_for(cleanup, timeout=1)
    assert killer.killed
    assert original.killed
