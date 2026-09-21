import asyncio
import os
import signal
import subprocess
from pathlib import Path

import pytest

from clawbench.schemas import BackgroundService
from clawbench import services as service_module
from clawbench.services import build_runtime_values, start_background_services, stop_background_services


@pytest.mark.asyncio
async def test_background_service_waits_for_ready_file(tmp_path: Path):
    script = tmp_path / "service.py"
    script.write_text(
        "from pathlib import Path\n"
        "import time\n"
        "Path('ready.txt').write_text('ok', encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    runtime_values = build_runtime_values(workspace=tmp_path, repo_root=Path.cwd())
    service = BackgroundService(
        name="ready_file_service",
        command="{python_exe} service.py",
        ready_file="ready.txt",
        startup_timeout_seconds=5,
    )

    services, _ = await start_background_services(
        [service],
        workspace=tmp_path,
        repo_root=Path.cwd(),
        runtime_values=runtime_values,
    )
    try:
        assert (tmp_path / "ready.txt").exists()
    finally:
        await stop_background_services(services)


@pytest.mark.asyncio
async def test_background_service_quotes_runtime_values_in_shell_command(tmp_path: Path):
    script = tmp_path / "service with space.py"
    script.write_text(
        "from pathlib import Path\n"
        "import time\n"
        "Path('ready with space.txt').write_text('ok', encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    runtime_values = build_runtime_values(
        workspace=tmp_path,
        repo_root=Path.cwd(),
        extra={"script_path": script.name},
    )
    service = BackgroundService(
        name="quoted_service",
        command="{python_exe} {script_path}",
        ready_file="ready with space.txt",
        startup_timeout_seconds=5,
    )

    services, _ = await start_background_services(
        [service],
        workspace=tmp_path,
        repo_root=Path.cwd(),
        runtime_values=runtime_values,
    )
    try:
        assert (tmp_path / "ready with space.txt").exists()
    finally:
        await stop_background_services(services)


@pytest.mark.asyncio
async def test_background_service_rejects_cwd_outside_workspace(tmp_path: Path):
    runtime_values = build_runtime_values(workspace=tmp_path, repo_root=Path.cwd())
    service = BackgroundService(
        name="bad_service",
        command="true",
        cwd="..",
        ready_path=None,
    )

    with pytest.raises(ValueError, match="escapes workspace"):
        await start_background_services(
            [service],
            workspace=tmp_path,
            repo_root=Path.cwd(),
            runtime_values=runtime_values,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cwd", "spawn", "readiness", "cancel"])
async def test_failed_service_start_reaps_every_started_process(tmp_path, monkeypatch, failure):
    spawned = []
    handles = []
    original_popen = subprocess.Popen

    def record_popen(*args, **kwargs):
        handles.append(kwargs["stdout"])
        if failure == "spawn" and spawned:
            raise OSError("synthetic spawn failure")
        process = original_popen(*args, **kwargs)
        spawned.append(process)
        return process

    ready_count = 0

    async def readiness(*args):
        nonlocal ready_count
        ready_count += 1
        if ready_count == 2:
            if failure == "cancel":
                raise asyncio.CancelledError
            raise TimeoutError("synthetic readiness failure")

    monkeypatch.setattr(service_module.subprocess, "Popen", record_popen)
    monkeypatch.setattr(service_module, "_wait_for_service_ready", readiness)
    specs = [
        BackgroundService(name="first", command="exec sleep 30"),
        BackgroundService(
            name="second", command="exec sleep 30", cwd=".." if failure == "cwd" else "."
        ),
    ]
    expected = {
        "cwd": ValueError,
        "spawn": OSError,
        "readiness": TimeoutError,
        "cancel": asyncio.CancelledError,
    }[failure]
    try:
        with pytest.raises(expected):
            await start_background_services(
                specs, workspace=tmp_path, repo_root=tmp_path, runtime_values={}
            )
        assert spawned
        assert all(process.poll() is not None for process in spawned)
        assert all(handle.closed for handle in handles)
    finally:
        for process in spawned:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)
        for handle in handles:
            handle.close()
