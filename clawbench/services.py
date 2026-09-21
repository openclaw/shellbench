"""Background service helpers for deterministic task environments."""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from clawbench.paths import resolve_workspace_path
from clawbench.render import render_shell_template, render_template, render_value
from clawbench.schemas import BackgroundService


@dataclass
class ManagedService:
    spec: BackgroundService
    process: subprocess.Popen[str]
    log_path: Path
    port: int | None
    base_url: str | None


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def build_runtime_values(
    *,
    workspace: Path,
    repo_root: Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = {
        "workspace": str(workspace),
        "workspace_name": workspace.name,
        "repo_root": str(repo_root),
        "benchmark_node_path": str(repo_root / "node_modules"),
        "openclaw_node_path": "/openclaw/node_modules",
        "python_exe": sys.executable,
    }
    if extra:
        values.update(extra)
    return values


async def start_background_services(
    specs: list[BackgroundService],
    *,
    workspace: Path,
    repo_root: Path,
    runtime_values: dict[str, Any],
) -> tuple[list[ManagedService], dict[str, Any]]:
    services: list[ManagedService] = []
    values = dict(runtime_values)

    try:
        for spec in specs:
            port = spec.port or _pick_free_port()
            base_url = render_template(spec.url_template, {"port": port}) if spec.url_template else None
            values[f"{spec.name}_port"] = port
            if base_url:
                values[f"{spec.name}_url"] = base_url

            rendered_env = render_value(spec.env, values)
            service_env = {
                **os.environ,
                **{key: str(value) for key, value in rendered_env.items()},
            }
            if spec.port_env:
                service_env[spec.port_env] = str(port)
            service_env.setdefault("PYTHONUNBUFFERED", "1")

            command = render_shell_template(spec.command, values)
            cwd = resolve_workspace_path(
                workspace,
                render_template(spec.cwd, values),
                field=f"background service cwd for {spec.name}",
            )
            log_dir = workspace / ".clawbench-services"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"{spec.name}.log"
            with log_path.open("w", encoding="utf-8") as log_file:
                process = subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=service_env,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,  # put shell + child in own process group so we can kill the whole tree
                )
            managed = ManagedService(
                spec=spec,
                process=process,
                log_path=log_path,
                port=port,
                base_url=base_url,
            )
            services.append(managed)
            await _wait_for_service_ready(managed, workspace, values)
    except BaseException:
        await stop_background_services(services)
        raise

    return services, values


async def _wait_for_service_ready(
    service: ManagedService,
    workspace: Path,
    runtime_values: dict[str, Any],
) -> None:
    spec = service.spec
    deadline = time.monotonic() + spec.startup_timeout_seconds
    ready_file = None
    if spec.ready_file:
        ready_file = resolve_workspace_path(
            workspace,
            render_template(spec.ready_file, runtime_values),
            field=f"background service ready_file for {spec.name}",
        )
    ready_url = None
    if service.base_url and spec.ready_path:
        ready_url = f"{service.base_url.rstrip('/')}/{spec.ready_path.lstrip('/')}"

    while time.monotonic() < deadline:
        if service.process.poll() is not None:
            log_tail = service.log_path.read_text(encoding="utf-8", errors="replace")[-2_000:]
            raise RuntimeError(
                f"Background service {spec.name} exited early with code {service.process.returncode}: {log_tail}"
            )
        if ready_file and ready_file.exists():
            return
        if ready_url:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(ready_url)
                if response.status_code == spec.ready_status:
                    if spec.ready_contains and spec.ready_contains not in response.text:
                        await asyncio.sleep(0.2)
                        continue
                    return
            except Exception:
                pass
        elif ready_file is None:
            await asyncio.sleep(0.2)
            return
        await asyncio.sleep(0.2)

    raise TimeoutError(f"Timed out waiting for background service {spec.name}")


def _kill_pgroup(process: subprocess.Popen, sig: int) -> None:
    """Signal the entire process group so shell-spawned children don't survive."""
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        pass


async def stop_background_services(services: list[ManagedService]) -> None:
    for service in reversed(services):
        process = service.process
        if process.poll() is not None:
            continue
        _kill_pgroup(process, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.to_thread(process.wait, 5), timeout=6)
        except Exception:
            _kill_pgroup(process, signal.SIGKILL)
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait, 5), timeout=6)
            except Exception:
                pass
