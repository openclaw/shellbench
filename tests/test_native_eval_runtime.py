from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from scripts.native_eval.runtime import DockerTaskEnvironment
from scripts.native_eval import runtime as native_runtime


def _environment(tmp_path: Path, *, compose: bool = False) -> DockerTaskEnvironment:
    compose_file = tmp_path / "docker-compose.yaml" if compose else None
    if compose_file:
        compose_file.write_text("services: {}\n", encoding="utf-8")
    return DockerTaskEnvironment(
        task=SimpleNamespace(compose_file=compose_file),  # type: ignore[arg-type]
        trial_dir=tmp_path,
        container_name="trial",
        project_name="trial",
        toolchain_root=tmp_path,
        container_id="container-id",
    )


def _hanging_run_process() -> object:
    async def hang(_args: list[str], **_kwargs: object) -> None:
        await asyncio.Event().wait()

    return hang


def _cleanup_deadline(monkeypatch) -> None:
    # Production uses 30s. Tests use a short bound so a hang is visible
    # as a wait_for miss instead of a minutes-long stall.
    monkeypatch.setattr(native_runtime, "DOCKER_CLEANUP_TIMEOUT_SEC", 0.05, raising=False)


async def _await_with_deadline(coro):
    outcome = {"kind": "hung"}
    try:

        async def run() -> None:
            try:
                await coro
                outcome["kind"] = "returned"
            except TimeoutError:
                outcome["kind"] = "timeout"

        await asyncio.wait_for(run(), timeout=1.0)
    except TimeoutError:
        outcome["kind"] = "hung"
    return outcome["kind"]


def test_exec_timeout_ends_when_docker_kill_hangs(tmp_path: Path, monkeypatch) -> None:
    _cleanup_deadline(monkeypatch)
    monkeypatch.setattr(native_runtime, "run_process", _hanging_run_process())
    environment = _environment(tmp_path)

    kind = asyncio.run(_await_with_deadline(environment.exec("true", timeout=0.01)))

    assert kind == "timeout"


def test_stop_ends_when_docker_rm_hangs(tmp_path: Path, monkeypatch) -> None:
    _cleanup_deadline(monkeypatch)
    monkeypatch.setattr(native_runtime, "run_process", _hanging_run_process())
    environment = _environment(tmp_path)

    kind = asyncio.run(_await_with_deadline(environment.stop()))

    assert kind == "timeout"


def test_stop_ends_when_compose_down_hangs(tmp_path: Path, monkeypatch) -> None:
    _cleanup_deadline(monkeypatch)
    monkeypatch.setattr(native_runtime, "run_process", _hanging_run_process())
    environment = _environment(tmp_path, compose=True)

    kind = asyncio.run(_await_with_deadline(environment.stop()))

    assert kind == "timeout"
