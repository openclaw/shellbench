"""Prepare a fresh provider dependency cache without credentials or model calls."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import secrets
import time

from clawbench.alignment import runtime
from clawbench.alignment.runtime import (
    ROOT,
    actor_config,
    approve_benchmark_device,
    docker,
    write_json,
)


async def prepare(context: str, image: str, volume: str, output: Path) -> dict:
    runtime.configure_runtime(context, image)
    if docker("volume", "inspect", volume, check=False).returncode == 0:
        raise ValueError("Use a new volume; never modify another run's dependency cache")
    resolved = docker("image", "inspect", image, "--format", "{{.Id}}").stdout.strip()
    output.mkdir(parents=True, exist_ok=False)
    state, workspace = output / "state", output / "workspace"
    state.mkdir()
    workspace.mkdir()
    (state / "home").mkdir()
    write_json(state / "openclaw.json", actor_config("openai/gpt-5.4-2026-03-05"))
    name = "shellbench-cache-" + secrets.token_hex(5)
    write_json(
        output / "resources.json",
        {"context": context, "image": resolved, "volume": volume, "container": name},
    )
    docker("volume", "create", volume)
    docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        "0",
        "--mount",
        f"type=volume,src={volume},dst=/runtime-deps",
        "--entrypoint",
        "chmod",
        resolved,
        "777",
        "/runtime-deps",
    )
    mounts = [(state, "/data"), (workspace, "/workspace")]
    sources = [
        (ROOT / source, target)
        for source, target in (
            ("clawbench/__init__.py", "/home/node/app/clawbench/__init__.py"),
            ("clawbench/client.py", "/home/node/app/clawbench/client.py"),
            ("clawbench/schemas.py", "/home/node/app/clawbench/schemas.py"),
            ("clawbench/alignment/bridge.py", "/bridge.py"),
        )
    ]
    command = [
        "run",
        "-d",
        "--name",
        name,
        "--network",
        "bridge",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--init",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--memory",
        "3g",
        "--cpus",
        "1",
        "--tmpfs",
        "/tmp:rw,nosuid,size=512m,mode=1777",
        "--mount",
        f"type=volume,src={volume},dst=/runtime-deps",
    ]
    for path, target in mounts:
        command += ["--mount", f"type=bind,src={path},dst={target}"]
    for path, target in sources:
        command += ["--mount", f"type=bind,src={path},dst={target},readonly"]
    for setting in (
        "OPENCLAW_STATE_DIR=/data",
        "OPENCLAW_CONFIG_PATH=/data/openclaw.json",
        "OPENCLAW_HOME=/data/home",
        "PYTHONPATH=/home/node/app",
        "OPENCLAW_PLUGIN_STAGE_DIR=/runtime-deps",
        "OPENCLAW_NO_RESPAWN=1",
        "OPENCLAW_SKIP_GMAIL_WATCHER=1",
        "OPENCLAW_SKIP_CANVAS_HOST=1",
        "OPENCLAW_SKIP_CHANNELS=1",
        "OPENCLAW_SKIP_PROVIDERS=1",
    ):
        command += ["-e", setting]
    ready = None
    try:
        # Only preparation has outbound network. No API key, relay or actor prompts.
        docker(*command, "--entrypoint", "openclaw", resolved, "gateway", "--port", "18789")
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if docker("inspect", "--format", "{{.State.Running}}", name).stdout.strip() != "true":
                raise RuntimeError("Preparation Gateway exited; inspect gateway.log")
            result = await asyncio.to_thread(
                docker,
                "exec",
                "-i",
                name,
                "python3",
                "/bridge.py",
                input=json.dumps({"preflight": True}),
                check=False,
                timeout=30,
            )
            (output / "preflight.stderr").write_text(result.stderr)
            if "NOT_PAIRED" in result.stderr and not (output / "device-bootstrap.json").exists():
                await asyncio.to_thread(approve_benchmark_device, name, state, output)
            if result.returncode == 0:
                ready = json.loads(result.stdout)
                if ready.get("authenticated_gateway"):
                    break
            await asyncio.sleep(2)
        if not ready or not ready.get("authenticated_gateway"):
            raise TimeoutError("Preparation never authenticated to Gateway")
        # Provider modules can load lazily only when a real native session is
        # created. Empty prompts warm that path without an inference request.
        session = await asyncio.to_thread(
            docker,
            "exec",
            "-i",
            name,
            "python3",
            "/bridge.py",
            input=json.dumps(
                {"workspace": "/workspace", "model": "openai/gpt-5.4-2026-03-05", "prompts": []}
            ),
            check=False,
            timeout=240,
        )
        (output / "session.stderr").write_text(session.stderr)
        (output / "session.json").write_text(session.stdout)
        if session.returncode or not json.loads(session.stdout).get("session_key"):
            raise RuntimeError("Native session preparation failed; inspect session.json")
    finally:
        docker("stop", "-t", "10", name, check=False, timeout=30)
        logs = docker("logs", name, check=False)
        (output / "gateway.log").write_text(logs.stdout + logs.stderr)
        docker("rm", name, check=False)
    runtime.configure_runtime(context, resolved, volume)
    manifest = runtime.dependency_manifest(resolved)
    write_json(output / "dependency-manifest.json", manifest)
    report = {
        "authenticated_gateway": True,
        "native_session_created": True,
        "image": resolved,
        "dependency_volume": volume,
        "dependency_entries": len(manifest),
        "model_calls": 0,
        "limits": "Gateway readiness alone is not model/tool E2E acceptance. Run the normal isolated task runner next.",
    }
    write_json(output / "result.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-context", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dependency-volume", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            asyncio.run(
                prepare(
                    args.docker_context, args.image, args.dependency_volume, args.output.resolve()
                )
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
