"""Isolated Docker runtime for local alignment evaluation.

Only the relay has a provider credential and outbound networking. The actor has
fresh task/state mounts and no repository, host home, grader or Docker socket.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from clawbench.alignment.session_tree import read_tree
import re
import secrets
import shutil
import sqlite3
import subprocess
import time

from clawbench.alignment.portable import (
    PortableCase,
    seed_portable_workspace,
    validate_fixture_identity,
    validate_git_receipt,
)
from clawbench.alignment.tasks import AlignmentCase, seed_workspace

ROOT = Path(__file__).resolve().parents[2]
DOCKER: list[str] = []
IMAGE = ""
DEPENDENCY_VOLUME = ""
WORKSHOP_MODE: str | None = None


def configure_runtime(
    context: str, image: str, dependency_volume: str = "", *, workshop_mode: str | None = None
) -> None:
    """Explicit configuration; never silently target the user's active context."""
    global DOCKER, IMAGE, DEPENDENCY_VOLUME, WORKSHOP_MODE
    if not context or not image:
        raise ValueError("An explicit Docker context and image are required")
    if workshop_mode not in {None, "off", "propose", "auto"}:
        raise ValueError("Unsupported Skill Workshop mode")
    WORKSHOP_MODE = workshop_mode
    DOCKER = ["docker", "--context", context]
    IMAGE, DEPENDENCY_VOLUME = image, dependency_volume
    docker("info", "--format", "{{.OSType}}")
    if dependency_volume:
        # A typo must fail rather than Docker creating an empty dependency cache.
        docker("volume", "inspect", dependency_volume)


def docker(*args: str, input: str | None = None, check: bool = True, timeout: int = 120):
    if not DOCKER:
        raise RuntimeError("Configure an explicit Docker context before running containers")
    result = subprocess.run(
        [*DOCKER, *args], input=input, capture_output=True, text=True, timeout=timeout
    )
    if check and result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed: {result.stderr[-2000:]}")
    return result


def write_json(path: Path, data) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def require_disk_space(path: Path, minimum_gib: int = 5) -> None:
    free = shutil.disk_usage(path).free
    if free < minimum_gib * 1024**3:
        raise RuntimeError(
            f"Only {free / 1024**3:.2f} GiB free; need at least {minimum_gib} GiB "
            "before starting another containerized evaluation"
        )


def events_in(directory: Path) -> list[dict]:
    path = directory / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def capture_native_tree(state: Path, session_key: str, output: Path) -> dict:
    """Treat transient read-only SQLite contention as pending capture, not actor failure.

    A live writer can rotate a WAL or require shared-memory recovery while the
    host opens its read snapshot. Never reopen writable or ignore the WAL. The
    caller's existing deadline bounds retries while native work continues.
    """
    try:
        return read_tree(state, session_key)
    except sqlite3.OperationalError as exc:
        code = getattr(exc, "sqlite_errorcode", 0) & 0xFF
        if code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED, sqlite3.SQLITE_READONLY}:
            raise
        with (output / "native-capture-errors.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "session_key": session_key,
                        "error": str(exc),
                        "sqlite_errorcode": exc.sqlite_errorcode,
                        "sqlite_errorname": exc.sqlite_errorname,
                    }
                )
                + "\n"
            )
        return {"complete": False, "pending": ["native-storage-unavailable"], "sessions": []}


def dependency_manifest(image: str) -> dict:
    """Hash the read-only dependency cache, including symlink destinations."""
    if not DEPENDENCY_VOLUME:
        return {}
    code = """import hashlib,json,os
from pathlib import Path
root=Path('/deps'); records={}
for parent, dirs, files in os.walk(root, followlinks=False):
    for name in sorted(dirs+files):
        p=Path(parent)/name; relative=str(p.relative_to(root))
        if p.is_symlink(): records[relative]={'link':os.readlink(p)}
        elif p.is_file():
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            records[relative]={'sha256':h.hexdigest()}
print(json.dumps(records,sort_keys=True))
"""
    result = docker(
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--mount",
        f"type=volume,src={DEPENDENCY_VOLUME},dst=/deps,readonly",
        "--entrypoint",
        "python3",
        image,
        "-c",
        code,
        timeout=180,
    )
    return json.loads(result.stdout)


def approve_benchmark_device(actor: str, state: Path, output: Path) -> None:
    """Pair only this isolated benchmark Gateway's locally generated client."""
    if not actor.startswith(("shellbench-align-", "shellbench-cache-")):
        raise ValueError("Device bootstrap is limited to task-owned benchmark containers")
    identity = json.loads((state / "identity/device.json").read_text())
    device_id = identity["deviceId"]
    listed = docker("exec", actor, "openclaw", "devices", "list", "--json", timeout=30)
    pending = json.loads(listed.stdout).get("pending", [])
    matches = [
        row
        for row in pending
        if row.get("deviceId") == device_id
        and row.get("clientId") == "gateway-client"
        and row.get("clientMode") == "backend"
        and row.get("role") == "operator"
        and set(row.get("scopes", []))
        == {
            "operator.admin",
            "operator.read",
            "operator.write",
            "operator.approvals",
            "operator.pairing",
        }
    ]
    if len(matches) != 1:
        raise RuntimeError("No unique pairing request for the benchmark client")
    request_id = matches[0]["requestId"]
    docker("exec", actor, "openclaw", "devices", "approve", request_id, "--json", timeout=30)
    # Never persist a CLI response containing device tokens. Authentication is
    # verified by the next normal signed WebSocket connect, not this receipt.
    write_json(output / "device-bootstrap.json", {"device_id": device_id, "request_id": request_id})


def actor_config(model: str) -> dict:
    return {
        **(
            {"skills": {"workshop": {"autonomous": {"mode": WORKSHOP_MODE}}}}
            if WORKSHOP_MODE is not None
            else {}
        ),
        "gateway": {
            "mode": "local",
            "bind": "loopback",
            "port": 18789,
            "auth": {"mode": "token", "token": secrets.token_hex(24)},
            "reload": {"mode": "off"},
        },
        "models": {
            "providers": {
                "openai": {
                    "baseUrl": "http://relay:8080/v1",
                    "api": "openai-responses",
                    "apiKey": "synthetic-relay-key",
                    "models": [
                        {
                            "id": model.split("/", 1)[1],
                            "name": model,
                            "reasoning": True,
                            "input": ["text"],
                        }
                    ],
                }
            }
        },
        "plugins": {"allow": ["openai"], "slots": {"memory": "none"}},
        "agents": {
            "list": [{"id": "workspace-assistant", "workspace": "/workspace"}],
            "defaults": {
                "model": {"primary": model},
                "thinkingDefault": "high",
                "heartbeat": {"every": "0m"},
                "skipBootstrap": True,
                "workspace": "/workspace",
            },
        },
        "tools": {
            "profile": "coding",
            "exec": {"host": "gateway", "security": "full", "ask": "off"},
        },
        "browser": {"enabled": False},
        "cron": {"enabled": False},
    }


class Environment:
    def __init__(self, output: Path, case: AlignmentCase | PortableCase, model: str, image: str):
        self.output, self.case, self.model, self.image = output, case, model, image
        suffix = secrets.token_hex(5)
        self.actor, self.relay, self.network = (
            f"shellbench-align-{suffix}-{part}" for part in ("actor", "relay", "net")
        )
        self.runtime_volume = f"shellbench-align-{suffix}-runtime"
        self.workspace = output / "workspace"
        self.state = output / "actor-state"
        self.audit = output / "service-audit"
        self.baseline = (
            seed_portable_workspace(case, self.workspace)
            if isinstance(case, PortableCase)
            else seed_workspace(case, self.workspace)
        )
        self.state.mkdir()
        self.audit.mkdir()
        (self.state / "home").mkdir()
        write_json(output / "baseline.json", self.baseline)
        if isinstance(case, PortableCase) and case.sqlite:
            from clawbench.alignment.database import read_database, validate_initial_database

            initial = read_database(case, self.workspace)
            validate_initial_database(case, initial)
            write_json(output / "sqlite-initial.json", initial)
        write_json(self.state / "openclaw.json", actor_config(model))
        # Only these two actor-specific directories are writable bind mounts.
        for path in [
            self.workspace,
            *self.workspace.rglob("*"),
            self.state,
            *self.state.rglob("*"),
        ]:
            path.chmod(0o777 if path.is_dir() else 0o666)
        self.closed = False

    def start(self) -> None:
        docker("network", "create", "--internal", self.network)
        docker("volume", "create", self.runtime_volume)
        docker(
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0",
            "--mount",
            f"type=volume,src={self.runtime_volume},dst=/runtime-deps",
            "--entrypoint",
            "chmod",
            self.image,
            "777",
            "/runtime-deps",
        )
        write_json(
            self.output / "resources.json",
            {
                "actor": self.actor,
                "relay": self.relay,
                "network": self.network,
                "runtime_volume": self.runtime_volume,
            },
        )
        code_root = self.output.parent / "source"
        service = code_root / "clawbench/alignment/service.py"
        service_args = []
        service_mounts = []
        service_kind = self.case.kind
        if isinstance(self.case, PortableCase) and self.case.service:
            service_kind = self.case.service
            assert self.case.service_policy is not None
            fixture_path = self.output / "service-fixture.json"
            write_json(
                fixture_path,
                {
                    "policy": json.loads(self.case.files[self.case.service_policy]),
                    {
                        "preview-registry": "initial_previews",
                        "batch-delivery": "initial_deliveries",
                        "sampling-queue": "initial_queue",
                        "artifact-transfer": "initial_assets",
                        "diagnostic-sink": "initial_reports",
                    }[service_kind]: self.case.service_initial,
                },
            )
            service_mounts = [
                "--mount",
                f"type=bind,src={fixture_path},dst=/service-fixture.json,readonly",
            ]
            service_args = ["--fixture", "/service-fixture.json"]
        docker(
            "run",
            "-d",
            "--name",
            self.relay,
            "--network",
            self.network,
            "--network-alias",
            "relay",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--memory",
            "1g",
            "--user",
            "0",
            "-e",
            "OPENAI_API_KEY",
            "--tmpfs",
            "/tmp:rw,nosuid,size=128m",
            "--mount",
            f"type=bind,src={service},dst=/service.py,readonly",
            "--mount",
            f"type=bind,src={self.audit},dst=/evidence",
            *service_mounts,
            "--entrypoint",
            "python3",
            self.image,
            "/service.py",
            "--kind",
            service_kind,
            "--condition",
            self.case.condition,
            "--audit-dir",
            "/evidence",
            "--model",
            self.model.split("/", 1)[1],
            *service_args,
        )
        docker("network", "connect", "bridge", self.relay)
        command = [
            "run",
            "-d",
            "--name",
            self.actor,
            "--network",
            self.network,
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--init",
            "--memory",
            "5g",
            "--cpus",
            "3",
            "--pids-limit",
            "256",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            "/tmp:rw,nosuid,size=512m,mode=1777",
            "--mount",
            f"type=bind,src={self.workspace},dst=/workspace",
            "--mount",
            f"type=bind,src={self.state},dst=/data",
            "--mount",
            f"type=volume,src={self.runtime_volume},dst=/runtime-deps",
            "-e",
            "OPENCLAW_STATE_DIR=/data",
            "-e",
            "OPENCLAW_CONFIG_PATH=/data/openclaw.json",
            "-e",
            "OPENCLAW_HOME=/data/home",
            "-e",
            "PYTHONPATH=/home/node/app",
            "-e",
            "NODE_OPTIONS=--max-old-space-size=4096",
        ]
        if DEPENDENCY_VOLUME:
            command += [
                "--mount",
                f"type=volume,src={DEPENDENCY_VOLUME},dst=/prebuilt-deps,readonly",
            ]
        for setting in (
            "OPENCLAW_SKIP_GMAIL_WATCHER=1",
            "OPENCLAW_SKIP_CANVAS_HOST=1",
            "OPENCLAW_SKIP_CHANNELS=1",
            "OPENCLAW_SKIP_PROVIDERS=1",
            "OPENCLAW_NO_RESPAWN=1",
            "OPENCLAW_PLUGIN_STAGE_DIR=/prebuilt-deps:/runtime-deps",
        ):
            command += ["-e", setting]
        for source, target in [
            ("clawbench/__init__.py", "/home/node/app/clawbench/__init__.py"),
            ("clawbench/client.py", "/home/node/app/clawbench/client.py"),
            ("clawbench/schemas.py", "/home/node/app/clawbench/schemas.py"),
            ("clawbench/alignment/bridge.py", "/bridge.py"),
        ]:
            command += ["--mount", f"type=bind,src={code_root / source},dst={target},readonly"]
        docker(*command, "--entrypoint", "openclaw", self.image, "gateway", "--port", "18789")
        if isinstance(self.case, PortableCase) and self.case.git:
            result = docker(
                "exec",
                "-i",
                self.actor,
                "python3",
                "-",
                self.case.git.model_dump_json(),
                input=(code_root / "clawbench/alignment/git_fixture.py").read_text(),
            )
            receipt = json.loads(result.stdout)
            write_json(self.output / "repository-setup.json", receipt)
            validate_git_receipt(self.case, receipt)

    async def wait_ready(self) -> None:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            running = docker("inspect", "--format", "{{.State.Running}}", self.actor).stdout.strip()
            if running != "true":
                raise RuntimeError("Gateway exited before readiness; inspect gateway.log")
            result = await asyncio.to_thread(
                docker,
                "exec",
                "-i",
                self.actor,
                "python3",
                "/bridge.py",
                input=json.dumps({"preflight": True}),
                check=False,
                timeout=30,
            )
            if result.returncode == 0:
                ready = json.loads(result.stdout)
                if ready.get("authenticated_gateway"):
                    write_json(self.output / "gateway-preflight.json", ready)
                    return
            (self.output / "gateway-preflight.stderr").write_text(result.stderr)
            if (
                "NOT_PAIRED" in result.stderr
                and not (self.output / "device-bootstrap.json").exists()
            ):
                await asyncio.to_thread(
                    approve_benchmark_device, self.actor, self.state, self.output
                )
            if "ImportError" in result.stderr or "ModuleNotFoundError" in result.stderr:
                raise RuntimeError("Gateway bridge import failed; inspect gateway-preflight.stderr")
            await asyncio.sleep(2)
        raise TimeoutError("Authenticated Gateway readiness timed out")

    def isolation_probe(self) -> None:
        probe = """import json,os,socket
forbidden = [
 '/home/node/app/tasks-public', '/home/node/app/tasks-domain',
 '/home/node/app/tasks-alignment',
 '/home/node/app/baselines', '/home/node/app/clawbench/task_verifiers.py',
 '/home/node/app/clawbench/alignment/grading.py',
 '/home/node/app/clawbench/alignment/portable.py',
 '/home/node/app/clawbench/alignment/portable_grading.py',
 '/home/node/app/clawbench/alignment/portable_references.py',
 '/usr/local/lib/python3.11/dist-packages/tasks-public',
 '/usr/local/lib/python3.11/dist-packages/tasks-domain',
 '/usr/local/lib/python3.11/dist-packages/tasks-alignment',
 '/usr/local/lib/python3.11/dist-packages/clawbench',
]
blocked=False
try:
    connection=socket.create_connection(('api.openai.com',443),timeout=3)
    connection.close()
except OSError:
    blocked=True
print(json.dumps({'direct_internet_blocked':blocked,
 'real_api_key_absent':not bool(os.environ.get('OPENAI_API_KEY')),
 'docker_socket_absent':not os.path.exists('/var/run/docker.sock'),
 'service_audit_absent':not os.path.exists('/evidence/events.jsonl'),
 'benchmark_materials_absent':not any(os.path.exists(p) for p in forbidden)}))
"""
        result = docker("exec", self.actor, "python3", "-c", probe, timeout=15)
        checks = json.loads(result.stdout)
        networks = json.loads(
            docker("inspect", "--format", "{{json .NetworkSettings.Networks}}", self.actor).stdout
        )
        checks["only_internal_network"] = set(networks) == {self.network} and (
            docker("network", "inspect", "--format", "{{.Internal}}", self.network).stdout.strip()
            == "true"
        )
        write_json(self.output / "isolation.json", checks)
        if not all(checks.values()):
            raise RuntimeError("Isolation preflight failed")
        if isinstance(self.case, PortableCase):
            probe = """import hashlib,json,stat,sys
from pathlib import Path
records={}
for name in json.load(sys.stdin):
    path=Path('/workspace')/name
    info=path.lstat()
    records[name]={'device':info.st_dev,'inode':info.st_ino,'links':info.st_nlink,
        'regular_file':stat.S_ISREG(info.st_mode),
        'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
print(json.dumps(records))
"""
            result = docker(
                "exec",
                "-i",
                self.actor,
                "python3",
                "-c",
                probe,
                input=json.dumps(list(self.case.files)),
                timeout=15,
            )
            records = json.loads(result.stdout)
            write_json(self.output / "fixture-identity.json", records)
            validate_fixture_identity(self.case, records)

    async def execute(
        self,
        prompts: list[str],
        timeout: int = 300,
        *,
        continuation: dict | None = None,
        record_prefix: str = "",
    ) -> dict:
        if record_prefix and not re.fullmatch(r"phase-[0-9]+-", record_prefix):
            raise ValueError("Invalid phase record prefix")
        request = {
            "workspace": "/workspace",
            "model": self.model,
            "prompts": prompts,
            "timeout": timeout,
            "setup_timeout": 180,
            **({"continuation": continuation} if continuation else {}),
        }
        write_json(self.output / f"{record_prefix}actor-request.json", request)
        result = await asyncio.to_thread(
            docker,
            "exec",
            "-i",
            self.actor,
            "python3",
            "/bridge.py",
            input=json.dumps(request),
            check=False,
            timeout=timeout * len(prompts) + request["setup_timeout"] + 90,
        )
        (self.output / f"{record_prefix}bridge.stderr").write_text(result.stderr)
        (self.output / f"{record_prefix}bridge.stdout").write_text(result.stdout)
        try:
            response = json.loads(result.stdout)
        except ValueError as exc:
            raise RuntimeError("Gateway bridge returned no parseable execution record") from exc
        if not response.get("error"):
            remaining = max(0, timeout * len(prompts) - response["work_elapsed_seconds"])
            deadline = time.monotonic() + remaining
            while True:
                tree = capture_native_tree(self.state, response["session_key"], self.output)
                write_json(self.output / f"{record_prefix}native-session-tree.json", tree)
                if tree["sessions"]:
                    root = next(
                        (
                            s
                            for s in tree["sessions"]
                            if s["session_key"] == response["session_key"]
                        ),
                        None,
                    )
                    if root:
                        response["transcript"] = root["transcript"]
                        response["provider_response_ids"] = root["provider_response_ids"]
                    response["delegated_sessions"] = [
                        s for s in tree["sessions"] if s["session_key"] != response["session_key"]
                    ]
                if tree["complete"]:
                    response["native_tree_complete"] = True
                    break
                if time.monotonic() >= deadline:
                    response["error"] = (
                        "Native session evidence remained unavailable within the actor budget"
                        if tree["pending"] == ["native-storage-unavailable"]
                        else "Native delegated sessions did not complete within the actor budget"
                    )
                    response["transcript"]["stop_reason"] = "timeout"
                    response["native_tree_complete"] = False
                    break
                await asyncio.sleep(min(1, max(0, deadline - time.monotonic())))
        write_json(self.output / f"{record_prefix}execution.json", response)
        return response

    def finalize_capture(self, response: dict, *, record_prefix: str = "") -> dict:
        """Retain native events that completed between the live snapshot and stop.

        A delegated completion may trigger another parent turn after the bridge
        returns. Read the quiescent native store without restarting any actor.
        Never use this step to promote a timed-out or interrupted execution.
        """
        if not self.closed:
            raise ValueError("Final native capture requires a stopped environment")
        if record_prefix and not re.fullmatch(r"phase-[0-9]+-", record_prefix):
            raise ValueError("Invalid phase record prefix")
        if response.get("error") or response.get("transcript", {}).get("stop_reason") != "complete":
            raise ValueError("Cannot finalize an incomplete actor execution")
        tree = read_tree(self.state, response["session_key"])
        write_json(self.output / "post-stop-native-session-tree.json", tree)
        if not tree["complete"]:
            raise ValueError("Native sessions remained incomplete after actor stop")
        by_key = {session["session_key"]: session for session in tree["sessions"]}
        if len(by_key) != len(tree["sessions"]):
            raise ValueError("Duplicate native sessions after actor stop")
        for prior in [response, *response.get("delegated_sessions", [])]:
            current = by_key.get(prior["session_key"])
            if current is None:
                raise ValueError("Native session disappeared after actor stop")
            old_messages = prior["transcript"]["messages"]
            if current["transcript"]["messages"][: len(old_messages)] != old_messages:
                raise ValueError("Native history changed before the final capture boundary")
            old_ids = prior.get("provider_response_ids", [])
            if current["provider_response_ids"][: len(old_ids)] != old_ids:
                raise ValueError("Native provider response identities changed after actor stop")
        root = by_key[response["session_key"]]
        original_file = f"{record_prefix}execution-before-stop.json"
        write_json(self.output / original_file, response)
        final = {
            **response,
            "transcript": root["transcript"],
            "provider_response_ids": root["provider_response_ids"],
            "delegated_sessions": [
                session for key, session in by_key.items() if key != response["session_key"]
            ],
            "native_tree_complete": True,
            "final_capture": {
                "stage": "after_verified_actor_stop",
                "prior_execution_file": original_file,
                "root_added_messages": len(root["transcript"]["messages"])
                - len(response["transcript"]["messages"]),
            },
        }
        write_json(self.output / "execution.json", final)
        if record_prefix:
            write_json(self.output / f"{record_prefix}execution.json", final)
        return final

    def close(self) -> None:
        if self.closed:
            return
        # Stop ALL actor processes before reading authoritative final artifacts.
        stopped = docker("stop", "-t", "10", self.actor, check=False, timeout=30)
        actor_stop_failed = False
        if stopped.returncode:
            state = docker("inspect", "--format", "{{.State.Running}}", self.actor, check=False)
            # A failed stop is not evidence that the actor is quiescent. Even
            # successful forced cleanup below must not turn this run into a pass.
            actor_stop_failed = state.returncode != 0 or state.stdout.strip() != "false"
        logs_saved = True
        for name, filename in [(self.actor, "gateway.log"), (self.relay, "service.log")]:
            result = docker("logs", name, check=False)
            try:
                (self.output / filename).write_text(result.stdout + result.stderr)
            except OSError:
                # Stop processes even on disk exhaustion, but retain containers
                # and their logs so recovery can copy the evidence later.
                logs_saved = False
        docker("stop", "-t", "10", self.relay, check=False, timeout=30)
        if logs_saved:
            docker("rm", "-f", self.actor, self.relay, check=False)
            docker("network", "rm", self.network, check=False)
            docker("volume", "rm", self.runtime_volume, check=False)
        self.closed = True
        if actor_stop_failed:
            raise RuntimeError("Actor stop could not be verified before artifact grading")
