from __future__ import annotations
import hashlib
import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")


@dataclass(frozen=True)
class McpServer:
    name: str
    transport: str
    url: str | None = None
    command: str | None = None
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    path: Path
    name: str
    title: str
    instruction: str
    raw_config: dict[str, Any]
    dockerfile: Path
    build_context: Path
    compose_file: Path | None
    verifier_command: str
    verifier_env: dict[str, str]
    environment_env: dict[str, str]
    mcp_servers: tuple[McpServer, ...]
    agent_timeout_sec: float
    verifier_timeout_sec: float
    build_timeout_sec: float
    checksum: str

    @classmethod
    def load(cls, path: Path) -> "TaskSpec":
        path = path.resolve()
        config_path = path / "task.toml"
        instruction_path = path / "instruction.md"
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
        environment = raw.get("environment") or {}
        verifier = raw.get("verifier") or {}
        task_meta = raw.get("task") or {}

        dockerfile_rel = environment.get("dockerfile", "environment/Dockerfile")
        context_rel = environment.get("context", "environment")
        dockerfile = _resolve_task_path(path, dockerfile_rel)
        build_context = _resolve_task_path(path, context_rel)
        compose_file = path / "environment" / "docker-compose.yaml"
        if not compose_file.is_file():
            compose_file = None

        title = str(raw.get("title") or task_meta.get("name") or path.name)
        servers = []
        for item in environment.get("mcp_servers") or []:
            servers.append(
                McpServer(
                    name=str(item["name"]),
                    transport=str(item.get("transport", "stdio")),
                    url=_optional_str(item.get("url")),
                    command=_optional_str(item.get("command")),
                    args=tuple(str(value) for value in item.get("args") or []),
                )
            )

        return cls(
            path=path,
            name=path.name,
            title=title,
            instruction=instruction_path.read_text(encoding="utf-8"),
            raw_config=raw,
            dockerfile=dockerfile,
            build_context=build_context,
            compose_file=compose_file,
            verifier_command=str(verifier.get("command") or "bash tests/test.sh"),
            verifier_env={
                str(key): str(value)
                for key, value in (verifier.get("env") or {}).items()
            },
            environment_env={
                str(key): str(value)
                for key, value in (environment.get("env") or {}).items()
            },
            mcp_servers=tuple(servers),
            agent_timeout_sec=float((raw.get("agent") or {}).get("timeout_sec", 900)),
            verifier_timeout_sec=float(verifier.get("timeout_sec", 300)),
            build_timeout_sec=float(environment.get("build_timeout_sec", 900)),
            checksum=_directory_checksum(path),
        )

    def resolved_verifier_env(self) -> dict[str, str]:
        return {
            key: _expand_env(value)
            for key, value in self.verifier_env.items()
        }


def validate_suite(tasks_root: Path) -> list[TaskSpec]:
    tasks_root = tasks_root.resolve()
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"Task suite not found: {tasks_root}")

    task_dirs = sorted(path for path in tasks_root.iterdir() if path.is_dir())
    if not task_dirs:
        raise ValueError(f"No task directories found under {tasks_root}")

    errors: list[str] = []
    tasks: list[TaskSpec] = []
    for task_dir in task_dirs:
        for relative in (
            "task.toml",
            "instruction.md",
            "environment/Dockerfile",
            "solution/solve.sh",
            "tests/test.sh",
        ):
            if not (task_dir / relative).is_file():
                errors.append(f"{task_dir.name}: missing {relative}")
        if errors and errors[-1].startswith(f"{task_dir.name}:"):
            continue
        try:
            task = TaskSpec.load(task_dir)
        except Exception as exc:
            errors.append(f"{task_dir.name}: invalid task definition: {exc}")
            continue
        if not task.dockerfile.is_file():
            errors.append(f"{task_dir.name}: dockerfile not found: {task.dockerfile}")
        if not task.build_context.is_dir():
            errors.append(
                f"{task_dir.name}: build context not found: {task.build_context}"
            )
        tasks.append(task)

    if errors:
        raise ValueError("Task suite validation failed:\n" + "\n".join(errors))
    return tasks


def _resolve_task_path(task_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    direct = task_dir / path
    if direct.exists():
        return direct
    environment_relative = task_dir / "environment" / path
    if environment_relative.exists():
        return environment_relative
    return direct


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _expand_env(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2) or ""
        return os.environ.get(name, default)

    return _ENV_PATTERN.sub(replace, value)


def _directory_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
