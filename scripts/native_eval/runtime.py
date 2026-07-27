from __future__ import annotations

import asyncio
import json
import os
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.native_eval.harnesses import TOOLCHAIN_ROOT, build_harness_command
from scripts.native_eval.models import RunSpec
from scripts.native_eval.tasks import TaskSpec


class NativeEvalError(RuntimeError):
    pass


class EnvironmentStartTimeoutError(NativeEvalError):
    pass


class DockerStartupError(NativeEvalError):
    pass


class AgentSetupTimeoutError(NativeEvalError):
    pass


class NonZeroAgentExitCodeError(NativeEvalError):
    pass


class RewardFileNotFoundError(NativeEvalError):
    pass


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    started_at: str
    finished_at: str


@dataclass
class DockerTaskEnvironment:
    task: TaskSpec
    trial_dir: Path
    container_name: str
    project_name: str
    toolchain_root: Path
    workdir: str = "/app"
    container_id: str | None = None
    compose_override: Path | None = None

    @property
    def agent_dir(self) -> Path:
        return self.trial_dir / "agent"

    @property
    def verifier_dir(self) -> Path:
        return self.trial_dir / "verifier"

    @property
    def artifacts_dir(self) -> Path:
        return self.trial_dir / "artifacts"

    async def start(self) -> CommandResult:
        for path in (
            self.agent_dir,
            self.verifier_dir,
            self.artifacts_dir / "logs" / "artifacts",
        ):
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o777)

        started_at = utc_now()
        try:
            async with asyncio.timeout(self.task.build_timeout_sec):
                if self.task.compose_file:
                    await self._start_compose()
                else:
                    await self._start_single()
        except TimeoutError as exc:
            raise EnvironmentStartTimeoutError(
                f"Environment start timed out after {self.task.build_timeout_sec}s"
            ) from exc
        except NativeEvalError:
            raise
        except Exception as exc:
            raise DockerStartupError(str(exc)) from exc
        return CommandResult(0, started_at, utc_now())

    async def _start_single(self) -> None:
        image = f"shellbench-task:{self.task.checksum[:16]}"
        build_log = self.trial_dir / "environment-build.log"
        build = await run_process(
            [
                "docker",
                "build",
                "--pull=false",
                "-f",
                str(self.task.dockerfile),
                "-t",
                image,
                str(self.task.build_context),
            ],
            stdout_path=build_log,
            stderr_path=build_log,
        )
        if build.returncode:
            raise DockerStartupError(f"Docker build exited {build.returncode}")

        command = [
            "docker",
            "run",
            "-d",
            "--name",
            self.container_name,
            "--add-host",
            "host.docker.internal:host-gateway",
            "-v",
            f"{self.agent_dir.resolve()}:/logs/agent",
            "-v",
            f"{self.verifier_dir.resolve()}:/logs/verifier",
            "-v",
            (
                f"{(self.artifacts_dir / 'logs' / 'artifacts').resolve()}"
                ":/logs/artifacts"
            ),
            "-v",
            f"{self.toolchain_root.resolve()}:{TOOLCHAIN_ROOT}:ro",
        ]
        for key, value in self.task.environment_env.items():
            command.extend(["-e", f"{key}={value}"])
        command.extend([image, "sleep", "infinity"])
        result = await run_process(
            command,
            stdout_path=self.trial_dir / "environment-start.log",
            stderr_path=self.trial_dir / "environment-start.log",
        )
        if result.returncode:
            raise DockerStartupError(f"docker run exited {result.returncode}")
        self.container_id = self.container_name
        await self._discover_workdir()

    async def _start_compose(self) -> None:
        self.compose_override = self.trial_dir / "docker-compose.native.json"
        override = {
            "services": {
                "main": {
                    "volumes": [
                        {
                            "type": "bind",
                            "source": str(self.agent_dir.resolve()),
                            "target": "/logs/agent",
                        },
                        {
                            "type": "bind",
                            "source": str(self.verifier_dir.resolve()),
                            "target": "/logs/verifier",
                        },
                        {
                            "type": "bind",
                            "source": str(
                                (self.artifacts_dir / "logs" / "artifacts").resolve()
                            ),
                            "target": "/logs/artifacts",
                        },
                        {
                            "type": "bind",
                            "source": str(self.toolchain_root.resolve()),
                            "target": str(TOOLCHAIN_ROOT),
                            "read_only": True,
                        },
                    ],
                    "extra_hosts": ["host.docker.internal:host-gateway"],
                    "environment": self.task.environment_env,
                }
            }
        }
        atomic_write_json(self.compose_override, override)
        command = self._compose_prefix() + ["up", "-d", "--build", "--wait"]
        result = await run_process(
            command,
            stdout_path=self.trial_dir / "environment-start.log",
            stderr_path=self.trial_dir / "environment-start.log",
        )
        if result.returncode:
            raise DockerStartupError(f"docker compose up exited {result.returncode}")
        ps = await capture_process(self._compose_prefix() + ["ps", "-q", "main"])
        self.container_id = ps.strip()
        if not self.container_id:
            raise DockerStartupError("docker compose did not return the main container")
        await self._discover_workdir()

    def _compose_prefix(self) -> list[str]:
        files = ["-f", str(self.task.compose_file)]
        if self.compose_override:
            files.extend(["-f", str(self.compose_override)])
        return ["docker", "compose", "-p", self.project_name, *files]

    async def _discover_workdir(self) -> None:
        assert self.container_id
        output = await capture_process(
            [
                "docker",
                "inspect",
                "--format",
                "{{.Config.WorkingDir}}",
                self.container_id,
            ]
        )
        self.workdir = output.strip() or "/app"

    async def copy_instruction(self, instruction: str) -> None:
        local = self.trial_dir / "instruction.md"
        local.write_text(instruction, encoding="utf-8")
        await self.copy_to(local, "/tmp/shellbench-instruction.md")

    async def install_tests(self) -> None:
        await self.exec("rm -rf /tests && mkdir -p /tests", user="root")
        assert self.container_id
        result = await run_process(
            [
                "docker",
                "cp",
                f"{self.task.path / 'tests'}/.",
                f"{self.container_id}:/tests",
            ],
            stdout_path=self.trial_dir / "trial.log",
            stderr_path=self.trial_dir / "trial.log",
        )
        if result.returncode:
            raise DockerStartupError(f"docker cp tests exited {result.returncode}")

    async def copy_to(self, source: Path, target: str) -> None:
        assert self.container_id
        result = await run_process(
            ["docker", "cp", str(source), f"{self.container_id}:{target}"],
            stdout_path=self.trial_dir / "trial.log",
            stderr_path=self.trial_dir / "trial.log",
        )
        if result.returncode:
            raise DockerStartupError(f"docker cp exited {result.returncode}")

    async def collect_artifacts(self) -> None:
        assert self.container_id
        diff_path = self.artifacts_dir / "container-diff.txt"
        diff = await capture_process(["docker", "diff", self.container_id])
        diff_path.write_text(diff, encoding="utf-8")

        collected: list[dict[str, str]] = []
        if self.task.compose_file or self.task.mcp_servers:
            for source in ("/workspace", "/app/output", "/downloads", "/screenshots"):
                destination = (
                    self.artifacts_dir
                    / "workspace"
                    / source.lstrip("/").replace("/", "__")
                )
                if await self._container_path_exists(source):
                    await self._copy_from(source, destination)
                    collected.append(
                        {
                            "source": source,
                            "destination": str(destination.relative_to(self.trial_dir)),
                            "type": "directory",
                            "status": "collected",
                            "service": "main",
                        }
                    )
        else:
            changed_files = []
            for line in diff.splitlines():
                if len(line) < 3 or line[0] not in {"A", "C"}:
                    continue
                source = line[2:]
                if source == self.workdir or not source.startswith(
                    self.workdir.rstrip("/") + "/"
                ):
                    continue
                if await self._container_file_exists(source):
                    changed_files.append(source)
            for source in sorted(set(changed_files)):
                relative = source.removeprefix(self.workdir.rstrip("/") + "/")
                destination = self.artifacts_dir / "workspace" / relative
                await self._copy_from(source, destination)
                collected.append(
                    {
                        "source": source,
                        "destination": str(destination.relative_to(self.trial_dir)),
                        "type": "file",
                        "status": "collected",
                        "service": "main",
                    }
                )
        atomic_write_json(self.artifacts_dir / "manifest.json", collected)

    async def _container_path_exists(self, source: str) -> bool:
        result = await self.exec(
            f"test -e {json.dumps(source)}",
            user="root",
        )
        return result.returncode == 0

    async def _container_file_exists(self, source: str) -> bool:
        result = await self.exec(
            f"test -f {json.dumps(source)} -o -L {json.dumps(source)}",
            user="root",
        )
        return result.returncode == 0

    async def _copy_from(self, source: str, destination: Path) -> None:
        assert self.container_id
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = await run_process(
            [
                "docker",
                "cp",
                f"{self.container_id}:{source}",
                str(destination),
            ],
            stdout_path=self.trial_dir / "trial.log",
            stderr_path=self.trial_dir / "trial.log",
        )
        if result.returncode:
            raise DockerStartupError(
                f"docker cp from {source} exited {result.returncode}"
            )

    async def exec(
        self,
        command: str,
        *,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
        stdout_path: Path | None = None,
        stderr_path: Path | None = None,
        workdir: str | None = None,
        user: str | None = None,
    ) -> CommandResult:
        assert self.container_id
        args = ["docker", "exec", "-i"]
        if user:
            args.extend(["--user", user])
        args.extend(["--workdir", workdir or self.workdir])
        for key, value in (env or {}).items():
            args.extend(["--env", f"{key}={value}"])
        args.extend([self.container_id, "sh", "-lc", command])
        try:
            if timeout is None:
                return await run_process(
                    args,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
            async with asyncio.timeout(timeout):
                return await run_process(
                    args,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
        except TimeoutError:
            await run_process(
                ["docker", "kill", self.container_id],
                stdout_path=self.trial_dir / "trial.log",
                stderr_path=self.trial_dir / "trial.log",
            )
            raise

    async def stop(self) -> None:
        if self.task.compose_file:
            await run_process(
                self._compose_prefix()
                + ["down", "--volumes", "--remove-orphans", "--timeout", "10"],
                stdout_path=self.trial_dir / "environment-stop.log",
                stderr_path=self.trial_dir / "environment-stop.log",
            )
        else:
            await run_process(
                ["docker", "rm", "-f", self.container_name],
                stdout_path=self.trial_dir / "environment-stop.log",
                stderr_path=self.trial_dir / "environment-stop.log",
            )


async def run_trial(
    task: TaskSpec,
    run: RunSpec,
    *,
    job_dir: Path,
    toolchain_root: Path,
    proxy_url: str,
    proxy_key: str,
) -> dict[str, Any]:
    trial_suffix = uuid.uuid5(uuid.NAMESPACE_URL, f"{run.run_label}/{task.name}").hex[:7]
    trial_name = f"{task.name[:32].rstrip('_-')}__{trial_suffix}"
    trial_dir = job_dir / trial_name
    trial_dir.mkdir(parents=True, exist_ok=False)
    (trial_dir / "agent").mkdir()
    (trial_dir / "verifier").mkdir()
    (trial_dir / "artifacts").mkdir()
    atomic_write_json(trial_dir / "artifacts" / "manifest.json", [])

    started_at = utc_now()
    result = _initial_trial_result(task, run, trial_name, trial_dir, started_at)
    atomic_write_json(trial_dir / "config.json", result["config"])
    atomic_write_json(trial_dir / "lock.json", _trial_lock(task, run))
    atomic_write_json(trial_dir / "result.json", result)

    environment = DockerTaskEnvironment(
        task=task,
        trial_dir=trial_dir,
        container_name=f"sb-{trial_suffix}-{uuid.uuid4().hex[:6]}",
        project_name=f"sb-{trial_suffix}-{uuid.uuid4().hex[:6]}",
        toolchain_root=toolchain_root,
    )
    recorded_exception: BaseException | None = None
    agent_command = build_harness_command(
        run,
        proxy_url=proxy_url,
        proxy_key=proxy_key,
        mcp_servers=task.mcp_servers,
    )

    try:
        env_start = await environment.start()
        result["environment_setup"] = _timing(env_start)
        atomic_write_json(trial_dir / "result.json", result)

        await environment.copy_instruction(task.instruction)

        setup_started = utc_now()
        try:
            setup = await environment.exec(
                agent_command.setup_command,
                env=agent_command.env,
                timeout=300,
                stdout_path=trial_dir / "agent" / "setup-stdout.txt",
                stderr_path=trial_dir / "agent" / "setup-stderr.txt",
            )
            if setup.returncode:
                raise NativeEvalError(f"Agent setup exited {setup.returncode}")
        except TimeoutError as exc:
            raise AgentSetupTimeoutError("Agent setup timed out after 300s") from exc
        finally:
            result["agent_setup"] = {
                "started_at": setup_started,
                "finished_at": utc_now(),
            }
            atomic_write_json(trial_dir / "result.json", result)

        agent_started = utc_now()
        try:
            agent = await environment.exec(
                agent_command.run_command,
                env=agent_command.env,
                timeout=task.agent_timeout_sec,
                stdout_path=trial_dir / "agent" / "stdout.txt",
                stderr_path=trial_dir / "agent" / "stderr.txt",
            )
            if agent.returncode:
                raise NonZeroAgentExitCodeError(
                    f"Agent exited with code {agent.returncode}"
                )
        except TimeoutError:
            recorded_exception = NonZeroAgentExitCodeError(
                f"Agent timed out after {task.agent_timeout_sec}s"
            )
        except BaseException as exc:
            recorded_exception = exc
        finally:
            result["agent_execution"] = {
                "started_at": agent_started,
                "finished_at": utc_now(),
            }
            try:
                await environment.exec(
                    agent_command.cleanup_command,
                    env=agent_command.env,
                    timeout=60,
                    stdout_path=trial_dir / "agent" / "cleanup-stdout.txt",
                    stderr_path=trial_dir / "agent" / "cleanup-stderr.txt",
                )
            except Exception:
                pass
            result["agent_result"] = collect_agent_metrics(
                run.harness, trial_dir / "agent"
            )
            write_minimal_trajectory(task, run, trial_dir / "agent")
            await environment.collect_artifacts()
            atomic_write_json(trial_dir / "result.json", result)

        await environment.install_tests()
        verifier_started = utc_now()
        verifier_env = task.resolved_verifier_env()
        judge_env = {
            "AGENT_JUDGE_API_URL": (
                f"{proxy_url.rstrip('/')}/v1/chat/completions"
            ),
            "AGENT_JUDGE_MODEL": "sb-gpt55",
            "AGENT_JUDGE_API_KEY": proxy_key,
            "LLM_JUDGE_API_URL": f"{proxy_url.rstrip('/')}/v1",
            "LLM_JUDGE_MODEL": "sb-gpt55",
            "LLM_JUDGE_API_KEY": proxy_key,
            "OPENAI_BASE_URL": f"{proxy_url.rstrip('/')}/v1",
            "OPENROUTER_API_KEY": proxy_key,
        }
        for key, value in judge_env.items():
            if not verifier_env.get(key):
                verifier_env[key] = value
        verifier = await environment.exec(
            _verifier_command(task.verifier_command),
            env=verifier_env,
            timeout=task.verifier_timeout_sec,
            stdout_path=trial_dir / "verifier" / "test-stdout.txt",
            stderr_path=trial_dir / "verifier" / "test-stderr.txt",
        )
        result["verifier"] = {
            "started_at": verifier_started,
            "finished_at": utc_now(),
        }
        reward = read_reward(trial_dir / "verifier")
        result["verifier_result"] = {"rewards": reward}
        if verifier.returncode and recorded_exception is None and not reward:
            recorded_exception = RewardFileNotFoundError(
                f"Verifier exited {verifier.returncode} without a reward"
            )
    except BaseException as exc:
        if recorded_exception is None:
            recorded_exception = exc
    finally:
        try:
            await environment.stop()
        except Exception as exc:
            if recorded_exception is None:
                recorded_exception = exc
        result["finished_at"] = utc_now()
        if recorded_exception is not None:
            result["exception_info"] = exception_info(recorded_exception)
            (trial_dir / "exception.txt").write_text(
                "".join(
                    traceback.format_exception(
                        type(recorded_exception),
                        recorded_exception,
                        recorded_exception.__traceback__,
                    )
                ),
                encoding="utf-8",
            )
        atomic_write_json(trial_dir / "result.json", result)
    return result


def _initial_trial_result(
    task: TaskSpec,
    run: RunSpec,
    trial_name: str,
    trial_dir: Path,
    started_at: str,
) -> dict[str, Any]:
    config = {
        "task": {"path": str(task.path)},
        "trial_name": trial_name,
        "trials_dir": str(trial_dir.parent),
        "timeout_multiplier": 1.0,
        "agent": {
            "name": run.harness,
            "model_name": f"{run.provider}/{run.model_id}",
            "override_timeout_sec": task.agent_timeout_sec,
            "override_setup_timeout_sec": 300,
            "kwargs": {"native_runner": True},
        },
        "environment": {"type": "docker", "delete": True},
        "verifier": {"override_timeout_sec": task.verifier_timeout_sec},
        "artifacts": [],
        "extra_instruction_paths": [],
    }
    return {
        "id": str(uuid.uuid4()),
        "task_name": task.title,
        "trial_name": trial_name,
        "trial_uri": trial_dir.resolve().as_uri(),
        "task_id": {"path": str(task.path)},
        "source": "shellbench-native",
        "task_checksum": task.checksum,
        "config": config,
        "agent_info": {
            "name": run.harness,
            "version": run.harness_version,
            "model_info": {"name": run.model_id, "provider": run.provider},
        },
        "agent_result": None,
        "verifier_result": None,
        "verifier_environment_mode": "shared",
        "exception_info": None,
        "started_at": started_at,
        "finished_at": None,
        "environment_setup": None,
        "agent_setup": None,
        "agent_execution": None,
        "verifier": None,
        "step_results": None,
    }


def _trial_lock(task: TaskSpec, run: RunSpec) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "task": {
            "name": task.name,
            "version": None,
            "type": "local",
            "digest": f"sha256:{task.checksum}",
            "source": "shellbench-native",
            "path": str(task.path),
            "git_url": None,
            "git_commit_id": None,
        },
        "install_only": False,
        "timeout_multiplier": 1.0,
        "agent": {
            "name": run.harness,
            "model_name": f"{run.provider}/{run.model_id}",
            "kwargs": {"native_runner": True},
        },
        "skills": [],
        "environment": {"type": "docker", "delete": True},
        "verifier": {"environment_mode": "shared"},
    }


def _verifier_command(command: str) -> str:
    normalized = command.replace("tests/", "/tests/")
    if normalized.startswith("bash /tests/"):
        return normalized
    if normalized.startswith("bash tests/"):
        return normalized.replace("bash tests/", "bash /tests/", 1)
    return normalized


def read_reward(verifier_dir: Path) -> dict[str, float | int]:
    reward_json = verifier_dir / "reward.json"
    reward_text = verifier_dir / "reward.txt"
    if reward_json.is_file() and reward_json.stat().st_size:
        value = json.loads(reward_json.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            return {
                str(key): number
                for key, number in value.items()
                if isinstance(number, (int, float))
            }
    if reward_text.is_file() and reward_text.stat().st_size:
        return {"reward": float(reward_text.read_text(encoding="utf-8").strip())}
    raise RewardFileNotFoundError(
        f"No reward file found under {verifier_dir}"
    )


def collect_agent_metrics(harness: str, agent_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n_input_tokens": None,
        "n_cache_tokens": None,
        "n_output_tokens": None,
        "cost_usd": None,
        "rollout_details": None,
        "metadata": {"native_harness": harness},
    }
    candidates = {
        "openclaw": agent_dir / "openclaw.txt",
        "codex": agent_dir / "codex.txt",
        "claude-code": agent_dir / "claude-code.txt",
        "hermes": agent_dir / "hermes-session.jsonl",
    }
    path = candidates.get(harness)
    if path is None or not path.is_file():
        return metrics
    events = _load_json_events(path)
    for event in events:
        usage = _find_usage(event)
        if not usage:
            continue
        _merge_usage(metrics, usage)
    return metrics


def _load_json_events(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return [parsed]
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    events = []
    for line in text.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            events.append(item)
    return events


def _find_usage(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        for key in ("usage", "token_usage", "final_metrics"):
            candidate = value.get(key)
            if isinstance(candidate, dict):
                return candidate
        for item in value.values():
            found = _find_usage(item)
            if found:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_usage(item)
            if found:
                return found
    return None


def _merge_usage(metrics: dict[str, Any], usage: dict[str, Any]) -> None:
    aliases = {
        "n_input_tokens": (
            "input_tokens",
            "input",
            "prompt_tokens",
            "total_prompt_tokens",
        ),
        "n_cache_tokens": (
            "cache_read_input_tokens",
            "cacheRead",
            "cached_tokens",
            "total_cached_tokens",
        ),
        "n_output_tokens": (
            "output_tokens",
            "output",
            "completion_tokens",
            "total_completion_tokens",
        ),
        "cost_usd": ("cost_usd", "total_cost_usd", "total_cost"),
    }
    for target, keys in aliases.items():
        for key in keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                metrics[target] = value
                break


def write_minimal_trajectory(task: TaskSpec, run: RunSpec, agent_dir: Path) -> None:
    raw_files = sorted(
        path.name
        for path in agent_dir.iterdir()
        if path.is_file() and path.name != "trajectory.json"
    )
    trajectory = {
        "schema_version": "ATIF-v1.7",
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": run.harness,
            "version": run.harness_version,
            "model_name": f"{run.provider}/{run.model_id}",
        },
        "steps": [
            {
                "step_id": 1,
                "timestamp": utc_now(),
                "source": "user",
                "message": task.instruction,
                "tool_calls": [],
                "observation": None,
            }
        ],
        "final_metrics": {
            "total_steps": 1,
            "native_raw_trace_files": raw_files,
        },
    }
    atomic_write_json(agent_dir / "trajectory.json", trajectory)


def exception_info(exc: BaseException) -> dict[str, str]:
    return {
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
        "exception_traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        ),
        "occurred_at": utc_now(),
    }


def _timing(result: CommandResult) -> dict[str, str]:
    return {
        "started_at": result.started_at,
        "finished_at": result.finished_at,
    }


async def run_process(
    args: list[str],
    *,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
) -> CommandResult:
    started_at = utc_now()
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await asyncio.gather(
        _drain(process.stdout, stdout_path),
        _drain(process.stderr, stderr_path),
    )
    returncode = await process.wait()
    return CommandResult(returncode, started_at, utc_now())


async def capture_process(args: list[str]) -> str:
    process = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    if process.returncode:
        raise DockerStartupError(
            stderr.decode("utf-8", errors="replace").strip()
            or f"Command exited {process.returncode}"
        )
    return stdout.decode("utf-8", errors="replace")


async def _drain(
    stream: asyncio.StreamReader | None,
    output_path: Path | None,
) -> None:
    if stream is None:
        return
    handle = None
    try:
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            handle = output_path.open("ab")
        while chunk := await stream.read(64 * 1024):
            if handle:
                handle.write(chunk)
                handle.flush()
    finally:
        if handle:
            handle.close()


def atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
