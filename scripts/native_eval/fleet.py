from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Protocol, Sequence

from scripts.native_eval.checkpoint_loop import count_result_json
from scripts.native_eval.models import RunSpec
from scripts.native_eval.runtime import atomic_write_json, utc_now


RUN_SPEC_FIELDS = (
    "run_label",
    "harness",
    "harness_version",
    "model_slug",
    "model_id",
    "provider",
    "proxy_model_name",
    "repetition",
    "expected_task_count",
    "run_date",
)
RESUMABLE_STATUSES = {
    "planned",
    "leasing",
    "bootstrapping",
    "ready",
    "running",
    "exported",
    "stop_pending",
}
RERUN_STATUSES = {"failed", "lease_lost"}
ACTIVE_RUN_STATUSES = {"leasing", "bootstrapping", "ready", "running"}
CLEANUP_STATUSES = {"exported", "stop_pending"}
REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
# Crabbox's coordinator release path retries five 60-second requests with
# bounded backoff. Give it enough time to finish instead of leaking live AWS
# leases after a verified export.
CRABBOX_STOP_TIMEOUT_SECONDS = 6 * 60
CRABBOX_READY_ATTEMPTS = 30
CRABBOX_READY_BACKOFF_SECONDS = 10
CRABBOX_INSPECT_ATTEMPTS = 4


class CommandExecutor(Protocol):
    def run(
        self,
        command: Sequence[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]: ...


class SubprocessExecutor:
    def run(
        self,
        command: Sequence[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            list(command),
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture_output else None,
            stderr=subprocess.PIPE if capture_output else None,
        )

    def run_with_timeout(
        self,
        command: Sequence[str],
        *,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                list(command),
                check=False,
                text=True,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or f"timed out after {timeout:g}s"
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            return subprocess.CompletedProcess(
                list(command),
                124,
                stdout=stdout,
                stderr=stderr,
            )


class FleetError(RuntimeError):
    pass


class LeaseUnavailableError(FleetError):
    pass


class LeaseNotReadyError(FleetError):
    pass


@dataclass(frozen=True)
class Lease:
    lease_id: str
    slug: str
    host: str
    user: str
    port: int
    identity_file: Path
    instance_type: str
    region: str

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"

    @classmethod
    def from_inspect(cls, value: dict[str, Any], *, region: str) -> Lease:
        state = str(value.get("state") or "")
        if not value.get("ready") and state in {"active", "provisioning"}:
            lease_id = value.get("id") or value.get("slug") or "unknown"
            raise LeaseNotReadyError(f"lease {lease_id} is {state} but not ready")
        if state != "active":
            lease_id = value.get("id") or value.get("slug") or "unknown"
            raise LeaseUnavailableError(f"lease {lease_id} is no longer active")
        required = ("id", "slug", "sshHost", "sshUser", "sshPort", "sshKey")
        missing = [field for field in required if not value.get(field)]
        if missing:
            raise FleetError(f"Crabbox inspect omitted fields: {', '.join(missing)}")
        return cls(
            lease_id=str(value["id"]),
            slug=str(value["slug"]),
            host=str(value["sshHost"]),
            user=str(value["sshUser"]),
            port=int(value["sshPort"]),
            identity_file=Path(str(value["sshKey"])),
            instance_type=str(value.get("serverType") or ""),
            region=region,
        )

    @classmethod
    def from_manifest(cls, value: dict[str, Any]) -> Lease:
        required = ("id", "slug", "host", "ssh_user", "ssh_port", "identity_file")
        missing = [field for field in required if not value.get(field)]
        if missing:
            raise FleetError(f"stored lease omitted fields: {', '.join(missing)}")
        return cls(
            lease_id=str(value["id"]),
            slug=str(value["slug"]),
            host=str(value["host"]),
            user=str(value["ssh_user"]),
            port=int(value["ssh_port"]),
            identity_file=Path(str(value["identity_file"])),
            instance_type=str(value.get("instance_type") or ""),
            region=str(value.get("region") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.lease_id,
            "slug": self.slug,
            "provider": "aws",
            "instance_type": self.instance_type,
            "region": self.region,
            "host": self.host,
            "ssh_user": self.user,
            "ssh_port": self.port,
            "identity_file": str(self.identity_file),
            "state": "active",
        }


@dataclass(frozen=True)
class FleetConfig:
    run_index: Path
    local_root: Path
    runner_root: Path
    task_archive: Path
    env_file: Path
    max_leases: int = 10
    max_attempts: int = 2
    task_concurrency: int = 16
    model_max_runs: dict[str, int] = field(default_factory=dict)
    provider_max_runs: dict[str, int] = field(default_factory=dict)
    model_task_concurrency: dict[str, int] = field(default_factory=dict)
    crabbox_bin: str = "crabbox"
    python_bin: str = sys.executable
    machine_class: str = "beast"
    instance_type: str = "c7a.24xlarge"
    market: str = "on-demand"
    region: str = "eu-west-1"
    ttl: str = "12h"
    idle_timeout: str = "12h"
    remote_root: str = "/work/crabbox/shellbench-native"
    checkpoint_poll_seconds: int = 30
    warmup_capacity_attempts: int = 12
    warmup_capacity_backoff_seconds: float = 5.0
    runner_archive: Path | None = None
    runner_commit: str | None = None
    harbor_reference_commit: str = ""
    judge_model_id: str = ""
    execution_mode: str = "native"
    parity_validated: bool = False
    parity_validated_routes: frozenset[tuple[str, str]] = frozenset()


class RunIndexStore:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._thread_lock = threading.RLock()
        lock_path = self.path.with_name(f"{self.path.name}.fleet.lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = lock_path.open("a+")
        try:
            fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock_handle.close()
            raise FleetError(f"another fleet controller owns {lock_path}") from exc
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def close(self) -> None:
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
        self._lock_handle.close()

    def __enter__(self) -> RunIndexStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def save(self) -> None:
        with self._thread_lock:
            self.data["updated_at_utc"] = utc_now()
            atomic_write_json(self.path, self.data)

    def update_root(self, **changes: Any) -> None:
        with self._thread_lock:
            self.data.update(changes)
            self.save()

    def get(self, run_label: str) -> dict[str, Any]:
        with self._thread_lock:
            for entry in self.data["runs"]:
                if entry["run_label"] == run_label:
                    return copy.deepcopy(entry)
        raise KeyError(run_label)

    def update(self, run_label: str, **changes: Any) -> dict[str, Any]:
        with self._thread_lock:
            for entry in self.data["runs"]:
                if entry["run_label"] == run_label:
                    entry.update(changes)
                    entry["updated_at_utc"] = utc_now()
                    self.save()
                    return copy.deepcopy(entry)
        raise KeyError(run_label)

    def append(self, entry: dict[str, Any]) -> None:
        with self._thread_lock:
            labels = {item["run_label"] for item in self.data["runs"]}
            if entry["run_label"] in labels:
                raise FleetError(f"duplicate run label: {entry['run_label']}")
            self.data["runs"].append(entry)
            self.save()

    def labels_with_status(self, statuses: set[str]) -> list[str]:
        with self._thread_lock:
            return [
                entry["run_label"] for entry in self.data["runs"] if entry.get("status") in statuses
            ]

    def all_entries(self) -> list[dict[str, Any]]:
        with self._thread_lock:
            return copy.deepcopy(self.data["runs"])


class FleetController:
    def __init__(
        self,
        config: FleetConfig,
        *,
        executor: CommandExecutor | None = None,
    ) -> None:
        if config.max_leases < 1:
            raise ValueError("max_leases must be at least 1")
        if config.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if config.task_concurrency < 1:
            raise ValueError("task_concurrency must be at least 1")
        if config.warmup_capacity_attempts < 1:
            raise ValueError("warmup_capacity_attempts must be at least 1")
        if config.warmup_capacity_backoff_seconds < 0:
            raise ValueError("warmup_capacity_backoff_seconds cannot be negative")
        _validate_model_values(config.model_max_runs, "model_max_runs")
        _validate_model_values(config.provider_max_runs, "provider_max_runs")
        _validate_model_values(config.model_task_concurrency, "model_task_concurrency")
        self.config = config
        self.executor = executor or SubprocessExecutor()
        self.store: RunIndexStore | None = None
        self.runner_archive: Path | None = None
        self.runner_commit = ""
        self.runner_archive_sha256 = ""
        self.task_archive_sha256 = ""
        self.crabbox_cli_version = ""
        self._dispatch_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()

    def run(self) -> int:
        self._prepare_local_layout()
        with RunIndexStore(self.config.run_index) as store:
            self.store = store
            self._validate_plan()
            self._prepare_inputs()
            self._record_fleet_metadata()
            self._resume_recovery_entries()
            self._schedule_existing_reruns()

            attempted_labels: set[str] = set()
            with ThreadPoolExecutor(max_workers=self.config.max_leases) as pool:
                futures: dict[Future[bool], str] = {}
                while True:
                    while len(futures) < self.config.max_leases:
                        label = self._next_schedulable_label(
                            attempted_labels,
                            set(futures.values()),
                        )
                        if label is None:
                            break
                        attempted_labels.add(label)
                        futures[pool.submit(self._execute_entry, label)] = label

                    if not futures:
                        break

                    completed, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in completed:
                        label = futures.pop(future)
                        try:
                            future.result()
                        except Exception as exc:
                            self._mark_recovery_required(
                                label,
                                f"unexpected controller error: {exc}",
                            )
                    self._schedule_existing_reruns()

            unfinished = self.store.labels_with_status(RESUMABLE_STATUSES | {"recovery_required"})
            return 1 if unfinished or not self._matrix_satisfied() else 0

    @property
    def _store(self) -> RunIndexStore:
        if self.store is None:
            raise RuntimeError("fleet store is not open")
        return self.store

    def _prepare_local_layout(self) -> None:
        for relative in ("raw", "logs", "manifests"):
            (self.config.local_root / relative).mkdir(parents=True, exist_ok=True)

    def _validate_plan(self) -> None:
        suite_expected = int(self._store.data["expected_task_count"])
        for entry in self._store.all_entries():
            run = self._run_spec(entry)
            task_names = entry.get("task_names")
            phase = str(entry.get("phase") or "full")
            if task_names is None:
                expected = suite_expected
            elif not isinstance(task_names, list) or not all(
                isinstance(name, str) and name for name in task_names
            ):
                raise FleetError(
                    f"{run.run_label} task_names must be a list of non-empty strings"
                )
            elif len(set(task_names)) != len(task_names):
                raise FleetError(f"{run.run_label} task_names contains duplicates")
            else:
                expected = len(task_names)
                if phase == "r0":
                    if run.repetition != 0:
                        raise FleetError(f"{run.run_label} r0 must use repetition 0")
                    if len(task_names) != 10:
                        raise FleetError(f"{run.run_label} r0 must select 10 tasks")
                    if entry.get("leaderboard_eligible") is not False:
                        raise FleetError(
                            f"{run.run_label} r0 must be leaderboard-ineligible"
                        )
                    if not entry.get("qualification_family"):
                        raise FleetError(
                            f"{run.run_label} r0 lacks qualification_family"
                        )
                elif not entry.get("rerun_of_canonical_run"):
                    raise FleetError(
                        f"{run.run_label} task subset lacks rerun_of_canonical_run"
                    )
            if phase == "r0" and task_names is None:
                raise FleetError(f"{run.run_label} r0 lacks task_names")
            if phase != "r0" and run.repetition == 0:
                raise FleetError(f"{run.run_label} repetition 0 requires phase r0")
            if run.expected_task_count != expected:
                raise FleetError(
                    f"{run.run_label} expects {run.expected_task_count} tasks, "
                    f"plan selects {expected}"
                )
            if run.provider == "openai":
                reasoning_effort = str(entry.get("reasoning_effort") or "")
                if reasoning_effort not in REASONING_EFFORTS:
                    raise FleetError(
                        f"{run.run_label} must set reasoning_effort to "
                        "low, medium, high, or xhigh"
                    )
            judge_reasoning_effort = str(
                entry.get("judge_reasoning_effort") or ""
            )
            if judge_reasoning_effort not in REASONING_EFFORTS:
                raise FleetError(
                    f"{run.run_label} must set judge_reasoning_effort to "
                    "low, medium, high, or xhigh"
                )

    def _prepare_inputs(self) -> None:
        for path, description in (
            (self.config.task_archive, "task archive"),
            (self.config.env_file, "environment file"),
        ):
            if not path.is_file():
                raise FleetError(f"{description} does not exist: {path}")

        if self.config.runner_commit:
            self.runner_commit = self.config.runner_commit
        else:
            result = self._checked(
                [
                    "git",
                    "-C",
                    str(self.config.runner_root),
                    "rev-parse",
                    "HEAD",
                ],
                capture_output=True,
                description="resolve runner commit",
            )
            self.runner_commit = result.stdout.strip()

        if self.config.runner_archive:
            self.runner_archive = self.config.runner_archive
        else:
            archive = (
                self.config.local_root
                / "manifests"
                / f"native-runner-{self.runner_commit[:12]}.tar.gz"
            )
            if not archive.exists():
                self._checked(
                    [
                        "git",
                        "-C",
                        str(self.config.runner_root),
                        "archive",
                        "--format=tar.gz",
                        f"--output={archive}",
                        self.runner_commit,
                    ],
                    capture_output=True,
                    description="archive pinned runner",
                )
            self.runner_archive = archive

        for archive in (self.runner_archive, self.config.task_archive):
            if archive is None or not archive.is_file():
                raise FleetError(f"archive does not exist: {archive}")
            try:
                with tarfile.open(archive, "r:gz") as handle:
                    handle.getmembers()
            except tarfile.TarError as exc:
                raise FleetError(f"invalid archive: {archive}") from exc
        self.runner_archive_sha256 = _sha256(self.runner_archive)
        self.task_archive_sha256 = _sha256(self.config.task_archive)
        version = self._checked(
            [self.config.crabbox_bin, "--version"],
            capture_output=True,
            description="resolve Crabbox CLI version",
        )
        self.crabbox_cli_version = version.stdout.strip()

    def _record_fleet_metadata(self) -> None:
        assert self.runner_archive is not None
        metadata = dict(self._store.data.get("fleet") or {})
        metadata.update(
            {
                "runner_commit": self.runner_commit,
                "runner_archive_sha256": self.runner_archive_sha256,
                "task_archive_sha256": self.task_archive_sha256,
                "harbor_reference_commit": self.config.harbor_reference_commit,
                "judge_model_id": self.config.judge_model_id,
                "execution_mode": self.config.execution_mode,
                "provider": "aws",
                "machine_class": self.config.machine_class,
                "instance_type": self.config.instance_type,
                "market": self.config.market,
                "region": self.config.region,
                "max_leases": self.config.max_leases,
                "task_concurrency": self.config.task_concurrency,
                "model_max_runs": self.config.model_max_runs,
                "provider_max_runs": self.config.provider_max_runs,
                "model_task_concurrency": self.config.model_task_concurrency,
                "warmup_capacity_attempts": self.config.warmup_capacity_attempts,
                "warmup_capacity_backoff_seconds": (
                    self.config.warmup_capacity_backoff_seconds
                ),
                "crabbox_cli_version": self.crabbox_cli_version,
                "controller_started_at_utc": utc_now(),
            }
        )
        self._store.update_root(fleet=metadata)

    def _execute_entry(self, run_label: str) -> bool:
        entry = self._store.get(run_label)
        run = self._run_spec(entry)
        try:
            verified, result_count, artifacts = self._verify_final(run.run_label)
            if verified:
                self._store.update(
                    run.run_label,
                    status="exported",
                    verified_final_export=True,
                    final_result_count=result_count,
                    artifacts=artifacts,
                )
                return self._finish_exported(
                    self._store.get(run.run_label),
                    run,
                )
            if entry["status"] in {"exported", "stop_pending"}:
                return self._finish_exported(entry, run)

            lease = self._ensure_lease(entry)
            remote_state = self._probe_remote(lease, run.run_label)
            if remote_state == "missing":
                if self._local_artifacts(run.run_label):
                    raise FleetError("local artifacts exist but the remote run state is missing")
                if not entry.get("bootstrapped_at_utc"):
                    self._hydrate_lease(lease)
                self._dispatch(lease, run)
            elif remote_state == "stale":
                raise FleetError("remote run state is stale and cannot be overwritten")

            self._store.update(
                run.run_label,
                status="running",
                started_at_utc=entry.get("started_at_utc") or utc_now(),
                last_error=None,
            )
            checkpoint = self._run_checkpoint_loop(lease, run.run_label)
            verified, result_count, artifacts = self._verify_final(run.run_label)
            if not verified:
                self._mark_recovery_required(
                    run.run_label,
                    f"checkpoint loop exited {checkpoint.returncode} without a verified final export",
                )
                return False

            self._store.update(
                run.run_label,
                status="exported",
                verified_final_export=True,
                final_result_count=result_count,
                run_exit_code=checkpoint.returncode,
                artifacts=artifacts,
            )
            return self._finish_exported(self._store.get(run.run_label), run)
        except LeaseUnavailableError as exc:
            self._store.update(
                run.run_label,
                status="lease_lost",
                finished_at_utc=utc_now(),
                last_error=str(exc),
            )
            self._schedule_rerun(self._store.get(run.run_label))
            return False
        except (FleetError, OSError, subprocess.SubprocessError, ValueError) as exc:
            current = self._store.get(run.run_label)
            if current.get("lease") or current.get("requested_lease_slug"):
                self._mark_recovery_required(run.run_label, str(exc))
            else:
                self._store.update(
                    run.run_label,
                    status="failed",
                    finished_at_utc=utc_now(),
                    last_error=str(exc),
                )
                self._schedule_rerun(self._store.get(run.run_label))
            return False

    def _ensure_lease(self, entry: dict[str, Any]) -> Lease:
        lease_value = entry.get("lease")
        if lease_value:
            try:
                stored = Lease.from_manifest(lease_value)
            except FleetError:
                stored = None
            if stored is not None and self._ssh_reachable(stored):
                return self._detect_region(stored)
            identifier = str(lease_value["id"])
            try:
                return self._inspect_lease(
                    identifier,
                    required=True,
                    region_hint=str(lease_value.get("region") or ""),
                )
            except LeaseNotReadyError as exc:
                if stored is None:
                    stored = Lease.from_manifest(lease_value)
                if self._ssh_reachable(stored):
                    return self._detect_region(stored)
                raise LeaseUnavailableError(
                    f"stored lease {identifier} lost readiness and SSH access"
                ) from exc

        slug = str(entry.get("requested_lease_slug") or self._lease_slug(entry["run_label"]))
        self._store.update(
            entry["run_label"],
            status="leasing",
            requested_lease_slug=slug,
        )
        try:
            existing = self._inspect_lease(slug, required=False)
        except LeaseNotReadyError:
            existing = self._wait_for_lease_ready(slug)
        if existing is None:
            command = [
                "env",
                f"CRABBOX_CAPACITY_MARKET={self.config.market}",
                self.config.crabbox_bin,
                "warmup",
                "--provider",
                "aws",
                "--class",
                self.config.machine_class,
                "--slug",
                slug,
                "--ttl",
                self.config.ttl,
                "--idle-timeout",
                self.config.idle_timeout,
            ]
            if self.config.instance_type:
                command.extend(["--type", self.config.instance_type])
            self._warmup_lease(command, slug)
            existing = self._wait_for_lease_ready(slug)
        self._store.update(
            entry["run_label"],
            status="bootstrapping",
            lease={**existing.to_dict(), "started_at_utc": utc_now()},
        )
        return existing

    def _wait_for_lease_ready(self, identifier: str) -> Lease:
        for attempt in range(1, CRABBOX_READY_ATTEMPTS + 1):
            try:
                lease = self._inspect_lease(identifier, required=True)
            except LeaseNotReadyError:
                if attempt == CRABBOX_READY_ATTEMPTS:
                    raise
                time.sleep(CRABBOX_READY_BACKOFF_SECONDS)
                continue
            assert lease is not None
            return lease
        raise AssertionError("unreachable")

    def _warmup_lease(self, command: Sequence[str], slug: str) -> None:
        for attempt in range(1, self.config.warmup_capacity_attempts + 1):
            result = self.executor.run(command, capture_output=True)
            if result.returncode == 0:
                return
            detail = (result.stderr or result.stdout or "").strip()
            capacity_pending = all(
                marker in detail.lower()
                for marker in ("429", "cost_limit_exceeded", "active lease limit")
            )
            if not capacity_pending or attempt == self.config.warmup_capacity_attempts:
                suffix = f": {detail[:500]}" if detail else ""
                raise FleetError(
                    f"lease Crabbox {slug} failed with exit {result.returncode}{suffix}"
                )
            if self.config.warmup_capacity_backoff_seconds:
                time.sleep(self.config.warmup_capacity_backoff_seconds)

    def _inspect_lease(
        self,
        identifier: str,
        *,
        required: bool,
        region_hint: str = "",
    ) -> Lease | None:
        command = [
            self.config.crabbox_bin,
            "inspect",
            "--provider",
            "aws",
            "--id",
            identifier,
            "--json",
        ]
        for attempt in range(1, CRABBOX_INSPECT_ATTEMPTS + 1):
            result = self.executor.run(command, capture_output=True)
            if result.returncode == 0:
                break
            detail = (result.stderr or result.stdout or "").strip()
            transient = any(
                marker in detail.lower()
                for marker in (
                    "http 500",
                    "http 502",
                    "http 503",
                    "http 504",
                    "error code: 1101",
                    "context deadline exceeded",
                    "connection timed out",
                    "connection reset",
                )
            )
            if not transient or attempt == CRABBOX_INSPECT_ATTEMPTS:
                break
            time.sleep(attempt * 5)
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()
            missing = any(
                marker in detail.lower()
                for marker in ("not found", "not_found", "http 404", "no claim", "unknown lease")
            )
            if not missing:
                raise FleetError(
                    f"Crabbox inspect failed for {identifier}: "
                    f"{detail[:500] or f'exit {result.returncode}'}"
                )
            if required:
                raise LeaseUnavailableError(f"Crabbox lease {identifier} could not be inspected")
            return None
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise FleetError(f"invalid Crabbox inspect JSON for {identifier}") from exc
        try:
            lease = Lease.from_inspect(
                value,
                region=region_hint or self.config.region,
            )
        except LeaseUnavailableError:
            if required:
                raise
            return None
        return self._detect_region(lease)

    def _ssh_reachable(self, lease: Lease) -> bool:
        with self._dispatch_lock:
            for attempt in range(1, 5):
                result = self.executor.run(
                    self._ssh_command(lease, ["true"]),
                    capture_output=True,
                )
                if result.returncode == 0:
                    return True
                if attempt < 4:
                    time.sleep(attempt * 5)
        return False

    def _detect_region(self, lease: Lease) -> Lease:
        script = """
set -eu
token=$(curl -fsS --max-time 3 -X PUT \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' \
  http://169.254.169.254/latest/api/token 2>/dev/null || true)
[ -n "$token" ] || exit 0
curl -fsS --max-time 3 \
  -H "X-aws-ec2-metadata-token: $token" \
  http://169.254.169.254/latest/dynamic/instance-identity/document \
  2>/dev/null \
  | sed -n 's/.*"region"[[:space:]]*:[[:space:]]*"\\([^"]*\\)".*/\\1/p'
"""
        result = self.executor.run(
            self._ssh_command(
                lease,
                ["bash", "-c", script, "fleet-region"],
            ),
            capture_output=True,
        )
        region = result.stdout.strip() if result.returncode == 0 else ""
        if not region:
            return lease
        return Lease(
            lease_id=lease.lease_id,
            slug=lease.slug,
            host=lease.host,
            user=lease.user,
            port=lease.port,
            identity_file=lease.identity_file,
            instance_type=lease.instance_type,
            region=region,
        )

    def _hydrate_lease(self, lease: Lease) -> None:
        assert self.runner_archive is not None
        remote_runner_archive = f"/tmp/{lease.slug}-runner.tar.gz"
        remote_task_archive = f"/tmp/{lease.slug}-tasks.tar.gz"
        remote_env = f"/tmp/{lease.slug}-provider.env"
        self._checked(
            self._ssh_command(
                lease,
                ["mkdir", "-p", self.config.remote_root],
            ),
            capture_output=True,
            description=f"prepare {lease.slug}",
        )
        for local_path, remote_path in (
            (self.runner_archive, remote_runner_archive),
            (self.config.task_archive, remote_task_archive),
            (self.config.env_file, remote_env),
        ):
            self._checked(
                self._scp_command(lease, local_path, remote_path),
                capture_output=True,
                description=f"sync {local_path.name} to {lease.slug}",
            )

        script = """
set -euo pipefail
root=$1
runner_archive=$2
task_archive=$3
source_env=$4
runner_commit=$5
runner_sha256=$6
task_sha256=$7
printf '%s  %s\n' "$runner_sha256" "$runner_archive" | sha256sum -c -
printf '%s  %s\n' "$task_sha256" "$task_archive" | sha256sum -c -
rm -rf "$root/runner.new" "$root/public-tasks.new"
mkdir -p "$root/runner.new" "$root/public-tasks.new" "$root/secrets"
tar -xzf "$runner_archive" -C "$root/runner.new"
tar -xzf "$task_archive" -C "$root/public-tasks.new"
install -m 0600 "$source_env" "$root/secrets/provider.env"
rm -rf "$root/runner" "$root/public-tasks"
mv "$root/runner.new" "$root/runner"
mv "$root/public-tasks.new" "$root/public-tasks"
printf '%s\n' "$runner_commit" > "$root/runner.commit"
rm -f "$runner_archive" "$task_archive" "$source_env"
bash "$root/runner/scripts/native_eval/bootstrap_beast.sh"
"""
        self._checked(
            self._ssh_command(
                lease,
                [
                    "bash",
                    "-c",
                    script,
                    "fleet-hydrate",
                    self.config.remote_root,
                    remote_runner_archive,
                    remote_task_archive,
                    remote_env,
                    self.runner_commit,
                    self.runner_archive_sha256,
                    self.task_archive_sha256,
                ],
            ),
            capture_output=False,
            description=f"bootstrap {lease.slug}",
        )
        self._store.update(
            self._run_label_for_lease(lease.lease_id),
            status="ready",
            bootstrapped_at_utc=utc_now(),
        )

    def _probe_remote(self, lease: Lease, run_label: str) -> str:
        script = """
set -eu
root=$1
label=$2
state="/tmp/shellbench-runs/$label"
job="$root/results/jobs/$label"
if [ -f "$state/done" ]; then
  printf 'done\n'
elif [ -f "$state/pid" ] && kill -0 "$(cat "$state/pid")" 2>/dev/null; then
  printf 'running\n'
elif [ -e "$state" ] || [ -e "$job" ]; then
  printf 'stale\n'
else
  printf 'missing\n'
fi
"""
        result = self._checked(
            self._ssh_command(
                lease,
                [
                    "bash",
                    "-c",
                    script,
                    "fleet-probe",
                    self.config.remote_root,
                    run_label,
                ],
            ),
            capture_output=True,
            description=f"probe {run_label}",
        )
        state = result.stdout.strip()
        if state not in {"missing", "running", "done", "stale"}:
            raise FleetError(f"unexpected remote state for {run_label}: {state!r}")
        return state

    def _dispatch(self, lease: Lease, run: RunSpec) -> None:
        script = """
set -euo pipefail
root=$1
label=$2
crabbox_cli_version=$3
crabbox_slug=$4
crabbox_lease_id=$5
crabbox_instance_type=$6
crabbox_ip=$7
crabbox_region=$8
runner_commit=$9
shift 9
harbor_reference_commit=$1
judge_model_id=$2
execution_mode=$3
reasoning_effort=$4
judge_reasoning_effort=$5
qualification_family=$6
run_phase=$7
leaderboard_eligible=$8
exclusion_reason=$9
shift 9
parity_validated=$1
parity_validation_json=$2
shift 2
mkdir -p "$root/run-logs"
stdout="$root/run-logs/$label.stdout.log"
stderr="$root/run-logs/$label.stderr.log"
nohup env \
  "CRABBOX_CLI_VERSION=$crabbox_cli_version" \
  "CRABBOX_SLUG=$crabbox_slug" \
  "CRABBOX_LEASE_ID=$crabbox_lease_id" \
  "CRABBOX_INSTANCE_TYPE=$crabbox_instance_type" \
  "CRABBOX_IP=$crabbox_ip" \
  "CRABBOX_REGION=$crabbox_region" \
  "SHELLBENCH_RUNNER_COMMIT=$runner_commit" \
  "SHELLBENCH_HARBOR_REFERENCE_COMMIT=$harbor_reference_commit" \
  "SHELLBENCH_JUDGE_MODEL_ID=$judge_model_id" \
  "SHELLBENCH_EXECUTION_MODE=$execution_mode" \
  "SHELLBENCH_REASONING_EFFORT=$reasoning_effort" \
  "SHELLBENCH_JUDGE_REASONING_EFFORT=$judge_reasoning_effort" \
  "SHELLBENCH_QUALIFICATION_FAMILY=$qualification_family" \
  "SHELLBENCH_RUN_PHASE=$run_phase" \
  "SHELLBENCH_LEADERBOARD_ELIGIBLE=$leaderboard_eligible" \
  "SHELLBENCH_EXCLUSION_REASON=$exclusion_reason" \
  "SHELLBENCH_PARITY_VALIDATED=$parity_validated" \
  "SHELLBENCH_PARITY_VALIDATION_JSON=$parity_validation_json" \
  "$root/runner/scripts/native_eval/remote_run.sh" "$@" \
  >"$stdout" 2>"$stderr" </dev/null &
pid=$!
sleep 1
kill -0 "$pid"
printf '%s\n' "$pid"
"""
        entry = self._store.get(run.run_label)
        args = [
            self.config.remote_root,
            f"{self.config.remote_root}/public-tasks/combined tasks/tasks",
            f"{self.config.remote_root}/secrets/provider.env",
            run.run_label,
            run.harness,
            run.model_slug,
            str(run.repetition),
            str(run.expected_task_count),
            str(self._store.data["public_tasks_commit"]),
            run.run_date,
            str(self._task_concurrency(run.model_slug)),
            run.harness_version,
            run.model_id,
            run.provider,
            run.proxy_model_name,
            str(entry.get("rerun_of_canonical_run") or ""),
            *[str(name) for name in entry.get("task_names") or []],
        ]
        reasoning_effort = str(entry.get("reasoning_effort") or "")
        judge_reasoning_effort = str(
            entry.get("judge_reasoning_effort") or reasoning_effort
        )
        qualification_family = str(entry.get("qualification_family") or "")
        run_phase = str(entry.get("phase") or "full")
        leaderboard_eligible = (
            "false" if entry.get("leaderboard_eligible") is False else ""
        )
        exclusion_reason = str(entry.get("exclusion_reason") or "")
        parity_validation = ""
        if (run.harness, run.model_slug) in self.config.parity_validated_routes:
            parity_validation = json.dumps(
                {
                    "scope": {
                        "harness": run.harness,
                        "model_slug": run.model_slug,
                    },
                    "validated": True,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        command = self._ssh_command(
            lease,
            [
                "bash",
                "-c",
                script,
                "fleet-dispatch",
                self.config.remote_root,
                run.run_label,
                self.crabbox_cli_version,
                lease.slug,
                lease.lease_id,
                lease.instance_type,
                lease.host,
                lease.region,
                self.runner_commit,
                self.config.harbor_reference_commit,
                self.config.judge_model_id,
                self.config.execution_mode,
                reasoning_effort,
                judge_reasoning_effort,
                qualification_family,
                run_phase,
                leaderboard_eligible,
                exclusion_reason,
                str(self.config.parity_validated).lower(),
                parity_validation,
                *args,
            ],
        )
        with self._dispatch_lock:
            for attempt in range(1, 5):
                result = self.executor.run(command, capture_output=True)
                if result.returncode == 0:
                    break
                detail = (result.stderr or result.stdout or "").strip()
                retryable = result.returncode == 255 and any(
                    marker in detail.lower()
                    for marker in (
                        "operation timed out",
                        "connection timed out",
                        "connection refused",
                        "connection closed",
                        "banner exchange",
                    )
                )
                if not retryable or attempt == 4:
                    suffix = f": {detail[:500]}" if detail else ""
                    raise FleetError(
                        f"dispatch {run.run_label} failed with exit "
                        f"{result.returncode}{suffix}"
                    )
                time.sleep(attempt * 5)
        self._store.update(
            run.run_label,
            dispatched_at_utc=utc_now(),
            task_concurrency=self._task_concurrency(run.model_slug),
        )

    def _next_schedulable_label(
        self,
        attempted_labels: set[str],
        in_flight_labels: set[str],
    ) -> str | None:
        entries = self._store.all_entries()
        candidates = [
            entry
            for entry in entries
            if entry.get("status") in RESUMABLE_STATUSES
            and entry["run_label"] not in attempted_labels
        ]
        for statuses in (CLEANUP_STATUSES, ACTIVE_RUN_STATUSES):
            for entry in candidates:
                if entry.get("lease") and entry.get("status") in statuses:
                    return str(entry["run_label"])

        active_entries = [
            entry
            for entry in entries
            if entry.get("lease") and entry.get("status") in ACTIVE_RUN_STATUSES
        ]
        cleanup_entries = [
            entry
            for entry in entries
            if entry.get("lease") and entry.get("status") in CLEANUP_STATUSES
        ]
        pending_reservations = [
            entry
            for entry in entries
            if entry["run_label"] in in_flight_labels
            and not entry.get("lease")
            and (
                entry.get("status") == "planned"
                or entry.get("status") in ACTIVE_RUN_STATUSES
            )
        ]
        occupied_slots = (
            len(active_entries) + len(cleanup_entries) + len(pending_reservations)
        )
        if occupied_slots >= self.config.max_leases:
            return None

        active_models = Counter(
            self._run_spec(entry).model_slug
            for entry in [*active_entries, *pending_reservations]
        )
        active_providers = Counter(
            self._run_spec(entry).provider
            for entry in [*active_entries, *pending_reservations]
        )
        for entry in candidates:
            pending = entry.get("status") == "planned" or (
                not entry.get("lease") and entry.get("status") in ACTIVE_RUN_STATUSES
            )
            if not pending:
                continue
            run = self._run_spec(entry)
            model_slug = run.model_slug
            model_limit = self.config.model_max_runs.get(model_slug)
            provider_limit = self.config.provider_max_runs.get(run.provider)
            model_available = (
                model_limit is None or active_models[model_slug] < model_limit
            )
            provider_available = (
                provider_limit is None
                or active_providers[run.provider] < provider_limit
            )
            if model_available and provider_available:
                return str(entry["run_label"])
        return None

    def _task_concurrency(self, model_slug: str) -> int:
        return self.config.model_task_concurrency.get(
            model_slug,
            self.config.task_concurrency,
        )

    def _run_checkpoint_loop(
        self,
        lease: Lease,
        run_label: str,
    ) -> subprocess.CompletedProcess[str]:
        control_hash = hashlib.sha256(lease.lease_id.encode()).hexdigest()[:16]
        control_path = Path("/tmp") / f"shellbench-fleet-{control_hash}.sock"
        return self.executor.run(
            [
                self.config.python_bin,
                "-m",
                "scripts.native_eval.checkpoint_loop",
                "--target",
                lease.target,
                "--port",
                str(lease.port),
                "--identity-file",
                str(lease.identity_file),
                "--control-path",
                str(control_path),
                "--runner-root",
                f"{self.config.remote_root}/runner",
                "--remote-root",
                self.config.remote_root,
                "--run-label",
                run_label,
                "--local-root",
                str(self.config.local_root),
                "--poll-seconds",
                str(self.config.checkpoint_poll_seconds),
            ],
            capture_output=False,
        )

    def _verify_final(self, run_label: str) -> tuple[bool, int, list[str]]:
        archive = self.config.local_root / "raw" / f"{run_label}-final-artifacts.tar.gz"
        if not archive.is_file():
            return False, 0, self._local_artifacts(run_label)
        try:
            with tarfile.open(archive, "r:gz") as handle:
                handle.getmembers()
            result_count = count_result_json(archive)
        except (OSError, tarfile.TarError):
            return False, 0, self._local_artifacts(run_label)
        return True, result_count, self._local_artifacts(run_label)

    def _archived_exit_status(self, run_label: str) -> int | None:
        archive = self.config.local_root / "raw" / f"{run_label}-final-artifacts.tar.gz"
        candidates: list[int] = []
        with tarfile.open(archive, "r:gz") as handle:
            for member in handle.getmembers():
                if not member.isfile():
                    continue
                parts = PurePosixPath(member.name).parts
                if not parts or parts[-1] != "exit_status":
                    continue
                if (
                    f"shellbench_meta-{run_label}" not in parts
                    and not (
                        len(parts) >= 3
                        and parts[-3:] == ("shellbench-runs", run_label, "exit_status")
                    )
                ):
                    continue
                extracted = handle.extractfile(member)
                if extracted is None:
                    continue
                raw_status = extracted.read().decode("utf-8").strip()
                try:
                    candidates.append(int(raw_status))
                except ValueError as exc:
                    raise FleetError(
                        f"invalid archived exit_status for {run_label}: {raw_status!r}"
                    ) from exc
        if len(set(candidates)) > 1:
            raise FleetError(f"conflicting archived exit_status values for {run_label}")
        return candidates[0] if candidates else None

    def _finish_exported(self, entry: dict[str, Any], run: RunSpec) -> bool:
        verified, result_count, artifacts = self._verify_final(run.run_label)
        if not verified:
            self._mark_recovery_required(
                run.run_label,
                "run was marked exported but its final archive is unavailable",
            )
            return False
        lease_value = entry.get("lease")
        if not lease_value:
            self._mark_recovery_required(
                run.run_label,
                "verified export has no lease metadata for cleanup",
            )
            return False
        raw_exit_code = entry.get("run_exit_code")
        run_exit_code = int(raw_exit_code) if raw_exit_code is not None else None
        exit_code_changes: dict[str, Any] = {}
        if run_exit_code is None:
            archived_exit_status = self._archived_exit_status(run.run_label)
            if archived_exit_status is not None:
                run_exit_code = archived_exit_status
                exit_code_changes = {
                    "run_exit_code": run_exit_code,
                    "run_exit_code_source": "archived_exit_status",
                }
        stop_command = [
            self.config.crabbox_bin,
            "stop",
            "--provider",
            "aws",
            "--id",
            str(lease_value["id"]),
        ]
        try:
            lease_still_exists = (
                self._inspect_lease(
                    str(lease_value["id"]),
                    required=False,
                    region_hint=str(lease_value.get("region") or ""),
                )
                is not None
            )
        except FleetError:
            lease_still_exists = True
        if lease_still_exists:
            with self._cleanup_lock:
                if isinstance(self.executor, SubprocessExecutor):
                    stop = self.executor.run_with_timeout(
                        stop_command,
                        capture_output=True,
                        timeout=CRABBOX_STOP_TIMEOUT_SECONDS,
                    )
                else:
                    stop = self.executor.run(stop_command, capture_output=True)
        else:
            stop = subprocess.CompletedProcess(stop_command, 0, stdout="", stderr="")
        if stop.returncode:
            try:
                stopped_after_error = (
                    self._inspect_lease(
                        str(lease_value["id"]),
                        required=False,
                        region_hint=str(lease_value.get("region") or ""),
                    )
                    is None
                )
            except FleetError:
                stopped_after_error = False
            if stopped_after_error:
                stop = subprocess.CompletedProcess(
                    stop_command,
                    0,
                    stdout=stop.stdout,
                    stderr=stop.stderr,
                )
        stop_detail = (stop.stderr or stop.stdout or "").strip().lower()
        already_stopped = any(
            marker in stop_detail
            for marker in ("not found", "not_found", "http 404", "already stopped")
        )
        if stop.returncode and not already_stopped:
            self._store.update(
                run.run_label,
                status="stop_pending",
                verified_final_export=True,
                final_result_count=result_count,
                artifacts=artifacts,
                last_error=f"Crabbox stop failed with exit {stop.returncode}",
                **exit_code_changes,
            )
            return False

        lease_value = dict(lease_value)
        lease_value.update({"state": "stopped", "stopped_at_utc": utc_now()})
        completed = run_exit_code == 0 and result_count == run.expected_task_count
        status = "completed" if completed else "failed"
        self._store.update(
            run.run_label,
            status=status,
            lease=lease_value,
            verified_final_export=True,
            final_result_count=result_count,
            artifacts=artifacts,
            finished_at_utc=utc_now(),
            last_error=(
                None
                if completed
                else (
                    f"run exit {run_exit_code if run_exit_code is not None else 'unknown'}; "
                    f"result coverage {result_count}/{run.expected_task_count}"
                )
            ),
            **exit_code_changes,
        )
        if not completed:
            self._schedule_rerun(self._store.get(run.run_label))
        return completed

    def _mark_recovery_required(self, run_label: str, error: str) -> None:
        self._store.update(
            run_label,
            status="recovery_required",
            last_error=error[:1000],
        )

    def _schedule_existing_reruns(self) -> None:
        for entry in self._store.all_entries():
            if entry.get("status") in RERUN_STATUSES:
                self._schedule_rerun(entry)

    def _resume_recovery_entries(self) -> None:
        for entry in self._store.all_entries():
            if entry.get("status") != "recovery_required":
                continue
            verified, result_count, artifacts = self._verify_final(entry["run_label"])
            if verified and entry.get("lease"):
                self._store.update(
                    entry["run_label"],
                    status="exported",
                    verified_final_export=True,
                    final_result_count=result_count,
                    artifacts=artifacts,
                    last_error=None,
                )
            elif entry.get("lease"):
                self._store.update(
                    entry["run_label"],
                    status="running",
                    last_error=None,
                )
            elif entry.get("requested_lease_slug"):
                self._store.update(
                    entry["run_label"],
                    status="planned",
                    last_error=None,
                )
            else:
                self._store.update(
                    entry["run_label"],
                    status="failed",
                    finished_at_utc=utc_now(),
                )

    def _schedule_rerun(self, entry: dict[str, Any]) -> str | None:
        root_label = str(entry.get("rerun_of") or entry["run_label"])
        existing = self._store.all_entries()
        if any(
            item.get("rerun_of") == root_label and item["status"] not in RERUN_STATUSES
            for item in existing
        ):
            return None
        attempts = {
            int(item.get("attempt", 0))
            for item in existing
            if item["run_label"] == root_label or item.get("rerun_of") == root_label
        }
        next_attempt = max(attempts, default=0) + 1
        if next_attempt >= self.config.max_attempts:
            return None
        label = f"{root_label}-rerun{next_attempt}"
        labels = {item["run_label"] for item in existing}
        while label in labels and next_attempt < self.config.max_attempts:
            next_attempt += 1
            label = f"{root_label}-rerun{next_attempt}"
        if next_attempt >= self.config.max_attempts:
            return None

        run = self._run_spec(entry)
        rerun = {
            **run.to_dict(),
            "run_label": label,
            "reasoning_effort": entry.get("reasoning_effort"),
            "judge_reasoning_effort": entry.get("judge_reasoning_effort"),
            "phase": entry.get("phase"),
            "qualification_family": entry.get("qualification_family"),
            "attempt": next_attempt,
            "status": "planned",
            "leaderboard_eligible": entry.get("leaderboard_eligible"),
            "exclusion_reason": entry.get("exclusion_reason"),
            "rerun_of": root_label,
            "lease": None,
            "artifacts": [],
            "created_at_utc": utc_now(),
        }
        for metadata_field in (
            "task_names",
            "rerun_of_canonical_run",
            "repair_classifications",
        ):
            if metadata_field in entry:
                rerun[metadata_field] = copy.deepcopy(entry[metadata_field])
        self._store.append(rerun)
        return label

    def _matrix_satisfied(self) -> bool:
        entries = self._store.all_entries()
        roots = [entry["run_label"] for entry in entries if not entry.get("rerun_of")]
        return all(
            any(
                candidate.get("status") == "completed"
                and (candidate["run_label"] == root or candidate.get("rerun_of") == root)
                for candidate in entries
            )
            for root in roots
        )

    def _run_spec(self, entry: dict[str, Any]) -> RunSpec:
        try:
            return RunSpec(**{field: entry[field] for field in RUN_SPEC_FIELDS})
        except KeyError as exc:
            raise FleetError(
                f"run entry {entry.get('run_label', '<unknown>')} lacks {exc.args[0]}"
            ) from exc

    def _run_label_for_lease(self, lease_id: str) -> str:
        for entry in self._store.all_entries():
            if (entry.get("lease") or {}).get("id") == lease_id:
                return str(entry["run_label"])
        raise FleetError(f"lease {lease_id} is not assigned to a run")

    def _lease_slug(self, run_label: str) -> str:
        digest = hashlib.sha256(run_label.encode()).hexdigest()[:12]
        return f"sb-native-{digest}"

    def _local_artifacts(self, run_label: str) -> list[str]:
        paths = [
            *sorted((self.config.local_root / "raw").glob(f"{run_label}-*.tar.gz")),
            *sorted((self.config.local_root / "logs").glob(f"{run_label}.*.log")),
        ]
        return [str(path.relative_to(self.config.local_root)) for path in paths if path.is_file()]

    def _ssh_options(self, lease: Lease) -> list[str]:
        return [
            "-i",
            str(lease.identity_file),
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=3",
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]

    def _ssh_command(self, lease: Lease, remote_command: Sequence[str]) -> list[str]:
        return [
            "ssh",
            "-p",
            str(lease.port),
            *self._ssh_options(lease),
            lease.target,
            shlex.join(remote_command),
        ]

    def _scp_command(
        self,
        lease: Lease,
        local_path: Path,
        remote_path: str,
    ) -> list[str]:
        return [
            "scp",
            "-P",
            str(lease.port),
            *self._ssh_options(lease),
            str(local_path),
            f"{lease.target}:{remote_path}",
        ]

    def _checked(
        self,
        command: Sequence[str],
        *,
        capture_output: bool,
        description: str,
    ) -> subprocess.CompletedProcess[str]:
        result = self.executor.run(command, capture_output=capture_output)
        if result.returncode:
            detail = (result.stderr or result.stdout or "").strip()[:500]
            suffix = f": {detail}" if detail else ""
            raise FleetError(f"{description} failed with exit {result.returncode}{suffix}")
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_values(values: dict[str, int], name: str) -> None:
    for model_slug, value in values.items():
        if not model_slug:
            raise ValueError(f"{name} contains an empty model slug")
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name}[{model_slug!r}] must be a positive integer")


def _parse_model_values(
    parser: argparse.ArgumentParser,
    values: list[str],
    option: str,
) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for raw in values:
        model_slug, separator, count_value = raw.partition("=")
        if not separator or not model_slug or not count_value:
            parser.error(f"{option} must use MODEL=COUNT: {raw!r}")
        if model_slug in parsed:
            parser.error(f"{option} repeats model {model_slug!r}")
        try:
            count = int(count_value)
        except ValueError:
            parser.error(f"{option} count must be an integer: {raw!r}")
        if count < 1:
            parser.error(f"{option} count must be positive: {raw!r}")
        parsed[model_slug] = count
    return parsed


def _parse_parity_routes(
    parser: argparse.ArgumentParser,
    values: list[str],
) -> frozenset[tuple[str, str]]:
    parsed: set[tuple[str, str]] = set()
    for raw in values:
        harness, separator, model_slug = raw.partition("=")
        if not separator or not harness or not model_slug:
            parser.error(
                f"--parity-validated-route must use HARNESS=MODEL: {raw!r}"
            )
        route = (harness, model_slug)
        if route in parsed:
            parser.error(f"--parity-validated-route repeats {raw!r}")
        parsed.add(route)
    return frozenset(parsed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-index", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--runner-root", type=Path, required=True)
    parser.add_argument("--task-archive", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--max-leases", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--task-concurrency", type=int, default=16)
    parser.add_argument("--model-max-runs", action="append", default=[], metavar="MODEL=COUNT")
    parser.add_argument(
        "--provider-max-runs",
        action="append",
        default=[],
        metavar="PROVIDER=COUNT",
    )
    parser.add_argument(
        "--model-task-concurrency",
        action="append",
        default=[],
        metavar="MODEL=COUNT",
    )
    parser.add_argument("--crabbox-bin", default="crabbox")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--machine-class", default="beast")
    parser.add_argument("--instance-type", default="c7a.24xlarge")
    parser.add_argument("--market", default="on-demand")
    parser.add_argument("--region", default="eu-west-1")
    parser.add_argument("--ttl", default="12h")
    parser.add_argument("--idle-timeout", default="12h")
    parser.add_argument(
        "--remote-root",
        default="/work/crabbox/shellbench-native",
    )
    parser.add_argument("--checkpoint-poll-seconds", type=int, default=30)
    parser.add_argument("--warmup-capacity-attempts", type=int, default=12)
    parser.add_argument("--warmup-capacity-backoff-seconds", type=float, default=5.0)
    parser.add_argument("--runner-archive", type=Path)
    parser.add_argument("--runner-commit")
    parser.add_argument("--harbor-reference-commit", default="")
    parser.add_argument("--judge-model-id", default="")
    parser.add_argument("--execution-mode", default="native")
    parser.add_argument("--parity-validated", action="store_true")
    parser.add_argument(
        "--parity-validated-route",
        action="append",
        default=[],
        metavar="HARNESS=MODEL",
    )
    args = parser.parse_args(argv)
    args.model_max_runs = _parse_model_values(
        parser,
        args.model_max_runs,
        "--model-max-runs",
    )
    args.provider_max_runs = _parse_model_values(
        parser,
        args.provider_max_runs,
        "--provider-max-runs",
    )
    args.model_task_concurrency = _parse_model_values(
        parser,
        args.model_task_concurrency,
        "--model-task-concurrency",
    )
    args.parity_validated_routes = _parse_parity_routes(
        parser,
        args.parity_validated_route,
    )
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = FleetConfig(
        run_index=args.run_index,
        local_root=args.local_root,
        runner_root=args.runner_root,
        task_archive=args.task_archive,
        env_file=args.env_file,
        max_leases=args.max_leases,
        max_attempts=args.max_attempts,
        task_concurrency=args.task_concurrency,
        model_max_runs=args.model_max_runs,
        provider_max_runs=args.provider_max_runs,
        model_task_concurrency=args.model_task_concurrency,
        crabbox_bin=args.crabbox_bin,
        python_bin=args.python_bin,
        machine_class=args.machine_class,
        instance_type=args.instance_type,
        market=args.market,
        region=args.region,
        ttl=args.ttl,
        idle_timeout=args.idle_timeout,
        remote_root=args.remote_root,
        checkpoint_poll_seconds=args.checkpoint_poll_seconds,
        warmup_capacity_attempts=args.warmup_capacity_attempts,
        warmup_capacity_backoff_seconds=args.warmup_capacity_backoff_seconds,
        runner_archive=args.runner_archive,
        runner_commit=args.runner_commit,
        harbor_reference_commit=args.harbor_reference_commit,
        judge_model_id=args.judge_model_id,
        execution_mode=args.execution_mode,
        parity_validated=args.parity_validated,
        parity_validated_routes=args.parity_validated_routes,
    )
    try:
        return FleetController(config).run()
    except (FleetError, OSError, json.JSONDecodeError) as exc:
        print(f"fleet controller failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
