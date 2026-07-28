from __future__ import annotations

import json
import shlex
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import Sequence

import pytest

from scripts.native_eval.fleet import (
    FleetConfig,
    FleetController,
    FleetError,
    Lease,
    LeaseNotReadyError,
    SubprocessExecutor,
    parse_args,
)
from scripts.native_eval.models import RunSpec


def _run_spec(
    label: str,
    *,
    expected_task_count: int = 2,
    model_slug: str = "gpt55",
) -> RunSpec:
    return RunSpec(
        run_label=label,
        harness="openclaw",
        harness_version="test",
        model_slug=model_slug,
        model_id=f"provider/{model_slug}",
        provider="anthropic" if model_slug == "fable5" else "openai",
        proxy_model_name=f"sb-{model_slug}",
        repetition=1,
        expected_task_count=expected_task_count,
        run_date="20260727",
    )


def _write_index(
    path: Path,
    runs: list[dict[str, object]],
    *,
    expected_task_count: int = 2,
) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "created_at_utc": "2026-07-27T00:00:00+00:00",
                "public_tasks_commit": "tasks-commit",
                "task_suite_path": "combined tasks/tasks",
                "expected_task_count": expected_task_count,
                "planned_run_count": len(runs),
                "runs": runs,
            }
        ),
        encoding="utf-8",
    )


def _planned(run: RunSpec) -> dict[str, object]:
    return {
        **run.to_dict(),
        "reasoning_effort": "high",
        "judge_reasoning_effort": "high",
        "attempt": 0,
        "status": "planned",
        "leaderboard_eligible": None,
        "rerun_of": None,
        "lease": None,
        "artifacts": [],
    }


def test_provisioning_lease_is_not_ready_before_ssh_details_exist() -> None:
    with pytest.raises(LeaseNotReadyError, match="provisioning but not ready"):
        Lease.from_inspect(
            {
                "id": "cbx_pending",
                "slug": "sb-native-pending",
                "state": "provisioning",
                "ready": False,
                "sshHost": "",
            },
            region="eu-west-1",
        )


def test_subprocess_timeout_output_is_normalized_to_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise subprocess.TimeoutExpired(
            ["crabbox", "stop"],
            45,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(subprocess, "run", timeout)
    result = SubprocessExecutor().run_with_timeout(
        ["crabbox", "stop"],
        capture_output=True,
        timeout=45,
    )

    assert result.returncode == 124
    assert result.stdout == "partial stdout"
    assert result.stderr == "partial stderr"


def _write_archive(path: Path) -> None:
    source = path.parent / f"{path.stem}-source"
    source.mkdir()
    (source / "marker.txt").write_text("pinned", encoding="utf-8")
    with tarfile.open(path, "w:gz") as handle:
        handle.add(source, arcname=".")


def _write_final(
    local_root: Path,
    run_label: str,
    result_count: int,
    *,
    exit_status: int | None = None,
) -> None:
    source = local_root / f".archive-{run_label}"
    job = source / "results" / "jobs" / run_label
    for index in range(result_count):
        trial = job / f"task-{index}__trial"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text("{}", encoding="utf-8")
    if exit_status is not None:
        meta = source / f"shellbench_meta-{run_label}"
        meta.mkdir(parents=True, exist_ok=True)
        (meta / "exit_status").write_text(f"{exit_status}\n", encoding="utf-8")
    archive = local_root / "raw" / f"{run_label}-final-artifacts.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(source, arcname=".")


class FakeExecutor:
    def __init__(
        self,
        local_root: Path,
        *,
        expected_counts: dict[str, int],
        checkpoint_codes: dict[str, int] | None = None,
        omit_final: set[str] | None = None,
        running_labels: set[str] | None = None,
        checkpoint_blocks: dict[str, threading.Event] | None = None,
        stop_code: int = 0,
        inspect_errors: set[str] | None = None,
        warmup_capacity_failures: dict[str, int] | None = None,
        dispatch_failures: dict[str, int] | None = None,
    ) -> None:
        self.local_root = local_root
        self.expected_counts = expected_counts
        self.checkpoint_codes = checkpoint_codes or {}
        self.omit_final = omit_final or set()
        self.remote_states = {label: "running" for label in (running_labels or set())}
        self.checkpoint_blocks = checkpoint_blocks or {}
        self.stop_code = stop_code
        self.inspect_errors = inspect_errors or set()
        self.warmup_capacity_failures = warmup_capacity_failures or {}
        self.dispatch_failures = dispatch_failures or {}
        self.warmup_attempts: dict[str, int] = {}
        self.dispatch_attempts: dict[str, int] = {}
        self.leases: dict[str, dict[str, object]] = {}
        self.commands: list[list[str]] = []
        self.events: list[tuple[str, str]] = []
        self.dispatches: list[str] = []
        self.dispatch_concurrency: dict[str, int] = {}
        self.dispatch_arguments: dict[str, list[str]] = {}
        self.stops: list[str] = []
        self._lock = threading.Lock()
        self._dispatch_condition = threading.Condition(self._lock)
        self._next_lease = 0
        self.active_leases = 0
        self.max_active_leases = 0

    def run(
        self,
        command: Sequence[str],
        *,
        capture_output: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        del capture_output
        argv = list(command)
        with self._lock:
            self.commands.append(argv)

        if argv == ["crabbox", "--version"]:
            return _result(argv, 0, stdout="crabbox version 0.36.0\n")

        if argv[:2] == ["env", "CRABBOX_CAPACITY_MARKET=on-demand"]:
            argv = argv[2:]

        if argv[:2] == ["crabbox", "inspect"]:
            identifier = argv[argv.index("--id") + 1]
            if identifier in self.inspect_errors:
                return _result(argv, 1, stderr="broker request timed out")
            lease = next(
                (
                    value
                    for value in self.leases.values()
                    if identifier in {value["id"], value["slug"]}
                ),
                None,
            )
            if lease is None:
                return _result(
                    argv,
                    1,
                    stderr='coordinator GET /v1/leases/example: http 404: {"error":"not_found"}',
                )
            return _result(argv, 0, stdout=json.dumps(lease))

        if argv[:2] == ["crabbox", "warmup"]:
            slug = argv[argv.index("--slug") + 1]
            with self._lock:
                attempt = self.warmup_attempts.get(slug, 0) + 1
                self.warmup_attempts[slug] = attempt
                if attempt <= self.warmup_capacity_failures.get(slug, 0):
                    return _result(
                        argv,
                        1,
                        stderr=(
                            "coordinator POST /v1/leases: http 429: "
                            '{"error":"cost_limit_exceeded",'
                            '"message":"Active lease limit 11/10"}'
                        ),
                    )
                self._next_lease += 1
                lease_id = f"cbx_{self._next_lease}"
                self.active_leases += 1
                self.max_active_leases = max(
                    self.max_active_leases,
                    self.active_leases,
                )
                self.leases[lease_id] = {
                    "id": lease_id,
                    "slug": slug,
                    "state": "active",
                    "ready": True,
                    "serverType": "c7a.24xlarge",
                    "sshHost": f"192.0.2.{self._next_lease}",
                    "sshUser": "crabbox",
                    "sshPort": "22",
                    "sshKey": f"/tmp/{lease_id}.key",
                }
            return _result(argv, 0)

        if argv[:2] == ["crabbox", "stop"]:
            lease_id = argv[argv.index("--id") + 1]
            with self._lock:
                self.stops.append(lease_id)
                self.events.append(("stop", lease_id))
                if not self.stop_code:
                    self.active_leases -= 1
            return _result(argv, self.stop_code)

        if "-m" in argv and "scripts.native_eval.checkpoint_loop" in argv:
            label = argv[argv.index("--run-label") + 1]
            with self._lock:
                self.events.append(("checkpoint", label))
            block = self.checkpoint_blocks.get(label)
            if block is not None:
                block.wait()
            if label not in self.omit_final:
                _write_final(
                    self.local_root,
                    label,
                    self.expected_counts.get(label, 0),
                )
            return _result(argv, self.checkpoint_codes.get(label, 0))

        joined = " ".join(argv)
        if argv and argv[0] == "ssh" and "fleet-probe" in joined:
            label = next(
                candidate
                for candidate in sorted(self.expected_counts, key=len, reverse=True)
                if candidate in joined
            )
            return _result(argv, 0, stdout=self.remote_states.get(label, "missing"))

        if argv and argv[0] == "ssh" and "fleet-dispatch" in joined:
            label = next(
                candidate
                for candidate in sorted(self.expected_counts, key=len, reverse=True)
                if candidate in joined
            )
            remote_command = shlex.split(argv[-1])
            dispatch_marker = remote_command.index("fleet-dispatch")
            remote_run_args = remote_command[dispatch_marker + 16 :]
            with self._dispatch_condition:
                attempt = self.dispatch_attempts.get(label, 0) + 1
                self.dispatch_attempts[label] = attempt
                if attempt <= self.dispatch_failures.get(label, 0):
                    return _result(
                        argv,
                        255,
                        stderr="ssh: connect to host 192.0.2.1 port 22: Operation timed out",
                    )
                self.dispatches.append(label)
                self.dispatch_concurrency[label] = int(remote_run_args[10])
                self.dispatch_arguments[label] = remote_run_args
                self.events.append(("dispatch", label))
                self.remote_states[label] = "running"
                self._dispatch_condition.notify_all()
            return _result(argv, 0, stdout="1234")

        return _result(argv, 0)

    def wait_for_dispatch(self, run_label: str, timeout: float) -> bool:
        with self._dispatch_condition:
            return self._dispatch_condition.wait_for(
                lambda: run_label in self.dispatches,
                timeout=timeout,
            )


def _result(
    command: Sequence[str],
    returncode: int,
    *,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        list(command),
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _config(
    tmp_path: Path,
    run_index: Path,
    *,
    max_leases: int = 1,
    max_attempts: int = 2,
    task_concurrency: int = 16,
    model_max_runs: dict[str, int] | None = None,
    provider_max_runs: dict[str, int] | None = None,
    model_task_concurrency: dict[str, int] | None = None,
    warmup_capacity_attempts: int = 12,
    warmup_capacity_backoff_seconds: float = 0,
) -> FleetConfig:
    runner_archive = tmp_path / "runner.tar.gz"
    task_archive = tmp_path / "tasks.tar.gz"
    _write_archive(runner_archive)
    _write_archive(task_archive)
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=TOPSECRET\n", encoding="utf-8")
    return FleetConfig(
        run_index=run_index,
        local_root=tmp_path / "runs",
        runner_root=tmp_path,
        runner_archive=runner_archive,
        runner_commit="runner-commit",
        task_archive=task_archive,
        env_file=env_file,
        max_leases=max_leases,
        max_attempts=max_attempts,
        task_concurrency=task_concurrency,
        model_max_runs=model_max_runs or {},
        provider_max_runs=provider_max_runs or {},
        model_task_concurrency=model_task_concurrency or {},
        checkpoint_poll_seconds=1,
        warmup_capacity_attempts=warmup_capacity_attempts,
        warmup_capacity_backoff_seconds=warmup_capacity_backoff_seconds,
    )


def test_controller_runs_bounded_wave_and_stops_after_verified_export(
    tmp_path: Path,
) -> None:
    labels = ["openclaw-gpt55-full-2-r1-20260727", "openclaw-gpt55-full-2-r2-20260727"]
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [_planned(_run_spec(label)) for label in labels])
    config = _config(tmp_path, run_index, max_leases=1)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2 for label in labels},
    )

    assert FleetController(config, executor=executor).run() == 0

    index = json.loads(run_index.read_text(encoding="utf-8"))
    assert [run["status"] for run in index["runs"]] == ["completed", "completed"]
    assert executor.dispatches == labels
    assert executor.max_active_leases == 1
    for label in labels:
        assert executor.dispatch_arguments[label][11:15] == [
            "test",
            "provider/gpt55",
            "openai",
            "sb-gpt55",
        ]
    warmup = next(command for command in executor.commands if "warmup" in command)
    assert warmup[:2] == ["env", "CRABBOX_CAPACITY_MARKET=on-demand"]
    assert "--market" not in warmup
    for label in labels:
        checkpoint_at = executor.events.index(("checkpoint", label))
        lease_id = index["runs"][labels.index(label)]["lease"]["id"]
        stop_at = executor.events.index(("stop", lease_id))
        assert checkpoint_at < stop_at
    assert "TOPSECRET" not in "\n".join(" ".join(command) for command in executor.commands)


def test_controller_accepts_xhigh_reasoning_effort(tmp_path: Path) -> None:
    label = "openclaw-gpt56-sol-xhigh-full-2-r1-20260728"
    run = _planned(_run_spec(label, model_slug="gpt56-sol"))
    run["reasoning_effort"] = "xhigh"
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(config.local_root, expected_counts={label: 2})

    assert FleetController(config, executor=executor).run() == 0

    completed = json.loads(run_index.read_text(encoding="utf-8"))["runs"][0]
    assert completed["status"] == "completed"
    assert completed["reasoning_effort"] == "xhigh"
    assert completed["judge_reasoning_effort"] == "high"


def test_controller_dispatches_targeted_repair_tasks_with_lineage(
    tmp_path: Path,
) -> None:
    label = "openclaw-gpt55-low-full-116-r1-20260728-repairtasks1"
    run = _planned(_run_spec(label, expected_task_count=2))
    run["task_names"] = ["task-a", "task-b"]
    run["rerun_of_canonical_run"] = (
        "openclaw-gpt55-low-full-116-r1-20260728-runnerd676-lifecyclefix1"
    )
    run["repair_classifications"] = ["infra", "agent_exit"]
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run], expected_task_count=116)
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(config.local_root, expected_counts={label: 2})

    assert FleetController(config, executor=executor).run() == 0

    dispatched = executor.dispatch_arguments[label]
    assert dispatched[15] == run["rerun_of_canonical_run"]
    assert dispatched[16:] == ["task-a", "task-b"]


def test_controller_retries_transient_dispatch_timeout(tmp_path: Path) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [_planned(_run_spec(label))])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2},
        dispatch_failures={label: 1},
    )

    assert FleetController(config, executor=executor).run() == 0

    assert executor.dispatch_attempts[label] == 2
    assert executor.dispatches == [label]


def test_controller_rejects_task_subset_without_canonical_parent(
    tmp_path: Path,
) -> None:
    label = "openclaw-gpt55-low-full-116-r1-20260728-repairtasks1"
    run = _planned(_run_spec(label, expected_task_count=1))
    run["task_names"] = ["task-a"]
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run], expected_task_count=116)
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(config.local_root, expected_counts={label: 1})

    with pytest.raises(FleetError, match="task subset lacks rerun_of_canonical_run"):
        FleetController(config, executor=executor).run()


def test_capacity_warmup_retries_same_run_without_recovery_churn(
    tmp_path: Path,
) -> None:
    retry_label = "openclaw-gpt55-full-2-r1-20260727"
    untouched_label = "openclaw-gpt55-full-2-r2-20260727"
    retry_run = _planned(_run_spec(retry_label))
    retry_run["requested_lease_slug"] = "capacity-retry"
    untouched_run = _planned(_run_spec(untouched_label))
    untouched_run["requested_lease_slug"] = "untouched"
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [retry_run, untouched_run])
    config = _config(
        tmp_path,
        run_index,
        max_leases=1,
        warmup_capacity_attempts=3,
        warmup_capacity_backoff_seconds=0,
    )
    executor = FakeExecutor(
        config.local_root,
        expected_counts={retry_label: 2, untouched_label: 2},
        warmup_capacity_failures={"capacity-retry": 2},
    )

    assert FleetController(config, executor=executor).run() == 0

    runs = json.loads(run_index.read_text(encoding="utf-8"))["runs"]
    assert [run["run_label"] for run in runs] == [retry_label, untouched_label]
    assert [run["status"] for run in runs] == ["completed", "completed"]
    assert all(run["rerun_of"] is None for run in runs)
    assert executor.warmup_attempts == {
        "capacity-retry": 3,
        "untouched": 1,
    }
    assert executor.dispatches == [retry_label, untouched_label]


def test_model_cap_counts_adopted_run_before_dispatching_same_model(
    tmp_path: Path,
) -> None:
    adopted_label = "openclaw-fable5-full-2-r1-20260727"
    pending_fable_label = "openclaw-fable5-full-2-r2-20260727"
    pending_gpt_label = "openclaw-gpt55-full-2-r1-20260727"
    adopted = _planned(_run_spec(adopted_label, model_slug="fable5"))
    adopted["status"] = "running"
    adopted["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(
        run_index,
        [
            _planned(_run_spec(pending_fable_label, model_slug="fable5")),
            _planned(_run_spec(pending_gpt_label)),
            adopted,
        ],
    )
    config = _config(
        tmp_path,
        run_index,
        max_leases=2,
        model_max_runs={"fable5": 1},
    )
    executor = FakeExecutor(
        config.local_root,
        expected_counts={
            adopted_label: 2,
            pending_fable_label: 2,
            pending_gpt_label: 2,
        },
        running_labels={adopted_label},
    )
    executor.leases["cbx_existing"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "state": "active",
        "ready": True,
        "serverType": "c7a.24xlarge",
        "sshHost": "192.0.2.50",
        "sshUser": "crabbox",
        "sshPort": "22",
        "sshKey": "/tmp/cbx_existing.key",
    }
    executor.active_leases = 1

    assert FleetController(config, executor=executor).run() == 0

    assert executor.dispatches == [pending_gpt_label, pending_fable_label]
    adopted_stop = executor.events.index(("stop", "cbx_existing"))
    pending_fable_dispatch = executor.events.index(("dispatch", pending_fable_label))
    assert adopted_stop < pending_fable_dispatch


def test_recovery_uses_recorded_ssh_endpoint_when_crabbox_ready_probe_is_stale(
    tmp_path: Path,
) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run = _planned(_run_spec(label))
    run["status"] = "recovery_required"
    run["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "instance_type": "c7a.24xlarge",
        "region": "eu-west-1",
        "host": "192.0.2.50",
        "ssh_user": "crabbox",
        "ssh_port": 22,
        "identity_file": "/tmp/cbx_existing.key",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2},
        running_labels={label},
    )
    executor.leases["cbx_existing"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "state": "active",
        "ready": False,
        "serverType": "c7a.24xlarge",
        "sshHost": "192.0.2.50",
        "sshUser": "crabbox",
        "sshPort": "2222",
        "sshKey": "/tmp/cbx_existing.key",
    }
    executor.active_leases = 1

    assert FleetController(config, executor=executor).run() == 0

    recovered = json.loads(run_index.read_text(encoding="utf-8"))["runs"][0]
    assert recovered["status"] == "completed"
    assert executor.dispatches == []
    reachability_probe = next(
        command
        for command in executor.commands
        if command and command[0] == "ssh" and command[-1] == "true"
    )
    assert "22" in reachability_probe


def test_model_task_concurrency_override_is_used_at_dispatch(tmp_path: Path) -> None:
    fable_label = "openclaw-fable5-full-2-r1-20260727"
    gpt_label = "openclaw-gpt55-full-2-r1-20260727"
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(
        run_index,
        [
            _planned(_run_spec(fable_label, model_slug="fable5")),
            _planned(_run_spec(gpt_label)),
        ],
    )
    config = _config(
        tmp_path,
        run_index,
        max_leases=2,
        task_concurrency=16,
        model_task_concurrency={"fable5": 2},
    )
    executor = FakeExecutor(
        config.local_root,
        expected_counts={fable_label: 2, gpt_label: 2},
    )

    assert FleetController(config, executor=executor).run() == 0

    assert executor.dispatch_concurrency == {
        fable_label: 2,
        gpt_label: 16,
    }


def test_provider_cap_counts_adopted_run_before_dispatching_same_provider(
    tmp_path: Path,
) -> None:
    adopted_label = "openclaw-fable5-full-2-r1-20260727"
    pending_fable_label = "openclaw-fable5-full-2-r2-20260727"
    pending_gpt_label = "openclaw-gpt55-full-2-r1-20260727"
    adopted = _planned(_run_spec(adopted_label, model_slug="fable5"))
    adopted["status"] = "running"
    adopted["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(
        run_index,
        [
            _planned(_run_spec(pending_fable_label, model_slug="fable5")),
            _planned(_run_spec(pending_gpt_label)),
            adopted,
        ],
    )
    config = _config(
        tmp_path,
        run_index,
        max_leases=2,
        provider_max_runs={"anthropic": 1},
    )
    executor = FakeExecutor(
        config.local_root,
        expected_counts={
            adopted_label: 2,
            pending_fable_label: 2,
            pending_gpt_label: 2,
        },
        running_labels={adopted_label},
    )
    executor.leases["cbx_existing"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "state": "active",
        "ready": True,
        "serverType": "c7a.24xlarge",
        "sshHost": "192.0.2.50",
        "sshUser": "crabbox",
        "sshPort": "22",
        "sshKey": "/tmp/cbx_existing.key",
    }
    executor.active_leases = 1

    assert FleetController(config, executor=executor).run() == 0

    assert executor.dispatches == [pending_gpt_label, pending_fable_label]
    adopted_stop = executor.events.index(("stop", "cbx_existing"))
    pending_fable_dispatch = executor.events.index(("dispatch", pending_fable_label))
    assert adopted_stop < pending_fable_dispatch


def test_slow_capped_model_does_not_block_refilling_eligible_slot(
    tmp_path: Path,
) -> None:
    slow_fable = "openclaw-fable5-full-2-r1-20260727"
    fast_gpt = "openclaw-gpt55-full-2-r1-20260727"
    capped_fable = "openclaw-fable5-full-2-r2-20260727"
    later_gpt = "openclaw-gpt55-full-2-r2-20260727"
    labels = [slow_fable, fast_gpt, capped_fable, later_gpt]
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(
        run_index,
        [
            _planned(_run_spec(slow_fable, model_slug="fable5")),
            _planned(_run_spec(fast_gpt)),
            _planned(_run_spec(capped_fable, model_slug="fable5")),
            _planned(_run_spec(later_gpt)),
        ],
    )
    config = _config(
        tmp_path,
        run_index,
        max_leases=2,
        model_max_runs={"fable5": 1},
    )
    release_slow = threading.Event()
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2 for label in labels},
        checkpoint_blocks={slow_fable: release_slow},
    )
    result: list[int] = []
    controller = threading.Thread(
        target=lambda: result.append(FleetController(config, executor=executor).run())
    )
    controller.start()
    try:
        assert executor.wait_for_dispatch(later_gpt, timeout=2)
        assert capped_fable not in executor.dispatches
        assert not release_slow.is_set()
    finally:
        release_slow.set()
        controller.join(timeout=5)

    assert not controller.is_alive()
    assert result == [0]
    assert executor.dispatches.index(later_gpt) < executor.dispatches.index(capped_fable)
    assert executor.max_active_leases == 2


def test_recovery_pending_entries_respect_capacity_behind_owned_runs(
    tmp_path: Path,
) -> None:
    owned_labels = [f"openclaw-gpt55-full-2-r{index}-20260727" for index in range(1, 10)]
    pending_labels = [
        f"openclaw-fable5-full-2-r{index}-20260727" for index in range(10, 15)
    ]
    owned_runs: list[dict[str, object]] = []
    for index, label in enumerate(owned_labels, start=1):
        run = _planned(_run_spec(label))
        run["status"] = "running"
        run["lease"] = {
            "id": f"cbx_owned_{index}",
            "slug": f"owned-{index}",
            "provider": "aws",
            "state": "active",
        }
        owned_runs.append(run)
    pending_runs: list[dict[str, object]] = []
    for index, label in enumerate(pending_labels, start=1):
        run = _planned(_run_spec(label, model_slug="fable5"))
        run["status"] = "leasing" if index == 1 else "recovery_required"
        run["requested_lease_slug"] = f"requested-{index}"
        pending_runs.append(run)

    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [*pending_runs, *owned_runs])
    config = _config(
        tmp_path,
        run_index,
        max_leases=10,
        model_max_runs={"fable5": 1},
    )
    release_runs = threading.Event()
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2 for label in [*owned_labels, *pending_labels]},
        running_labels=set(owned_labels),
        checkpoint_blocks={
            label: release_runs for label in [*owned_labels, pending_labels[0]]
        },
    )
    for index, label in enumerate(owned_labels, start=1):
        executor.leases[f"cbx_owned_{index}"] = {
            "id": f"cbx_owned_{index}",
            "slug": f"owned-{index}",
            "state": "active",
            "ready": True,
            "serverType": "c7a.24xlarge",
            "sshHost": f"192.0.2.{index}",
            "sshUser": "crabbox",
            "sshPort": "22",
            "sshKey": f"/tmp/cbx_owned_{index}.key",
        }
    executor.active_leases = len(owned_labels)
    executor.max_active_leases = len(owned_labels)

    result: list[int] = []
    controller = threading.Thread(
        target=lambda: result.append(FleetController(config, executor=executor).run())
    )
    controller.start()
    try:
        assert executor.wait_for_dispatch(pending_labels[0], timeout=2)
        assert executor.dispatches == [pending_labels[0]]
        index = json.loads(run_index.read_text(encoding="utf-8"))
        pending_statuses = {
            run["run_label"]: run["status"]
            for run in index["runs"]
            if run["run_label"] in pending_labels[1:]
        }
        assert set(pending_statuses.values()) == {"planned"}
        assert executor.active_leases == 10
    finally:
        release_runs.set()
        controller.join(timeout=10)

    assert not controller.is_alive()
    assert result == [0]
    assert not set(owned_labels) & set(executor.dispatches)
    assert all(("checkpoint", label) in executor.events for label in owned_labels)
    assert executor.max_active_leases == 10


def test_parse_args_accepts_repeatable_model_limits(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--run-index",
            str(tmp_path / "index.json"),
            "--local-root",
            str(tmp_path / "runs"),
            "--runner-root",
            str(tmp_path / "runner"),
            "--task-archive",
            str(tmp_path / "tasks.tar.gz"),
            "--env-file",
            str(tmp_path / ".env"),
            "--model-max-runs",
            "fable5=1",
            "--model-max-runs",
            "opus48=2",
            "--provider-max-runs",
            "anthropic=2",
            "--harbor-reference-commit",
            "harbor-commit",
            "--judge-model-id",
            "gpt-5.5",
            "--execution-mode",
            "native",
            "--parity-validated",
            "--model-task-concurrency",
            "fable5=2",
        ]
    )

    assert args.model_max_runs == {"fable5": 1, "opus48": 2}
    assert args.provider_max_runs == {"anthropic": 2}
    assert args.harbor_reference_commit == "harbor-commit"
    assert args.judge_model_id == "gpt-5.5"
    assert args.execution_mode == "native"
    assert args.parity_validated is True
    assert args.model_task_concurrency == {"fable5": 2}


@pytest.mark.parametrize(
    ("option", "values"),
    [
        ("--model-max-runs", ["fable5"]),
        ("--model-max-runs", ["=1"]),
        ("--model-max-runs", ["fable5=nope"]),
        ("--model-max-runs", ["fable5=0"]),
        ("--provider-max-runs", ["anthropic=0"]),
        ("--model-task-concurrency", ["fable5=-1"]),
        ("--model-task-concurrency", ["fable5=1", "fable5=2"]),
    ],
)
def test_parse_args_rejects_invalid_model_values(
    tmp_path: Path,
    option: str,
    values: list[str],
) -> None:
    argv = [
        "--run-index",
        str(tmp_path / "index.json"),
        "--local-root",
        str(tmp_path / "runs"),
        "--runner-root",
        str(tmp_path / "runner"),
        "--task-archive",
        str(tmp_path / "tasks.tar.gz"),
        "--env-file",
        str(tmp_path / ".env"),
    ]
    for value in values:
        argv.extend([option, value])

    with pytest.raises(SystemExit):
        parse_args(argv)


def test_failed_run_is_preserved_and_suffixed_rerun_completes(
    tmp_path: Path,
) -> None:
    base = "openclaw-gpt55-full-2-r1-20260727"
    rerun = f"{base}-rerun1"
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [_planned(_run_spec(base))])
    config = _config(tmp_path, run_index, max_attempts=2)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={base: 1, rerun: 2},
        checkpoint_codes={base: 1},
    )

    assert FleetController(config, executor=executor).run() == 0

    runs = json.loads(run_index.read_text(encoding="utf-8"))["runs"]
    assert [run["run_label"] for run in runs] == [base, rerun]
    assert runs[0]["status"] == "failed"
    assert runs[0]["final_result_count"] == 1
    assert runs[1]["status"] == "completed"
    assert runs[1]["rerun_of"] == base
    assert runs[1]["reasoning_effort"] == "high"
    assert runs[1]["judge_reasoning_effort"] == "high"
    assert executor.dispatches == [base, rerun]
    assert (config.local_root / "raw" / f"{base}-final-artifacts.tar.gz").is_file()
    assert (config.local_root / "raw" / f"{rerun}-final-artifacts.tar.gz").is_file()


def test_resume_attaches_to_running_lease_without_redispatch(tmp_path: Path) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run = _planned(_run_spec(label))
    run["status"] = "running"
    run["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2},
        running_labels={label},
    )
    executor.leases["cbx_existing"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "state": "active",
        "ready": True,
        "serverType": "c7a.24xlarge",
        "sshHost": "192.0.2.50",
        "sshUser": "crabbox",
        "sshPort": "22",
        "sshKey": "/tmp/cbx_existing.key",
    }
    executor.active_leases = 1

    assert FleetController(config, executor=executor).run() == 0

    assert executor.dispatches == []
    assert not any(command[:2] == ["crabbox", "warmup"] for command in executor.commands)
    assert not any(command and command[0] == "scp" for command in executor.commands)
    final = json.loads(run_index.read_text(encoding="utf-8"))["runs"][0]
    assert final["status"] == "completed"


def test_unverified_final_keeps_lease_for_recovery(tmp_path: Path) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [_planned(_run_spec(label))])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2},
        checkpoint_codes={label: 75},
        omit_final={label},
    )

    assert FleetController(config, executor=executor).run() == 1

    runs = json.loads(run_index.read_text(encoding="utf-8"))["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "recovery_required"
    assert runs[0]["lease"]["state"] == "active"
    assert executor.stops == []


def test_recovery_required_resumes_existing_remote_run(tmp_path: Path) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run = _planned(_run_spec(label))
    run["status"] = "recovery_required"
    run["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2},
        running_labels={label},
    )
    executor.leases["cbx_existing"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "state": "active",
        "ready": True,
        "serverType": "c7a.24xlarge",
        "sshHost": "192.0.2.50",
        "sshUser": "crabbox",
        "sshPort": "22",
        "sshKey": "/tmp/cbx_existing.key",
    }
    executor.active_leases = 1

    assert FleetController(config, executor=executor).run() == 0

    assert executor.dispatches == []
    final = json.loads(run_index.read_text(encoding="utf-8"))["runs"][0]
    assert final["status"] == "completed"


def test_recovery_infers_success_from_verified_full_archive_and_done_log(
    tmp_path: Path,
) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run = _planned(_run_spec(label))
    run["status"] = "recovery_required"
    run["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run])
    config = _config(tmp_path, run_index)
    _write_final(config.local_root, label, 2)
    checkpoint_log = config.local_root / "logs" / f"{label}.checkpoints.log"
    checkpoint_log.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_log.write_text(
        f"2026-07-27T00:00:00Z\tfinal\t{label}-final-artifacts.tar.gz\t2\t\n",
        encoding="utf-8",
    )
    executor = FakeExecutor(config.local_root, expected_counts={label: 2})
    executor.leases["cbx_existing"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "state": "active",
        "ready": True,
        "serverType": "c7a.24xlarge",
        "sshHost": "192.0.2.50",
        "sshUser": "crabbox",
        "sshPort": "22",
        "sshKey": "/tmp/cbx_existing.key",
    }
    executor.active_leases = 1

    assert FleetController(config, executor=executor).run() == 0

    recovered = json.loads(run_index.read_text(encoding="utf-8"))["runs"][0]
    assert recovered["status"] == "completed"
    assert recovered["run_exit_code"] == 0
    assert recovered["run_exit_code_source"] == "recovered_full_coverage_remote_done"
    assert "inferred zero" in recovered["run_exit_code_inference"]


def test_recovery_preserves_archived_nonzero_exit_status(tmp_path: Path) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run = _planned(_run_spec(label))
    run["status"] = "recovery_required"
    run["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run])
    config = _config(tmp_path, run_index, max_attempts=1)
    _write_final(config.local_root, label, 2, exit_status=7)
    executor = FakeExecutor(config.local_root, expected_counts={label: 2})
    executor.active_leases = 1

    assert FleetController(config, executor=executor).run() == 1

    recovered = json.loads(run_index.read_text(encoding="utf-8"))["runs"][0]
    assert recovered["status"] == "failed"
    assert recovered["run_exit_code"] == 7
    assert recovered["run_exit_code_source"] == "archived_exit_status"
    assert recovered["last_error"] == "run exit 7; result coverage 2/2"


def test_stop_failure_is_left_pending_without_tight_retry(tmp_path: Path) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [_planned(_run_spec(label))])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2},
        stop_code=1,
    )

    assert FleetController(config, executor=executor).run() == 1

    run = json.loads(run_index.read_text(encoding="utf-8"))["runs"][0]
    assert run["status"] == "stop_pending"
    assert run["verified_final_export"] is True
    assert len(executor.stops) == 1


def test_ambiguous_lease_inspect_error_does_not_duplicate_run(
    tmp_path: Path,
) -> None:
    label = "openclaw-gpt55-full-2-r1-20260727"
    run = _planned(_run_spec(label))
    run["status"] = "running"
    run["lease"] = {
        "id": "cbx_existing",
        "slug": "existing",
        "provider": "aws",
        "state": "active",
    }
    run_index = tmp_path / "manifests" / "run_index.json"
    _write_index(run_index, [run])
    config = _config(tmp_path, run_index)
    executor = FakeExecutor(
        config.local_root,
        expected_counts={label: 2},
        inspect_errors={"cbx_existing"},
    )

    assert FleetController(config, executor=executor).run() == 1

    runs = json.loads(run_index.read_text(encoding="utf-8"))["runs"]
    assert len(runs) == 1
    assert runs[0]["status"] == "recovery_required"
    assert executor.dispatches == []
    assert executor.stops == []
    assert not any(command[:2] == ["crabbox", "warmup"] for command in executor.commands)
