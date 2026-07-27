from __future__ import annotations

import json
import subprocess
import tarfile
import threading
from pathlib import Path
from typing import Sequence

from scripts.native_eval.fleet import FleetConfig, FleetController
from scripts.native_eval.models import RunSpec


def _run_spec(label: str, *, expected_task_count: int = 2) -> RunSpec:
    return RunSpec(
        run_label=label,
        harness="openclaw",
        harness_version="test",
        model_slug="gpt55",
        model_id="gpt-5.5",
        provider="openai",
        proxy_model_name="sb-gpt55",
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
        "attempt": 0,
        "status": "planned",
        "leaderboard_eligible": None,
        "rerun_of": None,
        "lease": None,
        "artifacts": [],
    }


def _write_archive(path: Path) -> None:
    source = path.parent / f"{path.stem}-source"
    source.mkdir()
    (source / "marker.txt").write_text("pinned", encoding="utf-8")
    with tarfile.open(path, "w:gz") as handle:
        handle.add(source, arcname=".")


def _write_final(local_root: Path, run_label: str, result_count: int) -> None:
    source = local_root / f".archive-{run_label}"
    job = source / "results" / "jobs" / run_label
    for index in range(result_count):
        trial = job / f"task-{index}__trial"
        trial.mkdir(parents=True, exist_ok=True)
        (trial / "result.json").write_text("{}", encoding="utf-8")
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
        stop_code: int = 0,
        inspect_errors: set[str] | None = None,
    ) -> None:
        self.local_root = local_root
        self.expected_counts = expected_counts
        self.checkpoint_codes = checkpoint_codes or {}
        self.omit_final = omit_final or set()
        self.remote_states = {label: "running" for label in (running_labels or set())}
        self.stop_code = stop_code
        self.inspect_errors = inspect_errors or set()
        self.leases: dict[str, dict[str, object]] = {}
        self.commands: list[list[str]] = []
        self.events: list[tuple[str, str]] = []
        self.dispatches: list[str] = []
        self.stops: list[str] = []
        self._lock = threading.Lock()
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
                return _result(argv, 1, stderr="not found")
            return _result(argv, 0, stdout=json.dumps(lease))

        if argv[:2] == ["crabbox", "warmup"]:
            slug = argv[argv.index("--slug") + 1]
            with self._lock:
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
            with self._lock:
                self.dispatches.append(label)
                self.events.append(("dispatch", label))
                self.remote_states[label] = "running"
            return _result(argv, 0, stdout="1234")

        return _result(argv, 0)


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
        checkpoint_poll_seconds=1,
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
    warmup = next(command for command in executor.commands if "warmup" in command)
    assert warmup[:2] == ["env", "CRABBOX_CAPACITY_MARKET=on-demand"]
    assert "--market" not in warmup
    for label in labels:
        checkpoint_at = executor.events.index(("checkpoint", label))
        lease_id = index["runs"][labels.index(label)]["lease"]["id"]
        stop_at = executor.events.index(("stop", lease_id))
        assert checkpoint_at < stop_at
    assert "TOPSECRET" not in "\n".join(" ".join(command) for command in executor.commands)


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
