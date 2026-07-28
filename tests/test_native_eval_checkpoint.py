from __future__ import annotations

import io
from argparse import Namespace
import subprocess
import tarfile
from pathlib import Path

import pytest

from scripts.native_eval import checkpoint_loop
from scripts.native_eval.checkpoint_loop import (
    RemoteState,
    partial_path,
    pull_checkpoint,
    pull_final,
)


def _archive_bytes(run_label: str) -> bytes:
    payload = io.BytesIO()
    result = b"{}"
    with tarfile.open(fileobj=payload, mode="w:gz") as handle:
        member = tarfile.TarInfo(
            f"results/jobs/{run_label}/task__trial/result.json"
        )
        member.size = len(result)
        handle.addfile(member, io.BytesIO(result))
    return payload.getvalue()


class InterruptedCopyClient:
    def __init__(self, run_label: str) -> None:
        self.run_label = run_label
        self.copy_attempts = 0

    def run(self, _command: list[str]) -> str:
        return f"/tmp/{self.run_label}-checkpoint.tar.gz"

    def copy_from(self, _remote_path: str, local_path: Path) -> None:
        self.copy_attempts += 1
        assert local_path.name.endswith(".partial")
        if self.copy_attempts == 1:
            local_path.write_bytes(b"interrupted")
            raise subprocess.CalledProcessError(1, ["scp"])
        local_path.write_bytes(_archive_bytes(self.run_label))


class RecordingCopyClient:
    def __init__(self, run_label: str, final_paths: list[Path]) -> None:
        self.run_label = run_label
        self.final_paths = final_paths
        self.destinations: list[Path] = []

    def copy_from(self, remote_path: str, local_path: Path) -> None:
        assert all(not path.exists() for path in self.final_paths)
        assert local_path.name.endswith(".partial")
        self.destinations.append(local_path)
        if remote_path.endswith(".tar.gz"):
            local_path.write_bytes(_archive_bytes(self.run_label))
        else:
            local_path.write_text(remote_path, encoding="utf-8")


def test_interrupted_checkpoint_copy_replaces_stale_partial_on_retry(
    tmp_path: Path,
) -> None:
    run_label = "openclaw-gpt55-full-1-r1-20260727"
    raw_dir = tmp_path / "raw"
    logs_dir = tmp_path / "logs"
    raw_dir.mkdir()
    logs_dir.mkdir()
    log_path = logs_dir / f"{run_label}.checkpoints.log"
    final_path = raw_dir / f"{run_label}-checkpoint-0001-artifacts.tar.gz"
    client = InterruptedCopyClient(run_label)

    with pytest.raises(subprocess.CalledProcessError):
        pull_checkpoint(
            client=client,
            runner_root="/runner",
            remote_root="/remote",
            run_label=run_label,
            raw_dir=raw_dir,
            log_path=log_path,
            sequence=1,
        )

    assert not final_path.exists()
    assert partial_path(final_path).read_bytes() == b"interrupted"

    assert pull_checkpoint(
        client=client,
        runner_root="/runner",
        remote_root="/remote",
        run_label=run_label,
        raw_dir=raw_dir,
        log_path=log_path,
        sequence=1,
    ) == 1
    assert final_path.is_file()
    assert not partial_path(final_path).exists()


def test_final_pull_publishes_only_after_all_partial_downloads_and_tar_verify(
    tmp_path: Path,
) -> None:
    run_label = "openclaw-gpt55-full-1-r1-20260727"
    raw_dir = tmp_path / "raw"
    logs_dir = tmp_path / "logs"
    raw_dir.mkdir()
    logs_dir.mkdir()
    log_path = logs_dir / f"{run_label}.checkpoints.log"
    final_paths = [
        raw_dir / f"{run_label}-final-artifacts.tar.gz",
        logs_dir / f"{run_label}.stdout.log",
        logs_dir / f"{run_label}.stderr.log",
    ]
    client = RecordingCopyClient(run_label, final_paths)

    assert pull_final(
        client=client,
        remote_root="/remote",
        run_label=run_label,
        raw_dir=raw_dir,
        logs_dir=logs_dir,
        log_path=log_path,
    ) == 1

    assert all(path.is_file() for path in final_paths)
    assert client.destinations == [partial_path(path) for path in final_paths]
    assert all(not partial_path(path).exists() for path in final_paths)


def test_done_final_copy_failure_logs_recovery_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_label = "openclaw-gpt55-full-1-r1-20260727"
    args = Namespace(
        local_root=tmp_path,
        control_path=tmp_path / "control.sock",
        target="example",
        port=22,
        identity_file=tmp_path / "identity",
        remote_root="/remote",
        runner_root="/runner",
        run_label=run_label,
        poll_seconds=0,
        interval_seconds=60,
        trial_increment=10,
        max_ssh_failures=3,
    )

    monkeypatch.setattr(checkpoint_loop, "SSHClient", lambda **_kwargs: object())
    monkeypatch.setattr(
        checkpoint_loop,
        "read_remote_state",
        lambda *_args: RemoteState(0, 0, True, 0),
    )

    def fail_final(**_kwargs: object) -> int:
        raise subprocess.CalledProcessError(1, ["scp"])

    diagnostic_checkpoints: list[int] = []

    def pull_diagnostic(**kwargs: object) -> int:
        diagnostic_checkpoints.append(int(kwargs["sequence"]))
        return 0

    monkeypatch.setattr(checkpoint_loop, "pull_final", fail_final)
    monkeypatch.setattr(checkpoint_loop, "pull_checkpoint", pull_diagnostic)

    assert checkpoint_loop.run_loop(args) == 75
    assert diagnostic_checkpoints == [1]
    log_path = tmp_path / "logs" / f"{run_label}.checkpoints.log"
    assert "\tfinal_error\t" in log_path.read_text(encoding="utf-8")
