from __future__ import annotations

import argparse
import re
import shlex
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath


@dataclass(frozen=True)
class RemoteState:
    result_count: int
    done: bool
    exit_status: int | None


def next_checkpoint_sequence(raw_dir: Path, run_label: str) -> int:
    pattern = re.compile(
        rf"^{re.escape(run_label)}-checkpoint-(\d{{4}})-artifacts\.tar\.gz$"
    )
    sequences = [
        int(match.group(1))
        for path in raw_dir.glob(f"{run_label}-checkpoint-*-artifacts.tar.gz")
        if (match := pattern.match(path.name))
    ]
    return max(sequences, default=0) + 1


def count_result_json(archive: Path) -> int:
    with tarfile.open(archive, "r:gz") as handle:
        count = 0
        for member in handle.getmembers():
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            for index, part in enumerate(parts):
                if part != "jobs":
                    continue
                relative = parts[index:]
                if len(relative) == 4 and relative[-1] == "result.json":
                    count += 1
                break
        return count


class SSHClient:
    def __init__(
        self,
        *,
        target: str,
        port: int,
        identity_file: Path,
        control_path: Path,
    ) -> None:
        common_options = [
            "-i",
            str(identity_file),
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
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=12h",
            "-o",
            f"ControlPath={control_path}",
        ]
        self.ssh_options = ["-p", str(port), *common_options]
        self.scp_options = ["-P", str(port), *common_options]
        self.target = target

    def run(self, command: list[str]) -> str:
        result = subprocess.run(
            ["ssh", *self.ssh_options, self.target, shlex.join(command)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()

    def copy_from(self, remote_path: str, local_path: Path) -> None:
        subprocess.run(
            [
                "scp",
                *self.scp_options,
                f"{self.target}:{remote_path}",
                str(local_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )


def read_remote_state(client: SSHClient, remote_root: str, run_label: str) -> RemoteState:
    script = """
set -eu
job_dir=$1
state_dir=$2
count=0
if [ -d "$job_dir" ]; then
  count=$(find "$job_dir" -mindepth 2 -maxdepth 2 -name result.json | wc -l | tr -d ' ')
fi
done_flag=0
[ -f "$state_dir/done" ] && done_flag=1
status=-
[ -f "$state_dir/exit_status" ] && status=$(cat "$state_dir/exit_status")
printf '%s\\t%s\\t%s\\n' "$count" "$done_flag" "$status"
"""
    output = client.run(
        [
            "bash",
            "-c",
            script,
            "checkpoint-state",
            f"{remote_root}/results/jobs/{run_label}",
            f"/tmp/shellbench-runs/{run_label}",
        ]
    )
    count, done, status = output.split("\t")
    return RemoteState(
        result_count=int(count),
        done=done == "1",
        exit_status=None if status == "-" else int(status),
    )


def append_checkpoint_log(
    log_path: Path,
    *,
    event: str,
    archive_name: str,
    result_count: int,
    detail: str = "",
) -> None:
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with log_path.open("a") as handle:
        handle.write(
            f"{timestamp}\t{event}\t{archive_name}\t{result_count}\t{detail}\n"
        )


def pull_checkpoint(
    *,
    client: SSHClient,
    runner_root: str,
    remote_root: str,
    run_label: str,
    raw_dir: Path,
    log_path: Path,
    sequence: int,
) -> int:
    archive_name = (
        f"{run_label}-checkpoint-{sequence:04d}-artifacts.tar.gz"
    )
    remote_script = f"{runner_root}/scripts/native_eval/remote_checkpoint.sh"
    remote_path = client.run(
        [remote_script, remote_root, run_label, archive_name]
    )
    local_path = raw_dir / archive_name
    if local_path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {local_path}")
    client.copy_from(remote_path, local_path)
    result_count = count_result_json(local_path)
    append_checkpoint_log(
        log_path,
        event="checkpoint",
        archive_name=archive_name,
        result_count=result_count,
    )
    return result_count


def pull_final(
    *,
    client: SSHClient,
    run_label: str,
    raw_dir: Path,
    log_path: Path,
) -> int:
    archive_name = f"{run_label}-final-artifacts.tar.gz"
    local_path = raw_dir / archive_name
    if local_path.exists():
        raise FileExistsError(f"refusing to overwrite final archive: {local_path}")
    client.copy_from(f"/tmp/{archive_name}", local_path)
    result_count = count_result_json(local_path)
    append_checkpoint_log(
        log_path,
        event="final",
        archive_name=archive_name,
        result_count=result_count,
    )
    return result_count


def run_loop(args: argparse.Namespace) -> int:
    local_root = args.local_root.resolve()
    raw_dir = local_root / "raw"
    logs_dir = local_root / "logs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    args.control_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{args.run_label}.checkpoints.log"
    client = SSHClient(
        target=args.target,
        port=args.port,
        identity_file=args.identity_file,
        control_path=args.control_path,
    )

    sequence = next_checkpoint_sequence(raw_dir, args.run_label)
    last_count = 0
    last_checkpoint_at = 0.0
    failures = 0

    while True:
        try:
            state = read_remote_state(client, args.remote_root, args.run_label)
            failures = 0
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            failures += 1
            append_checkpoint_log(
                log_path,
                event="ssh_error",
                archive_name="",
                result_count=last_count,
                detail=str(exc).replace("\n", " ")[:500],
            )
            if failures >= args.max_ssh_failures:
                append_checkpoint_log(
                    log_path,
                    event="lease_lost",
                    archive_name="",
                    result_count=last_count,
                )
                return 75
            time.sleep(args.poll_seconds)
            continue

        now = time.monotonic()
        checkpoint_due = (
            state.result_count >= 1
            and (
                last_checkpoint_at == 0
                or state.result_count - last_count >= args.trial_increment
                or now - last_checkpoint_at >= args.interval_seconds
            )
        )
        if checkpoint_due:
            try:
                last_count = pull_checkpoint(
                    client=client,
                    runner_root=args.runner_root,
                    remote_root=args.remote_root,
                    run_label=args.run_label,
                    raw_dir=raw_dir,
                    log_path=log_path,
                    sequence=sequence,
                )
                sequence += 1
                last_checkpoint_at = time.monotonic()
            except (OSError, subprocess.SubprocessError, tarfile.TarError) as exc:
                append_checkpoint_log(
                    log_path,
                    event="checkpoint_error",
                    archive_name="",
                    result_count=last_count,
                    detail=str(exc).replace("\n", " ")[:500],
                )

        if state.done:
            if state.result_count != last_count and state.result_count:
                last_count = pull_checkpoint(
                    client=client,
                    runner_root=args.runner_root,
                    remote_root=args.remote_root,
                    run_label=args.run_label,
                    raw_dir=raw_dir,
                    log_path=log_path,
                    sequence=sequence,
                )
            final_count = pull_final(
                client=client,
                run_label=args.run_label,
                raw_dir=raw_dir,
                log_path=log_path,
            )
            return state.exit_status or (0 if final_count else 1)

        time.sleep(args.poll_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--control-path", type=Path, required=True)
    parser.add_argument("--runner-root", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--local-root", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--trial-increment", type=int, default=10)
    parser.add_argument("--max-ssh-failures", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    return run_loop(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
