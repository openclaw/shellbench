from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from scripts.native_eval.models import (
    RunSpec,
    harness_by_name,
    model_by_slug,
    trajectory_mode_for_harness,
)
from scripts.native_eval.runtime import atomic_write_json, run_trial, utc_now
from scripts.native_eval.tasks import TaskSpec, validate_suite


async def run_job(
    *,
    tasks_root: Path,
    jobs_dir: Path,
    run: RunSpec,
    public_tasks_commit: str,
    task_suite_path: str,
    toolchain_root: Path,
    proxy_url: str,
    proxy_key: str,
    concurrency: int,
    task_names: set[str] | None = None,
) -> dict[str, Any]:
    tasks = validate_suite(tasks_root)
    if task_names:
        tasks = [task for task in tasks if task.name in task_names]
        missing = task_names - {task.name for task in tasks}
        if missing:
            raise ValueError(f"Unknown task names: {', '.join(sorted(missing))}")
    if len(tasks) != run.expected_task_count:
        raise ValueError(
            f"Expected {run.expected_task_count} tasks, found {len(tasks)}"
        )
    if not toolchain_root.is_dir():
        raise FileNotFoundError(f"Toolchain not found: {toolchain_root}")
    if not proxy_key:
        raise ValueError("SHELLBENCH_PROXY_KEY is required")

    job_dir = jobs_dir / run.run_label
    job_dir.mkdir(parents=True, exist_ok=False)
    job_id = str(uuid.uuid4())
    started_at = utc_now()
    manifest = _run_manifest(
        run,
        public_tasks_commit=public_tasks_commit,
        task_suite_path=task_suite_path,
        concurrency=concurrency,
        started_at=started_at,
        tasks_root=tasks_root,
        tasks=tasks,
    )
    atomic_write_json(job_dir / "run_manifest.json", manifest)
    atomic_write_json(
        job_dir / "config.json",
        {
            "job_name": run.run_label,
            "jobs_dir": str(jobs_dir),
            "n_concurrent_trials": concurrency,
            "trials": [],
            "native_run": asdict(run),
        },
    )
    atomic_write_json(
        job_dir / "lock.json",
        {
            "schema_version": 3,
            "created_at": started_at,
            "harbor": {
                "version": None,
                "git_commit_hash": None,
                "is_editable": None,
            },
            "n_concurrent_trials": concurrency,
            "retry": {"max_retries": 0},
            "trials": [],
            "native_runner": {
                "git_commit": manifest["runner_commit"],
                "patch_hash": manifest["runner_patch_hash"],
            },
        },
    )

    state: dict[str, Any] = {
        "id": job_id,
        "started_at": started_at,
        "updated_at": started_at,
        "finished_at": None,
        "n_total_trials": len(tasks),
        "stats": _empty_stats(len(tasks)),
    }
    atomic_write_json(job_dir / "result.json", state)

    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    result_lock = asyncio.Lock()

    async def execute(task: TaskSpec) -> None:
        async with semaphore:
            result = await run_trial(
                task,
                run,
                job_dir=job_dir,
                toolchain_root=toolchain_root,
                proxy_url=proxy_url,
                proxy_key=proxy_key,
            )
        async with result_lock:
            results.append(result)
            _update_job_result(state, results, len(tasks))
            atomic_write_json(job_dir / "result.json", state)

    await asyncio.gather(*(execute(task) for task in tasks))
    state["finished_at"] = utc_now()
    state["updated_at"] = state["finished_at"]
    _update_job_result(state, results, len(tasks))
    atomic_write_json(job_dir / "result.json", state)
    manifest["finished_at_utc"] = state["finished_at"]
    manifest["result_json_count"] = len(results)
    agent_results = [
        result.get("agent_result")
        for result in results
        if isinstance(result.get("agent_result"), dict)
    ]
    manifest["canonical_model_identity"] = bool(agent_results) and all(
        result.get("canonical_model_identity") is True for result in agent_results
    )
    manifest["observed_model_ids"] = sorted(
        {
            str(result.get("runtime_model_name"))
            for result in agent_results
            if result.get("runtime_model_name")
        }
    )
    manifest["trajectory_complete"] = bool(agent_results) and all(
        result.get("trajectory_status") == "real" for result in agent_results
    )
    atomic_write_json(job_dir / "run_manifest.json", manifest)
    return state


def _empty_stats(total: int) -> dict[str, Any]:
    return {
        "n_completed_trials": 0,
        "n_errored_trials": 0,
        "n_running_trials": 0,
        "n_pending_trials": total,
        "n_cancelled_trials": 0,
        "n_retries": 0,
        "evals": {},
        "n_input_tokens": None,
        "n_cache_tokens": None,
        "n_output_tokens": None,
        "cost_usd": None,
    }


def _update_job_result(
    state: dict[str, Any],
    results: list[dict[str, Any]],
    total: int,
) -> None:
    stats = state["stats"]
    stats["n_completed_trials"] = len(results)
    stats["n_errored_trials"] = sum(
        result.get("exception_info") is not None for result in results
    )
    stats["n_pending_trials"] = total - len(results)
    for field in (
        "n_input_tokens",
        "n_cache_tokens",
        "n_output_tokens",
        "cost_usd",
    ):
        values = [
            (result.get("agent_result") or {}).get(field)
            for result in results
            if isinstance((result.get("agent_result") or {}).get(field), (int, float))
        ]
        stats[field] = sum(values) if values else None
    state["updated_at"] = utc_now()


def _run_manifest(
    run: RunSpec,
    *,
    public_tasks_commit: str,
    task_suite_path: str,
    concurrency: int,
    started_at: str,
    tasks_root: Path,
    tasks: list[TaskSpec],
) -> dict[str, Any]:
    return {
        "run_label": run.run_label,
        "harness": run.harness,
        "harness_version": run.harness_version,
        "model_slug": run.model_slug,
        "model_id": run.model_id,
        "provider_model_id": run.model_id,
        "model_provider": run.provider,
        "proxy_model_name": run.proxy_model_name,
        "repetition": run.repetition,
        "task_suite": task_suite_path,
        "task_suite_root": str(tasks_root.resolve()),
        "expected_task_count": run.expected_task_count,
        "tasks": [
            {
                "name": task.name,
                "path": str(task.path.resolve()),
                "checksum": task.checksum,
            }
            for task in tasks
        ],
        "task_concurrency": concurrency,
        "agent_concurrency": concurrency,
        "provider": "aws",
        "runner": "shellbench-native",
        "execution_mode": os.environ.get("SHELLBENCH_EXECUTION_MODE", "native"),
        "canonical_model_identity": None,
        "intended_model_identity": {
            "provider": run.provider,
            "provider_model_id": run.model_id,
            "proxy_model_name": run.proxy_model_name,
        },
        "observed_model_ids": [],
        "trajectory_mode": trajectory_mode_for_harness(run.harness),
        "trajectory_complete": False,
        "parity_validated": (
            os.environ.get("SHELLBENCH_PARITY_VALIDATED", "").lower() == "true"
        ),
        "harbor_reference_commit": os.environ.get(
            "SHELLBENCH_HARBOR_REFERENCE_COMMIT"
        ),
        "judge_model_id": os.environ.get("SHELLBENCH_JUDGE_MODEL_ID"),
        "reasoning_effort": os.environ.get("SHELLBENCH_REASONING_EFFORT"),
        "judge_reasoning_effort": os.environ.get(
            "SHELLBENCH_JUDGE_REASONING_EFFORT"
        ),
        "runner_commit": _git_commit(),
        "runner_patch_hash": _runner_patch_hash(),
        "public_tasks_commit": public_tasks_commit,
        "crabbox_cli_version": os.environ.get("CRABBOX_CLI_VERSION"),
        "crabbox_slug": os.environ.get("CRABBOX_SLUG"),
        "crabbox_lease_id": os.environ.get("CRABBOX_LEASE_ID"),
        "crabbox_instance_type": os.environ.get("CRABBOX_INSTANCE_TYPE"),
        "crabbox_ip": os.environ.get("CRABBOX_IP"),
        "crabbox_region": os.environ.get("CRABBOX_REGION"),
        "started_at_utc": started_at,
        "finished_at_utc": None,
        "result_json_count": 0,
    }


def _git_commit() -> str:
    configured = os.environ.get("SHELLBENCH_RUNNER_COMMIT", "").strip()
    if configured:
        return configured
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def _runner_patch_hash() -> str:
    paths = sorted(Path("scripts/native_eval").glob("*"))
    digest = subprocess.run(
        ["git", "hash-object", *[str(path) for path in paths if path.is_file()]],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    ).stdout
    import hashlib

    return hashlib.sha256(digest.encode("utf-8")).hexdigest()


def build_run_spec(args: argparse.Namespace) -> RunSpec:
    harness = harness_by_name(args.harness)
    model = model_by_slug(args.model_slug)
    return RunSpec(
        run_label=args.run_label,
        harness=harness.name,
        harness_version=args.harness_version or harness.version,
        model_slug=model.slug,
        model_id=args.model_id or model.provider_model_id,
        provider=args.model_provider or model.provider,
        proxy_model_name=args.proxy_model_name or model.proxy_model_name,
        repetition=args.repetition,
        expected_task_count=args.expected_task_count,
        run_date=args.run_date,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--jobs-dir", type=Path, required=True)
    parser.add_argument("--run-label", required=True)
    parser.add_argument("--harness", required=True)
    parser.add_argument("--harness-version")
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--model-provider")
    parser.add_argument("--proxy-model-name")
    parser.add_argument("--repetition", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--expected-task-count", type=int, required=True)
    parser.add_argument("--public-tasks-commit", required=True)
    parser.add_argument("--task-suite-path", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument(
        "--toolchain-root",
        type=Path,
        default=Path("/opt/shellbench-native"),
    )
    parser.add_argument(
        "--proxy-url",
        default="http://host.docker.internal:4000",
    )
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument(
        "--task",
        action="append",
        dest="task_names",
        help="Run only the named task. Repeat for multiple tasks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    proxy_key = os.environ.get("SHELLBENCH_PROXY_KEY", "")
    state = asyncio.run(
        run_job(
            tasks_root=args.tasks_root,
            jobs_dir=args.jobs_dir,
            run=build_run_spec(args),
            public_tasks_commit=args.public_tasks_commit,
            task_suite_path=args.task_suite_path,
            toolchain_root=args.toolchain_root,
            proxy_url=args.proxy_url,
            proxy_key=proxy_key,
            concurrency=args.concurrency,
            task_names=set(args.task_names or []),
        )
    )
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
