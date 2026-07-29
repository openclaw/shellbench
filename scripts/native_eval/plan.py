from __future__ import annotations

import argparse
from pathlib import Path

from scripts.native_eval.models import build_matrix_plan
from scripts.native_eval.runtime import atomic_write_json, utc_now
from scripts.native_eval.tasks import validate_suite


def write_run_index(
    *,
    tasks_root: Path,
    output: Path,
    public_tasks_commit: str,
    run_date: str,
) -> list[dict[str, object]]:
    tasks = validate_suite(tasks_root)
    runs = build_matrix_plan(len(tasks), run_date=run_date)
    entries = [
        {
            **run.to_dict(),
            "attempt": 0,
            "status": "planned",
            "leaderboard_eligible": None,
            "rerun_of": None,
            "lease": None,
            "artifacts": [],
        }
        for run in runs
    ]
    atomic_write_json(
        output,
        {
            "created_at_utc": utc_now(),
            "public_tasks_commit": public_tasks_commit,
            "task_suite_path": "combined tasks/tasks",
            "expected_task_count": len(tasks),
            "planned_run_count": len(entries),
            "runs": entries,
        },
    )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-tasks-commit", required=True)
    parser.add_argument("--run-date", required=True)
    args = parser.parse_args()
    entries = write_run_index(
        tasks_root=args.tasks_root,
        output=args.output,
        public_tasks_commit=args.public_tasks_commit,
        run_date=args.run_date,
    )
    print(f"wrote {len(entries)} planned runs to {args.output}")


if __name__ == "__main__":
    main()
