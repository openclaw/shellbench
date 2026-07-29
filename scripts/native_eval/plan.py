from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from scripts.native_eval.models import HARNESSES, MODELS, build_matrix_plan
from scripts.native_eval.runtime import atomic_write_json, utc_now
from scripts.native_eval.tasks import validate_suite


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def write_run_index(
    *,
    tasks_root: Path,
    output: Path,
    public_tasks_commit: str,
    run_date: str,
    reasoning_effort: str,
    judge_reasoning_effort: str,
    judge_model_id: str = "gpt-5.6-sol",
    repetitions: int = 3,
    harness_names: Sequence[str] | None = None,
    model_slugs: Sequence[str] | None = None,
) -> list[dict[str, object]]:
    tasks = validate_suite(tasks_root)
    selected_harnesses = tuple(
        harness
        for harness in HARNESSES
        if harness_names is None or harness.name in harness_names
    )
    selected_models = tuple(
        model
        for model in MODELS
        if model_slugs is None or model.slug in model_slugs
    )
    if not selected_harnesses:
        raise ValueError("no harnesses selected")
    if not selected_models:
        raise ValueError("no models selected")
    runs = build_matrix_plan(
        len(tasks),
        run_date=run_date,
        harnesses=selected_harnesses,
        models=selected_models,
        repetitions=range(1, repetitions + 1),
        reasoning_effort=reasoning_effort,
    )
    entries = [
        {
            **run.to_dict(),
            "reasoning_effort": reasoning_effort,
            "judge_model_id": judge_model_id,
            "judge_reasoning_effort": judge_reasoning_effort,
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
            "repetition_count": repetitions,
            "reasoning_effort": reasoning_effort,
            "judge_model_id": judge_model_id,
            "judge_reasoning_effort": judge_reasoning_effort,
            "harnesses": [harness.name for harness in selected_harnesses],
            "models": [model.slug for model in selected_models],
            "runs": entries,
        },
    )
    return entries


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-tasks-commit", required=True)
    parser.add_argument("--run-date", required=True)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        required=True,
    )
    parser.add_argument(
        "--judge-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
    )
    parser.add_argument("--judge-model-id", default="gpt-5.6-sol")
    parser.add_argument("--repetitions", type=_positive_int, default=3)
    parser.add_argument(
        "--harness",
        action="append",
        choices=tuple(harness.name for harness in HARNESSES),
        dest="harness_names",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=tuple(model.slug for model in MODELS),
        dest="model_slugs",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    entries = write_run_index(
        tasks_root=args.tasks_root,
        output=args.output,
        public_tasks_commit=args.public_tasks_commit,
        run_date=args.run_date,
        reasoning_effort=args.reasoning_effort,
        judge_model_id=args.judge_model_id,
        judge_reasoning_effort=(
            args.judge_reasoning_effort or args.reasoning_effort
        ),
        repetitions=args.repetitions,
        harness_names=args.harness_names,
        model_slugs=args.model_slugs,
    )
    print(f"wrote {len(entries)} planned runs to {args.output}")


if __name__ == "__main__":
    main()
