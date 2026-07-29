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
    phase: str = "full",
    task_names: Sequence[str] | None = None,
    qualification_family: str | None = None,
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
    selected_task_names = list(task_names or [])
    suite_task_names = {task.name for task in tasks} if selected_task_names else set()
    unknown_tasks = set(selected_task_names) - suite_task_names
    if unknown_tasks:
        raise ValueError(f"unknown task names: {', '.join(sorted(unknown_tasks))}")
    if len(set(selected_task_names)) != len(selected_task_names):
        raise ValueError("task names must not contain duplicates")
    if phase == "r0":
        if len(selected_harnesses) != 1 or len(selected_models) != 1:
            raise ValueError("r0 plans require exactly one harness and one model")
        if len(selected_task_names) != 10:
            raise ValueError("r0 plans require exactly 10 named tasks")
        if not qualification_family:
            raise ValueError("r0 plans require qualification_family")
        planned_repetitions = (0,)
        expected_task_count = len(selected_task_names)
        run_kind = "smoke"
    elif phase == "full":
        if selected_task_names:
            raise ValueError("full plans do not accept task subsets")
        if qualification_family:
            raise ValueError("qualification_family is only valid for r0 plans")
        planned_repetitions = tuple(range(1, repetitions + 1))
        expected_task_count = len(tasks)
        run_kind = "full"
    else:
        raise ValueError(f"unsupported phase: {phase}")
    runs = build_matrix_plan(
        expected_task_count,
        run_date=run_date,
        harnesses=selected_harnesses,
        models=selected_models,
        repetitions=planned_repetitions,
        reasoning_effort=reasoning_effort,
        run_kind=run_kind,
    )
    entries = [
        {
            **run.to_dict(),
            "reasoning_effort": reasoning_effort,
            "judge_model_id": judge_model_id,
            "judge_reasoning_effort": judge_reasoning_effort,
            "phase": phase,
            "qualification_family": qualification_family,
            "task_names": selected_task_names or None,
            "attempt": 0,
            "status": "planned",
            "leaderboard_eligible": False if phase == "r0" else None,
            "exclusion_reason": (
                "r0_non_scoring_qualification" if phase == "r0" else None
            ),
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
            "repetition_count": len(planned_repetitions),
            "planned_repetitions": list(planned_repetitions),
            "phase": phase,
            "qualification_family": qualification_family,
            "qualification_task_names": selected_task_names,
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
    parser.add_argument("--phase", choices=("r0", "full"), default="full")
    parser.add_argument("--qualification-family")
    parser.add_argument(
        "--task",
        action="append",
        dest="task_names",
        help="Select one r0 qualification task. Repeat exactly ten times.",
    )
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
        phase=args.phase,
        task_names=args.task_names,
        qualification_family=args.qualification_family,
    )
    print(f"wrote {len(entries)} planned runs to {args.output}")


if __name__ == "__main__":
    main()
