"""Task-level analysis for native ShellBench matrix exports."""

from __future__ import annotations

import csv
import itertools
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


RUN_REQUIRED_FIELDS = {
    "run_label",
    "harness",
    "model_slug",
    "reasoning_effort",
    "task_revision",
    "expected_task_count",
    "score",
    "eligible",
}
TASK_REQUIRED_FIELDS = {
    "run_label",
    "task_name",
    "classification",
    "reward",
}
REASONING_ORDER = ("low", "medium", "high", "xhigh")

CELL_FIELDS = (
    "dataset_key",
    "task_revision",
    "suite_task_count",
    "harness",
    "model_slug",
    "model_id",
    "reasoning_effort",
    "task_name",
    "repetitions",
    "mean_reward",
    "reward_stdev",
    "min_reward",
    "max_reward",
    "nonzero_repetitions",
    "exact_passes",
)
TASK_VARIANCE_FIELDS = (
    "dataset_key",
    "task_revision",
    "suite_task_count",
    "task_name",
    "cell_count",
    "nonzero_cells",
    "mean_cell_reward",
    "cell_reward_stdev",
    "min_cell_reward",
    "max_cell_reward",
    "range_across_cells",
    "best_cell",
    "worst_cell",
)
HARNESS_DELTA_FIELDS = (
    "dataset_key",
    "task_revision",
    "suite_task_count",
    "model_slug",
    "reasoning_effort",
    "task_name",
    "left_harness",
    "right_harness",
    "left_reward",
    "right_reward",
    "delta",
)
REASONING_DELTA_FIELDS = (
    "dataset_key",
    "task_revision",
    "suite_task_count",
    "harness",
    "model_slug",
    "task_name",
    "lower_reasoning",
    "higher_reasoning",
    "lower_reward",
    "higher_reward",
    "delta",
)


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required analysis input does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(required - fields)
        if missing:
            raise ValueError(f"{path.name} is missing required columns: {', '.join(missing)}")
        return list(reader)


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _integer(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _boolean(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _stdev(values: Sequence[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _dataset_key(run: dict[str, str]) -> str:
    revision = run["task_revision"].strip() or "unknown-revision"
    count = _integer(run["expected_task_count"])
    return f"{revision}:{count}"


def _cell_label(row: dict[str, Any]) -> str:
    return "/".join(
        (
            str(row["harness"]),
            str(row["model_slug"]),
            str(row["reasoning_effort"]),
        )
    )


def _reasoning_rank(value: str) -> tuple[int, str]:
    try:
        return REASONING_ORDER.index(value), value
    except ValueError:
        return len(REASONING_ORDER), value


def _task_cells(
    runs: Sequence[dict[str, str]],
    tasks: Sequence[dict[str, str]],
) -> tuple[list[dict[str, Any]], set[str]]:
    eligible_runs = {row["run_label"]: row for row in runs if _boolean(row["eligible"])}
    grouped: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(list)
    model_ids: dict[tuple[str, str, str, str, str], set[str]] = defaultdict(set)

    for task in tasks:
        run = eligible_runs.get(task["run_label"])
        if run is None:
            continue
        key = (
            _dataset_key(run),
            run["harness"],
            run["model_slug"],
            run["reasoning_effort"],
            task["task_name"],
        )
        grouped[key].append(_number(task["reward"]))
        if run.get("model_id"):
            model_ids[key].add(run["model_id"])

    rows: list[dict[str, Any]] = []
    for key, rewards in sorted(grouped.items()):
        dataset_key, harness, model_slug, reasoning, task_name = key
        revision, count = dataset_key.rsplit(":", 1)
        ids = sorted(model_ids[key])
        rows.append(
            {
                "dataset_key": dataset_key,
                "task_revision": revision,
                "suite_task_count": int(count),
                "harness": harness,
                "model_slug": model_slug,
                "model_id": ",".join(ids),
                "reasoning_effort": reasoning,
                "task_name": task_name,
                "repetitions": len(rewards),
                "mean_reward": statistics.mean(rewards),
                "reward_stdev": _stdev(rewards),
                "min_reward": min(rewards),
                "max_reward": max(rewards),
                "nonzero_repetitions": sum(value > 0 for value in rewards),
                "exact_passes": sum(value >= 1 for value in rewards),
            }
        )
    return rows, set(eligible_runs)


def _task_variance(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        grouped[(str(cell["dataset_key"]), str(cell["task_name"]))].append(cell)

    rows = []
    for (dataset_key, task_name), task_cells in grouped.items():
        values = [float(cell["mean_reward"]) for cell in task_cells]
        best_reward = max(values)
        worst_reward = min(values)
        best_cells = sorted(
            _cell_label(cell)
            for cell in task_cells
            if float(cell["mean_reward"]) == best_reward
        )
        worst_cells = sorted(
            _cell_label(cell)
            for cell in task_cells
            if float(cell["mean_reward"]) == worst_reward
        )
        revision, count = dataset_key.rsplit(":", 1)
        rows.append(
            {
                "dataset_key": dataset_key,
                "task_revision": revision,
                "suite_task_count": int(count),
                "task_name": task_name,
                "cell_count": len(values),
                "nonzero_cells": sum(value > 0 for value in values),
                "mean_cell_reward": statistics.mean(values),
                "cell_reward_stdev": _stdev(values),
                "min_cell_reward": min(values),
                "max_cell_reward": max(values),
                "range_across_cells": max(values) - min(values),
                "best_cell": ";".join(best_cells),
                "worst_cell": ";".join(worst_cells),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            str(row["dataset_key"]),
            -float(row["range_across_cells"]),
            -float(row["cell_reward_stdev"]),
            str(row["task_name"]),
        ),
    )


def _harness_deltas(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for cell in cells:
        key = (
            str(cell["dataset_key"]),
            str(cell["model_slug"]),
            str(cell["reasoning_effort"]),
            str(cell["task_name"]),
        )
        grouped[key][str(cell["harness"])] = float(cell["mean_reward"])

    rows = []
    for (dataset_key, model, reasoning, task_name), values in sorted(grouped.items()):
        revision, count = dataset_key.rsplit(":", 1)
        for left, right in itertools.combinations(sorted(values), 2):
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "task_revision": revision,
                    "suite_task_count": int(count),
                    "model_slug": model,
                    "reasoning_effort": reasoning,
                    "task_name": task_name,
                    "left_harness": left,
                    "right_harness": right,
                    "left_reward": values[left],
                    "right_reward": values[right],
                    "delta": values[left] - values[right],
                }
            )
    return rows


def _reasoning_deltas(cells: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(dict)
    for cell in cells:
        key = (
            str(cell["dataset_key"]),
            str(cell["harness"]),
            str(cell["model_slug"]),
            str(cell["task_name"]),
        )
        grouped[key][str(cell["reasoning_effort"])] = float(cell["mean_reward"])

    rows = []
    for (dataset_key, harness, model, task_name), values in sorted(grouped.items()):
        revision, count = dataset_key.rsplit(":", 1)
        reasoning = sorted(values, key=_reasoning_rank)
        for lower, higher in zip(reasoning, reasoning[1:]):
            rows.append(
                {
                    "dataset_key": dataset_key,
                    "task_revision": revision,
                    "suite_task_count": int(count),
                    "harness": harness,
                    "model_slug": model,
                    "task_name": task_name,
                    "lower_reasoning": lower,
                    "higher_reasoning": higher,
                    "lower_reward": values[lower],
                    "higher_reward": values[higher],
                    "delta": values[higher] - values[lower],
                }
            )
    return rows


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-").lower()


def _short_cell_list(value: Any, limit: int = 3) -> str:
    cells = str(value).split(";")
    if len(cells) <= limit:
        return "; ".join(cells)
    return f"{'; '.join(cells[:limit])}; +{len(cells) - limit} tied"


def _plot_box(
    values_by_label: dict[str, list[float]],
    *,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = sorted(values_by_label)
    values = [values_by_label[label] for label in labels]
    width = min(24, max(9, len(labels) * 0.55))
    fig, ax = plt.subplots(figsize=(width, 6))
    ax.boxplot(values, tick_labels=labels, showmeans=True, patch_artist=True)
    ax.axhline(0, color="#374151", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def _generate_plots(
    runs: Sequence[dict[str, str]],
    cells: Sequence[dict[str, Any]],
    harness_deltas: Sequence[dict[str, Any]],
    reasoning_deltas: Sequence[dict[str, Any]],
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "plot generation requires matplotlib; install the project with .[analysis]"
        ) from exc

    paths = []
    datasets = sorted({str(cell["dataset_key"]) for cell in cells})
    for dataset_key in datasets:
        prefix = _safe_name(dataset_key)
        run_scores: dict[str, list[float]] = defaultdict(list)
        for run in runs:
            if _boolean(run["eligible"]) and _dataset_key(run) == dataset_key:
                label = "/".join(
                    (run["harness"], run["model_slug"], run["reasoning_effort"])
                )
                run_scores[label].append(_number(run["score"]))
        path = output_dir / f"{prefix}-run-score-distributions.svg"
        _plot_box(
            run_scores,
            title=f"Run score distributions ({dataset_key})",
            ylabel="Score",
            path=path,
        )
        paths.append(path)

        harness_values: dict[str, list[float]] = defaultdict(list)
        for row in harness_deltas:
            if row["dataset_key"] == dataset_key:
                label = f"{row['left_harness']} - {row['right_harness']}"
                harness_values[label].append(float(row["delta"]))
        if harness_values:
            path = output_dir / f"{prefix}-harness-task-deltas.svg"
            _plot_box(
                harness_values,
                title=f"Paired harness task deltas ({dataset_key})",
                ylabel="Left minus right task reward",
                path=path,
            )
            paths.append(path)

        reasoning_values: dict[str, list[float]] = defaultdict(list)
        for row in reasoning_deltas:
            if row["dataset_key"] == dataset_key:
                label = f"{row['lower_reasoning']} -> {row['higher_reasoning']}"
                reasoning_values[label].append(float(row["delta"]))
        if reasoning_values:
            path = output_dir / f"{prefix}-reasoning-task-deltas.svg"
            _plot_box(
                reasoning_values,
                title=f"Paired reasoning task deltas ({dataset_key})",
                ylabel="Higher minus lower task reward",
                path=path,
            )
            paths.append(path)

        task_values: dict[str, list[float]] = defaultdict(list)
        dataset_cells = [cell for cell in cells if cell["dataset_key"] == dataset_key]
        ranges: dict[str, float] = defaultdict(float)
        for cell in dataset_cells:
            task_values[str(cell["task_name"])].append(float(cell["mean_reward"]))
        for task_name, values in task_values.items():
            ranges[task_name] = max(values) - min(values)
        top_tasks = {
            task: task_values[task]
            for task in sorted(ranges, key=lambda task: (-ranges[task], task))[:20]
        }
        if top_tasks:
            path = output_dir / f"{prefix}-highest-variance-tasks.svg"
            _plot_box(
                top_tasks,
                title=f"Highest-variance task cell distributions ({dataset_key})",
                ylabel="Mean reward per experiment cell",
                path=path,
            )
            paths.append(path)
    return paths


def _dataset_summaries(
    runs: Sequence[dict[str, str]],
    cells: Sequence[dict[str, Any]],
    task_variance: Sequence[dict[str, Any]],
    harness_deltas: Sequence[dict[str, Any]],
    reasoning_deltas: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for dataset_key in sorted({_dataset_key(run) for run in runs}):
        dataset_runs = [run for run in runs if _dataset_key(run) == dataset_key]
        dataset_cells = [cell for cell in cells if cell["dataset_key"] == dataset_key]
        variance_rows = [
            row for row in task_variance if row["dataset_key"] == dataset_key
        ]
        harness_rows = [
            row for row in harness_deltas if row["dataset_key"] == dataset_key
        ]
        reasoning_rows = [
            row for row in reasoning_deltas if row["dataset_key"] == dataset_key
        ]
        revision, count = dataset_key.rsplit(":", 1)
        summaries.append(
            {
                "dataset_key": dataset_key,
                "task_revision": revision,
                "suite_task_count": int(count),
                "total_runs": len(dataset_runs),
                "eligible_runs": sum(_boolean(run["eligible"]) for run in dataset_runs),
                "excluded_runs": sum(not _boolean(run["eligible"]) for run in dataset_runs),
                "experiment_cells": len(
                    {
                        (
                            cell["harness"],
                            cell["model_slug"],
                            cell["reasoning_effort"],
                        )
                        for cell in dataset_cells
                    }
                ),
                "tasks": len({cell["task_name"] for cell in dataset_cells}),
                "high_variance_tasks": sum(
                    float(row["range_across_cells"]) >= 0.5 for row in variance_rows
                ),
                "harness_task_comparisons": len(harness_rows),
                "large_harness_deltas": sum(
                    abs(float(row["delta"])) >= 0.5 for row in harness_rows
                ),
                "reasoning_task_comparisons": len(reasoning_rows),
                "reasoning_improvements": sum(
                    float(row["delta"]) > 0 for row in reasoning_rows
                ),
                "reasoning_regressions": sum(
                    float(row["delta"]) < 0 for row in reasoning_rows
                ),
            }
        )
    return summaries


def _write_markdown(
    path: Path,
    datasets: Sequence[dict[str, Any]],
    task_variance: Sequence[dict[str, Any]],
    plot_paths: Sequence[Path],
) -> None:
    lines = [
        "# ShellBench task matrix analysis",
        "",
        "## TL;DR",
        "",
    ]
    for dataset in datasets:
        lines.append(
            "- `{dataset_key}`: {eligible_runs}/{total_runs} eligible runs, "
            "{experiment_cells} experiment cells, {tasks} tasks, "
            "{high_variance_tasks} tasks with a cross-cell range of at least 0.5, and "
            "{large_harness_deltas}/{harness_task_comparisons} paired harness "
            "comparisons with an absolute delta of at least 0.5.".format(**dataset)
        )
        lines.append(
            "  Reasoning changes improved {reasoning_improvements} matched task cells "
            "and regressed {reasoning_regressions}; treat reasoning as an experimental "
            "condition, not a monotonic quality ladder.".format(**dataset)
        )
    if len(datasets) > 1:
        lines.append(
            "- Multiple task revisions are present. They are analyzed separately and "
            "must not be pooled into one leaderboard."
        )

    lines.extend(
        [
            "",
            "## Datasets",
            "",
            "| Revision | Tasks | Eligible runs | Cells | High-variance tasks |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for dataset in datasets:
        lines.append(
            "| `{task_revision}` | {suite_task_count} | {eligible_runs}/{total_runs} | "
            "{experiment_cells} | {high_variance_tasks} |".format(**dataset)
        )

    lines.extend(["", "## Highest-variance tasks", ""])
    for dataset in datasets:
        lines.extend(
            [
                f"### `{dataset['dataset_key']}`",
                "",
                "| Task | Cells | Mean | Stdev | Range | Best cell | Worst cell |",
                "| --- | ---: | ---: | ---: | ---: | --- | --- |",
            ]
        )
        rows = [
            row
            for row in task_variance
            if row["dataset_key"] == dataset["dataset_key"]
        ][:20]
        for row in rows:
            lines.append(
                "| {task} | {cells} | {mean:.4f} | {stdev:.4f} | {span:.4f} | "
                "{best} | {worst} |".format(
                    task=row["task_name"],
                    cells=row["cell_count"],
                    mean=row["mean_cell_reward"],
                    stdev=row["cell_reward_stdev"],
                    span=row["range_across_cells"],
                    best=_short_cell_list(row["best_cell"]),
                    worst=_short_cell_list(row["worst_cell"]),
                )
            )
        lines.append("")

    if plot_paths:
        lines.extend(["", "## Plots", ""])
        for plot_path in plot_paths:
            lines.append(f"![{plot_path.stem}]({plot_path.name})")
            lines.append("")

    lines.extend(
        [
            "## Appendix: how to read this report",
            "",
            "| Term | Meaning |",
            "| --- | --- |",
            "| Dataset | One exact task revision and task count. Different datasets are never pooled. |",
            "| Experiment cell | One harness, model, and reasoning-effort combination. |",
            "| Run | One independent full-suite repetition within an experiment cell. |",
            "| Task cell score | Mean reward for one task across eligible repetitions of one experiment cell. |",
            "| Harness delta | Paired task-cell score for the left harness minus the right harness, holding model and reasoning fixed. |",
            "| Reasoning delta | Paired higher-effort task-cell score minus the adjacent lower-effort score, holding harness and model fixed. |",
            "| High variance | A task whose best and worst experiment-cell means differ by at least 0.5. This is a triage signal, not proof of a harness defect. |",
            "| Eligible run | A complete run accepted by native aggregation after coverage, infrastructure, model-identity, and trajectory checks. |",
            "",
            "Raw failed and excluded runs remain in the source aggregate exports; this report "
            "uses eligible runs for score distributions and paired task comparisons.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_task_matrix(
    summaries_dir: str | Path,
    output_dir: str | Path,
    *,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Build task-level diagnostics from native aggregate CSV exports."""

    source = Path(summaries_dir)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    runs = _read_csv(source / "aggregate_results.csv", RUN_REQUIRED_FIELDS)
    tasks = _read_csv(source / "per_task_results.csv", TASK_REQUIRED_FIELDS)

    cells, eligible_run_labels = _task_cells(runs, tasks)
    if not cells:
        raise ValueError("no task rows belong to eligible runs")
    task_variance = _task_variance(cells)
    harness_deltas = _harness_deltas(cells)
    reasoning_deltas = _reasoning_deltas(cells)
    plots = (
        _generate_plots(runs, cells, harness_deltas, reasoning_deltas, destination)
        if generate_plots
        else []
    )
    datasets = _dataset_summaries(
        runs,
        cells,
        task_variance,
        harness_deltas,
        reasoning_deltas,
    )

    _write_csv(destination / "task_cell_summary.csv", CELL_FIELDS, cells)
    _write_csv(
        destination / "task_variance_diagnostics.csv",
        TASK_VARIANCE_FIELDS,
        task_variance,
    )
    _write_csv(
        destination / "harness_task_deltas.csv",
        HARNESS_DELTA_FIELDS,
        harness_deltas,
    )
    _write_csv(
        destination / "reasoning_task_deltas.csv",
        REASONING_DELTA_FIELDS,
        reasoning_deltas,
    )
    report = {
        "schema_version": 1,
        "source": {
            "summaries_dir": source.name,
            "run_count": len(runs),
            "task_row_count": len(tasks),
            "eligible_run_count": len(eligible_run_labels),
        },
        "datasets": datasets,
        "outputs": {
            "task_cell_summary": "task_cell_summary.csv",
            "task_variance_diagnostics": "task_variance_diagnostics.csv",
            "harness_task_deltas": "harness_task_deltas.csv",
            "reasoning_task_deltas": "reasoning_task_deltas.csv",
            "markdown": "ANALYSIS.md",
            "plots": [path.name for path in plots],
        },
    }
    (destination / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(destination / "ANALYSIS.md", datasets, task_variance, plots)
    return report
