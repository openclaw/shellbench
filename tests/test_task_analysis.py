import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from clawbench.cli import cli
from clawbench.task_analysis import analyze_task_matrix


RUN_FIELDS = (
    "run_label",
    "harness",
    "model_slug",
    "model_id",
    "reasoning_effort",
    "task_revision",
    "expected_task_count",
    "score",
    "eligible",
)
TASK_FIELDS = ("run_label", "task_name", "classification", "reward")


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _matrix_fixture(tmp_path: Path) -> Path:
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    runs = []
    tasks = []

    conditions = [
        ("openclaw", "low", {"task-a": 1.0, "task-b": 0.0}),
        ("hermes", "low", {"task-a": 0.0, "task-b": 0.5}),
        ("openclaw", "medium", {"task-a": 1.0, "task-b": 0.5}),
    ]
    for harness, reasoning, rewards in conditions:
        for repetition in (1, 2):
            run_label = f"{harness}-model-{reasoning}-r{repetition}"
            runs.append(
                {
                    "run_label": run_label,
                    "harness": harness,
                    "model_slug": "model",
                    "model_id": "provider/model",
                    "reasoning_effort": reasoning,
                    "task_revision": "revision-a",
                    "expected_task_count": 2,
                    "score": sum(rewards.values()) / 2,
                    "eligible": "true",
                }
            )
            for task_name, reward in rewards.items():
                tasks.append(
                    {
                        "run_label": run_label,
                        "task_name": task_name,
                        "classification": "pass" if reward >= 1 else "partial",
                        "reward": reward,
                    }
                )

    runs.append(
        {
            "run_label": "openclaw-other-low-r1",
            "harness": "openclaw",
            "model_slug": "other",
            "model_id": "provider/other",
            "reasoning_effort": "low",
            "task_revision": "revision-b",
            "expected_task_count": 1,
            "score": 1.0,
            "eligible": "true",
        }
    )
    tasks.append(
        {
            "run_label": "openclaw-other-low-r1",
            "task_name": "task-a",
            "classification": "pass",
            "reward": 1.0,
        }
    )
    runs.append(
        {
            "run_label": "excluded-run",
            "harness": "codex",
            "model_slug": "model",
            "model_id": "provider/model",
            "reasoning_effort": "low",
            "task_revision": "revision-a",
            "expected_task_count": 2,
            "score": 1.0,
            "eligible": "false",
        }
    )
    tasks.extend(
        [
            {
                "run_label": "excluded-run",
                "task_name": "task-a",
                "classification": "pass",
                "reward": 1.0,
            },
            {
                "run_label": "excluded-run",
                "task_name": "task-b",
                "classification": "pass",
                "reward": 1.0,
            },
        ]
    )
    _write_csv(summaries / "aggregate_results.csv", RUN_FIELDS, runs)
    _write_csv(summaries / "per_task_results.csv", TASK_FIELDS, tasks)
    return summaries


def test_analyze_task_matrix_builds_separate_dataset_diagnostics(tmp_path: Path):
    summaries = _matrix_fixture(tmp_path)
    output = tmp_path / "analysis"

    report = analyze_task_matrix(summaries, output, generate_plots=False)

    assert report["source"]["eligible_run_count"] == 7
    assert len(report["datasets"]) == 2
    revision_a = next(
        dataset
        for dataset in report["datasets"]
        if dataset["task_revision"] == "revision-a"
    )
    assert revision_a["experiment_cells"] == 3
    assert revision_a["tasks"] == 2
    assert revision_a["excluded_runs"] == 1
    assert revision_a["high_variance_tasks"] == 2
    assert revision_a["reasoning_improvements"] == 1
    assert revision_a["reasoning_regressions"] == 0

    with (output / "task_variance_diagnostics.csv").open(newline="") as handle:
        variance = list(csv.DictReader(handle))
    task_a = next(
        row
        for row in variance
        if row["task_revision"] == "revision-a" and row["task_name"] == "task-a"
    )
    assert float(task_a["range_across_cells"]) == 1.0
    assert task_a["best_cell"] == "openclaw/model/low;openclaw/model/medium"
    assert task_a["worst_cell"] == "hermes/model/low"

    with (output / "harness_task_deltas.csv").open(newline="") as handle:
        harness_deltas = list(csv.DictReader(handle))
    assert sorted(float(row["delta"]) for row in harness_deltas) == [-1.0, 0.5]

    markdown = (output / "ANALYSIS.md").read_text(encoding="utf-8")
    assert "Multiple task revisions are present" in markdown
    assert "Reasoning effort" not in markdown
    assert (output / "analysis_summary.json").is_file()


def test_task_analysis_cli_supports_no_plots(tmp_path: Path):
    summaries = _matrix_fixture(tmp_path)
    output = tmp_path / "analysis"

    result = CliRunner().invoke(
        cli,
        [
            "task-analysis",
            "--summaries-dir",
            str(summaries),
            "--output-dir",
            str(output),
            "--no-plots",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Analyzed 7 eligible runs across 2 task dataset(s)" in result.output
    assert list(output.glob("*.svg")) == []


def test_analyze_task_matrix_renders_svg_box_plots(tmp_path: Path):
    pytest.importorskip("matplotlib")
    summaries = _matrix_fixture(tmp_path)
    output = tmp_path / "analysis"

    report = analyze_task_matrix(summaries, output, generate_plots=True)

    plot_names = report["outputs"]["plots"]
    assert plot_names
    assert all((output / name).read_text(encoding="utf-8").startswith("<?xml") for name in plot_names)
