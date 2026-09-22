"""Tests for offline dynamics archive helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawbench.dynamics_archive import build_dynamics_report, load_task_runs_archive, safe_model_name, write_dynamics_report
from clawbench.schemas import TaskRunResult, TokenUsage, ToolCall, Transcript, TranscriptMessage


def _msg(role: str, text: str = "", family: str | None = None, ts: int = 0) -> TranscriptMessage:
    tool_calls = []
    if family is not None:
        tool_calls.append(
            ToolCall(
                name=f"tool_{family}",
                family=family,
                success=True,
                error="",
                mutating=family == "edit",
            )
        )
    return TranscriptMessage(
        role=role,
        text=text,
        tool_calls=tool_calls,
        timestamp_ms=ts,
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _run(task_id: str, score: float = 0.5, run_index: int = 0) -> TaskRunResult:
    transcript = Transcript(
        messages=[
            _msg("user", f"Solve {task_id}"),
            _msg("assistant", "inspect", family="read", ts=1000),
            _msg("assistant", "edit", family="edit", ts=2000),
            _msg("assistant", "verify", family="execute", ts=3000),
        ]
    )
    return TaskRunResult(
        task_id=task_id,
        run_index=run_index,
        transcript=transcript,
        run_score=score,
        duration_ms=3000,
        token_usage=transcript.total_usage,
    )


def test_load_task_runs_archive_filters_model_and_tier(tmp_path: Path):
    model_dir = tmp_path / safe_model_name("ollama/gpt-oss:20b")
    other_dir = tmp_path / safe_model_name("openai/gpt-5.4")
    for root, task_id in ((model_dir, "t1-demo-task"), (other_dir, "t2-other-task")):
        task_dir = root / task_id
        task_dir.mkdir(parents=True)
        run = _run(task_id)
        (task_dir / "run0.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")

    loaded = load_task_runs_archive(
        archive_dir=tmp_path,
        model="ollama/gpt-oss:20b",
        tier="tier1",
    )

    assert list(loaded) == ["t1-demo-task"]
    assert loaded["t1-demo-task"][0].task_id == "t1-demo-task"


def test_write_dynamics_report_creates_report_without_plots(tmp_path: Path):
    task_runs = {
        "t1-demo-task": [_run("t1-demo-task", score=0.8)],
        "t2-demo-task": [_run("t2-demo-task", score=0.4)],
    }

    report_path, plots = write_dynamics_report(task_runs, tmp_path, generate_plots=False)

    assert report_path.exists()
    assert report_path.name == "dynamics.json"
    assert plots == []

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "sensitivity" in report
    assert report["sensitivity"]["same_task"]["n_pairs"] == 0


def test_build_dynamics_report_includes_pairwise_sensitivity():
    task_runs = {
        "t1-demo-task": [
            _run("t1-demo-task", score=0.8, run_index=0),
            TaskRunResult(
                task_id="t1-demo-task",
                run_index=1,
                transcript=Transcript(
                    messages=[
                        _msg("user", "Solve t1-demo-task"),
                        _msg("assistant", "inspect", family="search", ts=1000),
                        _msg("assistant", "edit", family="edit", ts=2000),
                        _msg("assistant", "verify", family="execute", ts=3000),
                    ]
                ),
                run_score=0.5,
                duration_ms=3000,
                token_usage=TokenUsage(input_tokens=30, output_tokens=15, total_tokens=45),
            ),
        ]
    }

    report, _plot_data = build_dynamics_report(task_runs, include_pca=False)

    same_task = report["sensitivity"]["same_task"]
    assert same_task["n_pairs"] == 1
    assert "t1-demo-task" in same_task["per_task"]
    assert same_task["per_task"]["t1-demo-task"]["mean_score_delta"] > 0


@pytest.mark.parametrize("contents", ['{"task_id":', '{}', b"\xff"])
def test_load_task_runs_archive_rejects_corrupt_selected_run(tmp_path, contents):
    task_dir = tmp_path / "model" / "t1-demo"
    task_dir.mkdir(parents=True)
    (task_dir / "run0.json").write_text(_run("t1-demo").model_dump_json())
    corrupt = task_dir / "run1.json"
    corrupt.write_bytes(contents if isinstance(contents, bytes) else contents.encode())

    with pytest.raises(ValueError, match="run1.json"):
        load_task_runs_archive(tmp_path)


def test_load_task_runs_archive_rejects_unreadable_selected_run(tmp_path, monkeypatch):
    task_dir = tmp_path / "model" / "t1-demo"
    task_dir.mkdir(parents=True)
    run_file = task_dir / "run0.json"
    run_file.write_text(_run("t1-demo").model_dump_json())
    original = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == run_file:
            raise PermissionError("synthetic read failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    with pytest.raises(ValueError, match="run0.json"):
        load_task_runs_archive(tmp_path)


@pytest.mark.parametrize("filters", [{"tier": "tier1"}, {"task_ids": ["t1-demo"]}])
def test_load_task_runs_archive_does_not_read_excluded_corrupt_runs(tmp_path, filters):
    model_dir = tmp_path / "model"
    for task_id in ["t1-demo", "t2-broken"]:
        (model_dir / task_id).mkdir(parents=True)
    (model_dir / "t1-demo" / "run0.json").write_text(_run("t1-demo").model_dump_json())
    (model_dir / "t2-broken" / "run0.json").write_text("{broken")

    assert list(load_task_runs_archive(tmp_path, **filters)) == ["t1-demo"]
