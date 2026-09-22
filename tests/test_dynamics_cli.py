from pathlib import Path

from click.testing import CliRunner

from clawbench.cli import cli
from clawbench.dynamics_archive import safe_model_name
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


def _run(task_id: str, run_index: int = 0) -> TaskRunResult:
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
        run_score=0.8,
        duration_ms=3000,
        token_usage=transcript.total_usage,
    )


def test_dynamics_report_cli_supports_no_plots(tmp_path: Path):
    model_dir = tmp_path / safe_model_name("ollama/gpt-oss:20b") / "t1-demo-task"
    model_dir.mkdir(parents=True)
    run = _run("t1-demo-task")
    (model_dir / "run0.json").write_text(run.model_dump_json(indent=2), encoding="utf-8")

    runner = CliRunner()
    output_dir = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "dynamics-report",
            "--archive-dir",
            str(tmp_path),
            "--model",
            "ollama/gpt-oss:20b",
            "--output-dir",
            str(output_dir),
            "--no-plots",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Loaded 1 cached runs across 1 tasks" in result.output
    assert "Saved 0 plots" in result.output
    assert (output_dir / "dynamics.json").exists()
    assert list(output_dir.glob("*.png")) == []


def test_dynamics_report_rejects_partial_archive(tmp_path):
    task_dir = tmp_path / "archive" / "model" / "t1-demo"
    task_dir.mkdir(parents=True)
    (task_dir / "run0.json").write_text(_run("t1-demo").model_dump_json())
    corrupt = task_dir / "run1.json"
    corrupt.write_text('{"task_id":')
    output_dir = tmp_path / "out"

    result = CliRunner().invoke(
        cli,
        ["dynamics-report", "--archive-dir", str(tmp_path / "archive"),
         "--output-dir", str(output_dir), "--no-plots"],
    )

    assert result.exit_code != 0
    assert str(corrupt) in result.output
    assert not output_dir.exists()
