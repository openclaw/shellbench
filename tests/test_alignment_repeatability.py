import json

import pytest

from scripts.summarize_alignment_repeatability import (
    alignment_summary,
    bounds,
    distribution,
    evidence_qualified_rows,
    scorecard_summary,
)


def test_unknowns_and_invalid_runs_stay_in_denominator():
    rows = [
        {"run_valid": True, "safe_success": True},
        {"run_valid": True, "safe_success": False},
        {"run_valid": True, "safe_success": None},
        {"run_valid": False, "safe_success": True},
    ]
    assert bounds(rows) == {
        "attempts": 4,
        "true": 1,
        "false": 1,
        "unknown": 2,
        "lower": 0.25,
        "upper": 0.75,
    }


def test_known_effect_failure_survives_missing_semantic_grade():
    assert (
        bounds([{"run_valid": True, "safe_success": False, "grading_complete": False}])["upper"]
        == 0
    )


def test_numeric_scores_keep_missing_and_invalid_attempts_in_bounds():
    rows = [
        {"run_valid": True, "scorecard": {"composite": 0.5}},
        {"run_valid": True, "scorecard": {"composite": 1}},
        {"run_valid": True, "scorecard": {"composite": None}},
        {"run_valid": False, "scorecard": {"composite": 1}},
    ]
    result = scorecard_summary(rows)
    assert (result["planned"], result["scored"], result["unknown"]) == (4, 2, 2)
    assert result["observed"]["mean"] == 0.75
    assert result["mean_bounds"] == {"lower": 0.375, "upper": 0.875}


def test_sample_spread_and_no_single_run_variance_claim():
    assert distribution([0.5, 0.75, 1.0]) == {
        "n": 3,
        "mean": 0.75,
        "min": 0.5,
        "max": 1.0,
        "sample_sd": 0.25,
    }
    assert distribution([0.5])["sample_sd"] is None
    assert distribution([])["mean"] is None


def test_one_successful_integration_run_cannot_pass_repeatability_screen(tmp_path, monkeypatch):
    provenance = {"model": "synthetic", "thinking": "high", "tasks": ["task"], "runs_per_task": 1}
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    (tmp_path / "results.json").write_text(
        json.dumps(
            [
                {
                    "task_id": "task",
                    "repetition": 0,
                    "condition": "control",
                    "run_valid": True,
                    "grading_complete": True,
                    "safe_success": True,
                }
            ]
        )
    )
    monkeypatch.setattr(
        "scripts.summarize_alignment_repeatability.audit_batch",
        lambda _: {"issues": [], "runs": []},
    )
    summary = alignment_summary(tmp_path)
    assert summary["round_score_distribution"]["mean"] == 1
    assert summary["round_score_distribution"]["sample_sd"] is None
    assert summary["narrow_band_screen"] is False


def test_independent_rejection_remains_unknown_in_rounds_and_case_scores(tmp_path, monkeypatch):
    provenance = {"model": "synthetic", "thinking": "high", "tasks": ["task"], "runs_per_task": 5}
    rows = [
        {
            "task_id": "task",
            "repetition": i,
            "condition": "control",
            "run_valid": True,
            "grading_complete": True,
            "safe_success": True,
            "scorecard": {"composite": 1},
        }
        for i in range(5)
    ]
    (tmp_path / "provenance.json").write_text(json.dumps(provenance))
    (tmp_path / "results.json").write_text(json.dumps(rows))
    audit = {
        "issues": ["At least one attempt failed evidence audit"],
        "runs": [
            {
                "task_id": "task",
                "repetition": i,
                "errors": ["Evidence changed: workspace/output.json"] if i == 2 else [],
            }
            for i in range(5)
        ],
    }
    monkeypatch.setattr("scripts.summarize_alignment_repeatability.audit_batch", lambda _: audit)
    summary = alignment_summary(tmp_path)
    assert summary["valid_executions"] == 5
    assert summary["evidence_qualified_executions"] == 4
    assert summary["mean_score_bounds"] == {"lower": 0.8, "upper": 1.0}
    assert summary["rounds"][2]["all"]["unknown"] == 1
    assert summary["rounds"][2]["scorecard"]["unknown"] == 1
    assert summary["per_case"]["task"]["safe_success"]["unknown"] == 1
    assert summary["per_case"]["task"]["scorecard"]["unknown"] == 1
    assert summary["round_composite_distribution"] is None
    assert summary["round_score_distribution"] is None
    assert summary["narrow_band_screen"] is False
    assert json.loads((tmp_path / "results.json").read_text()) == rows


@pytest.mark.parametrize(
    "error",
    ["Execution or grading incomplete", "Quality grading missing", "Quality score incomplete"],
)
def test_missing_judgment_does_not_erase_known_failure(error):
    rows = [
        {
            "task_id": "task",
            "repetition": 0,
            "run_valid": True,
            "safe_success": False,
            "scorecard": {"composite": 0},
        }
    ]
    audit = {
        "issues": ["At least one attempt failed evidence audit"],
        "runs": [{"task_id": "task", "repetition": 0, "errors": [error]}],
    }
    qualified = evidence_qualified_rows(rows, audit)
    assert bounds(qualified)["false"] == 1
    assert scorecard_summary(qualified)["mean_bounds"] == {"lower": 0.0, "upper": 0.0}


@pytest.mark.parametrize(
    "issue", ["Dependency cache changed", "Original actor evidence failed independent audit"]
)
def test_shared_or_unlocalized_revision_errors_withhold_all_results(issue):
    rows = [{"task_id": "task", "repetition": 0, "run_valid": True, "safe_success": True}]
    qualified = evidence_qualified_rows(rows, {"issues": [issue]})
    assert bounds(qualified)["unknown"] == 1
    assert rows[0]["run_valid"] is True


def test_incomplete_audit_map_cannot_qualify_results():
    rows = [{"task_id": "task", "repetition": 0, "run_valid": True}]
    with pytest.raises(ValueError, match="audit has incomplete or duplicate"):
        evidence_qualified_rows(
            rows, {"issues": ["At least one attempt failed evidence audit"], "runs": []}
        )
