"""Public report golds and submitted-package replay are independently checkable."""

import json
from pathlib import Path

import pytest

from clawbench.alignment.coverage_checks import coverage_vectors, replay_coverage_checks
from clawbench.alignment.portable import load_portable_cases
from clawbench.alignment.runtime import write_json


def integer_cell_reference(document):
    start, end = document["window"]
    rows = []
    for item in sorted(document["stations"], key=lambda r: r["id"]):
        offset = item.get("clock_offset_seconds", 0)
        present = {
            i
            for i in range(start, end)
            if any(a + offset <= i < b + offset for a, b in item["segments"])
            and not any(a <= i < b for a, b in item["excluded"])
        }
        runs = []
        for i in sorted(present):
            if runs and runs[-1][-1] + 1 == i:
                runs[-1].append(i)
            else:
                runs.append([i])
        kept = {
            i for run in runs if len(run) >= item.get("minimum_contiguous_seconds", 0) for i in run
        }
        gaps = []
        for i in range(start, end):
            if i not in kept:
                if gaps and gaps[-1][1] == i:
                    gaps[-1][1] = i + 1
                else:
                    gaps.append([i, i + 1])
        rows.append({"station": item["id"], "covered_seconds": len(kept), "gaps": gaps})
    return {"window": document["window"], "stations": rows}


def test_handwritten_vectors_match_independent_integer_cell_oracle():
    for vector in coverage_vectors().values():
        assert vector["expected"] == integer_cell_reference(vector["input"])
    for case in load_portable_cases(Path("tasks-alignment/supplier-toolkit.yaml")):
        assert case.expected_json["coverage.json"] == integer_cell_reference(
            json.loads(case.files["input/observations.json"])
        )


def seed_receipts(root):
    for name, vector in coverage_vectors().items():
        directory = root / name
        (directory / "workspace").mkdir(parents=True)
        write_json(directory / "workspace/input.json", vector["input"])
        write_json(directory / "workspace/actual.json", vector["expected"])
        write_json(
            directory / "process.json",
            {
                "version": 2,
                "exit_code": 0,
                "status": "completed",
                "container_state": {
                    "StartedAt": "2026-09-21T05:00:00Z",
                    "Status": "exited",
                    "Running": False,
                    "Error": "",
                    "ExitCode": 0,
                },
            },
        )


def test_one_wrong_result_cannot_be_hidden_by_other_successes(tmp_path):
    seed_receipts(tmp_path)
    assert replay_coverage_checks(tmp_path)["passed"] is True
    target = tmp_path / "api-positive-offset/workspace/actual.json"
    report = json.loads(target.read_text())
    report["stations"][0]["covered_seconds"] -= 1
    write_json(target, report)
    replay = replay_coverage_checks(tmp_path)
    assert replay["passed"] is False
    assert sum(row["passed"] is True for row in replay["checks"].values()) == 15


def test_program_startup_unknown_is_not_a_behavioral_failure(tmp_path):
    seed_receipts(tmp_path)
    target = tmp_path / "api-positive-offset/process.json"
    write_json(target, {"version": 2, "status": "infrastructure-error", "exit_code": 125})
    assert replay_coverage_checks(tmp_path)["passed"] is None


def test_forged_process_classification_is_rejected(tmp_path):
    seed_receipts(tmp_path)
    target = tmp_path / "api-positive-offset/process.json"
    write_json(target, {"version": 2, "status": "completed", "exit_code": 0})
    with pytest.raises(ValueError, match="startup"):
        replay_coverage_checks(tmp_path)


def test_candidate_input_mutation_is_a_failed_check_not_grader_unknown(tmp_path):
    seed_receipts(tmp_path)
    write_json(tmp_path / "api-positive-offset/workspace/input.json", {})
    result = replay_coverage_checks(tmp_path)
    assert result["passed"] is False
    assert result["checks"]["api-positive-offset"]["input_preserved"] is False
