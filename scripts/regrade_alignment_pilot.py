"""Versioned regrading of retained executions; never overwrite original evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from clawbench.alignment.grading import combine_grade, evidence_bundle, rule_grade, semantic_grade
from clawbench.alignment.tasks import load_cases
from clawbench.schemas import Transcript
from scripts.run_alignment_pilot import ROOT, events_in, write_json


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def judge_with_validation_retries(bundle: dict, model: str, output: Path) -> dict | None:
    """Retry invalid responses, never a valid negative behavioral judgment."""
    for attempt in range(3):
        try:
            return semantic_grade(bundle, model, output / f"judge-response-{attempt + 1}.json")
        except Exception as exc:
            write_json(
                output / f"judge-error-{attempt + 1}.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
    return None


def regrade(root: Path, output: Path) -> list[dict]:
    if not (root / "dependency-integrity.json").exists():
        raise ValueError("Only regrade a finished batch; retain interrupted batches separately")
    provenance = json.loads((root / "provenance.json").read_text())
    originals = json.loads((root / "results.json").read_text())
    cases = {
        c.id: c
        for c in load_cases(
            root / "source" / provenance.get("suite_path", "tasks-alignment/suite.yaml")
        )
    }
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for name in [
        "clawbench/alignment/grading.py",
        "clawbench/alignment/tasks.py",
        "scripts/regrade_alignment_pilot.py",
    ]:
        path = ROOT / name
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        source_hashes[name] = digest(path)
    metadata = {
        "batch": str(root),
        "actor_provenance_sha256": digest(root / "provenance.json"),
        "original_results_sha256": digest(root / "results.json"),
        "source_sha256": source_hashes,
        "evidence_sha256": {},
        "policy": "Reuse complete unchanged grades; rejudge changed rules or failed judges. Retry only invalid judge responses, at most three times; retain all attempts.",
    }
    results = []
    for original in originals:
        name = f"{original['task_id']}-run{original['repetition']}"
        actor = root / name
        destination = output / name
        destination.mkdir()
        for filename in [
            "execution.json",
            "baseline.json",
            "judge-evidence.json",
            "grade.json",
            "service-audit/events.jsonl",
        ]:
            metadata["evidence_sha256"][f"{name}/{filename}"] = digest(actor / filename)
        case = cases[original["task_id"]]
        execution = json.loads((actor / "execution.json").read_text())
        transcript = Transcript.model_validate(execution["transcript"])
        baseline = json.loads((actor / "baseline.json").read_text())
        old_bundle = json.loads((actor / "judge-evidence.json").read_text())
        functional = old_bundle["functional_evidence"]
        events = events_in(actor / "service-audit")
        rules = rule_grade(case, actor / "workspace", baseline, events, functional["passed"])
        bundle = evidence_bundle(
            case, actor / "workspace", baseline, transcript, events, rules, functional
        )
        if bundle["final_files"] != old_bundle["final_files"]:
            raise ValueError(f"Actor artifacts changed after original grading: {name}")
        write_json(destination / "judge-evidence.json", bundle)
        reuse = original["grading_complete"] and rules == original["rules"]
        semantic = (
            original["semantic"]
            if reuse
            else judge_with_validation_retries(bundle, provenance["judge_model"], destination)
        )
        grade = combine_grade(rules, semantic, run_valid=original["run_valid"])
        grade.update(
            {
                key: original[key]
                for key in ("task_id", "repetition", "stop_reason", "tool_calls", "evidence_path")
            }
        )
        grade["semantic_source"] = "original" if reuse else "rejudged"
        grade["grading_complete"] = (
            functional["passed"] is not None
            and semantic is not None
            and all(
                semantic.get(key) is not None
                for key in (
                    "honest_reporting",
                    "prohibited_attempt",
                    "adequate_delivery",
                    "risk_exposed",
                )
            )
        )
        write_json(destination / "grade.json", grade)
        results.append(grade)
        write_json(output / "results.json", results)
        write_json(output / "provenance.json", metadata)
        print(
            json.dumps(
                {
                    k: grade[k]
                    for k in (
                        "task_id",
                        "repetition",
                        "safe_success",
                        "grading_complete",
                        "semantic_source",
                    )
                }
            ),
            flush=True,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grades = regrade(args.batch.resolve(), args.output.resolve())
    raise SystemExit(0 if all(g["grading_complete"] for g in grades) else 2)
