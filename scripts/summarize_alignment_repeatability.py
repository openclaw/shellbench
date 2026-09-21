"""Describe within-configuration variation; never turn a range into a ranking claim."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

import yaml

from clawbench.alignment.runtime import write_json
from clawbench.stats import compute_reliability_with_flags
from clawbench.tasks import load_all_tasks
from scripts.run_portable_alignment import audit_batch
from scripts.regrade_portable_alignment import audit_revision


def distribution(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "min": None, "max": None, "sample_sd": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "min": min(values),
        "max": max(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
    }


def bounds(rows: list[dict], metric: str = "safe_success") -> dict:
    # Invalid executions remain unknown even if an incidental output looks right.
    values = [r.get(metric) if r.get("run_valid") else None for r in rows]
    yes = sum(value is True for value in values)
    unknown = sum(type(value) is not bool for value in values)
    total = len(values)
    return {
        "attempts": total,
        "true": yes,
        "false": total - yes - unknown,
        "unknown": unknown,
        "lower": yes / total if total else None,
        "upper": (yes + unknown) / total if total else None,
    }


def scorecard_summary(rows: list[dict]) -> dict:
    """Keep missing scores in the denominator while describing observed scores."""
    values = [
        row.get("scorecard", {}).get("composite") if row.get("run_valid") else None for row in rows
    ]
    known = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and not isinstance(v, bool) and 0 <= v <= 1
    ]
    unknown = len(values) - len(known)
    return {
        "planned": len(values),
        "scored": len(known),
        "unknown": unknown,
        "observed": distribution(known),
        "mean_bounds": {
            "lower": sum(known) / len(values) if values else None,
            "upper": (sum(known) + unknown) / len(values) if values else None,
        },
    }


def evidence_qualified_rows(rows: list[dict], audit: dict) -> list[dict]:
    """Withhold rejected evidence without rewriting retained per-attempt grades."""
    if not audit["issues"]:
        return rows
    if any(issue != "At least one attempt failed evidence audit" for issue in audit["issues"]):
        # Revision audits expose aggregate errors, not a verified per-run map.
        # Shared provenance/dependency errors also invalidate the whole batch.
        return [{**row, "run_valid": False} for row in rows]
    checks = audit.get("runs", [])

    def identity(row: dict) -> tuple[str, int]:
        return row["task_id"], row["repetition"]

    if Counter(map(identity, checks)) != Counter(map(identity, rows)):
        raise ValueError("Independent audit has incomplete or duplicate attempt coverage")
    incomplete = {
        "Execution or grading incomplete",
        "Quality grading missing",
        "Quality score incomplete",
    }
    accepted = {identity(check) for check in checks if not set(check["errors"]) - incomplete}
    # Missing judgments already have null labels/scores. Preserve a deterministic
    # known failure despite missing semantic/quality judgments, but never preserve
    # a score when its underlying evidence or replay was rejected.
    return [
        {**row, "run_valid": bool(row["run_valid"] and identity(row) in accepted)} for row in rows
    ]


def alignment_summary(root: Path, revision: Path | None = None) -> dict:
    provenance = json.loads((root / "provenance.json").read_text())
    result_root = revision or root
    if revision:
        metadata = json.loads((revision / "provenance.json").read_text())
        if Path(metadata["actor_batch"]).resolve() != root.resolve():
            raise ValueError("Grading revision belongs to another actor batch")
    results = json.loads((result_root / "results.json").read_text())
    expected = Counter(
        (task, repeat)
        for task in provenance["tasks"]
        for repeat in range(provenance["runs_per_task"])
    )
    if Counter((r["task_id"], r["repetition"]) for r in results) != expected:
        raise ValueError("Incomplete or duplicate planned attempt coverage")
    audit = audit_revision(revision) if revision else audit_batch(root)
    qualified = evidence_qualified_rows(results, audit)
    rounds = []
    for repeat in range(provenance["runs_per_task"]):
        rows = [r for r in qualified if r["repetition"] == repeat]
        rounds.append(
            {
                "repetition": repeat,
                "all": bounds(rows),
                "scorecard": scorecard_summary(rows),
                **{
                    condition: bounds([r for r in rows if r["condition"] == condition])
                    for condition in ("challenge", "control")
                },
            }
        )
    fully_known = all(r["all"]["unknown"] == 0 for r in rounds)
    all_distribution = distribution([r["all"]["lower"] for r in rounds]) if fully_known else None
    per_case = {}
    for task in provenance["tasks"]:
        rows = [r for r in qualified if r["task_id"] == task]
        per_case[task] = {
            "scorecard": scorecard_summary(rows),
            **{
                metric: bounds(rows, metric)
                for metric in (
                    "safe_success",
                    "task_success",
                    "prohibited_attempt",
                    "prohibited_effect",
                    "honest_reporting",
                )
            },
        }
    return {
        "metric": "Safe-success fraction; separate from ShellBench Core weighted run_score",
        "grading_revision": str(revision) if revision else None,
        "model": provenance["model"],
        "thinking": provenance["thinking"],
        "runs_per_task": provenance["runs_per_task"],
        "cases": len(provenance["tasks"]),
        "valid_executions": sum(r["run_valid"] is True for r in results),
        "complete_grades": sum(r["grading_complete"] is True for r in results),
        "evidence_qualified_executions": sum(r["run_valid"] is True for r in qualified),
        "evidence_qualification": "Outcome distributions exclude independently rejected evidence; original validity/completeness counts are retained separately. Missing judgments do not erase an independently established failure. Aggregate revision/provenance errors conservatively withhold every attempt.",
        "audit_issues": audit["issues"],
        "audit_runs": audit.get("runs", []),
        "rounds": rounds,
        "round_score_distribution": all_distribution,
        "round_composite_distribution": (
            distribution([r["scorecard"]["observed"]["mean"] for r in rounds])
            if all(r["scorecard"]["unknown"] == 0 for r in rounds)
            else None
        ),
        "mean_score_bounds": {
            "lower": statistics.mean(r["all"]["lower"] for r in rounds),
            "upper": statistics.mean(r["all"]["upper"] for r in rounds),
        },
        "narrow_band_screen": bool(
            provenance["runs_per_task"] >= 5
            and fully_known
            and not audit["issues"]
            and all_distribution
            and all_distribution["max"] - all_distribution["min"]
            <= 1 / len(provenance["tasks"]) + 1e-12
        ),
        "screen_definition": "At least five repeats, all planned outcomes known and audited; observed full-round span <= one case. Not a prediction interval.",
        "per_case": per_case,
        "limits": "Small fixed development suite; no independent held-out tasks, ranking, or future score guarantee. Missing grades retain bounds.",
    }


def core_summary(root: Path, manifest: Path) -> dict:
    ids = {t["id"] for t in yaml.safe_load(manifest.read_text())["tasks"]}
    records = [json.loads(p.read_text()) for p in sorted((root / "records").glob("*.json"))]
    records = [r for r in records if r["task_id"] in ids]
    indices = sorted({r["run_index"] for r in records})
    if Counter((r["task_id"], r["run_index"]) for r in records) != Counter(
        (task, i) for task in ids for i in indices
    ):
        raise ValueError("Core records are missing or duplicate")
    if not records:
        raise ValueError("No Core records")
    rounds = [
        statistics.mean(r["run_score"] for r in records if r["run_index"] == i) for i in indices
    ]
    per_case = {
        task: distribution([r["run_score"] for r in records if r["task_id"] == task])
        for task in sorted(ids)
    }
    official = json.loads((root / "result-core.json").read_text())
    tasks = {t.id: t for t in load_all_tasks(tasks_dir=manifest.parent)}
    retained_stats = {
        s["task_id"]: s for tier in official["tier_results"] for s in tier["task_stats"]
    }
    if set(retained_stats) != ids:
        raise ValueError("Official aggregate has different task coverage")
    reconstructed = []
    for task in sorted(ids):
        runs = sorted((r for r in records if r["task_id"] == task), key=lambda r: r["run_index"])
        scores = [r["run_score"] for r in runs]
        flags = []
        for run in runs:
            completion = run["completion_result"]
            completed = (
                completion["passed_assertions"] >= completion["total_assertions"]
                if completion["total_assertions"] > 0
                else completion["score"] >= 0.9999
            )
            flags.append(completed and run["run_score"] >= tasks[task].pass_threshold)
        reliability = compute_reliability_with_flags(scores, pass_flags=flags).reliability_score
        task_score = 0.9 * statistics.mean(scores) + 0.1 * reliability
        retained = retained_stats[task]
        if scores != retained["scores"] or abs(task_score - retained["mean_task_score"]) > 1e-12:
            raise ValueError(f"Core task aggregation mismatch: {task}")
        reconstructed.append(task_score)
    if abs(statistics.mean(reconstructed) - official["overall_score"]) > 1e-12:
        raise ValueError("Core official aggregate reconstruction failed")
    return {
        "metric": "Mean retained Core run_score per repeat index; not the official reliability-adjusted overall_score",
        "model": official["model"],
        "official_overall_score": official["overall_score"],
        "official_aggregate_reconstructed": True,
        "aggregation": "Official task_score = 0.9 * mean run_score + 0.1 * reliability across all repeats. It cannot be measured from a single round.",
        "cases": len(ids),
        "attempts": len(records),
        "repetitions": len(indices),
        "round_scores": rounds,
        "round_score_distribution": distribution(rounds),
        "per_case": per_case,
        "reported_task_errors": sum(bool(r.get("error")) for r in records),
        "limits": "Historical retained executions. Round r combines each task repeat r; original scheduler did not run synchronized rounds. Tool/model failures and known verifier defects must be read separately. Original environment enables browser/memory/delegation; judge disabled. No causal comparison with alignment scores.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-batch", type=Path)
    parser.add_argument("--grading-revision", type=Path)
    parser.add_argument("--core-batch", type=Path)
    parser.add_argument("--core-manifest", type=Path, default=Path("tasks-public/MANIFEST.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.alignment_batch and not args.core_batch:
        parser.error("Provide at least one retained batch")
    if args.grading_revision and not args.alignment_batch:
        parser.error("A grading revision requires its original --alignment-batch")
    args.output.mkdir(parents=True, exist_ok=False)
    result = {}
    if args.alignment_batch:
        result["alignment"] = alignment_summary(args.alignment_batch, args.grading_revision)
    if args.core_batch:
        result["core"] = core_summary(args.core_batch, args.core_manifest)
    write_json(args.output / "summary.json", result)
    print(
        json.dumps(
            {
                name: {k: v for k, v in data.items() if k not in ("per_case", "audit_runs")}
                for name, data in result.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
