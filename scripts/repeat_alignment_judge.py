"""Repeat a judge on fixed retained evidence without resampling actors or replacing grades."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import shutil

from clawbench.alignment.grading import combine_grade, semantic_grade
from clawbench.alignment.portable_grading import PORTABLE_JUDGE_INSTRUCTIONS
from clawbench.alignment.runtime import ROOT, write_json
from scripts.run_portable_alignment import digest


LABELS = ("honest_reporting", "prohibited_attempt", "adequate_delivery", "risk_exposed")


def summarize(results: list[dict]) -> dict:
    cases = {}
    for name in sorted({r["name"] for r in results}):
        rows = [r for r in results if r["name"] == name]
        labels = {}
        for label in (*LABELS, "safe_success"):
            values = [r.get("grade", {}).get(label) for r in rows]
            counts = Counter("unknown" if type(v) is not bool else str(v).lower() for v in values)
            labels[label] = {
                "counts": dict(counts),
                "all_known_and_equal": "unknown" not in counts and len(counts) == 1,
            }
        cases[name] = {"calls": len(rows), "labels": labels}
    return {
        "planned_calls_retained": len(results),
        "validated_responses": sum(r.get("validated") is True for r in results),
        "errors": sum("error" in r for r in results),
        "unknown_label_responses": sum(
            r.get("validated") is True
            and any(type(r.get("grade", {}).get(k)) is not bool for k in LABELS)
            for r in results
        ),
        "cases_with_all_safe_labels_equal": sum(
            c["labels"]["safe_success"]["all_known_and_equal"] for c in cases.values()
        ),
        "cases": cases,
        "limits": "Purposively selected development traces, not a judge accuracy estimate. No majority-vote grade replacement; invalid responses remain unknown.",
    }


async def run(manifest: Path, output: Path, model: str, repetitions: int) -> dict:
    selection = json.loads(manifest.read_text())
    if not selection or len({row["name"] for row in selection}) != len(selection):
        raise ValueError("Need nonempty selection with unique names")
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    frozen = {}
    for row in selection:
        name = row["name"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("Unsafe selection name")
        bundle_path, grade_path = Path(row["bundle"]), Path(row["original_grade"])
        bundle = json.loads(bundle_path.read_text())
        original = json.loads(grade_path.read_text())
        if original.get("run_valid") is not True:
            raise ValueError(f"Invalid actor execution: {name}")
        folder = output / name
        folder.mkdir()
        shutil.copy2(bundle_path, folder / "evidence.json")
        shutil.copy2(grade_path, folder / "original-grade.json")
        sources[name] = {
            **row,
            "bundle_sha256": digest(bundle_path),
            "original_grade_sha256": digest(grade_path),
        }
        frozen[name] = bundle
    source_files = [
        Path(__file__).resolve(),
        ROOT / "clawbench/alignment/grading.py",
        ROOT / "clawbench/alignment/portable_grading.py",
    ]
    hashes = {}
    for path in source_files:
        relative = path.relative_to(ROOT)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        hashes[str(relative)] = digest(destination)
    write_json(
        output / "provenance.json",
        {
            "model": model,
            "reasoning": "low",
            "repetitions": repetitions,
            "source_sha256": hashes,
            "selection": sources,
            "instructions": PORTABLE_JUDGE_INSTRUCTIONS,
            "policy": "Exactly one call per fixed repetition; no outcome-based retries or actor execution.",
        },
    )
    results = []
    semaphore = asyncio.Semaphore(2)

    async def one(name: str, repeat: int) -> None:
        async with semaphore:
            folder = output / name / f"judge-{repeat}"
            folder.mkdir()
            row = {"name": name, "repetition": repeat, "validated": False}
            try:
                semantic = await asyncio.to_thread(
                    semantic_grade,
                    frozen[name],
                    model,
                    folder / "response.json",
                    instructions=PORTABLE_JUDGE_INSTRUCTIONS,
                )
                row["validated"] = True
                row["semantic"] = semantic
                combined = combine_grade(frozen[name]["rules"], semantic, run_valid=True)
                row["grade"] = {**semantic, "safe_success": combined["safe_success"]}
            except Exception as exc:
                row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            write_json(folder / "result.json", row)
            results.append(row)
            write_json(output / "results.json", results)
            print(json.dumps({k: row[k] for k in ("name", "repetition", "validated")}), flush=True)

    await asyncio.gather(*(one(name, repeat) for repeat in range(repetitions) for name in frozen))
    if any(digest(Path(row["bundle"])) != row["bundle_sha256"] for row in sources.values()):
        raise ValueError("Retained source evidence changed during experiment")
    summary = summarize(results)
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model", default="gpt-5.4-2026-03-05")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("Repetitions must be positive")
    result = asyncio.run(run(args.manifest, args.output, args.model, args.repetitions))
    print(json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2))


if __name__ == "__main__":
    main()
