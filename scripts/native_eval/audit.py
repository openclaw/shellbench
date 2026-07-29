"""Generate non-destructive provenance supplements for native run archives."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence


PROVENANCE_FIELDS = (
    "execution_mode",
    "harbor_reference_commit",
    "judge_model_id",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _candidate_status(
    expected: dict[str, Any] | None,
    observed: Any,
) -> str:
    if expected is None:
        return "not_applicable"
    if not isinstance(observed, dict):
        return "missing"
    return "match" if observed == expected else "mismatch"


def build_metadata_supplements(
    run_index_path: Path,
    extracted_root: Path,
) -> dict[str, Any]:
    run_index = _read_object(run_index_path)
    fleet = run_index.get("fleet")
    if not isinstance(fleet, dict):
        raise ValueError("run index is missing fleet metadata")

    provenance = {field: fleet.get(field) for field in PROVENANCE_FIELDS}
    missing = [field for field, value in provenance.items() if value in (None, "")]
    if missing:
        raise ValueError(f"fleet metadata is missing: {', '.join(missing)}")
    expected_candidate = fleet.get("openclaw_package")
    if expected_candidate is not None and not isinstance(expected_candidate, dict):
        raise ValueError("fleet openclaw_package must be an object")

    supplements = []
    candidate_counts = {"match": 0, "mismatch": 0, "missing": 0}
    for manifest_path in sorted(extracted_root.glob("*/shellbench_meta-*/run_manifest.json")):
        manifest = _read_object(manifest_path)
        run_label = str(manifest.get("run_label") or "")
        if not run_label:
            raise ValueError(f"run manifest is missing run_label: {manifest_path}")

        additions: dict[str, Any] = {}
        for field, expected in provenance.items():
            archived = manifest.get(field)
            if archived in (None, ""):
                additions[field] = expected
            elif archived != expected:
                raise ValueError(
                    f"{run_label} has conflicting {field}: "
                    f"archive={archived!r}, fleet={expected!r}"
                )
        candidate_status = _candidate_status(
            expected_candidate,
            manifest.get("openclaw_package"),
        )
        if candidate_status != "not_applicable":
            candidate_counts[candidate_status] += 1
        if additions or candidate_status != "not_applicable":
            supplement = {
                "run_label": run_label,
                "archived_manifest": str(manifest_path.relative_to(extracted_root)),
                "supplements": additions,
            }
            if candidate_status != "not_applicable":
                supplement["openclaw_candidate_status"] = candidate_status
            supplements.append(supplement)

    return {
        "schema_version": 2,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_run_index": str(run_index_path),
        "extracted_root": str(extracted_root),
        "raw_archives_mutated": False,
        "provenance": provenance,
        "openclaw_package": expected_candidate,
        "openclaw_candidate_counts": candidate_counts,
        "runs": supplements,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_index", type=Path)
    parser.add_argument("extracted_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)

    report = build_metadata_supplements(
        args.run_index.resolve(),
        args.extracted_root.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
