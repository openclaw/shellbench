#!/usr/bin/env python3
"""Run the full posterior dynamical analysis pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clawbench.dynamics_archive import discover_model_roots, load_task_runs_archive, write_dynamics_report


def _run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else (REPO_ROOT / path)


def _write_dynamics_reports(
    archive_dir: Path,
    output_dir: Path,
    tier: str | None,
) -> None:
    roots = discover_model_roots(archive_dir)
    if not roots:
        raise SystemExit(f"No cached runs found under {archive_dir}")

    multiple_models = len(roots) > 1
    wrote_any = False
    for model_name, model_dir in roots.items():
        task_runs = load_task_runs_archive(model_dir, tier=tier)
        if not task_runs:
            continue

        wrote_any = True
        model_output_dir = output_dir / model_name if multiple_models else output_dir
        report_path, plots = write_dynamics_report(task_runs, model_output_dir)
        n_runs = sum(len(runs) for runs in task_runs.values())

        print(f"[dynamics] {model_name}: loaded {n_runs} cached runs across {len(task_runs)} tasks")
        print(f"[dynamics] {model_name}: wrote {report_path}")
        print(f"[dynamics] {model_name}: saved {len(plots)} plots to {model_output_dir}/")

    if not wrote_any:
        raise SystemExit(f"No cached runs found under {archive_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run posterior dynamics pipeline end to end")
    parser.add_argument("--archive-dir", type=Path, default=Path(".clawbench/run_cache"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/posterior_dynamics"))
    parser.add_argument(
        "--include-dynamics-report",
        action="store_true",
        help="Also build per-model dynamics.json files and plots from the archive.",
    )
    parser.add_argument("--tier", choices=["tier1", "tier2", "tier3", "tier4", "tier5"], default=None)
    args = parser.parse_args()

    py = sys.executable
    archive_dir = _resolve_path(args.archive_dir)
    reports_dir = _resolve_path(args.reports_dir)
    output_dir = _resolve_path(args.output_dir)
    tier_args = ["--tier", args.tier] if args.tier else []
    scripts_dir = REPO_ROOT / "scripts"

    _run([
        py,
        str(scripts_dir / "posterior" / "2_compute_constraint_index.py"),
        "--archive-dir",
        str(archive_dir),
        "--reports-dir",
        str(reports_dir),
        *tier_args,
    ])
    _run([py, str(scripts_dir / "classify_regimes.py"), "--archive-dir", str(archive_dir), "--reports-dir", str(reports_dir), *tier_args])
    _run([py, str(scripts_dir / "variance_decomp.py"), "--archive-dir", str(archive_dir), "--reports-dir", str(reports_dir), *tier_args])
    _run([py, str(scripts_dir / "survival_analysis.py"), "--archive-dir", str(archive_dir), "--reports-dir", str(reports_dir), *tier_args])
    _run([py, str(scripts_dir / "snr_weighted_ranking.py"), "--archive-dir", str(archive_dir), "--reports-dir", str(reports_dir), *tier_args])
    _run([
        py,
        str(scripts_dir / "violation_time_decomposition.py"),
        "--archive-dir",
        str(archive_dir),
        "--reports-dir",
        str(reports_dir / "violation_time_decomposition"),
        *tier_args,
    ])
    _run([py, str(scripts_dir / "generate_dynamical_report.py"), "--reports-dir", str(reports_dir)])
    if args.include_dynamics_report:
        _write_dynamics_reports(archive_dir, output_dir, args.tier)


if __name__ == "__main__":
    main()
