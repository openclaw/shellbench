#!/usr/bin/env python3
"""Assemble a combined posterior dynamical-systems markdown report.

Inputs:
    - constraint_index.json
    - regimes.json
    - variance_decomposition.json
    - survival_analysis.json
    - snr_weighted_ranking.json (optional)

Output:
    - EVAL_REPORT_DYNAMICAL.md

The goal is to keep a compact human-readable summary next to the machine
outputs produced by the posterior analysis pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _read_json(path: Path):
    if not path.exists():
        raise SystemExit(f"Missing required report file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a combined dynamical report markdown")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown output path; defaults to <reports-dir>/EVAL_REPORT_DYNAMICAL.md",
    )
    args = parser.parse_args()

    reports = args.reports_dir
    output_path = args.output or (reports / "EVAL_REPORT_DYNAMICAL.md")
    cq = _read_json(reports / "constraint_index.json")
    regimes = _read_json(reports / "regimes.json")
    variance = _read_json(reports / "variance_decomposition.json")
    survival = _read_json(reports / "survival_analysis.json")
    ranking_path = reports / "snr_weighted_ranking.json"
    ranking = json.loads(ranking_path.read_text(encoding="utf-8")) if ranking_path.exists() else None

    lines: list[str] = []
    L = lines.append

    L("# ClawBench Posterior Dynamical Report")
    L("")
    L("This report combines posterior-only diagnostics from cached run artifacts.")
    L("")

    L("## 1. Constraint Index C(q)")
    L("")
    values = [(task, float(data.get("C_q", 0.0))) for task, data in cq.items()]
    values.sort(key=lambda row: row[1], reverse=True)
    highs = [row for row in values if row[1] > 0.5]
    lows = [row for row in values if row[1] < -0.5]
    L(f"- High-constraint tasks (C > 0.5): {len(highs)}")
    L(f"- Low-constraint tasks (C < -0.5): {len(lows)}")
    L("")
    if values:
        L("Top tasks by C(q):")
        L("")
        L("| Task | C(q) |")
        L("|---|---:|")
        for task, c_q in values[:10]:
            L(f"| {task} | {c_q:+.3f} |")
        L("")

    L("## 2. Regime Classification")
    L("")
    by_model = defaultdict(Counter)
    for key, row in regimes.items():
        model = row.get("model") or key.rsplit("/", 2)[0]
        regime = row.get("regime", "unknown")
        by_model[model][regime] += 1

    L("| Model | too_short | trapped | limit_cycle | diffusive | mixed |")
    L("|---|---:|---:|---:|---:|---:|")
    for model in sorted(by_model):
        c = by_model[model]
        L(
            f"| {model} | {c['too_short']} | {c['trapped']} | {c['limit_cycle']} | "
            f"{c['diffusive']} | {c['mixed']} |"
        )
    L("")

    L("## 3. Variance Decomposition")
    L("")
    agg = variance.get("aggregate", {})
    L(f"- Mean seed variance: {agg.get('mean_seed_var', 0.0):.6f}")
    L(f"- Mean capability variance: {agg.get('mean_cap_var', 0.0):.6f}")
    L(f"- Capability fraction: {agg.get('capability_fraction', 0.0):.1%}")
    L(f"- High-SNR tasks: {agg.get('high_snr_tasks', 0)}")
    L(f"- Mid-SNR tasks: {agg.get('mid_snr_tasks', 0)}")
    L(f"- Low-SNR tasks: {agg.get('low_snr_tasks', 0)}")
    L("")

    L("## 4. Survival Analysis")
    L("")
    L("| Model | Runs | Events | Median failure turn | S(3) | S(5) | S(8) |")
    L("|---|---:|---:|---:|---:|---:|---:|")
    for model in sorted(survival):
        row = survival[model]
        surv = row.get("survival", [0.0] * 8)
        med = row.get("median_fail_turn", "inf")
        if isinstance(med, float) and med == float("inf"):
            med_display = "inf"
        else:
            med_display = f"{float(med):.1f}"
        L(
            f"| {model} | {row.get('n_runs', 0)} | {row.get('n_events', 0)} | "
            f"{med_display} | {surv[2] if len(surv) > 2 else 0.0:.2f} | "
            f"{surv[4] if len(surv) > 4 else 0.0:.2f} | {surv[7] if len(surv) > 7 else 0.0:.2f} |"
        )
    L("")

    if ranking is not None:
        L("## 5. SNR-weighted Ranking")
        L("")
        L("| Rank | Model | Flat | SNR x |C(q)| | Winsorized | Coverage |")
        L("|---:|---|---:|---:|---:|---:|")
        for idx, row in enumerate(ranking.get("results", []), start=1):
            L(
                f"| {idx} | {row.get('model', '')} | {row.get('flat', 0.0):.4f} | "
                f"{row.get('snr_x_abs_cq', 0.0):.4f} | {row.get('snr_x_abs_cq_winsorized', 0.0):.4f} | "
                f"{row.get('coverage', 0)} |"
            )
        L("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
