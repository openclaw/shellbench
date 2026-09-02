#!/usr/bin/env python3
"""Classify posterior run trajectories into dynamical regimes.

We embed each assistant turn using bag-of-words text plus tool-call summaries,
then compute simple geometric proxies:

    drift_mean = mean ||x_t - x_{t-1}||
    from_start = max ||x_t - x_0||
    recurrence = max cosine(x_i, x_j) for non-adjacent turns
    vol_log    = log det(Sigma + eps I)

Runs are then bucketed into coarse regimes such as trapped, limit_cycle, and
diffusive using quartile-based thresholds estimated from the observed archive.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clawbench.dynamics_archive import load_task_runs_by_model

WORD_RE = re.compile(r"[a-z]{3,}")
STOPWORDS = set(
    "the and that with this have from what your will can but not "
    "was are been one would there they their has had its were only some "
    "than about these which into also each when where them how who very "
    "much more most other then here such does like just make many want need take".split()
)


def tokenize(text: str) -> list[str]:
    return [w for w in WORD_RE.findall((text or "").lower()) if w not in STOPWORDS]


def build_vocab(texts: list[str], top_k: int = 500) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(set(tokenize(text)))
    return {w: i for i, (w, _) in enumerate(counter.most_common(top_k))}


def vectorize(text: str, vocab: dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab), dtype=np.float32)
    for word, cnt in Counter(tokenize(text)).items():
        if word in vocab:
            vec[vocab[word]] = cnt
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def turn_texts(run, fallback_any_message: bool = False) -> list[str]:
    source = run.transcript.messages if fallback_any_message else run.transcript.assistant_messages
    out = []
    for msg in source:
        parts = []
        if msg.text:
            parts.append(msg.text)
        for tc in msg.tool_calls:
            parts.append(tc.name)
            if tc.input:
                parts.append(json.dumps(tc.input, sort_keys=True)[:200])
        if parts:
            out.append(" ".join(parts))
    return out


def trajectory_metrics(vecs: np.ndarray) -> dict[str, float]:
    """Compute drift, recurrence, and support-volume proxies for one run."""
    n = vecs.shape[0]
    if n < 2:
        return {
            "n_turns": float(n),
            "drift_mean": 0.0,
            "from_start": 0.0,
            "recurrence": 0.0,
            "vol_log": -12.0,
        }

    diffs = np.linalg.norm(np.diff(vecs, axis=0), axis=1)
    drift_mean = float(diffs.mean())
    from_start = float(np.linalg.norm(vecs - vecs[0:1], axis=1).max())

    recurrence = 0.0
    for i in range(n):
        for j in range(i + 2, n):
            ni = np.linalg.norm(vecs[i])
            nj = np.linalg.norm(vecs[j])
            if ni > 0 and nj > 0:
                sim = float(vecs[i] @ vecs[j] / (ni * nj))
                recurrence = max(recurrence, sim)

    if n >= 3:
        sigma = np.cov(vecs.T)
        eigs = np.linalg.eigvalsh(sigma + 1e-6 * np.eye(vecs.shape[1], dtype=np.float32))
        vol_log = float(np.log(np.clip(eigs, 1e-12, None)).sum())
    else:
        vol_log = -12.0

    return {
        "n_turns": float(n),
        "drift_mean": drift_mean,
        "from_start": from_start,
        "recurrence": recurrence,
        "vol_log": vol_log,
    }


def classify(metrics: dict[str, float], thresholds: dict[str, float]) -> str:
    """Map trajectory metrics to a coarse regime label."""
    n_turns = int(metrics["n_turns"])
    if n_turns < 3:
        return "too_short"
    drift = metrics["drift_mean"]
    recurrence = metrics["recurrence"]
    vol = metrics["vol_log"]

    if drift < thresholds["drift_low"] and vol < thresholds["vol_low"]:
        return "trapped"
    if recurrence > thresholds["rec_hi"] and drift < thresholds["drift_med"]:
        return "limit_cycle"
    if drift > thresholds["drift_hi"] and vol > thresholds["vol_hi"]:
        return "diffusive"
    return "mixed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify cached run regimes")
    parser.add_argument("--archive-dir", type=Path, default=Path(".clawbench/run_cache"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--tier", choices=["tier1", "tier2", "tier3", "tier4", "tier5"], default=None)
    args = parser.parse_args()

    grouped = load_task_runs_by_model(args.archive_dir, tier=args.tier)
    if not grouped:
        raise SystemExit(f"No cached runs found under {args.archive_dir}")

    all_turn_texts: list[str] = []
    run_turns: dict[str, list[str]] = {}
    run_metadata: dict[str, dict[str, str | int]] = {}

    for model_name, task_runs in grouped.items():
        for task_id, runs in task_runs.items():
            for run in runs:
                ts = turn_texts(run, fallback_any_message=False)
                key = f"{model_name}/{task_id}/run{run.run_index}"
                run_turns[key] = ts
                run_metadata[key] = {
                    "model": model_name,
                    "task_id": task_id,
                    "run_index": run.run_index,
                }
                all_turn_texts.extend(ts)

    used_fallback_messages = False
    if not all_turn_texts:
        used_fallback_messages = True
        all_turn_texts = []
        run_turns = {}
        run_metadata = {}
        for model_name, task_runs in grouped.items():
            for task_id, runs in task_runs.items():
                for run in runs:
                    ts = turn_texts(run, fallback_any_message=True)
                    key = f"{model_name}/{task_id}/run{run.run_index}"
                    run_turns[key] = ts
                    run_metadata[key] = {
                        "model": model_name,
                        "task_id": task_id,
                        "run_index": run.run_index,
                    }
                    all_turn_texts.extend(ts)

    if not all_turn_texts:
        raise SystemExit("No usable turn text found in archive.")

    vocab = build_vocab(all_turn_texts, top_k=500)

    per_run: dict[str, dict[str, float | str]] = {}
    for key, ts in run_turns.items():
        if not ts:
            continue
        vecs = np.stack([vectorize(text, vocab) for text in ts])
        per_run[key] = trajectory_metrics(vecs)

    eligible = [r for r in per_run.values() if int(r["n_turns"]) >= 3]
    if eligible:
        drifts = np.array([float(v["drift_mean"]) for v in eligible])
        recs = np.array([float(v["recurrence"]) for v in eligible])
        vols = np.array([float(v["vol_log"]) for v in eligible])
        thresholds = {
            "drift_low": float(np.percentile(drifts, 25)),
            "drift_med": float(np.percentile(drifts, 50)),
            "drift_hi": float(np.percentile(drifts, 75)),
            "vol_low": float(np.percentile(vols, 25)),
            "vol_hi": float(np.percentile(vols, 75)),
            "rec_hi": float(np.percentile(recs, 75)),
        }
    else:
        thresholds = {
            "drift_low": 0.15,
            "drift_med": 0.25,
            "drift_hi": 0.35,
            "vol_low": -6.0,
            "vol_hi": -3.0,
            "rec_hi": 0.8,
        }

    for key, metrics in per_run.items():
        metrics["regime"] = classify(metrics, thresholds)
        metrics["turn_source"] = "any_message" if used_fallback_messages else "assistant"
        metrics.update(run_metadata.get(key, {}))

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    out = args.reports_dir / "regimes.json"
    out.write_text(json.dumps(per_run, indent=2), encoding="utf-8")

    counts = Counter(str(v["regime"]) for v in per_run.values())
    print(f"Wrote: {out}")
    print(f"Regime counts: {dict(counts)}")


if __name__ == "__main__":
    main()
