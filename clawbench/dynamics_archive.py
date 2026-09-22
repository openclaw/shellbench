"""Offline dynamics analysis helpers for cached ClawBench runs."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

from clawbench.dynamics import (
    build_strata,
    compute_dynamics,
    compute_pca_bundle,
    compute_sensitivity,
    find_event_step,
    kaplan_meier,
    stratify_by_regime,
    stratify_by_scenario,
    stratify_by_tier,
    stratify_by_tool_mix,
)
from clawbench.schemas import TaskRunResult

_TIER_PREFIXES = {
    "tier1": ("t1-", "t1_"),
    "tier2": ("t2-", "t2_"),
    "tier3": ("t3-", "t3_"),
    "tier4": ("t4-", "t4_"),
    "tier5": ("t5-", "t5_"),
}


def safe_model_name(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _candidate_model_dir_names(model: str) -> set[str]:
    return {
        model,
        safe_model_name(model),
        model.replace("/", "_"),
        model.replace("/", "-").replace(":", "-"),
    }


def _has_run_files(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_file() and child.name.startswith("run") and child.suffix == ".json":
                return True
    except FileNotFoundError:
        return False
    return False


def _is_task_collection_root(path: Path) -> bool:
    try:
        for child in path.iterdir():
            if child.is_dir() and _has_run_files(child):
                return True
    except FileNotFoundError:
        return False
    return False


def _resolve_model_roots(archive_dir: Path, model: str | None) -> list[Path]:
    if _is_task_collection_root(archive_dir):
        if model is not None and archive_dir.name not in _candidate_model_dir_names(model):
            raise ValueError(
                f"Archive dir {archive_dir} does not match requested model {model}."
            )
        return [archive_dir]

    roots = [
        child
        for child in sorted(archive_dir.iterdir())
        if child.is_dir() and _is_task_collection_root(child)
    ]
    if model is not None:
        candidates = _candidate_model_dir_names(model)
        roots = [root for root in roots if root.name in candidates]
    elif len(roots) > 1:
        raise ValueError(
            "Archive root contains multiple model directories. Pass --model or point "
            "--archive-dir at a specific model directory."
        )
    return roots


def discover_model_roots(archive_dir: Path) -> dict[str, Path]:
    """Discover model directories inside an archive root.

    Returns a mapping of model directory name to its path. If archive_dir is
    itself a model cache root (contains task directories with run*.json), the
    mapping contains a single entry.
    """
    if not archive_dir.exists():
        raise ValueError(f"Archive dir does not exist: {archive_dir}")

    if _is_task_collection_root(archive_dir):
        return {archive_dir.name: archive_dir}

    roots = {}
    for child in sorted(archive_dir.iterdir()):
        if not child.is_dir():
            continue
        if _is_task_collection_root(child):
            roots[child.name] = child
        else:
            for subchild in sorted(child.iterdir()):
                if subchild.is_dir() and _is_task_collection_root(subchild):
                    roots[f"{child.name}/{subchild.name}"] = subchild
    return roots


def _matches_tier(task_id: str, tier: str | None) -> bool:
    if tier is None:
        return True
    return task_id.lower().startswith(_TIER_PREFIXES[tier])


def load_task_runs_archive(
    archive_dir: Path,
    model: str | None = None,
    task_ids: Iterable[str] | None = None,
    tier: str | None = None,
) -> dict[str, list[TaskRunResult]]:
    """Load cached TaskRunResult objects from a run cache/archive directory."""
    task_filter = set(task_ids or [])
    task_runs: dict[str, list[TaskRunResult]] = {}

    if not archive_dir.exists():
        raise ValueError(f"Archive dir does not exist: {archive_dir}")

    roots = _resolve_model_roots(archive_dir, model)
    if not roots:
        return {}

    for root in roots:
        for task_dir in sorted(child for child in root.iterdir() if child.is_dir()):
            task_id = task_dir.name
            if task_filter and task_id not in task_filter:
                continue
            if not _matches_tier(task_id, tier):
                continue

            runs = []
            for run_file in sorted(task_dir.glob("run*.json")):
                try:
                    run = TaskRunResult.model_validate_json(
                        run_file.read_text(encoding="utf-8")
                    )
                except (OSError, ValueError) as exc:
                    raise ValueError(f"Cannot load cached run {run_file}") from exc
                runs.append(run)

            if runs:
                task_runs.setdefault(task_id, []).extend(runs)

    for task_id, runs in task_runs.items():
        runs.sort(key=lambda run: run.run_index)

    return task_runs


def _aligned_mean_std(arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not arrays:
        return np.array([]), np.array([])
    max_len = max(len(arr) for arr in arrays)
    if max_len == 0:
        return np.array([]), np.array([])
    mat = np.full((len(arrays), max_len), np.nan)
    for idx, arr in enumerate(arrays):
        mat[idx, :len(arr)] = arr
    return np.nanmean(mat, axis=0), np.nanstd(mat, axis=0)


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values.tolist()]


def _empty_sensitivity_summary() -> dict[str, object]:
    return {
        "n_pairs": 0,
        "mean_score_delta": 0.0,
        "mean_tool_edit_distance": 0.0,
        "mean_family_js_divergence": 0.0,
        "mean_lyapunov_proxy": 0.0,
        "mean_initial_divergence": 0.0,
        "mean_final_divergence": 0.0,
        "mean_contraction_delta": 0.0,
        "mean_contraction_ratio": 0.0,
        "fraction_converging_pairs": 0.0,
        "mean_divergence_curve": [],
        "std_divergence_curve": [],
        "pair_points": [],
    }


def _summarize_sensitivity_group(pairs: list) -> dict[str, object]:
    if not pairs:
        return _empty_sensitivity_summary()

    divergence_curves = [pair.embedding_divergence for pair in pairs if len(pair.embedding_divergence) > 0]
    curve_mean, curve_std = _aligned_mean_std(divergence_curves)

    pair_points = []
    for pair in pairs:
        if len(pair.embedding_divergence) > 0:
            initial_divergence = float(pair.embedding_divergence[0])
            final_divergence = float(pair.embedding_divergence[-1])
            contraction_delta = final_divergence - initial_divergence
            contraction_ratio = final_divergence / max(initial_divergence, 1e-6)
        else:
            initial_divergence = 0.0
            final_divergence = 0.0
            contraction_delta = 0.0
            contraction_ratio = 0.0
        pair_points.append(
            {
                "score_delta": round(float(pair.score_delta), 4),
                "tool_edit_distance": int(pair.tool_edit_distance),
                "family_js_divergence": round(float(pair.family_js_divergence), 4),
                "lyapunov_proxy": round(float(pair.lyapunov_proxy), 4),
                "initial_divergence": round(initial_divergence, 4),
                "final_divergence": round(final_divergence, 4),
                "contraction_delta": round(contraction_delta, 4),
                "contraction_ratio": round(contraction_ratio, 4),
            }
        )

    converging_pairs = sum(
        1 for point in pair_points if point["final_divergence"] < point["initial_divergence"]
    )

    return {
        "n_pairs": len(pairs),
        "mean_score_delta": round(float(np.mean([pair.score_delta for pair in pairs])), 4),
        "mean_tool_edit_distance": round(float(np.mean([pair.tool_edit_distance for pair in pairs])), 4),
        "mean_family_js_divergence": round(float(np.mean([pair.family_js_divergence for pair in pairs])), 4),
        "mean_lyapunov_proxy": round(float(np.mean([pair.lyapunov_proxy for pair in pairs])), 4),
        "mean_initial_divergence": round(float(np.mean([point["initial_divergence"] for point in pair_points])), 4),
        "mean_final_divergence": round(float(np.mean([point["final_divergence"] for point in pair_points])), 4),
        "mean_contraction_delta": round(float(np.mean([point["contraction_delta"] for point in pair_points])), 4),
        "mean_contraction_ratio": round(float(np.mean([point["contraction_ratio"] for point in pair_points])), 4),
        "fraction_converging_pairs": round(converging_pairs / len(pair_points), 4),
        "mean_divergence_curve": _round_list(curve_mean),
        "std_divergence_curve": _round_list(curve_std),
        "pair_points": pair_points,
    }


def _build_sensitivity_sections(
    valid_runs_by_task: dict[str, list[TaskRunResult]],
) -> tuple[list, dict[str, object]]:
    same_task_pairs = []
    per_task: dict[str, object] = {}
    for task_id, runs in sorted(valid_runs_by_task.items()):
        if len(runs) < 2:
            continue
        task_pairs = [
            compute_sensitivity(run_a, run_b, task_id=task_id)
            for run_a, run_b in combinations(runs, 2)
        ]
        if task_pairs:
            same_task_pairs.extend(task_pairs)
            per_task[task_id] = _summarize_sensitivity_group(task_pairs)

    same_task_summary = _summarize_sensitivity_group(same_task_pairs)
    same_task_summary["per_task"] = per_task

    perturbation_pairs = []
    per_variant_group: dict[str, object] = {}
    runs_by_variant_group: dict[str, list[TaskRunResult]] = {}
    for runs in valid_runs_by_task.values():
        for run in runs:
            runs_by_variant_group.setdefault(run.variant_group or run.task_id, []).append(run)

    for variant_group, runs in sorted(runs_by_variant_group.items()):
        distinct_members = {
            (run.task_id, run.prompt_variant, run.variant_id)
            for run in runs
        }
        if len(distinct_members) < 2:
            continue

        group_pairs = []
        for run_a, run_b in combinations(runs, 2):
            if (
                run_a.task_id == run_b.task_id
                and run_a.prompt_variant == run_b.prompt_variant
                and run_a.variant_id == run_b.variant_id
            ):
                continue
            group_pairs.append(compute_sensitivity(run_a, run_b, task_id=variant_group))

        if not group_pairs:
            continue

        perturbation_pairs.extend(group_pairs)
        group_summary = _summarize_sensitivity_group(group_pairs)
        group_summary["members"] = [
            {
                "task_id": task_id,
                "prompt_variant": prompt_variant,
                "variant_id": variant_id,
            }
            for task_id, prompt_variant, variant_id in sorted(distinct_members)
        ]
        per_variant_group[variant_group] = group_summary

    perturbation_summary = _summarize_sensitivity_group(perturbation_pairs)
    perturbation_summary["per_variant_group"] = per_variant_group

    return same_task_pairs, {
        "same_task": same_task_summary,
        "prompt_perturbation": perturbation_summary,
    }


def build_dynamics_report(
    task_runs: dict[str, list[TaskRunResult]],
    include_pca: bool = True,
) -> tuple[dict[str, object], dict[str, object]]:
    """Compute stratified dynamics report data from cached runs."""
    all_runs = [run for runs in task_runs.values() for run in runs]
    if not all_runs:
        raise ValueError("No cached runs were loaded.")

    dynamics_list = []
    scores = []
    valid_runs = []
    for run in all_runs:
        if not run.transcript.messages:
            continue
        dynamics_list.append(compute_dynamics(run.transcript))
        scores.append(run.run_score)
        valid_runs.append(run)

    if not valid_runs:
        raise ValueError("No runs with transcripts were found in the archive.")

    valid_runs_by_task: dict[str, list[TaskRunResult]] = {}
    for run in valid_runs:
        valid_runs_by_task.setdefault(run.task_id, []).append(run)

    same_task_sensitivities, sensitivity_summary = _build_sensitivity_sections(valid_runs_by_task)

    stratifiers = {
        "tier": stratify_by_tier,
        "regime": stratify_by_regime,
        "tool_mix": stratify_by_tool_mix,
        "scenario": stratify_by_scenario,
    }

    report: dict[str, object] = {
        "n_runs": len(valid_runs),
        "n_tasks": len(task_runs),
        "strata": {},
    }

    stratified = {}
    for name, fn in stratifiers.items():
        assessment = build_strata(
            valid_runs,
            dynamics_list,
            scores,
            fn,
            name,
            sensitivities=same_task_sensitivities,
        )
        stratified[name] = assessment
        strata_summary = []
        for stratum in assessment.strata:
            strata_summary.append(
                {
                    "name": stratum.name,
                    "n_runs": stratum.n_runs,
                    "weight": round(stratum.weight, 4),
                    "score_mean": round(stratum.score_mean, 4),
                    "score_std": round(stratum.score_std, 4),
                    "score_quantiles": {
                        key: round(value, 4)
                        for key, value in stratum.score_quantiles.items()
                    },
                    "entropy_mean": round(float(stratum.entropy_dist.mean()), 4)
                    if len(stratum.entropy_dist)
                    else 0.0,
                    "error_rate_mean": round(float(stratum.error_rate_dist.mean()), 4)
                    if len(stratum.error_rate_dist)
                    else 0.0,
                    "constraint_mean": round(float(stratum.constraint_dist.mean()), 4)
                    if len(stratum.constraint_dist)
                    else 0.0,
                    "memory_depth_mean": round(float(stratum.memory_depth_dist.mean()), 4)
                    if len(stratum.memory_depth_dist)
                    else 0.0,
                    "sensitivity_pairs": int(len(stratum.sensitivity_deltas)),
                    "sensitivity_mean_score_delta": round(float(stratum.sensitivity_deltas.mean()), 4)
                    if len(stratum.sensitivity_deltas)
                    else 0.0,
                    "regime_counts": stratum.regime_counts,
                }
            )
        report["strata"][name] = {
            "observed_mean_score": round(assessment.observed_mean_score, 4),
            "observed_std_score": round(assessment.observed_std_score, 4),
            "strata": strata_summary,
        }

    report["per_run"] = [
        {
            "task_id": run.task_id,
            "run_index": run.run_index,
            "score": round(run.run_score, 4),
            "regime": dynamics.regime.value,
            "entropy": round(dynamics.tool_entropy, 4),
            "error_rate": round(dynamics.error_rate, 4),
            "constraint_index": round(dynamics.constraint_index, 4),
            "memory_depth": round(dynamics.memory_depth, 4),
            "n_steps": dynamics.n_steps,
            "mean_drift": round(dynamics.mean_drift, 4),
            "mean_step_size": round(dynamics.mean_step_size, 4),
        }
        for run, dynamics in zip(valid_runs, dynamics_list)
    ]
    report["sensitivity"] = sensitivity_summary

    if include_pca:
        compute_pca_bundle(dynamics_list)

    events = []
    censored = []
    for run in valid_runs:
        step = find_event_step(run.transcript, "first_correct_write")
        if step is not None:
            events.append(step)
            censored.append(False)
        else:
            events.append(float(len(run.transcript.assistant_messages)))
            censored.append(True)
    km_points = kaplan_meier(events, censored)
    return report, {
        "valid_runs": valid_runs,
        "dynamics_list": dynamics_list,
        "stratified": stratified,
        "km_points": km_points,
        "sensitivity": sensitivity_summary,
    }


def write_dynamics_report(
    task_runs: dict[str, list[TaskRunResult]],
    out_dir: Path,
    report_name: str = "dynamics.json",
    generate_plots: bool = True,
) -> tuple[Path, list[Path]]:
    """Write the dynamics report JSON and plots to an output directory."""
    report, plot_data = build_dynamics_report(task_runs, include_pca=generate_plots)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / report_name
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    plots: list[Path] = []
    if generate_plots:
        from clawbench.dynamics_plots import generate_all_plots

        plots = generate_all_plots(
            plot_data["dynamics_list"],
            plot_data["valid_runs"],
            plot_data["stratified"],
            km_points=plot_data["km_points"],
            event_name="first_correct_write",
            out_dir=out_dir,
            sensitivity_summary=plot_data["sensitivity"],
        )
    return report_path, plots


def load_task_runs_by_model(
    archive_dir: Path,
    tier: str | None = None,
    task_ids: Iterable[str] | None = None,
) -> dict[str, dict[str, list[TaskRunResult]]]:
    """Load cached TaskRunResult objects grouped by model directory name."""
    grouped: dict[str, dict[str, list[TaskRunResult]]] = {}
    for model_name, model_dir in discover_model_roots(archive_dir).items():
        task_runs = load_task_runs_archive(
            archive_dir=model_dir,
            model=None,
            task_ids=task_ids,
            tier=tier,
        )
        if task_runs:
            grouped[model_name] = task_runs
    return grouped
