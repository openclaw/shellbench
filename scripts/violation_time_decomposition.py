#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clawbench.dynamics_archive import load_task_runs_by_model, safe_model_name
from clawbench.trajectory import extract_shell_command


def get_first_violation_turn(run):
    """Return a 1-based event turn or the final observed censoring turn.

    A recorded violation with unknown timing returns (None, True). If any
    violation cannot be localized, the first violation time is also unknown.
    """
    messages = run.transcript.assistant_messages
    violations = run.trajectory_result.forbidden_violations
    if not violations:
        return len(messages), False

    event_turns = []
    for violation in set(violations):
        prefix, separator, value = violation.partition(": ")
        if not separator:
            return None, True
        matched_turn = None
        for turn, message in enumerate(messages, 1):
            for call in message.tool_calls:
                command = extract_shell_command(call)
                matched = False
                if prefix == "Forbidden tool called":
                    matched = call.name == value
                elif prefix == "Dangerous shell command":
                    matched = command == value
                elif prefix == "Forbidden shell pattern matched" and command:
                    try:
                        matched = re.search(value, command, re.IGNORECASE) is not None
                    except re.error:
                        return None, True
                if matched:
                    matched_turn = turn
                    break
            if matched_turn is not None:
                break
        if matched_turn is None:
            return None, True
        event_turns.append(matched_turn)
    return min(event_turns), True


def compute_decomposition(events_by_topic, max_t=20):
    """
    events_by_topic: dict mapping topic (e.g., scenario) to list of (turn, has_violation)
    Returns decomposition metrics.
    """
    if max_t < 1:
        raise ValueError("max_t must be positive")
    observed_events = [event for events in events_by_topic.values() for event in events]
    if not observed_events:
        return {}
    unknown_count = sum(turn is None and event for turn, event in observed_events)
    events_by_topic = {
        topic: [(turn, event) for turn, event in events if turn is not None]
        for topic, events in events_by_topic.items()
    }
    # Unknown violation times are excluded, never treated as clean observations.
    all_events = []
    for evs in events_by_topic.values():
        all_events.extend(evs)

    metrics = {
        "unknown_violation_count": unknown_count,
        "timed_run_count": len(all_events),
        "at_risk_counts": [],
        "event_counts": [],
        "marginal_hazard": [],
        "marginal_survival": [],
        "conditional_hazards": defaultdict(list),
        "mutual_information": [],
    }

    survival = 1.0
    # Discrete Kaplan–Meier: censoring changes exposure, not survival.
    for t in range(1, max_t + 1):
        at_risk_total = sum(1 for tf, _ in all_events if tf >= t)
        events_total = sum(1 for tf, is_event in all_events if is_event and tf == t)

        h_t = events_total / at_risk_total if at_risk_total > 0 else 0.0
        survival *= 1.0 - h_t

        metrics["marginal_hazard"].append(h_t)
        metrics["marginal_survival"].append(survival)
        metrics["at_risk_counts"].append(at_risk_total)
        metrics["event_counts"].append(events_total)

        # Calculate conditional hazards
        mi_t = 0.0
        for topic, evs in events_by_topic.items():
            at_risk_topic = sum(1 for tf, _ in evs if tf >= t)
            events_topic = sum(1 for tf, is_event in evs if is_event and tf == t)
            h_t_given_s = events_topic / at_risk_topic if at_risk_topic > 0 else 0.0
            metrics["conditional_hazards"][topic].append(h_t_given_s)

            # P(S = topic | T >= t)
            if at_risk_total > 0 and at_risk_topic > 0:
                p_s_given_at_risk = at_risk_topic / at_risk_total

                # MI term: P(S|T>=t) * D_KL( P(V|S) || P(V) )
                kl = 0.0
                if h_t_given_s > 0 and h_t > 0:
                    kl += h_t_given_s * math.log2(h_t_given_s / h_t)
                if (1 - h_t_given_s) > 0 and (1 - h_t) > 0:
                    kl += (1 - h_t_given_s) * math.log2((1 - h_t_given_s) / (1 - h_t))

                mi_t += p_s_given_at_risk * kl

        metrics["mutual_information"].append(mi_t)

    return metrics


def plot_metrics(metrics, model_name, output_dir):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    max_t = len(metrics["marginal_hazard"])
    turns = list(range(1, max_t + 1))

    plt.figure(figsize=(15, 5))

    # Plot 1: Marginal Hazard
    plt.subplot(1, 3, 1)
    plt.plot(turns, metrics["marginal_hazard"], marker="o", color="red")
    plt.title("Marginal Hazard $h(t)$")
    plt.xlabel("Turn $t$")
    plt.ylabel("P(Violation | No Prior Violation)")
    plt.grid(True)

    # Plot 2: Conditional Hazards
    plt.subplot(1, 3, 2)
    for topic, h_cond in metrics["conditional_hazards"].items():
        if max(h_cond) > 0:  # Only plot if there's non-zero hazard
            plt.plot(turns, h_cond, alpha=0.5, label=topic)
    plt.title("Conditional Hazards $h(t | Topic)$")
    plt.xlabel("Turn $t$")
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.grid(True)

    # Plot 3: Mutual Information
    plt.subplot(1, 3, 3)
    plt.plot(turns, metrics["mutual_information"], marker="o", color="purple")
    plt.title(r"Mutual Information $I(Topic; V_t | T \geq t)$")
    plt.xlabel("Turn $t$")
    plt.ylabel("Bits")
    plt.grid(True)

    plt.tight_layout()
    plot_path = output_dir / f"violation_metrics_{model_name}.png"
    plt.savefig(plot_path)
    plt.close()
    return plot_path


def generate_markdown(metrics, model_name, plot_path, out_file):
    max_t = len(metrics["marginal_hazard"])

    md = [
        f"# Spatio-Temporal Violation Dynamics: {model_name}",
        "",
        "## Theoretical Decomposition",
        "This report decomposes the time-to-first-violation for safety triggers and tool misuses.",
        "To connect the long-term behavior of agent risk to its spatial risk conditioned on context semantics, we decompose the first occurrence probability:",
        r"$$ P(T = t) = h(t) \cdot S(t-1) $$",
        "where $h(t)$ is the conditional hazard rate at turn $t$, and $S(t-1)$ is the macro survival probability.",
        "",
        "Furthermore, we examine the mutual information between the semantic spatial context (scenario) and the violation event at each turn to determine if localized contexts explain hazard spikes.",
        "",
        "## Observation coverage",
        f"Timed runs: {metrics['timed_run_count']}. "
        f"Violations with unknown timing excluded: {metrics['unknown_violation_count']}.",
        "Clean runs are right-censored at their last observed assistant turn. "
        "Survival is the cumulative product of hazard complements (Kaplan–Meier). "
        "Unknown event times are excluded from all timed estimates; these estimates "
        "describe the retained runs and can be biased if missing timing is informative. "
        "After the risk set is empty, hazard and mutual information use zero placeholders "
        "and survival is carried forward; these are not evidence of zero future risk. "
        "Censoring-aware estimates assume non-informative censoring.",
        "",
        "## Visualization",
        f"![Violation Metrics]({plot_path.name})"
        if plot_path
        else "Plot generation skipped because matplotlib is not installed.",
        "",
        "## Empirical Metrics Table",
        "| Turn $t$ | Marginal $S(t)$ | Marginal $h(t)$ | Mutual Info (bits) | At risk | Events |",
        "|----------|-----------------|-----------------|--------------------|---------|--------|",
    ]

    for i in range(max_t):
        t = i + 1
        s = metrics["marginal_survival"][i]
        h = metrics["marginal_hazard"][i]
        mi = metrics["mutual_information"][i]
        at_risk = metrics["at_risk_counts"][i]
        events = metrics["event_counts"][i]
        md.append(f"| {t} | {s:.4f} | {h:.4f} | {mi:.4f} | {at_risk} | {events} |")

    out_file.write_text("\n".join(md), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-dir", type=Path, default=Path(".clawbench/run_cache"))
    parser.add_argument(
        "--reports-dir",
        "--results-dir",
        dest="reports_dir",
        type=Path,
        default=Path("reports/violation_time_decomposition"),
    )
    parser.add_argument(
        "--tier", choices=["tier1", "tier2", "tier3", "tier4", "tier5"], default=None
    )
    parser.add_argument("--max-turn", type=int, default=15)
    args = parser.parse_args()
    if args.max_turn < 1:
        parser.error("--max-turn must be positive")

    print(f"Loading runs from {args.archive_dir}...")
    grouped = load_task_runs_by_model(args.archive_dir, tier=args.tier)
    print(f"Loaded models: {list(grouped.keys())}")

    args.reports_dir.mkdir(parents=True, exist_ok=True)

    for model_name, task_runs in grouped.items():
        print(f"Processing model: {model_name}")
        events_by_topic = defaultdict(list)
        for task_id, runs in task_runs.items():
            for run in runs:
                topic = run.scenario if run.scenario else "unknown"
                events_by_topic[topic].append(get_first_violation_turn(run))

        metrics = compute_decomposition(events_by_topic, max_t=args.max_turn)
        if not metrics:
            print(f"No events for {model_name}")
            continue

        safe_model = safe_model_name(model_name)
        model_out_dir = args.reports_dir / safe_model
        model_out_dir.mkdir(parents=True, exist_ok=True)

        plot_path = plot_metrics(metrics, safe_model, model_out_dir)
        doc_path = model_out_dir / "dynamics_violation_decomposition.md"
        generate_markdown(metrics, model_name, plot_path, doc_path)
        print(f"Generated doc for {model_name}: {doc_path}")

        # Dump JSON
        json_path = model_out_dir / "violation_metrics.json"
        json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
