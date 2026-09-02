#!/usr/bin/env python3
import json
import argparse
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def compute_debiased_dynamics(regimes_path, constraint_path, weights_path, topics_path, output_path):
    """
    Computes the Horvitz-Thompson / Hajek estimators for the temporal
    dynamical properties (Regime Distributions, Constraint Index)
    using the Radon-Nikodym derivatives (weights).
    """
    with open(weights_path, 'r') as f:
        weights = json.load(f)

    with open(topics_path, 'r') as f:
        topics_data = json.load(f)

    # Extract topics
    task_topics = {}
    for task_id, data in topics_data.items():
        if isinstance(data, dict):
            task_topics[task_id] = data.get("topic", "unknown")
        else:
            task_topics[task_id] = str(data)

    # 1. Debiased Regimes
    with open(regimes_path, 'r') as f:
        regimes = json.load(f)

    model_regimes_weighted = defaultdict(lambda: defaultdict(float))
    model_regimes_weight_sum = defaultdict(float)

    for key, data in regimes.items():
        key_parts = key.rsplit("/", 2)
        model = data.get("model") or (key_parts[0] if len(key_parts) == 3 else key)
        task_id = data.get("task_id") or (key_parts[1] if len(key_parts) == 3 else key)

        # Match task to topic
        matched_topic = "unknown"
        for t_id in task_topics:
            if task_id.startswith(t_id):
                matched_topic = task_topics[t_id]
                break

        rho = weights.get(matched_topic, 1.0)
        regime = data.get("regime", "unknown")

        model_regimes_weighted[model][regime] += rho
        model_regimes_weight_sum[model] += rho

    debiased_regimes = {}
    for model, r_counts in model_regimes_weighted.items():
        total_w = model_regimes_weight_sum[model]
        if total_w > 0:
            debiased_regimes[model] = {r: float(w / total_w) for r, w in r_counts.items()}
        else:
            debiased_regimes[model] = {}

    # 2. Debiased Constraint Index (Expected Predictability)
    with open(constraint_path, 'r') as f:
        constraints = json.load(f)

    weighted_cq_sum = 0.0
    cq_weight_sum = 0.0
    for task_id, data in constraints.items():
        matched_topic = "unknown"
        for t_id in task_topics:
            if task_id.startswith(t_id):
                matched_topic = task_topics[t_id]
                break

        rho = weights.get(matched_topic, 1.0)
        cq = data.get("C_q", 0.0)
        weighted_cq_sum += rho * cq
        cq_weight_sum += rho

    debiased_cq = float(weighted_cq_sum / cq_weight_sum) if cq_weight_sum > 0 else 0.0

    output = {
        "debiased_expected_C_q": debiased_cq,
        "debiased_regimes_probability": debiased_regimes
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)
    logging.info(f"Wrote debiased Space-Time dynamics to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Debiased Dynamics")
    parser.add_argument("--regimes", required=True, help="Path to empirical regimes JSON")
    parser.add_argument("--constraint", required=True, help="Path to empirical constraint index JSON")
    parser.add_argument("--weights", required=True, help="Path to importance weights JSON")
    parser.add_argument("--topics", required=True, help="Path to task-to-topic mapping JSON (e.g. mock results)")
    parser.add_argument("--output", required=True, help="Path to output debiased JSON")
    args = parser.parse_args()

    compute_debiased_dynamics(
        args.regimes,
        args.constraint,
        args.weights,
        args.topics,
        args.output
    )
