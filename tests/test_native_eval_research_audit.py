from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.native_eval.research_audit import export_research_tables


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_research_audit_exports_identity_turn_tool_and_usage_tables(
    tmp_path: Path,
) -> None:
    run_label = "openclaw-gpt56-sol-high-full-1-r1-20260729"
    run_index = tmp_path / "run-index.json"
    extracted = tmp_path / "extracted"
    output = tmp_path / "analysis"
    job_dir = extracted / run_label
    trial_dir = job_dir / "task__abc"
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    _write_json(
        run_index,
        {
            "runs": [
                {
                    "run_label": run_label,
                    "harness": "openclaw",
                    "harness_version": "2026.7.1-2",
                    "model_slug": "gpt56-sol",
                    "model_id": "gpt-5.6-sol",
                    "reasoning_effort": "high",
                    "judge_model_id": "gpt-5.6-sol",
                    "judge_reasoning_effort": "high",
                    "repetition": 1,
                    "phase": "full",
                    "expected_task_count": 1,
                }
            ]
        },
    )
    _write_json(job_dir / "run_manifest.json", {"run_label": run_label})
    _write_json(
        extracted / f"shellbench_meta-{run_label}" / "toolchain_manifest.json",
        {"openclaw": "openclaw 2026.7.1-2"},
    )
    (extracted / "proxy" / run_label).mkdir(parents=True)
    (extracted / "proxy" / run_label / "proxy.log").write_text(
        "proxy output\n",
        encoding="utf-8",
    )
    _write_json(
        trial_dir / "result.json",
        {
            "task_id": {"path": "/tasks/example-task"},
            "verifier_result": {"rewards": {"reward": 1}},
            "agent_result": {
                "trajectory_status": "real",
                "runtime_model_name": "gpt-5.6-sol",
                "canonical_model_identity": True,
                "n_input_tokens": 120,
                "n_cache_tokens": 20,
                "n_output_tokens": 30,
                "cost_usd": 0.25,
            },
        },
    )
    _write_json(
        trajectory_path,
        {
            "agent": {
                "name": "openclaw",
                "version": "2026.7.1-2",
                "model_name": "openai/gpt-5.6-sol",
            },
            "steps": [
                {
                    "source": "agent",
                    "message": "",
                    "usage": {
                        "input_tokens": 12,
                        "output_tokens": 3,
                        "cost_usd": 0.01,
                    },
                    "tool_calls": [
                        {
                            "tool_call_id": "call-1",
                            "function_name": "shell",
                            "arguments": {"command": "pwd"},
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "call-1",
                                "content": "/workspace",
                            }
                        ]
                    },
                }
            ],
            "final_metrics": {
                "total_prompt_tokens": 120,
                "total_cached_tokens": 20,
                "total_completion_tokens": 30,
                "total_cost_usd": 0.25,
            },
            "extra": {"observed_models": ["gpt-5.6-sol"]},
        },
    )

    summary = export_research_tables(
        run_index_path=run_index,
        extracted_root=extracted,
        output_dir=output,
    )

    assert summary["identity_audit_pass_count"] == 1
    assert summary["task_result_count"] == 1
    assert summary["turn_count"] == 1
    assert summary["tool_call_count"] == 1
    with (output / "trace_inventory.csv").open(newline="", encoding="utf-8") as handle:
        trace_row = next(csv.DictReader(handle))
    assert trace_row["model_identity_status"] == "match"
    assert trace_row["harness_version_status"] == "match"
    assert trace_row["installed_harness_version"] == "openclaw 2026.7.1-2"
    assert trace_row["judge_identity_status"] == ("unverified_requires_proxy_request_evidence")
    assert trace_row["phase"] == "full"
    assert trace_row["cost_provenance"] == "reported_harness"
    with (output / "turn_usage.csv").open(newline="", encoding="utf-8") as handle:
        turn_row = next(csv.DictReader(handle))
    assert turn_row["n_input_tokens"] == "12"
    assert turn_row["cost_provenance"] == "reported_trace_turn"
    with (output / "tool_calls.csv").open(newline="", encoding="utf-8") as handle:
        tool_row = next(csv.DictReader(handle))
    assert tool_row["function_name"] == "shell"
    assert tool_row["observation_excerpt"] == "/workspace"


def test_research_audit_fails_identity_when_trace_is_missing(tmp_path: Path) -> None:
    run_label = "hermes-gpt56-sol-low-full-1-r1-20260729"
    run_index = tmp_path / "run-index.json"
    extracted = tmp_path / "extracted"
    job_dir = extracted / run_label
    _write_json(
        run_index,
        {
            "runs": [
                {
                    "run_label": run_label,
                    "harness": "hermes",
                    "model_slug": "gpt56-sol",
                    "model_id": "gpt-5.6-sol",
                    "reasoning_effort": "low",
                    "repetition": 1,
                    "expected_task_count": 1,
                }
            ]
        },
    )
    _write_json(job_dir / "run_manifest.json", {"run_label": run_label})
    _write_json(
        job_dir / "task__abc" / "result.json",
        {
            "task_id": {"path": "/tasks/example-task"},
            "agent_result": {"trajectory_status": "unavailable"},
        },
    )

    summary = export_research_tables(
        run_index_path=run_index,
        extracted_root=extracted,
        output_dir=tmp_path / "analysis",
    )

    assert summary["identity_audit_pass_count"] == 0
    assert summary["identity_audit_fail_count"] == 1


def _export_family(
    tmp_path: Path,
    trajectory: dict,
    *,
    receipt: dict | None = None,
    agent_result: dict | None = None,
    result_extra: dict | None = None,
):
    run_index = tmp_path / "run-index.json"
    job = tmp_path / "extracted" / "run"
    agent = job / "task" / "agent"
    _write_json(
        run_index,
        {
            "runs": [
                {
                    "run_label": "run",
                    "harness": "openclaw",
                    "model_id": "parent",
                    "expected_task_count": 1,
                }
            ]
        },
    )
    _write_json(job / "run_manifest.json", {"run_label": "run"})
    _write_json(
        job / "task" / "result.json",
        {"task_id": "synthetic", "agent_result": agent_result or {}, **(result_extra or {})},
    )
    _write_json(agent / "trajectory.json", trajectory)
    if receipt is not None:
        _write_json(agent / "receipt.json", receipt)
    output = tmp_path / "output"
    summary = export_research_tables(
        run_index_path=run_index, extracted_root=job.parent, output_dir=output
    )
    tables = {}
    for path in output.glob("*.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            tables[path.stem] = list(csv.DictReader(handle))
    return summary, tables


def _node(name: str, model: str = "parent", **kwargs) -> dict:
    return {
        "trajectory_id": f"trajectory-{name}",
        "session_id": f"session-{name}",
        "agent": {"model_name": f"provider/{model}"},
        "extra": {"openclaw": {"session_key": f"key-{name}", "leaf_id": "leaf"}},
        "steps": [{"step_id": 1, "source": "agent", "model_name": f"provider/{model}"}],
        **kwargs,
    }


def _receipt(trajectory: dict, nodes: list[dict]) -> dict:
    import hashlib

    return {
        "schema": "openclaw-atif-receipt-v1",
        "status": "complete",
        "source": {"stable": True},
        "diagnostics": [],
        "root": {
            "sessionKey": "key-root",
            "sessionId": "session-root",
            "trajectoryId": "trajectory-root",
        },
        "nodes": [
            {
                "sessionKey": node["extra"]["openclaw"]["session_key"],
                "sessionId": node["session_id"],
                "leafId": "leaf",
            }
            for node in nodes
        ],
        "familyMetrics": {"promptTokens": 90, "completionTokens": 0},
        "output": {"trajectorySha256": hashlib.sha256(json.dumps(trajectory).encode()).hexdigest()},
    }


def test_nested_family_exports_unique_nodes_models_and_tool_lineage(tmp_path: Path) -> None:
    grandchild = _node("grandchild", "switched")
    child = _node("child", "child", subagent_trajectories=[grandchild])
    root = _node("root", subagent_trajectories=[child, child])
    root["steps"][0].update(
        {
            "tool_calls": [{"tool_call_id": "spawn", "function_name": "spawn"}],
            "observation": {
                "results": [
                    {
                        "source_call_id": "spawn",
                        "content": "ok",
                        "subagent_trajectory_ref": [
                            {"trajectory_id": "trajectory-child", "session_id": "session-child"}
                        ],
                    }
                ]
            },
        }
    )
    summary, tables = _export_family(
        tmp_path, root, receipt=_receipt(root, [root, child, grandchild])
    )
    assert summary["turn_count"] == 3
    assert len(tables["trajectory_nodes"]) == 3
    trace = tables["trace_inventory"][0]
    assert json.loads(trace["observed_model_ids"]) == ["child", "parent", "switched"]
    assert trace["model_identity_status"] == "mismatch"
    child_turn = next(row for row in tables["turn_usage"] if row["session_id"] == "session-child")
    assert child_turn["trajectory_id"] == "trajectory-child"
    assert json.loads(child_turn["parent_tool_call_ids_json"]) == ["spawn"]
    assert tables["tool_calls"][0]["trajectory_id"] == "trajectory-root"
    assert any(row["tool_call_id"] == "spawn" for row in tables["trajectory_links"])


def test_explicit_zero_does_not_fall_through_to_root_or_family_totals(tmp_path: Path) -> None:
    root = _node(
        "root",
        final_metrics={
            "total_prompt_tokens": 10,
            "total_cached_tokens": 11,
            "total_completion_tokens": 12,
            "total_cost_usd": 1,
        },
    )
    _, tables = _export_family(
        tmp_path,
        root,
        receipt=_receipt(root, [root]),
        agent_result={
            "n_input_tokens": 0,
            "n_cache_tokens": 0,
            "n_output_tokens": 0,
            "cost_usd": 0,
        },
    )
    row = tables["trace_inventory"][0]
    assert (row["n_input_tokens"], row["n_cache_tokens"], row["n_output_tokens"]) == ("0", "0", "0")
    assert row["cost_usd"] == "0"
    assert row["cost_provenance"] == "reported_harness"
    assert json.loads(row["receipt_family_metrics_json"])["promptTokens"] == 90
    assert json.loads(row["root_metrics_json"])["total_prompt_tokens"] == 10


def test_digest_coverage_and_execution_reward_are_independent(tmp_path: Path) -> None:
    root = _node("root")
    receipt = _receipt(root, [root, _node("not-embedded")])
    _, tables = _export_family(
        tmp_path,
        root,
        receipt=receipt,
        result_extra={
            "execution_outcome": {"kind": "clean"},
            "verifier_result": {"rewards": {"reward": 0}},
        },
    )
    row = tables["trace_inventory"][0]
    assert row["receipt_digest_status"] == "match"
    assert row["receipt_node_coverage_status"] == "partial"
    assert row["capture_status"] == "partial"
    assert row["execution_status"] == "accepted"
    assert row["reward_status"] == "observed"
    assert row["reward"] == "0"
    assert row["resource_status"] == "not_observed"
    receipt["output"]["trajectorySha256"] = "0" * 64
    _, tables = _export_family(tmp_path, root, receipt=receipt)
    assert tables["trace_inventory"][0]["capture_status"] == "invalid"
    assert tables["trace_inventory"][0]["execution_status"] == "not_observed"


def test_per_model_coverage_keeps_missing_usage_and_step_model_switch(tmp_path: Path) -> None:
    root = _node(
        "root",
        steps=[
            {
                "source": "agent",
                "model_name": "provider/parent",
                "metrics": {
                    "prompt_tokens": 0,
                    "cached_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0,
                },
            },
            {"source": "agent", "model_name": "provider/parent", "metrics": {}},
            {"source": "agent", "model_name": "provider/switched", "metrics": {"prompt_tokens": 2}},
        ],
    )
    _, tables = _export_family(tmp_path, root)
    rows = {row["observed_model_name"]: row for row in tables["model_usage_coverage"]}
    assert rows["provider/parent"]["agent_step_count"] == "2"
    assert rows["provider/parent"]["input_observed_step_count"] == "1"
    assert rows["provider/parent"]["n_input_tokens"] == "0"
    assert rows["provider/switched"]["n_output_tokens"] == ""
    assert tables["trace_inventory"][0]["token_status"] == "partial"
    assert tables["trace_inventory"][0]["model_identity_status"] == "mismatch"


def test_conflicting_duplicate_and_unknown_child_cannot_pass_identity(tmp_path: Path) -> None:
    child = _node("child")
    conflicting = {**child, "steps": [{"source": "agent", "model_name": "other"}]}
    root = _node("root", subagent_trajectories=[child, conflicting])
    _, tables = _export_family(tmp_path, root, receipt=_receipt(root, [root, child]))
    assert tables["trace_inventory"][0]["capture_status"] == "invalid"
    assert tables["trace_inventory"][0]["model_identity_status"] != "match"
    root["subagent_trajectories"] = [{"session_id": "unknown", "steps": [{"source": "agent"}]}]
    _, tables = _export_family(tmp_path, root)
    assert tables["trace_inventory"][0]["model_identity_status"] == "not_observed"


def test_unmatched_tool_observation_does_not_borrow_another_call_result(tmp_path: Path) -> None:
    root = _node(
        "root",
        steps=[
            {
                "source": "agent",
                "tool_calls": [
                    {"tool_call_id": "present", "function_name": "one"},
                    {"tool_call_id": "missing", "function_name": "two"},
                ],
                "observation": {"results": [{"source_call_id": "present", "content": "only-one"}]},
            }
        ],
    )
    _, tables = _export_family(tmp_path, root)
    assert tables["tool_calls"][0]["observation_excerpt"] == "only-one"
    assert tables["tool_calls"][1]["observation_excerpt"] == ""


def test_reported_step_cost_is_not_billing_exact_and_unknown_is_not_zero(tmp_path: Path) -> None:
    root = _node("root", steps=[{"source": "agent", "metrics": {"cost_usd": 0}}])
    summary, tables = _export_family(tmp_path, root)
    trace = tables["trace_inventory"][0]
    assert trace["cost_usd"] == ""
    assert trace["cost_status"] == "reported_unreconciled"
    assert trace["token_status"] == "not_observed"
    assert trace["n_input_tokens"] == ""
    assert summary["exact_task_cost_count"] == 0
    assert tables["turn_usage"][0]["cost_provenance"] == "reported_trace_turn"


def test_invalid_usage_is_missing_not_complete(tmp_path: Path) -> None:
    root = _node(
        "root",
        steps=[
            {
                "source": "agent",
                "metrics": {"prompt_tokens": -1, "cached_tokens": True, "completion_tokens": "NaN"},
            }
        ],
    )
    _, tables = _export_family(tmp_path, root)
    assert tables["trace_inventory"][0]["token_status"] == "not_observed"
    assert tables["model_usage_coverage"][0]["n_input_tokens"] == ""
