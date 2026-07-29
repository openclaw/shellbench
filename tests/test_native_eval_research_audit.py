from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.native_eval.research_audit import (
    _discovery_status,
    _openclaw_code_discovery_rows,
    _structured_discovery_row,
    export_research_tables,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_structured_discovery_success_stays_unknown_without_result_status() -> None:
    row = _structured_discovery_row(
        base={},
        function_name="tool_call",
        arguments={"id": "openclaw:core:read"},
    )

    assert row is not None
    assert row["success"] == ""


def test_zero_count_code_telemetry_is_observed_but_not_exercised() -> None:
    rows, telemetry_observed = _openclaw_code_discovery_rows(
        base={},
        observation=json.dumps(
            {
                "telemetry": {
                    "catalogSize": 12,
                    "searchCount": 0,
                    "describeCount": 0,
                    "callCount": 0,
                }
            }
        ),
    )

    assert rows == []
    assert telemetry_observed is True
    assert (
        _discovery_status(
            harness="openclaw",
            openclaw_mode="code",
            tool_names=["tool_search_code"],
            discovery_rows=rows,
            telemetry_observed=telemetry_observed,
        )
        == "supported_not_exercised"
    )


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
    assert trace_row["judge_identity_status"] == (
        "unverified_requires_proxy_request_evidence"
    )
    assert trace_row["phase"] == "full"
    assert trace_row["cost_provenance"] == "exact_harness"
    assert trace_row["discovery_status"] == "disabled"
    with (output / "turn_usage.csv").open(newline="", encoding="utf-8") as handle:
        turn_row = next(csv.DictReader(handle))
    assert turn_row["n_input_tokens"] == "12"
    assert turn_row["cost_provenance"] == "exact_trace_turn"
    with (output / "tool_calls.csv").open(newline="", encoding="utf-8") as handle:
        tool_row = next(csv.DictReader(handle))
    assert tool_row["function_name"] == "shell"
    assert tool_row["observation_excerpt"] == "/workspace"
    with (output / "discovery_events.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert list(csv.DictReader(handle)) == []


def test_research_audit_exports_openclaw_code_discovery_telemetry(
    tmp_path: Path,
) -> None:
    run_label = "openclaw-gpt55-high-ablation-1-r1-20260729"
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
                    "model_slug": "gpt55",
                    "model_id": "gpt-5.5",
                    "reasoning_effort": "high",
                    "repetition": 1,
                    "expected_task_count": 1,
                    "openclaw_tool_search_mode": "code",
                }
            ]
        },
    )
    _write_json(
        job_dir / "run_manifest.json",
        {
            "run_label": run_label,
            "runner_commit": "runner-sha",
            "openclaw_tool_search_mode": "code",
        },
    )
    _write_json(
        trial_dir / "result.json",
        {
            "task_id": {"path": "/tasks/example-task"},
            "agent_result": {
                "trajectory_status": "real",
                "runtime_model_name": "gpt-5.5",
                "canonical_model_identity": True,
            },
        },
    )
    _write_json(
        trajectory_path,
        {
            "agent": {
                "name": "openclaw",
                "version": "2026.7.1-2",
                "model_name": "openai/gpt-5.5",
            },
            "steps": [
                {
                    "source": "agent",
                    "tool_calls": [
                        {
                            "tool_call_id": "call-1",
                            "function_name": "tool_search_code",
                            "arguments": {
                                "code": "return await openclaw.tools.search('mail')"
                            },
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": "call-1",
                                "content": json.dumps(
                                    {
                                        "ok": True,
                                        "telemetry": {
                                            "catalogSize": 42,
                                            "sources": {
                                                "openclaw": 30,
                                                "mcp": 10,
                                                "client": 2,
                                            },
                                            "searchCount": 2,
                                            "describeCount": 1,
                                            "callCount": 1,
                                        },
                                    }
                                ),
                            }
                        ]
                    },
                }
            ],
            "extra": {
                "observed_models": ["gpt-5.5"],
                "trace_fidelity": "session",
            },
        },
    )

    summary = export_research_tables(
        run_index_path=run_index,
        extracted_root=extracted,
        output_dir=output,
    )

    assert summary["discovery_event_count"] == 4
    with (output / "trace_inventory.csv").open(newline="", encoding="utf-8") as handle:
        trace_row = next(csv.DictReader(handle))
    assert trace_row["runner_commit"] == "runner-sha"
    assert trace_row["trace_fidelity"] == "session"
    assert trace_row["discovery_status"] == "observed"
    assert trace_row["discovery_event_count"] == "4"
    with (output / "discovery_events.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["operation"], row["count"]) for row in rows] == [
        ("search", "2"),
        ("describe", "1"),
        ("call", "1"),
    ]
    assert {row["catalog_size"] for row in rows} == {"42"}
    assert {row["success"] for row in rows} == {""}
    assert {row["trace_fidelity"] for row in rows} == {"session"}


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
