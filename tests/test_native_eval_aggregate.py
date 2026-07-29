import csv
import json
from pathlib import Path

import pytest

from scripts.native_eval.aggregate import aggregate, main


def _write_run(
    jobs_root: Path,
    run_label: str,
    *,
    expected_task_count: int,
    results: list[dict],
    pair_label: str | None = "openclaw-model",
    repetition: int = 1,
    tasks: list[str] | None = None,
    harness: str = "openclaw",
    model_slug: str = "model",
    native: bool = False,
    parity_validated: bool = False,
    parity_validation: dict | None = None,
    leaderboard_eligible: bool | None = None,
    exclusion_reason: str = "",
    manifest_extra: dict | None = None,
) -> None:
    run_dir = jobs_root / "jobs" / run_label
    run_dir.mkdir(parents=True)
    manifest = {
        "run_label": run_label,
        "harness": harness,
        "model_slug": model_slug,
        "repetition": repetition,
        "expected_task_count": expected_task_count,
    }
    if native:
        scoped_validation = parity_validation
        if parity_validated and scoped_validation is None:
            scoped_validation = {
                "validated": True,
                "scope": {
                    "harness": harness,
                    "model_slug": model_slug,
                },
            }
        manifest.update(
            {
                "runner": "shellbench-native",
                "canonical_model_identity": True,
                "trajectory_mode": "real_harness_events",
                "parity_validated": parity_validated,
                "parity_validation": scoped_validation,
            }
        )
    if pair_label is not None:
        manifest["pair_label"] = pair_label
    if leaderboard_eligible is not None:
        manifest["leaderboard_eligible"] = leaderboard_eligible
        manifest["exclusion_reason"] = exclusion_reason
    if tasks is not None:
        manifest["tasks"] = tasks
    if manifest_extra is not None:
        manifest.update(manifest_extra)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest))

    for index, result in enumerate(results, start=1):
        trial_dir = run_dir / result.get("trial_name", f"trial-{index}")
        trial_dir.mkdir()
        (trial_dir / "result.json").write_text(json.dumps(result))


def _result(
    task: str,
    *,
    reward: float | None = None,
    exception_type: str | None = None,
    exception_message: str | None = None,
    **extra: object,
) -> dict:
    result = {
        "task_name": task,
        "task_id": {"path": f"/benchmark/tasks/{task}"},
        "trial_name": f"{task}__trial",
        "started_at": "2026-07-27T10:00:00+00:00",
        "finished_at": "2026-07-27T10:01:00+00:00",
        "environment_setup": {
            "started_at": "2026-07-27T10:00:00+00:00",
            "finished_at": "2026-07-27T10:00:05+00:00",
        },
        "agent_execution": {
            "started_at": "2026-07-27T10:00:10+00:00",
            "finished_at": "2026-07-27T10:00:50+00:00",
        },
        "verifier": {
            "started_at": "2026-07-27T10:00:50+00:00",
            "finished_at": "2026-07-27T10:01:00+00:00",
        },
        "agent_result": {
            "n_input_tokens": 100,
            "n_cache_tokens": 20,
            "n_output_tokens": 30,
            "cost_usd": 0.0125,
        },
        **extra,
    }
    if reward is not None:
        result["verifier_result"] = {"rewards": {"reward": reward}}
    if exception_type is not None:
        result["exception_info"] = {
            "exception_type": exception_type,
            "exception_message": exception_message or f"{exception_type} happened",
            "exception_traceback": "traceback",
            "occurred_at": "2026-07-27T10:00:55+00:00",
        }
    return result


def _native_result(task: str, *, reward: float = 1.0) -> dict:
    result = _result(task, reward=reward, source="shellbench-native")
    result["agent_result"].update(
        {
            "trajectory_status": "real",
            "runtime_model_name": "gpt-5.5",
            "canonical_model_identity": True,
        }
    )
    return result


def test_aggregate_classifies_harbor_results_and_writes_all_outputs(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "model-rep-1",
        expected_task_count=6,
        results=[
            _result("pass-task", reward=1.0),
            _result("partial-task", reward=0.4),
            _result("fail-task", reward=0.0),
            _result("missing-task", exception_type="RewardFileNotFoundError"),
            _result("agent-exit-task", reward=0.0, exception_type="AgentTimeoutError"),
            _result("infra-task", exception_type="EnvironmentStartTimeoutError"),
        ],
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["score"] == pytest.approx(1.4 / 6)
    assert run["coverage"] == 1.0
    assert run["completed_result_count"] == 6
    assert run["exact_passes"] == 1
    assert run["partials"] == 1
    assert run["nonzero"] == 2
    assert run["infra"] == 1
    assert run["agent_exits"] == 1
    assert run["clean_completed"] == 3
    assert run["missing_reward"] == 1
    assert run["eligible"] is True

    with (summaries_dir / "per_task_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["classification"] for row in rows} == {
        "infra",
        "agent_exit",
        "verifier_missing_reward",
        "clean_fail",
        "partial",
        "pass",
    }
    pass_row = next(row for row in rows if row["task_name"] == "pass-task")
    assert pass_row["task_path"] == "/benchmark/tasks/pass-task"
    assert pass_row["duration_sec"] == "60.0"
    assert pass_row["agent_execution_sec"] == "40.0"
    assert pass_row["n_input_tokens"] == "100"
    assert pass_row["n_cache_tokens"] == "20"
    assert pass_row["n_output_tokens"] == "30"
    assert pass_row["cost_usd"] == "0.0125"

    with (summaries_dir / "infra_failures.csv").open(newline="") as handle:
        infra_rows = list(csv.DictReader(handle))
    assert [row["task_name"] for row in infra_rows] == ["infra-task"]
    assert "EnvironmentStartTimeoutError happened" in infra_rows[0]["exception_message"]

    assert (summaries_dir / "aggregate_results.csv").is_file()
    assert (summaries_dir / "aggregate_results.json").is_file()
    assert (summaries_dir / "cleaned_leaderboard.md").is_file()


def test_aggregate_uses_task_path_for_canonical_task_name(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    result = _result("display-name", reward=0.0)
    result["task_name"] = "computer-use/display-name"
    result["task_id"] = {"path": "/benchmark/tasks/canonical-task"}
    _write_run(
        jobs_root,
        "model-rep-1",
        expected_task_count=1,
        results=[result],
    )

    aggregate(jobs_root, summaries_dir)

    with (summaries_dir / "per_task_results.csv").open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["task_name"] == "canonical-task"
    assert row["task_path"] == "/benchmark/tasks/canonical-task"


def test_pair_aggregates_exclude_incomplete_and_infra_dominated_runs(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    pair = "openclaw-model"

    _write_run(
        jobs_root,
        "model-rep-1",
        expected_task_count=2,
        results=[
            _result("a", reward=1.0),
            _result("b", reward=0.0, exception_type="AgentTimeoutError"),
        ],
        pair_label=pair,
        repetition=1,
    )
    _write_run(
        jobs_root,
        "model-rep-2",
        expected_task_count=2,
        results=[_result("a", reward=1.0), _result("b", reward=1.0)],
        pair_label=pair,
        repetition=2,
    )
    _write_run(
        jobs_root,
        "model-rep-3",
        expected_task_count=2,
        results=[_result("a", reward=1.0)],
        pair_label=pair,
        repetition=3,
        tasks=["a", "b"],
    )
    _write_run(
        jobs_root,
        "model-rep-4",
        expected_task_count=2,
        results=[
            _result("a", exception_type="VerifierTimeoutError"),
            _result("b", exception_type="EnvironmentStartTimeoutError"),
        ],
        pair_label=pair,
        repetition=4,
    )

    report = aggregate(jobs_root, summaries_dir)

    pair_summary = report["pairs"][0]
    assert pair_summary["pair_label"] == pair
    assert pair_summary["total_repetitions"] == 4
    assert pair_summary["eligible_repetitions"] == 2
    assert pair_summary["excluded_repetitions"] == 2
    assert pair_summary["mean_score"] == pytest.approx(0.75)
    assert pair_summary["score_stdev"] == pytest.approx(0.3535533906)
    assert pair_summary["min_score"] == 0.5
    assert pair_summary["max_score"] == 1.0
    assert pair_summary["mean_exact_passes"] == 1.5
    assert pair_summary["pass_rate"] == 0.75
    assert pair_summary["clean_complete_repetitions"] == 2

    runs = {run["run_label"]: run for run in report["runs"]}
    assert runs["model-rep-3"]["eligible"] is False
    assert runs["model-rep-3"]["exclusion_reason"] == "incomplete"
    assert runs["model-rep-4"]["eligible"] is False
    assert runs["model-rep-4"]["exclusion_reason"] == "infra_dominated"

    leaderboard = (summaries_dir / "cleaned_leaderboard.md").read_text()
    assert "openclaw-model" in leaderboard
    assert "0.7500" in leaderboard


def test_pair_label_uses_manifest_harness_and_model(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    for repetition in (1, 2):
        _write_run(
            jobs_root,
            f"openclaw-gpt55-full-1-r{repetition}-20260727",
            expected_task_count=1,
            results=[_result("a", reward=1.0)],
            pair_label=None,
            repetition=repetition,
            harness="openclaw",
            model_slug="gpt55",
        )

    report = aggregate(jobs_root, summaries_dir)

    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["pair_label"] == "openclaw-gpt55"
    assert report["pairs"][0]["eligible_repetitions"] == 2

    adjacent_summaries = jobs_root / "jobs" / "summaries"
    assert main([str(jobs_root / "jobs"), str(adjacent_summaries)]) == 0
    rerun = aggregate(jobs_root / "jobs", adjacent_summaries)
    assert len(rerun["runs"]) == 2


def test_native_run_requires_parity_validation_for_leaderboard(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "codex-gpt55-calibration",
        expected_task_count=1,
        results=[_native_result("a")],
        harness="codex",
        model_slug="gpt55",
        native=True,
    )

    report = aggregate(jobs_root, summaries_dir)

    assert report["runs"][0]["eligible"] is False
    assert report["runs"][0]["exclusion_reason"] == "parity_not_validated"


def test_native_run_rejects_legacy_unscoped_parity_claim(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "hermes-gpt56-terra",
        expected_task_count=1,
        results=[_native_result("a")],
        harness="hermes",
        model_slug="gpt56-terra",
        native=True,
        parity_validated=True,
        manifest_extra={"parity_validation": None},
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["parity_validated"] is False
    assert run["parity_validation_status"] == "legacy_unscoped"
    assert run["eligible"] is False


def test_native_run_accepts_known_legacy_codex_gpt55_route(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "codex-gpt55-calibration",
        expected_task_count=20,
        results=[_native_result(f"task-{index}") for index in range(20)],
        harness="codex",
        model_slug="gpt55",
        native=True,
        parity_validated=True,
        manifest_extra={"parity_validation": None},
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["parity_validated"] is True
    assert run["parity_validation_status"] == "legacy_route_compatible"
    assert run["eligible"] is True


def test_native_run_rejects_legacy_codex_gpt55_full_suite(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "codex-gpt55-full-116",
        expected_task_count=116,
        results=[_native_result(f"task-{index}") for index in range(116)],
        harness="codex",
        model_slug="gpt55",
        native=True,
        parity_validated=True,
        manifest_extra={"parity_validation": None},
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["parity_validated"] is False
    assert run["parity_validation_status"] == "legacy_scope_mismatch"
    assert run["eligible"] is False
    assert run["exclusion_reason"] == "parity_not_validated"


def test_native_run_rejects_mismatched_parity_scope(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "hermes-gpt55",
        expected_task_count=1,
        results=[_native_result("a")],
        harness="hermes",
        model_slug="gpt55",
        native=True,
        parity_validated=True,
        parity_validation={
            "validated": True,
            "scope": {"harness": "codex", "model_slug": "gpt55"},
        },
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["parity_validated"] is False
    assert run["parity_validation_status"] == "scope_mismatch"
    assert run["eligible"] is False


def test_native_identity_checks_ignore_pre_agent_infra_failures(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "hermes-gpt56-terra",
        expected_task_count=3,
        results=[
            _native_result("a"),
            _native_result("b", reward=0.0),
            _result(
                "environment-timeout",
                source="shellbench-native",
                exception_type="EnvironmentStartTimeoutError",
            ),
        ],
        harness="hermes",
        model_slug="gpt56-terra",
        native=True,
        parity_validated=True,
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["infra"] == 1
    assert run["infra_dominated"] is False
    assert run["canonical_model_identity"] is True
    assert run["trajectory_complete"] is True
    assert run["eligible"] is True


def test_native_identity_checks_ignore_unavailable_agent_exit_trajectory(
    tmp_path: Path,
):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    timed_out = _result(
        "agent-timeout",
        source="shellbench-native",
        exception_type="NonZeroAgentExitCodeError",
        exception_message="Agent timed out after 900.0s",
    )
    timed_out["agent_result"].update(
        {
            "trajectory_status": "unavailable",
            "runtime_model_name": None,
            "canonical_model_identity": False,
        }
    )
    _write_run(
        jobs_root,
        "hermes-gpt56-sol-timeout",
        expected_task_count=2,
        results=[_native_result("a"), timed_out],
        harness="hermes",
        model_slug="gpt56-sol",
        native=True,
        parity_validated=True,
    )
    manifest_path = (
        jobs_root
        / "jobs"
        / "hermes-gpt56-sol-timeout"
        / "run_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["canonical_model_identity"] = False
    manifest_path.write_text(json.dumps(manifest))

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["agent_exits"] == 1
    assert run["canonical_model_identity"] is True
    assert run["trajectory_complete"] is True
    assert run["eligible"] is True


def test_native_identity_checks_real_agent_exit_trajectory(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    wrong_model = _result(
        "agent-exit",
        source="shellbench-native",
        exception_type="NonZeroAgentExitCodeError",
    )
    wrong_model["agent_result"].update(
        {
            "trajectory_status": "real",
            "runtime_model_name": "wrong-model",
            "canonical_model_identity": False,
        }
    )
    _write_run(
        jobs_root,
        "hermes-gpt56-sol-wrong-model",
        expected_task_count=2,
        results=[_native_result("a"), wrong_model],
        harness="hermes",
        model_slug="gpt56-sol",
        native=True,
        parity_validated=True,
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    assert run["canonical_model_identity"] is False
    assert run["eligible"] is False
    assert run["exclusion_reason"] == "canonical_model_identity_not_preserved"


def test_manifest_can_explicitly_exclude_complete_run(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "openclaw-gpt55-rerun1",
        expected_task_count=1,
        results=[_result("a", reward=1.0)],
        leaderboard_eligible=False,
        exclusion_reason="manual_intervention",
    )

    report = aggregate(jobs_root, summaries_dir)

    assert report["runs"][0]["eligible"] is False
    assert report["runs"][0]["exclusion_reason"] == "manual_intervention"


def test_agent_setup_error_is_classified_as_infrastructure(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "openclaw-gpt55-setup-failure",
        expected_task_count=1,
        results=[_result("a", exception_type="AgentSetupError")],
    )

    report = aggregate(jobs_root, summaries_dir)

    assert report["runs"][0]["infra"] == 1
    assert report["runs"][0]["agent_exits"] == 0


def test_unknown_exception_is_agent_exit_and_gateway_signature_is_excluded(
    tmp_path: Path,
):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    results = [
        _result(
            f"gateway-{index}",
            exception_type="NonZeroAgentExitCodeError",
            exception_message="LiteLLM gateway connection refused on port 4000",
        )
        for index in range(3)
    ]
    results.extend(
        _result(f"unknown-{index}", exception_type="UnexpectedAgentError")
        for index in range(5)
    )
    _write_run(
        jobs_root,
        "openclaw-model-full-8-r1-20260727",
        expected_task_count=8,
        results=results,
    )

    report = aggregate(jobs_root, summaries_dir)

    run = report["runs"][0]
    with (summaries_dir / "per_task_results.csv").open(newline="") as handle:
        rows = {row["task_name"]: row for row in csv.DictReader(handle)}
    assert rows["unknown-0"]["classification"] == "agent_exit"
    assert run["harness_wide_failure"] is True
    assert run["eligible"] is False


def test_scorecard_infra_error_is_not_counted_as_clean_failure(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "model-rep-1",
        expected_task_count=1,
        results=[_result("judge-task", reward=0.0)],
    )
    scorecard_path = (
        jobs_root
        / "jobs"
        / "model-rep-1"
        / "judge-task__trial"
        / "verifier"
        / "scorecard.json"
    )
    scorecard_path.parent.mkdir()
    scorecard_path.write_text(
        json.dumps(
            {
                "status": "infra_error",
                "judge_reasons": [
                    "judge HTTP 400: temperature is unsupported",
                ],
            }
        )
    )

    report = aggregate(jobs_root, summaries_dir)

    with (summaries_dir / "per_task_results.csv").open(newline="") as handle:
        task = list(csv.DictReader(handle))[0]
    run = report["runs"][0]
    assert task["classification"] == "infra"
    assert task["exception_type"] == "VerifierJudgeInfraError"
    assert "temperature" in task["exception_message"]
    assert run["infra"] == 1
    assert run["clean_completed"] == 0
    assert run["eligible"] is False
    assert run["exclusion_reason"] == "infra_dominated"


def test_reward_above_one_counts_as_exact_pass(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "openclaw-model-full-1-r1-20260727",
        expected_task_count=1,
        results=[_result("bonus", reward=1.25)],
    )

    report = aggregate(jobs_root, summaries_dir)

    assert report["runs"][0]["exact_passes"] == 1


def test_extra_or_duplicate_task_results_cannot_inflate_score(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "model-rep-1",
        expected_task_count=2,
        tasks=["a", "b"],
        results=[
            _result("a", reward=1.0, trial_name="a__first"),
            _result("a", reward=1.0, trial_name="a__duplicate"),
            _result("b", reward=1.0, trial_name="b__only"),
        ],
    )

    report = aggregate(jobs_root, summaries_dir)
    run = report["runs"][0]

    assert run["score"] == 1.0
    assert run["result_file_count"] == 3
    assert run["infra"] == 1
    assert run["eligible"] is False
    assert run["exclusion_reason"] == "incomplete"
    with (summaries_dir / "per_task_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    duplicate = next(
        row
        for row in rows
        if row["exception_type"] == "DuplicateTaskResultError"
    )
    assert duplicate["classification"] == "infra"


def test_unexpected_task_result_is_excluded_from_reward_sum(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "model-rep-1",
        expected_task_count=2,
        tasks=["a", "b"],
        results=[
            _result("a", reward=1.0),
            _result("b", reward=0.0),
            _result("c", reward=100.0),
        ],
    )

    report = aggregate(jobs_root, summaries_dir)
    run = report["runs"][0]

    assert run["score"] == 0.5
    assert run["eligible"] is False
    assert run["infra"] == 1


def test_repair_overlay_sensitivity_reports_available_score_delta(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    original_label = "codex-gpt55-r1"
    repaired_label = f"{original_label}-repair-abc"
    _write_run(
        jobs_root,
        original_label,
        expected_task_count=2,
        results=[_result("a", reward=0.0), _result("b", reward=0.0)],
    )
    _write_run(
        jobs_root,
        repaired_label,
        expected_task_count=2,
        results=[_result("a", reward=1.0), _result("b", reward=0.0)],
        manifest_extra={
            "rerun_of_canonical_run": original_label,
            "repair_task_names": ["a"],
            "repair_overlay": {
                "rerun_of_canonical_run": original_label,
                "repair_task_names": ["a"],
                "task_lineage": [
                    {"task_name": "a", "source_kind": "repair"},
                    {"task_name": "b", "source_kind": "canonical"},
                ],
            },
        },
    )

    report = aggregate(jobs_root, summaries_dir)

    sensitivity = report["repair_overlay_sensitivity"]
    assert sensitivity["overlay_run_count"] == 1
    row = sensitivity["runs"][0]
    assert row["run_label"] == repaired_label
    assert row["original_task_count"] == 1
    assert row["repaired_task_count"] == 1
    assert row["original_result_count"] == 2
    assert row["repaired_result_count"] == 2
    assert row["original_score"] == 0.0
    assert row["repaired_score"] == 0.5
    assert row["score_delta"] == 0.5
    assert row["delta_status"] == "available"

    written = json.loads(
        (summaries_dir / "repair_overlay_sensitivity.json").read_text()
    )
    assert written == sensitivity
    with (summaries_dir / "repair_overlay_sensitivity.csv").open(
        newline=""
    ) as handle:
        csv_row = next(csv.DictReader(handle))
    assert csv_row["delta_status"] == "available"
    assert csv_row["score_delta"] == "0.5"


def test_repair_overlay_sensitivity_marks_missing_original(tmp_path: Path):
    jobs_root = tmp_path / "native"
    summaries_dir = tmp_path / "summaries"
    _write_run(
        jobs_root,
        "openclaw-gpt55-r1-repair-abc",
        expected_task_count=2,
        results=[_result("a", reward=1.0), _result("b", reward=0.0)],
        manifest_extra={
            "rerun_of_canonical_run": "openclaw-gpt55-r1",
            "repair_task_names": ["a"],
        },
    )

    report = aggregate(jobs_root, summaries_dir)

    row = report["repair_overlay_sensitivity"]["runs"][0]
    assert row["original_task_count"] == 1
    assert row["repaired_task_count"] == 1
    assert row["original_score"] is None
    assert row["score_delta"] is None
    assert row["delta_status"] == "original_unavailable"
