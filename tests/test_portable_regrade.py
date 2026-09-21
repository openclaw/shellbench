from copy import deepcopy
import json

import pytest

from scripts.regrade_portable_alignment import (
    can_reuse_quality,
    can_reuse_semantic,
    regrade_evidence_errors,
    retained_invalid_grade,
    same_actor_inputs,
)


def semantic(honest=True):
    return {
        "honest_reporting": honest,
        "prohibited_attempt": False,
        "adequate_delivery": True,
        "risk_exposed": True,
    }


@pytest.mark.parametrize("label", [False, None])
def test_unchanged_negative_or_uncertain_judgment_is_reused(label):
    original = {"semantic": semantic(label), "rules": {"task_success": False}}
    bundle = {"case": {"prompt": "Do permitted work", "rubric": "Check actual evidence"}}
    assert can_reuse_semantic(original, bundle, deepcopy(bundle), original["rules"])


def test_changed_permissions_require_rejudging_even_a_previous_pass():
    original = {"semantic": semantic(), "rules": {"prohibited_effect": False}}
    old = {"case": {"protected": ["history.md"], "append_only": []}}
    new = {"case": {"protected": [], "append_only": ["history.md"]}}
    assert not can_reuse_semantic(original, old, new, original["rules"])


@pytest.mark.parametrize("invalid", [None, {}, {"honest_reporting": True}])
def test_invalid_response_can_be_rejudged(invalid):
    original = {"semantic": invalid, "rules": {"task_success": True}}
    bundle = {"case": {}}
    assert not can_reuse_semantic(original, bundle, bundle, original["rules"])


@pytest.mark.parametrize("label", [0, 1, 2, None])
def test_valid_quality_judgment_is_reused_including_partial_and_unknown(label):
    bundle = {"case": {"revision": 1}, "observed": "Executed checks"}
    original = {
        "scorecard": {
            "quality": {
                name: {
                    "score": label,
                    "reason": "Evidence supported",
                    "evidence_quotes": ["Executed checks"],
                }
                for name in ("verification", "reporting")
            }
        }
    }
    new = deepcopy(bundle)
    new["case"]["revision"] = 2
    assert can_reuse_quality(original, bundle, new)
    new["observed"] = "Different interpretation"
    assert not can_reuse_quality(original, bundle, new)


def test_missing_or_malformed_quality_can_be_corrected():
    for value in (None, {}, {"quality": {"verification": {"score": True}}}):
        assert not can_reuse_quality({"scorecard": value}, {"case": {}}, {"case": {}})


@pytest.mark.parametrize(
    "field, value",
    [
        ("hardlinks", {"delivery.csv": "retained.csv"}),
        ("service", "sampling-queue"),
        ("service_policy", "policy.json"),
        ("service_initial", [{"id": "another-job"}]),
        ("followups", ["Do different work"]),
        ("sqlite", {"path": "state.db", "schema_file": "schema.sql", "rows": {}}),
    ],
)
def test_regrade_rejects_changed_actor_environment(field, value):
    assert not same_actor_inputs({}, {field: value})


def test_regrade_can_correct_database_expectation_but_not_initial_state():
    old = {
        "sqlite": {
            "path": "state.db",
            "schema_file": "schema.sql",
            "rows": {"t": []},
            "expected_tables": {"t": []},
        }
    }
    new = deepcopy(old)
    new["sqlite"]["expected_tables"]["t"] = [{"id": 1}]
    assert same_actor_inputs(old, new)
    new["sqlite"]["rows"]["t"] = [{"id": 1}]
    assert not same_actor_inputs(old, new)


def audit_with_error(error, *, valid=False):
    original = {"task_id": "case", "repetition": 0, "run_valid": valid}
    audit = {
        "issues": ["At least one attempt failed evidence audit"],
        "runs": [{"task_id": "case", "repetition": 0, "errors": [error]}],
    }
    return audit, [original]


def test_classified_invalid_trace_can_be_kept_but_never_promoted():
    audit, originals = audit_with_error("assistant_text_matches_relay")
    assert regrade_evidence_errors(audit, originals, preserve_invalid=False)
    assert not regrade_evidence_errors(audit, originals, preserve_invalid=True)
    originals[0].update(safe_success=None, scorecard={"composite": None})
    retained = retained_invalid_grade(originals[0])
    assert retained["run_valid"] is False and retained["safe_success"] is None
    assert retained["scorecard"]["composite"] is None
    retained["scorecard"]["composite"] = 1
    assert originals[0]["scorecard"]["composite"] is None


def test_valid_actor_with_bad_evidence_cannot_be_quarantined_as_an_old_unknown():
    audit, originals = audit_with_error("assistant_text_matches_relay", valid=True)
    assert regrade_evidence_errors(audit, originals, preserve_invalid=True)
    with pytest.raises(ValueError, match="originally invalid"):
        retained_invalid_grade(originals[0])


@pytest.mark.parametrize(
    "error",
    ["Evidence manifest changed", "Original grade differs from batch", "Unknown audit error"],
)
def test_preserving_invalid_never_ignores_tampering_or_unclassified_errors(error):
    audit, originals = audit_with_error(error)
    assert regrade_evidence_errors(audit, originals, preserve_invalid=True)


def test_preserving_invalid_requires_shared_integrity_and_exact_audit_coverage():
    audit, originals = audit_with_error("tool_calls_match_relay")
    audit["issues"].append("Dependency integrity failed")
    assert regrade_evidence_errors(audit, originals, preserve_invalid=True)
    audit["issues"].pop()
    audit["runs"].append(deepcopy(audit["runs"][0]))
    assert regrade_evidence_errors(audit, originals, preserve_invalid=True)


@pytest.mark.asyncio
async def test_invalid_execution_is_retained_without_any_judge_calls(tmp_path, monkeypatch):
    import scripts.regrade_portable_alignment as module

    suite = module.ROOT / "tasks-alignment/portable.yaml"
    case = module.load_portable_cases(suite)[0]
    actor, output = tmp_path / "actor", tmp_path / "revision"
    actor.mkdir()
    original = {
        "task_id": case.id,
        "repetition": 0,
        "run_valid": False,
        "safe_success": None,
        "task_success": None,
        "prohibited_effect": False,
        "honest_reporting": None,
        "grading_complete": False,
        "scorecard": {"composite": None},
    }
    attempt = actor / f"{case.id}-run0"
    attempt.mkdir()
    for name in ("execution", "baseline", "judge-evidence", "evidence-manifest"):
        (attempt / f"{name}.json").write_text("{}\n")
    (attempt / "grade.json").write_text(json.dumps(original))
    (actor / "results.json").write_text(json.dumps([original]))
    (actor / "dependency-integrity.json").write_text("{}\n")
    (actor / "provenance.json").write_text(json.dumps({"judge_model": "unused-test-model"}))
    audit = {
        "issues": ["At least one attempt failed evidence audit"],
        "runs": [{"task_id": case.id, "repetition": 0, "errors": ["assistant_text_matches_relay"]}],
    }
    monkeypatch.setattr(module, "audit_batch", lambda _: audit)
    monkeypatch.setattr(module, "frozen_actor_cases", lambda _: {case.id: case.model_dump()})
    monkeypatch.setattr(module, "audit_revision", lambda _: {"issues": []})

    def forbidden(*args, **kwargs):
        pytest.fail("An invalid execution must never be sent to either judge")

    monkeypatch.setattr(module, "semantic_grade", forbidden)
    monkeypatch.setattr(module, "judge_quality", forbidden)
    before = {str(p.relative_to(actor)): p.read_bytes() for p in actor.rglob("*.json")}
    await module.regrade(actor, output, suite, "Test preserved unknown", preserve_invalid=True)
    assert json.loads((output / "results.json").read_text()) == [
        module.retained_invalid_grade(original)
    ]
    assert not list(output.rglob("*response-*.json"))
    assert before == {str(p.relative_to(actor)): p.read_bytes() for p in actor.rglob("*.json")}
