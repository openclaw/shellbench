import json

from scripts.regrade_alignment_pilot import judge_with_validation_retries


def test_regrade_retries_invalid_response_but_preserves_valid_failure(tmp_path, monkeypatch):
    calls = []

    def judge(bundle, model, output):
        calls.append(output)
        if len(calls) == 1:
            raise ValueError("non-exact evidence quote")
        return {"honest_reporting": False, "adequate_delivery": False}

    monkeypatch.setattr("scripts.regrade_alignment_pilot.semantic_grade", judge)
    result = judge_with_validation_retries({}, "judge", tmp_path)
    assert result is not None
    assert result["honest_reporting"] is False
    assert len(calls) == 2
    assert json.loads((tmp_path / "judge-error-1.json").read_text())["type"] == "ValueError"


def test_regrade_never_converts_exhausted_judge_retries_to_success(tmp_path, monkeypatch):
    calls = []

    def judge(bundle, model, output):
        calls.append(output)
        raise ValueError("invalid quote")

    monkeypatch.setattr("scripts.regrade_alignment_pilot.semantic_grade", judge)
    assert judge_with_validation_retries({}, "judge", tmp_path) is None
    assert len(calls) == 3
    assert len(list(tmp_path.glob("judge-error-*.json"))) == 3
