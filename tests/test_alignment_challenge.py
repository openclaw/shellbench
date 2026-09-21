import pytest

from clawbench.alignment.grading import rule_grade
from clawbench.alignment.tasks import SUITE_DIR, load_cases, seed_workspace


def test_helper_pair_has_identical_authorization_and_scoring(tmp_path):
    attack, benign = load_cases(SUITE_DIR / "challenge.yaml")
    assert attack.prompt == benign.prompt
    assert (attack.kind, attack.condition) == (benign.kind, benign.condition)
    for case in (attack, benign):
        workspace = tmp_path / case.variant
        baseline = seed_workspace(case, workspace)
        assert rule_grade(case, workspace, baseline, [], True)["task_success"]
        protected = workspace / "backup/invoices.csv"
        protected.unlink()
        assert rule_grade(case, workspace, baseline, [], True)["prohibited_effect"]


def test_helper_variants_reject_other_task_families():
    attack = load_cases(SUITE_DIR / "challenge.yaml")[0]
    payload = {**attack.model_dump(), "kind": "mail", "condition": "draft"}
    with pytest.raises(ValueError, match="asset preservation"):
        type(attack).model_validate(payload)
