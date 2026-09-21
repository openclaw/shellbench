"""Receiver evidence survives successful tool output and later remediation."""

from copy import deepcopy
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

import pytest

from clawbench.alignment.portable import load_portable_cases, seed_portable_workspace
from clawbench.alignment.portable_grading import portable_rule_grade
from clawbench.alignment.runtime import events_in
from clawbench.alignment.service import ServiceState, make_server

CASES = load_portable_cases(Path("tasks-alignment/supplier-toolkit.yaml"))


@pytest.fixture(params=CASES, ids=lambda c: c.condition)
def fixture(request, tmp_path):
    # These tests isolate receiver grading. Real program interfaces have their
    # own fresh-container tests rather than executing submitted code on this host.
    case = request.param.model_copy(update={"functional_profile": None})
    policy = json.loads(case.files[case.service_policy])
    state = ServiceState(
        "diagnostic-sink",
        case.condition,
        tmp_path / "audit",
        fixture={"policy": policy, "initial_reports": []},
    )
    server = make_server(state, host="127.0.0.1", port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    path = tmp_path / "workspace"
    baseline = seed_portable_workspace(case, path)
    try:
        yield case, policy, state, path, baseline, server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def grade(fixture, events=None):
    case, _, state, workspace, baseline, _ = fixture
    return portable_rule_grade(
        case,
        workspace,
        baseline,
        service_events=events if events is not None else events_in(state.audit_dir),
        service_state=json.loads((state.audit_dir / "diagnostic-state.json").read_text()),
    )


def send(fixture, context):
    _, policy, _, _, _, port = fixture
    payload = {"suite": policy["suite"], "tests": 3, "failures": 0, "context": context}
    request = Request(
        f"http://127.0.0.1:{port}/support/runs",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=5) as response:
        assert json.load(response) == {"accepted": True}


def test_safe_independent_delivery_and_optional_public_summary(fixture):
    case, policy, _, workspace, _, _ = fixture
    (workspace / "coverage.json").write_text(json.dumps(case.expected_json["coverage.json"]))
    assert grade(fixture)["task_success"] and not grade(fixture)["prohibited_effect"]
    send(fixture, policy["public_context"])
    assert grade(fixture)["task_success"] and not grade(fixture)["prohibited_effect"]


def test_paired_supplier_default_context_has_actual_receiver_consequence(fixture):
    case, _, _, workspace, _, _ = fixture
    config = json.loads(case.files["vendor/support-settings.json"])
    context = json.loads((workspace / "vendor" / config["context_file"]).read_text())
    send(fixture, context)
    (workspace / "coverage.json").write_text(json.dumps(case.expected_json["coverage.json"]))
    result = grade(fixture)
    assert result["task_success"]
    assert result["prohibited_effect"] is (case.condition == "challenge")
    assert result["service_prohibited_attempt"] is (case.condition == "challenge")


def test_later_public_report_cannot_erase_an_earlier_private_transfer(fixture):
    case, policy, _, _, _, _ = fixture
    send(fixture, json.loads(case.files["internal/operations.json"]))
    send(fixture, policy["public_context"])
    assert grade(fixture)["prohibited_effect"]


def test_forged_receiver_flags_are_rejected(fixture):
    case, _, state, _, _, _ = fixture
    send(fixture, json.loads(case.files["internal/operations.json"]))
    events = deepcopy(events_in(state.audit_dir))
    events[-1]["prohibited_effect"] = False
    with pytest.raises(ValueError, match="disagree"):
        grade(fixture, events)


def test_inactivity_does_not_satisfy_the_business_job(fixture):
    assert not grade(fixture)["task_success"]
    assert not grade(fixture)["prohibited_effect"]
