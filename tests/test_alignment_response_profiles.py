"""Verify native child thinking choices without relaxing root configuration."""

from copy import deepcopy
import json

import pytest

from clawbench.alignment.audit import response_profile_audit


def fixture():
    child = {
        "session_key": "child",
        "provider_response_ids": ["response-child"],
        "transcript": {"messages": []},
    }
    execution = {
        "session_key": "root",
        "provider_response_ids": ["response-root"],
        "transcript": {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": "sessions_spawn",
                            "input": {"thinking": "medium"},
                            "output": json.dumps(
                                {"status": "accepted", "childSessionKey": "child"}
                            ),
                        }
                    ],
                }
            ],
        },
        "delegated_sessions": [child],
    }
    responses = [
        {"id": "response-root", "model": "test", "reasoning": {"effort": "high"}},
        {"id": "response-child", "model": "test", "reasoning": {"effort": "medium"}},
    ]
    return execution, responses


def test_explicit_child_effort_is_bound_to_native_response_identity():
    execution, responses = fixture()
    result = response_profile_audit(execution, responses, "test")
    assert result["match"] is True
    assert result["session_profiles"]["child"] == {"reasoning": "medium", "responses": 1}
    # Equal global effort counts do not excuse a changed root configuration.
    responses[0]["id"], responses[1]["id"] = responses[1]["id"], responses[0]["id"]
    assert response_profile_audit(execution, responses, "test")["match"] is False


def test_absent_override_inherits_parent_effort():
    execution, responses = fixture()
    execution["transcript"]["messages"][0]["tool_calls"][0]["input"] = {}
    responses[1]["reasoning"]["effort"] = "high"
    assert response_profile_audit(execution, responses, "test")["match"] is True


def test_nested_child_inherits_explicit_parent_choice():
    execution, responses = fixture()
    child = execution["delegated_sessions"][0]
    child["transcript"]["messages"] = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "name": "sessions_spawn",
                    "input": {},
                    "output": json.dumps({"status": "accepted", "childSessionKey": "grandchild"}),
                }
            ],
        }
    ]
    grandchild = deepcopy(child)
    grandchild.update(
        session_key="grandchild",
        provider_response_ids=["response-grandchild"],
        transcript={"messages": []},
    )
    execution["delegated_sessions"].append(grandchild)
    responses.append(
        {"id": "response-grandchild", "model": "test", "reasoning": {"effort": "medium"}}
    )
    assert response_profile_audit(execution, responses, "test")["match"] is True


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "unknown", "wrong-model", "unaccepted", "untrusted-message"]
)
def test_unexplained_provider_identity_or_configuration_fails_closed(fault):
    execution, responses = fixture()
    if fault == "missing":
        execution["delegated_sessions"][0].pop("provider_response_ids")
    elif fault == "duplicate":
        responses.append(deepcopy(responses[-1]))
    elif fault == "unknown":
        responses[-1]["id"] = "unrelated-response"
    elif fault == "wrong-model":
        responses[-1]["model"] = "other"
    elif fault == "untrusted-message":
        execution["transcript"]["messages"][0]["role"] = "user"
    else:
        execution["transcript"]["messages"] = []
    assert response_profile_audit(execution, responses, "test")["match"] is False


def test_legacy_evidence_does_not_gain_an_unproven_effort_exception():
    execution, responses = fixture()
    execution.pop("provider_response_ids")
    assert response_profile_audit(execution, responses, "test")["match"] is False
    responses[1]["reasoning"]["effort"] = "high"
    assert response_profile_audit(execution, responses, "test")["match"] is True
