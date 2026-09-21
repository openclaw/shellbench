from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.native_eval.research_evidence import receipt_evidence, trajectory_family


def _node(name: str, *, leaf: str = "leaf", **extra) -> dict:
    return {
        "trajectory_id": f"t-{name}",
        "session_id": f"s-{name}",
        "extra": {"openclaw": {"session_key": f"key-{name}", "leaf_id": leaf}},
        "agent": {"model_name": "provider/model"},
        "steps": [{"source": "agent"}],
        **extra,
    }


def _audit(tmp_path: Path, root: dict, *, modify=None) -> dict:
    path = tmp_path / "trajectory.json"
    raw = (json.dumps(root, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(raw)
    family = trajectory_family(root)
    receipt = {
        "schema": "openclaw-atif-receipt-v1",
        "status": "complete",
        "source": {"stable": True},
        "diagnostics": [],
        "root": {
            "trajectoryId": root["trajectory_id"],
            "sessionId": root["session_id"],
            "sessionKey": root["extra"]["openclaw"]["session_key"],
        },
        "nodes": [
            {
                "sessionKey": node.data["extra"]["openclaw"]["session_key"],
                "sessionId": node.data["session_id"],
                "leafId": node.data["extra"]["openclaw"]["leaf_id"],
            }
            for node in family.nodes
        ],
        "output": {"trajectorySha256": hashlib.sha256(raw).hexdigest()},
    }
    if modify:
        modify(receipt)
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    return receipt_evidence(path, family)


def test_receipt_binds_actual_bytes_not_json_reserialization(tmp_path: Path) -> None:
    root = _node("root", steps=[{"source": "agent", "message": "π"}])
    assert _audit(tmp_path, root)["capture_status"] == "complete"
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(root))
    evidence = receipt_evidence(path, trajectory_family(root))
    assert evidence["receipt_digest_status"] == "mismatch"
    assert evidence["capture_status"] == "invalid"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda receipt: receipt["nodes"].append(dict(receipt["nodes"][0])),
        lambda receipt: receipt["nodes"].append(None),
        lambda receipt: receipt["nodes"].__setitem__(0, {}),
        lambda receipt: receipt["nodes"][0].update(leafId={"invalid": True}),
        lambda receipt: receipt["root"].update(sessionId="wrong"),
        lambda receipt: receipt.update(schema="unknown"),
    ],
)
def test_invalid_receipt_structure_cannot_certify_capture(tmp_path: Path, mutation) -> None:
    assert _audit(tmp_path, _node("root"), modify=mutation)["capture_status"] == "invalid"


def test_receipt_partial_and_opaque_wrapper_remain_partial(tmp_path: Path) -> None:
    root = _node("root")
    assert (
        _audit(tmp_path, root, modify=lambda r: r.update(status="partial"))["capture_status"]
        == "partial"
    )
    assert (
        _audit(tmp_path, root, modify=lambda r: r["source"].update(stable=False))["capture_status"]
        == "partial"
    )
    root["extra"]["openclaw"]["wrapper_only"] = True
    evidence = _audit(tmp_path, root)
    assert evidence["capture_status"] == "partial"
    assert "opaque_wrapper" in json.loads(evidence["capture_reasons_json"])


def test_same_session_and_leaf_not_double_counted_under_conflicting_trajectory_ids() -> None:
    child = _node("child")
    alias = {**child, "trajectory_id": "t-conflicting"}
    family = trajectory_family(_node("root", subagent_trajectories=[child, alias]))
    assert len(family.nodes) == 2
    assert "conflicting_session_identity" in family.issues
    assert not family.identity_observed


def test_distinct_leaves_and_generations_remain_distinct() -> None:
    child = _node("child")
    new_leaf = _node("child", leaf="other", trajectory_id="t-other-leaf")
    new_generation = _node(
        "child", session_id="s-other-generation", trajectory_id="t-new-generation"
    )
    family = trajectory_family(
        _node("root", subagent_trajectories=[child, new_leaf, new_generation])
    )
    assert len(family.nodes) == 4
    assert not family.issues


@pytest.mark.parametrize("steps", [None, {}, [], [None]])
def test_malformed_steps_cannot_claim_identity(steps) -> None:
    family = trajectory_family(_node("root", steps=steps))
    assert "missing_or_malformed_steps" in family.issues
    assert not family.identity_observed


def test_duplicate_dag_node_keeps_both_parent_links_and_only_one_step_set() -> None:
    child = _node("child")
    left = _node("left", subagent_trajectories=[child])
    right = _node("right", subagent_trajectories=[child])
    family = trajectory_family(_node("root", subagent_trajectories=[left, right]))
    assert len(family.nodes) == 4
    node = next(node for node in family.nodes if node.ref == "t-child")
    assert node.parents == {"t-left", "t-right"}
    assert len(family.links) == 4


def test_external_reference_is_retained_but_not_read_or_counted(tmp_path: Path) -> None:
    root = _node(
        "root",
        steps=[
            {
                "step_id": 1,
                "source": "agent",
                "observation": {
                    "results": [
                        {
                            "source_call_id": "spawn",
                            "subagent_trajectory_ref": [
                                {
                                    "trajectory_id": "missing",
                                    "session_id": "deleted",
                                    "trajectory_path": "../../private.json",
                                }
                            ],
                        }
                    ]
                },
            }
        ],
    )
    family = trajectory_family(root)
    assert len(family.nodes) == 1
    assert family.links[0]["reference_status"] == "not_embedded"
    assert family.links[0]["tool_call_id"] == "spawn"
    assert family.links[0]["child_trajectory_path"] == "../../private.json"
    assert _audit(tmp_path, root)["capture_status"] == "partial"


def test_receipt_node_diagnostics_and_unresolved_relationships_remain_visible(
    tmp_path: Path,
) -> None:
    def modify(receipt):
        receipt["nodes"][0]["diagnostics"] = [{"code": "deleted-source"}]
        receipt["relationships"] = [
            {
                "parentKey": "key-root",
                "childKey": "missing",
                "toolCallId": "spawn",
                "referenceStatus": "unresolved",
            }
        ]

    result = _audit(tmp_path, _node("root"), modify=modify)
    assert result["capture_status"] == "partial"
    assert "deleted-source" in result["receipt_node_diagnostics_json"]
    assert "unresolved_relationships" in result["capture_reasons_json"]
    assert "spawn" in result["receipt_relationships_json"]


def test_path_only_reference_does_not_alias_an_embedded_link(tmp_path: Path) -> None:
    child = _node("child")
    root = _node(
        "root",
        subagent_trajectories=[child],
        steps=[
            {
                "source": "agent",
                "observation": {
                    "results": [
                        {
                            "source_call_id": None,
                            "subagent_trajectory_ref": [{"trajectory_path": "unavailable.json"}],
                        }
                    ]
                },
            }
        ],
    )
    family = trajectory_family(root)
    assert any(edge["reference_status"] == "not_embedded" for edge in family.links)
    assert _audit(tmp_path, root)["capture_status"] == "partial"
