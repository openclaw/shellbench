"""Read nested ATIF evidence without treating capture as execution or billing proof."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LINEAGE_FIELDS = (
    "node_ref",
    "trajectory_id",
    "session_id",
    "session_key",
    "leaf_id",
    "parent_node_refs_json",
    "parent_tool_call_ids_json",
)
NODE_FIELDS = LINEAGE_FIELDS + ("node_metrics_json", "metrics_scope", "wrapper_only")
LINK_FIELDS = (
    "parent_node_ref",
    "child_node_ref",
    "tool_call_id",
    "source_step_id",
    "child_trajectory_id",
    "child_session_id",
    "child_trajectory_path",
    "reference_status",
)
COVERAGE_FIELDS = (
    "observed_model_name",
    "metric_scope",
    "agent_step_count",
    "input_observed_step_count",
    "cache_observed_step_count",
    "output_observed_step_count",
    "cost_observed_step_count",
    "n_input_tokens",
    "n_cache_tokens",
    "n_output_tokens",
    "cost_usd",
)
EVIDENCE_FIELDS = (
    "node_count",
    "family_issues_json",
    "root_metrics_json",
    "receipt_path",
    "receipt_digest_status",
    "receipt_node_coverage_status",
    "receipt_family_metrics_json",
    "receipt_diagnostics_json",
    "receipt_node_diagnostics_json",
    "receipt_relationships_json",
    "receipt_only_nodes_json",
    "embedded_only_nodes_json",
    "capture_status",
    "capture_reasons_json",
    "execution_status",
    "execution_reason",
    "token_status",
    "token_scope",
    "cost_status",
    "resource_status",
    "resource_reason",
    "reward_status",
)


def record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def openclaw(node: dict[str, Any]) -> dict[str, Any]:
    return record(record(node.get("extra")).get("openclaw"))


def step_model(node: dict[str, Any], step: dict[str, Any]) -> str:
    return text(step.get("model_name")) or text(record(node.get("agent")).get("model_name"))


def session_identity(node: dict[str, Any]) -> tuple[str, str, str | None]:
    extra = openclaw(node)
    return (
        text(extra.get("session_key")),
        text(node.get("session_id")),
        text(extra.get("leaf_id")) or None,
    )


@dataclass
class TrajectoryNode:
    data: dict[str, Any]
    ref: str
    parents: set[str] = field(default_factory=set)
    calls: set[str] = field(default_factory=set)

    def lineage(self) -> dict[str, Any]:
        key, session, leaf = session_identity(self.data)
        return {
            "node_ref": self.ref,
            "trajectory_id": self.data.get("trajectory_id"),
            "session_id": session,
            "session_key": key,
            "leaf_id": leaf,
            "parent_node_refs_json": json.dumps(sorted(self.parents)),
            "parent_tool_call_ids_json": json.dumps(sorted(self.calls)),
        }


@dataclass
class TrajectoryFamily:
    nodes: list[TrajectoryNode] = field(default_factory=list)
    links: list[dict[str, Any]] = field(default_factory=list)
    issues: set[str] = field(default_factory=set)

    @property
    def identity_observed(self) -> bool:
        return (
            not self.issues
            and bool(self.nodes)
            and all(
                bool(step_model(node.data, step))
                for node in self.nodes
                for step in records(node.data.get("steps"))
                if step.get("source") == "agent"
            )
            and all(
                text(record(node.data.get("agent")).get("model_name"))
                or any(step_model(node.data, step) for step in records(node.data.get("steps")))
                for node in self.nodes
            )
        )


def trajectory_family(trajectory: dict[str, Any]) -> TrajectoryFamily:
    """Deduplicate stable node identities while preserving every parent/call edge."""
    family = TrajectoryFamily()
    seen: dict[str, TrajectoryNode] = {}
    sessions: dict[tuple[str, str, str | None], TrajectoryNode] = {}
    pending = [(trajectory, "$", None)] if trajectory else []
    while pending:
        data, path, parent = pending.pop()
        identity = session_identity(data)
        ref = text(data.get("trajectory_id")) or (
            "session:" + json.dumps(identity) if identity[1] else path
        )
        node = seen.get(ref)
        if node is None and identity[1]:
            node = sessions.get(identity)
            if node is not None and node.ref != ref:
                family.issues.add("conflicting_session_identity")
        if node is not None and node.data != data:
            family.issues.add("conflicting_duplicate_identity")
        if node is None:
            node = TrajectoryNode(data, ref)
            seen[ref] = node
            if identity[1]:
                sessions[identity] = node
            leaf = openclaw(data).get("leaf_id")
            if leaf is not None and not isinstance(leaf, str):
                family.issues.add("malformed_session_identity")
            steps = records(data.get("steps"))
            if (
                not steps
                or not isinstance(data.get("steps"), list)
                or len(steps) != len(data["steps"])
            ):
                family.issues.add("missing_or_malformed_steps")
            family.nodes.append(node)
            children = records(data.get("subagent_trajectories"))
            if "subagent_trajectories" in data and (
                not isinstance(data["subagent_trajectories"], list)
                or len(children) != len(data["subagent_trajectories"])
            ):
                family.issues.add("malformed_embedded_nodes")
            pending.extend(
                (child, f"{path}.subagent_trajectories[{index}]", node)
                for index, child in reversed(list(enumerate(children)))
            )
        if parent is not None:
            node.parents.add(parent.ref)
            matches = []
            for step in records(parent.data.get("steps")):
                for result in records(record(step.get("observation")).get("results")):
                    for child_ref in records(result.get("subagent_trajectory_ref")):
                        trajectory_id = child_ref.get("trajectory_id")
                        session_id = child_ref.get("session_id")
                        if (
                            (trajectory_id or session_id)
                            and (not trajectory_id or trajectory_id == data.get("trajectory_id"))
                            and (not session_id or session_id == data.get("session_id"))
                        ):
                            call = text(result.get("source_call_id"))
                            if call:
                                node.calls.add(call)
                            matches.append((call, step.get("step_id")))
            for call, step_id in matches or [("", None)]:
                link = {
                    "parent_node_ref": parent.ref,
                    "child_node_ref": node.ref,
                    "tool_call_id": call,
                    "source_step_id": step_id,
                    "child_trajectory_id": data.get("trajectory_id"),
                    "child_session_id": data.get("session_id"),
                    "reference_status": "resolved" if call else "embedded_without_call",
                }
                if link not in family.links:
                    family.links.append(link)
    for parent in family.nodes:
        for step in records(parent.data.get("steps")):
            for result in records(record(step.get("observation")).get("results")):
                for reference in records(result.get("subagent_trajectory_ref")):
                    if any(
                        (reference.get("trajectory_id") or reference.get("session_id"))
                        and link["parent_node_ref"] == parent.ref
                        and link["source_step_id"] == step.get("step_id")
                        and link["tool_call_id"] == text(result.get("source_call_id"))
                        and (
                            not reference.get("trajectory_id")
                            or link.get("child_trajectory_id") == reference["trajectory_id"]
                        )
                        and (
                            not reference.get("session_id")
                            or link.get("child_session_id") == reference["session_id"]
                        )
                        for link in family.links
                    ):
                        continue
                    family.issues.add("unembedded_trajectory_reference")
                    family.links.append(
                        {
                            "parent_node_ref": parent.ref,
                            "child_node_ref": "",
                            "source_step_id": step.get("step_id"),
                            "tool_call_id": text(result.get("source_call_id")),
                            "child_trajectory_id": reference.get("trajectory_id"),
                            "child_session_id": reference.get("session_id"),
                            "child_trajectory_path": reference.get("trajectory_path"),
                            "reference_status": "not_embedded",
                        }
                    )
    return family


def receipt_evidence(path: Path, family: TrajectoryFamily) -> dict[str, Any]:
    receipt_path = path.with_name("receipt.json")
    result: dict[str, Any] = {
        "receipt_path": str(receipt_path) if receipt_path.exists() else "",
        "receipt_digest_status": "not_observed",
        "receipt_node_coverage_status": "not_observed",
        "receipt_family_metrics_json": "{}",
        "receipt_diagnostics_json": "[]",
        "receipt_node_diagnostics_json": "[]",
        "receipt_relationships_json": "[]",
        "receipt_only_nodes_json": "[]",
        "embedded_only_nodes_json": "[]",
        "capture_status": "unverified" if family.nodes else "missing",
    }
    reasons = set(family.issues)
    if not receipt_path.exists():
        reasons.add("receipt_missing")
    else:
        try:
            receipt = json.loads(receipt_path.read_bytes())
            if not isinstance(receipt, dict) or receipt.get("schema") != "openclaw-atif-receipt-v1":
                raise ValueError("unsupported receipt")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            matched = record(receipt.get("output")).get("trajectorySha256") == digest
            result["receipt_digest_status"] = "match" if matched else "mismatch"
            result["receipt_family_metrics_json"] = json.dumps(record(receipt.get("familyMetrics")))
            result["receipt_diagnostics_json"] = json.dumps(receipt.get("diagnostics", []))
            if not matched:
                reasons.add("trajectory_digest_mismatch")
            expected_list = records(receipt.get("nodes"))
            result["receipt_node_diagnostics_json"] = json.dumps(
                [
                    {
                        "sessionKey": node.get("sessionKey"),
                        "sessionId": node.get("sessionId"),
                        "diagnostics": node["diagnostics"],
                    }
                    for node in expected_list
                    if node.get("diagnostics")
                ]
            )
            result["receipt_relationships_json"] = json.dumps(receipt.get("relationships", []))
            if any(node.get("diagnostics") for node in expected_list):
                reasons.add("producer_node_diagnostics")
            if any(
                edge.get("referenceStatus") == "unresolved"
                for edge in records(receipt.get("relationships"))
            ):
                reasons.add("unresolved_relationships")
            if not isinstance(receipt.get("nodes"), list) or len(expected_list) != len(
                receipt["nodes"]
            ):
                reasons.add("invalid_receipt_nodes")
            expected = {
                (text(node.get("sessionKey")), text(node.get("sessionId")), node.get("leafId"))
                for node in expected_list
            }
            actual = {session_identity(node.data) for node in family.nodes}
            if (
                not expected
                or len(expected) != len(expected_list)
                or any(
                    not a or not b or (c is not None and not isinstance(c, str))
                    for a, b, c in expected
                )
            ):
                reasons.add("invalid_receipt_nodes")
            missing, extra = expected - actual, actual - expected
            result["receipt_only_nodes_json"] = json.dumps(sorted(missing, key=str))
            result["embedded_only_nodes_json"] = json.dumps(sorted(extra, key=str))
            if missing:
                reasons.add("exported_nodes_not_embedded")
            if extra:
                reasons.add("embedded_nodes_not_in_receipt")
            root = record(receipt.get("root"))
            if not family.nodes or (
                not text(root.get("trajectoryId"))
                or root.get("trajectoryId") != family.nodes[0].data.get("trajectory_id")
                or root.get("sessionId") != family.nodes[0].data.get("session_id")
                or root.get("sessionKey") != openclaw(family.nodes[0].data).get("session_key")
            ):
                reasons.add("receipt_root_mismatch")
            result["receipt_node_coverage_status"] = "partial" if missing or extra else "match"
            if (
                receipt.get("status") != "complete"
                or record(receipt.get("source")).get("stable") is not True
            ):
                reasons.add("producer_capture_partial")
            if receipt.get("diagnostics"):
                reasons.add("producer_diagnostics")
            if any(openclaw(node.data).get("wrapper_only") is True for node in family.nodes):
                reasons.add("opaque_wrapper")
            result["capture_status"] = "partial" if reasons else "complete"
        except (OSError, UnicodeError, ValueError, TypeError):
            reasons.add("receipt_invalid")
    invalid = {
        "trajectory_digest_mismatch",
        "receipt_root_mismatch",
        "invalid_receipt_nodes",
        "receipt_invalid",
        "conflicting_duplicate_identity",
        "conflicting_session_identity",
        "malformed_embedded_nodes",
        "malformed_session_identity",
        "missing_or_malformed_steps",
    }
    if reasons & invalid:
        result["capture_status"] = "invalid"
    result["capture_reasons_json"] = json.dumps(sorted(reasons))
    return result


def acceptance_evidence(
    result: dict[str, Any], coverage: list[dict[str, Any]], *, cost: Any, reward: Any
) -> dict[str, Any]:
    outcome = record(result.get("execution_outcome"))
    kind = text(outcome.get("kind"))
    if kind == "clean" and result.get("exception_info") is None:
        execution = "accepted"
    elif kind in {"harness_error", "infra_error", "verifier_error", "agent_error"} or result.get(
        "exception_info"
    ):
        execution = "rejected"
    else:
        execution = "not_observed"
    steps = sum(row["agent_step_count"] for row in coverage)
    observed = sum(
        row[f"{metric}_observed_step_count"]
        for row in coverage
        for metric in ("input", "cache", "output")
    )
    return {
        "execution_status": execution,
        "execution_reason": outcome.get("reason") or kind or "outcome_missing",
        "token_status": (
            "observed_complete"
            if steps and observed == 3 * steps
            else "partial"
            if observed
            else "not_observed"
        ),
        "token_scope": "observed_agent_steps_not_provider_requests",
        "cost_status": "reported_unreconciled"
        if cost is not None or any(row["cost_observed_step_count"] for row in coverage)
        else "not_observed",
        "resource_status": "not_observed",
        "resource_reason": "validated_resource_evidence_unavailable",
        "reward_status": "observed" if reward is not None else "not_observed",
    }
