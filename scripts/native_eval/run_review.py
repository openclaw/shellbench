"""Freeze and review native trial evidence without changing task rewards.

The files in agent/, artifacts/, and verifier/ are actor-controlled observations.
In particular, a verifier sharing the actor's container is not an independent
effects monitor. Only read them after the environment has stopped, through the
bounded reader that rejects symlinks and special files.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clawbench.review_report import persist_run_review
from clawbench.run_review import (
    build_review_evidence,
    evidence_sha256,
    read_evidence_text,
    review_with_http,
    unreviewed_result,
)
from scripts.native_eval.models import RunSpec
from scripts.native_eval.proxy import JUDGE_PROXY_MODEL_NAME
from scripts.native_eval.tasks import TaskSpec


MAX_TRAJECTORY_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_FILES = 64
MAX_ARTIFACT_CHARS = 100_000
RAW_TRACE_FILES = (
    "codex.txt",
    "openclaw.session.jsonl",
    "openclaw.txt",
    "hermes-session.jsonl",
    "claude-code.txt",
    "stdout.txt",
    "stderr.txt",
    "setup-stdout.txt",
    "setup-stderr.txt",
)


@dataclass
class _Capture:
    notes: list[str] = field(default_factory=list)
    incomplete: bool = False

    def gap(self, reason: str) -> None:
        self.incomplete = True
        self.notes.append(reason)


def _read(
    trial_dir: Path,
    relative: str,
    capture: _Capture,
    *,
    max_bytes: int = 100_000,
    required: bool = False,
) -> str | None:
    # lexists notices a dangling symlink too; the safe reader rejects it.
    if not os.path.lexists(trial_dir / relative):
        if required:
            capture.gap(f"Required evidence is missing at {relative}.")
        return None
    try:
        value = read_evidence_text(trial_dir, relative, max_bytes=max_bytes)
    except (OSError, ValueError) as exc:
        capture.gap(f"Evidence unavailable at {relative}: {exc}")
        return None
    if getattr(value, "truncated", False):
        capture.gap(f"Evidence truncated at {relative} (limit {max_bytes} bytes).")
    return value


def _trace(
    trial_dir: Path,
    agent_result: dict[str, Any],
    capture: _Capture,
) -> tuple[list[dict[str, Any]], bool]:
    text = _read(
        trial_dir,
        "agent/trajectory.json",
        capture,
        max_bytes=MAX_TRAJECTORY_BYTES,
        required=agent_result.get("trajectory_status") == "real",
    )
    steps: list[dict[str, Any]] = []
    if text is not None and not getattr(text, "truncated", False):
        try:
            trajectory = json.loads(text)
            raw_steps = trajectory.get("steps") if isinstance(trajectory, dict) else None
            if not isinstance(raw_steps, list) or not all(isinstance(s, dict) for s in raw_steps):
                raise ValueError("trajectory steps must be a list of objects")
            for index, step in enumerate(raw_steps, start=1):
                step_id = step.get("step_id", index)
                steps.append(
                    {
                        **step,
                        "source_id": f"step:{step_id}",
                        "source_ref": f"agent/trajectory.json#step_id={step_id}",
                    }
                )
        except (ValueError, TypeError) as exc:
            capture.gap(f"Normalized trajectory unavailable: {exc}")

    validation = agent_result.get("trajectory_validation") or {}
    complete = bool(
        steps
        and agent_result.get("trajectory_status") == "real"
        and validation.get("terminal_event_seen") is True
        and not validation.get("malformed_event_lines")
        and validation.get("trace_fidelity") != "envelope"
        and validation.get("session_tree_complete") is not False
        and not validation.get("session_tree_missing_transcript_count")
    )
    if not complete:
        capture.gap(
            "Trajectory capture is partial or lacks a validated terminal event; "
            "retained raw logs are additional observations, not proof of full coverage."
        )
        for name in RAW_TRACE_FILES:
            relative = f"agent/{name}"
            raw = _read(trial_dir, relative, capture)
            if raw:
                steps.append(
                    {
                        "source_id": f"raw:{name}",
                        "source_ref": relative,
                        "source": "raw_trace",
                        "content": raw,
                        "truncated": bool(getattr(raw, "truncated", False)),
                    }
                )
    return steps, complete


def _artifacts(trial_dir: Path, capture: _Capture) -> dict[str, str]:
    collected: dict[str, str] = {}
    manifest = _read(trial_dir, "artifacts/manifest.json", capture, required=True)
    if manifest is not None:
        collected["artifacts/manifest.json"] = manifest
        if not getattr(manifest, "truncated", False):
            try:
                entries = json.loads(manifest)
                if not isinstance(entries, list) or not all(
                    isinstance(entry, dict) for entry in entries
                ):
                    raise ValueError("manifest must be a list of artifact entries")
                for entry in entries:
                    destination = entry.get("destination")
                    if (
                        not isinstance(destination, str)
                        or not destination.startswith("artifacts/")
                        or ".." in Path(destination).parts
                        or "\x00" in destination
                    ):
                        raise ValueError("manifest contains an invalid artifact destination")
                    if entry.get("status") != "collected" or not os.path.lexists(
                        trial_dir / destination
                    ):
                        capture.gap(f"Manifest artifact is not available at {destination}.")
            except (ValueError, TypeError) as exc:
                capture.gap(f"Artifact manifest is malformed: {exc}")
    artifact_root = trial_dir / "artifacts"
    if artifact_root.is_symlink():
        capture.gap("Artifact directory is a symlink; its contents were not read.")
        return collected
    remaining = MAX_ARTIFACT_CHARS
    count = 0
    directories = 0

    # os.walk does not follow symlink directories. Every file is still opened
    # using directory descriptors and O_NOFOLLOW by read_evidence_text.
    def scan_error(error: OSError) -> None:
        capture.gap(f"Artifact directory could not be scanned ({type(error).__name__}).")

    for directory, dirs, names in os.walk(artifact_root, followlinks=False, onerror=scan_error):
        directories += 1
        if directories > MAX_ARTIFACT_FILES:
            capture.gap("Artifact directory scan reached its bounded selection limit.")
            return collected
        dirs.sort()
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                capture.gap("A symlink artifact directory was skipped.")
        for name in sorted(names):
            relative = (Path(directory) / name).relative_to(trial_dir).as_posix()
            if relative == "artifacts/manifest.json":
                continue
            count += 1
            if count > MAX_ARTIFACT_FILES or remaining <= 0:
                capture.gap("Only a bounded selection of retained artifact text was reviewed.")
                return collected
            text = _read(
                trial_dir,
                relative,
                capture,
                max_bytes=min(remaining, 25_000),
                required=True,
            )
            if text is not None:
                collected[relative] = text
                remaining -= len(text)
    return collected


async def review_native_trial(
    *,
    trial_dir: Path,
    task: TaskSpec,
    run: RunSpec,
    result: dict[str, Any],
    proxy_url: str,
    proxy_key: str,
    actor_stopped: bool = True,
) -> dict[str, Any]:
    """Persist a versioned review; callers retain their execution/score outcome."""
    capture = _Capture(
        notes=[
            "Native verifier files share the actor environment and are untrusted observations, "
            "not independent effect checks.",
            "No independent external-effects monitor was captured; absence of a logged effect "
            "does not establish that it did not occur.",
        ]
    )
    events: list[dict[str, Any]] = []
    artifacts: dict[str, str] = {}
    verifier = None
    trace_complete = False
    agent_result = result.get("agent_result") or {}
    if actor_stopped:
        events, trace_complete = _trace(trial_dir, agent_result, capture)
        artifacts = _artifacts(trial_dir, capture)
        verifier = {
            "provenance": "shared_actor_environment",
            "independent": False,
            "result": result.get("verifier_result"),
        }
        for name in (
            "scorecard.json",
            "reward.json",
            "reward.txt",
            "test-stdout.txt",
            "test-stderr.txt",
        ):
            content = _read(trial_dir, f"verifier/{name}", capture)
            if content is not None:
                verifier[name] = content
                if name.endswith(".json") and not getattr(content, "truncated", False):
                    try:
                        if not isinstance(json.loads(content), dict):
                            raise ValueError("expected a JSON object")
                    except ValueError as exc:
                        capture.gap(f"Verifier observation {name} is malformed: {exc}")
        if result.get("verifier_result") is not None and not any(
            name in verifier for name in ("reward.txt", "reward.json")
        ):
            capture.gap("Verifier result has no retained reward source.")
        if result.get("verifier_result") is None and len(verifier) == 3:
            verifier = None
    else:
        capture.gap("Actor shutdown was not confirmed; mutable actor files were not read.")

    outcome = result.get("execution_outcome") or {}
    status = "completed" if actor_stopped and outcome.get("kind") == "clean" else "incomplete"
    capture.notes.append(f"Native execution outcome: {json.dumps(outcome, sort_keys=True)}")
    if result.get("exception_info"):
        capture.notes.append(
            f"Execution exception: {result['exception_info'].get('exception_type', 'unknown')}"
        )
    evidence = build_review_evidence(
        run_id=str(result.get("id") or run.run_label),
        task_id=task.name,
        instruction=task.instruction,
        events=events,
        artifacts=artifacts,
        verifier=verifier,
        execution_status=status,
        coverage_notes=capture.notes,
        trace_complete=trace_complete,
        capture_incomplete=capture.incomplete,
    )
    if os.environ.get("SHELLBENCH_RUN_REVIEW", "").lower() == "off":
        review = unreviewed_result(evidence)
    else:
        try:
            review = await review_with_http(
                evidence,
                api_url=os.environ.get("SHELLBENCH_RUN_REVIEW_API_URL")
                or f"{proxy_url.rstrip('/')}/v1/chat/completions",
                api_key=proxy_key,
                model=JUDGE_PROXY_MODEL_NAME,
            )
        except Exception as exc:
            review = unreviewed_result(
                evidence,
                model=JUDGE_PROXY_MODEL_NAME,
                error=f"Native review failed ({type(exc).__name__}).",
            )
    persist_run_review(trial_dir / "review", evidence, review)
    return {
        "schema_version": review.schema_version,
        "status": review.status,
        "evidence_sha256": evidence_sha256(evidence),
        "violation_count": sum(item.verdict == "violation" for item in review.assessments),
        "coverage": evidence.coverage.model_dump(mode="json"),
        "paths": {
            "evidence": "review/evidence.json",
            "review": "review/review.json",
            "html": "review/index.html",
        },
    }
