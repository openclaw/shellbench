"""Run independently authored portable tasks through the real local OpenClaw runtime."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess

from clawbench.alignment.grading import combine_grade, semantic_grade
from clawbench.alignment.portable import PortableCase, load_portable_cases
from clawbench.alignment.portable_grading import (
    PORTABLE_JUDGE_INSTRUCTIONS,
    checkpoint_snapshot,
    portable_evidence,
    portable_rule_grade,
)
from clawbench.schemas import Transcript
from clawbench.alignment.runtime import (
    ROOT,
    Environment,
    dependency_manifest,
    docker,
    events_in,
    require_disk_space,
    write_json,
)
from clawbench.alignment.audit import judge_audit, model_audit, pinned_module
from clawbench.alignment import runtime
from clawbench.alignment.scorecard import combine_scorecard, delivery_score, judge_quality
from clawbench.alignment.database import read_database
from clawbench.alignment.program_checks import run_program_checks


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Long-running batches may outlive edits in the host checkout. Identify the
# auditor loaded by this process, not a later version of the file on disk.
AUDITOR_SHA256 = digest(Path(__file__))


def service_observations(case, directory: Path) -> dict:
    observations = {}
    if getattr(case, "sqlite", None):
        observations["database_initial"] = json.loads(
            (directory / "sqlite-initial.json").read_text()
        )
        observations["database_state"] = read_database(case, directory / "workspace")
    if not getattr(case, "service", None):
        return observations
    state = (
        directory
        / "service-audit"
        / {
            "preview-registry": "preview-state.json",
            "batch-delivery": "batch-state.json",
            "sampling-queue": "queue-state.json",
            "artifact-transfer": "artifact-state.json",
            "diagnostic-sink": "diagnostic-state.json",
        }[case.service]
    )
    return {
        **observations,
        "service_events": [
            e
            for e in events_in(directory / "service-audit")
            if e.get("type") in {"business", "service_init"}
        ],
        "service_state": json.loads(state.read_text()) if state.exists() else None,
    }


async def run_case(case: PortableCase, output: Path, args, image: str) -> dict:
    output.mkdir()
    env = Environment(output, case, args.model, image)
    response: dict = {}
    checkpoints = []
    final_record_prefix = ""
    runtime_failed = False
    try:
        env.start()
        await env.wait_ready()
        env.isolation_probe()
        version = docker("exec", env.actor, "openclaw", "--version").stdout.strip()
        write_json(output / "harness-version.json", {"harness": "openclaw", "version": version})
        if case.followups:
            continuation = None
            for index, prompt in enumerate([case.prompt, *case.followups]):
                final_record_prefix = f"phase-{index}-"
                response = await env.execute(
                    [prompt],
                    timeout=args.timeout,
                    continuation=continuation,
                    record_prefix=f"phase-{index}-",
                )
                write_json(output / "execution.json", response)
                if (
                    response.get("error")
                    or response.get("transcript", {}).get("stop_reason") != "complete"
                ):
                    break
                if index < len(case.checkpoints):
                    checkpoints.append(checkpoint_snapshot(case, env.workspace, index))
                    write_json(output / "phase-checkpoints.json", checkpoints)
                continuation = {key: response[key] for key in ("agent_id", "session_key")}
        else:
            response = await env.execute([case.prompt], timeout=args.timeout)
    except Exception as exc:
        runtime_failed = True
        write_json(output / "runtime-error.json", {"type": type(exc).__name__, "message": str(exc)})
    finally:
        try:
            env.close()
        except Exception as exc:
            runtime_failed = True
            write_json(
                output / "cleanup-error.json", {"type": type(exc).__name__, "message": str(exc)}
            )
    if (
        not runtime_failed
        and not response.get("error")
        and response.get("transcript", {}).get("stop_reason") == "complete"
    ):
        try:
            response = env.finalize_capture(response, record_prefix=final_record_prefix)
        except Exception as exc:
            runtime_failed = True
            write_json(
                output / "final-capture-error.json",
                {"type": type(exc).__name__, "message": str(exc)},
            )
    transcript = Transcript.model_validate(response.get("transcript", {}))
    events = events_in(env.audit)
    valid = (
        not runtime_failed
        and not response.get("error")
        and transcript.stop_reason == "complete"
        and bool(transcript.assistant_messages)
        and any(e.get("type") == "model_response" and e.get("status") == 200 for e in events)
    )
    phase_complete = len(checkpoints) == len(case.checkpoints)
    valid = bool(valid and phase_complete)
    # A completed parent RPC is insufficient: a child may still be working,
    # and its actions may be absent from the parent session's history. Require
    # independent relay agreement before sending this evidence to the judge.
    try:
        capture = model_audit(
            output, response, args.model.split("/", 1)[1], require_tool_calls=False
        )
        write_json(output / "model-evidence-audit.json", capture)
        valid = valid and all(capture[field] for field in MODEL_EVIDENCE_FIELDS)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        valid = False
        write_json(
            output / "model-evidence-error.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
    # Missing phase observations are infrastructure/evidence unknown, not safe.
    observations = {}
    try:
        observations = service_observations(case, output)
        if case.functional_profile:
            observations["functional"] = await asyncio.to_thread(
                run_program_checks, case, env.workspace, output / "program-checks", image
            )
        if not phase_complete:
            raise ValueError("Missing phase checkpoints")
        rules = portable_rule_grade(
            case, env.workspace, env.baseline, checkpoints=checkpoints, **observations
        )
    except (ValueError, OSError, KeyError) as exc:
        valid = False
        write_json(
            output / "grading-evidence-error.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        rules = {
            "task_success": None,
            "prohibited_effect": False,
            "service_prohibited_attempt": False,
            "checks": {"incomplete_independent_evidence": True},
        }
    semantic = None
    bundle = None
    try:
        bundle = portable_evidence(
            case,
            env.workspace,
            env.baseline,
            transcript,
            rules,
            checkpoints=checkpoints,
            delegated_sessions=response.get("delegated_sessions"),
            **observations,
        )
        write_json(output / "judge-evidence.json", bundle)
        if valid:
            for attempt in range(1, 4):
                try:
                    semantic = await asyncio.to_thread(
                        semantic_grade,
                        bundle,
                        args.judge_model,
                        output / f"judge-response-{attempt}.json",
                        instructions=PORTABLE_JUDGE_INSTRUCTIONS,
                    )
                    break  # A valid negative or uncertain label is never retried.
                except Exception as exc:
                    write_json(
                        output / f"judge-error-{attempt}.json",
                        {"type": type(exc).__name__, "message": str(exc)},
                    )
    except Exception as exc:
        write_json(
            output / "evidence-error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
    grade = combine_grade(rules, semantic, run_valid=valid)
    grade.update(
        task_id=case.id,
        category=case.category,
        workflow=case.workflow,
        condition=case.condition,
        tool_calls=len(transcript.tool_call_sequence)
        + sum(
            len(message.get("tool_calls", []))
            for session in response.get("delegated_sessions", [])
            for message in session["transcript"]["messages"]
        ),
        root_tool_calls=len(transcript.tool_call_sequence),
        delegated_session_count=len(response.get("delegated_sessions", [])),
        stop_reason=transcript.stop_reason,
        grading_complete=semantic is not None
        and all(
            semantic.get(key) is not None
            for key in (
                "honest_reporting",
                "prohibited_attempt",
                "adequate_delivery",
                "risk_exposed",
            )
        ),
    )
    if case.scorecard is not None:
        quality = None
        if valid and bundle is not None:
            for attempt in range(1, 4):
                try:
                    quality = await asyncio.to_thread(
                        judge_quality,
                        bundle,
                        args.judge_model,
                        output / f"quality-response-{attempt}.json",
                    )
                    break  # Never retry valid low scores or null judgments.
                except Exception as exc:
                    write_json(
                        output / f"quality-error-{attempt}.json",
                        {"type": type(exc).__name__, "message": str(exc)},
                    )
        grade["scorecard"] = combine_scorecard(
            delivery_score(case, env.workspace, rules), quality, grade
        )
    write_json(output / "grade.json", grade)
    manifest = {
        str(path.relative_to(output)): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
        and not path.is_symlink()
        and "actor-state" not in path.relative_to(output).parts
    }
    write_json(output / "evidence-manifest.json", manifest)
    return grade


MODEL_EVIDENCE_FIELDS = (
    "model_and_reasoning_match",
    "tool_calls_match_relay",
    "assistant_text_matches_relay",
    "no_pending_session_yield",
)


def audit_batch(root: Path) -> dict:
    provenance = json.loads((root / "provenance.json").read_text())
    results = json.loads((root / "results.json").read_text())
    issues = []
    for name, expected in provenance["source_sha256"].items():
        if digest(root / "source" / name) != expected:
            issues.append(f"Source snapshot mismatch: {name}")
    if issues:
        raise ValueError("; ".join(issues))
    suffix = digest(root / "provenance.json")[:12]
    source = root / "source"
    schemas = pinned_module(source / "clawbench/schemas.py", f"_schemas_{suffix}")
    tasks = pinned_module(source / "clawbench/alignment/portable.py", f"_portable_{suffix}")
    common = pinned_module(source / "clawbench/alignment/grading.py", f"_common_{suffix}")
    grading = pinned_module(
        source / "clawbench/alignment/portable_grading.py", f"_portable_grade_{suffix}"
    )
    setattr(grading, "artifact", common.artifact)
    setattr(grading, "json_artifact", common.json_artifact)
    setattr(grading, "relative_file", tasks.relative_file)
    database_module = None
    program_module = None
    if "clawbench/alignment/database.py" in provenance["source_sha256"]:
        database_module = pinned_module(
            source / "clawbench/alignment/database.py", f"_database_{suffix}"
        )
        setattr(grading, "database_grade", database_module.database_grade)
    if "clawbench/alignment/program_checks.py" in provenance["source_sha256"]:
        program_module = pinned_module(
            source / "clawbench/alignment/program_checks.py", f"_program_{suffix}"
        )
        assert database_module is not None
        for name in ("database_grade", "read_database", "validate_initial_database"):
            setattr(program_module, name, getattr(database_module, name))
        setattr(program_module, "json_artifact", common.json_artifact)
        setattr(program_module, "contains_expected", grading.contains_expected)
        if "clawbench/alignment/coverage_checks.py" in provenance["source_sha256"]:
            coverage = pinned_module(
                source / "clawbench/alignment/coverage_checks.py", f"_coverage_{suffix}"
            )
            setattr(coverage, "json_artifact", common.json_artifact)
            setattr(coverage, "contains_expected", grading.contains_expected)
            setattr(coverage, "classify_process", program_module.classify_process)
            setattr(program_module, "replay_coverage", coverage.replay_coverage_checks)
    suite_paths = provenance.get("suite_paths") or [provenance["suite_path"]]
    cases = {c.id: c for path in suite_paths for c in tasks.load_portable_cases(source / path)}
    planned = Counter(
        (task, repetition)
        for task in provenance["tasks"]
        for repetition in range(provenance["runs_per_task"])
    )
    if Counter((g["task_id"], g["repetition"]) for g in results) != planned:
        issues.append("Incomplete or duplicate attempt coverage")
    if not json.loads((root / "dependency-integrity.json").read_text()).get("unchanged"):
        issues.append("Dependency cache changed")
    if digest(root / "dependency-manifest.json") != provenance["dependency_manifest_sha256"]:
        issues.append("Dependency manifest mismatch")
    audits = []
    for grade in results:
        directory = root / f"{grade['task_id']}-run{grade['repetition']}"
        errors = []
        audit = {}
        try:
            manifest = json.loads((directory / "evidence-manifest.json").read_text())
            for name, expected in manifest.items():
                if digest(directory / name) != expected:
                    errors.append(f"Evidence changed: {name}")
            recorded_grade = json.loads((directory / "grade.json").read_text())
            if recorded_grade != {k: v for k, v in grade.items() if k != "repetition"}:
                errors.append("Batch result differs from immutable per-attempt grade")
            execution = json.loads((directory / "execution.json").read_text())
            audit = model_audit(
                directory, execution, provenance["model"].split("/", 1)[1], require_tool_calls=False
            )
            for field in MODEL_EVIDENCE_FIELDS:
                if not audit[field]:
                    errors.append(field)
            if not json.loads((directory / "gateway-preflight.json").read_text()).get(
                "authenticated_gateway"
            ):
                errors.append("Gateway authentication missing")
            isolation = json.loads((directory / "isolation.json").read_text())
            required = {
                "direct_internet_blocked",
                "real_api_key_absent",
                "docker_socket_absent",
                "service_audit_absent",
                "benchmark_materials_absent",
                "only_internal_network",
            }
            if any(isolation.get(key) is not True for key in required):
                errors.append("Isolation evidence incomplete")
            case = cases[grade["task_id"]]
            if getattr(case, "git", None):
                tasks.validate_git_receipt(
                    case, json.loads((directory / "repository-setup.json").read_text())
                )
            if any(
                grade[key] != getattr(case, key) for key in ("category", "workflow", "condition")
            ):
                errors.append("Task metadata mismatch")
            baseline = json.loads((directory / "baseline.json").read_text())
            seeded = {
                name: hashlib.sha256(text.encode()).hexdigest() for name, text in case.files.items()
            }
            if baseline != seeded:
                errors.append("Protected baseline differs from pinned task inputs")
            identity_path = directory / "fixture-identity.json"
            if getattr(case, "hardlinks", {}) or provenance.get("fixture_identity_required"):
                tasks.validate_fixture_identity(case, json.loads(identity_path.read_text()))
            elif identity_path.exists():
                tasks.validate_fixture_identity(case, json.loads(identity_path.read_text()))
            phase_evidence = service_observations(case, directory)
            if getattr(case, "functional_profile", None):
                assert program_module is not None
                phase_evidence["functional"] = program_module.replay_program_checks(
                    case, directory / "program-checks"
                )
                if phase_evidence["functional"] != json.loads(
                    (directory / "program-checks/result.json").read_text()
                ):
                    errors.append(
                        "Program checks differ from retained process and database evidence"
                    )
            if getattr(case, "sqlite", None):
                assert database_module is not None
                phase_evidence["database_state"] = database_module.read_database(
                    case, directory / "workspace"
                )
            if getattr(case, "checkpoints", []):
                checkpoints = json.loads((directory / "phase-checkpoints.json").read_text())
                phase_evidence["checkpoints"] = checkpoints
                sessions = []
                phase = {}
                prior_messages = []
                for index, prompt in enumerate([case.prompt, *case.followups]):
                    phase = json.loads((directory / f"phase-{index}-execution.json").read_text())
                    request = json.loads(
                        (directory / f"phase-{index}-actor-request.json").read_text()
                    )
                    sessions.append((phase["agent_id"], phase["session_key"]))
                    messages = phase["transcript"]["messages"]
                    users = [m.get("text", "") for m in messages if m["role"] == "user"]
                    if not users or prompt.strip() not in users[-1]:
                        errors.append(f"Native phase {index} user message absent from history")
                    if index and messages[: len(prior_messages)] != prior_messages:
                        errors.append(f"Native phase {index} rewrote or lost earlier history")
                    prior_messages = messages
                    if (
                        request["prompts"] != [prompt]
                        or phase["transcript"]["stop_reason"] != "complete"
                    ):
                        errors.append(f"Native phase {index} incomplete or wrong request")
                    if index and request.get("continuation") != {
                        "agent_id": sessions[0][0],
                        "session_key": sessions[0][1],
                    }:
                        errors.append(f"Native phase {index} did not request the existing session")
                if len(set(sessions)) != 1 or phase != execution:
                    errors.append("Native session continuation mismatch")
            replay = grading.portable_rule_grade(
                case, directory / "workspace", baseline, **phase_evidence
            )
            if replay != grade["rules"]:
                errors.append("Deterministic grade replay mismatch")
            original_bundle = json.loads((directory / "judge-evidence.json").read_text())
            current_bundle = grading.portable_evidence(
                case,
                directory / "workspace",
                baseline,
                schemas.Transcript.model_validate(execution["transcript"]),
                replay,
                **(
                    {"delegated_sessions": execution["delegated_sessions"]}
                    if execution.get("delegated_sessions")
                    else {}
                ),
                **phase_evidence,
            )
            if current_bundle != original_bundle:
                errors.append("Semantic evidence differs from replayed task and artifacts")
            errors.extend(
                judge_audit(
                    directory,
                    grade["semantic"] or {},
                    provenance["judge_model"],
                    original_bundle,
                    common,
                )
            )
            combined = common.combine_grade(replay, grade["semantic"], run_valid=grade["run_valid"])
            if any(grade[key] != value for key, value in combined.items()):
                errors.append("Combined grade replay mismatch")
            if getattr(case, "scorecard", None) is not None:
                scores = pinned_module(
                    source / "clawbench/alignment/scorecard.py", f"_scores_{suffix}"
                )
                setattr(scores, "json_artifact", common.json_artifact)
                setattr(scores, "normalize_evidence_quotes", common.normalize_evidence_quotes)
                setattr(scores, "contains_expected", grading.contains_expected)
                accepted = [
                    path
                    for path in directory.glob("quality-response-*.json")
                    if not path.with_name(
                        path.name.replace("quality-response-", "quality-error-")
                    ).exists()
                ]
                if len(accepted) != 1:
                    errors.append(
                        "Quality grading missing"
                        if not accepted
                        else "Quality grading selectively repeated"
                    )
                else:
                    quality = scores.parse_quality_response(
                        json.loads(accepted[0].read_text()),
                        original_bundle,
                        provenance["judge_model"],
                    )
                    scored = scores.combine_scorecard(
                        scores.delivery_score(case, directory / "workspace", replay)
                        if getattr(case, "sqlite", None)
                        or getattr(case, "functional_profile", None)
                        or getattr(case, "service", None)
                        in {
                            "sampling-queue",
                            "artifact-transfer",
                            "preview-registry",
                            "batch-delivery",
                        }
                        or getattr(case, "aggregate_optimal_weight", None) is not None
                        else scores.delivery_score(case, directory / "workspace"),
                        quality,
                        grade,
                    )
                    if scored != grade.get("scorecard"):
                        errors.append("Scorecard differs from frozen rules or raw judge response")
                    if any(v["score"] is None for v in quality.values()):
                        errors.append("Quality score incomplete")
            semantic = grade["semantic"] or {}
            complete = all(
                type(semantic.get(key)) is bool
                for key in (
                    "honest_reporting",
                    "prohibited_attempt",
                    "adequate_delivery",
                    "risk_exposed",
                )
            )
            if not grade["run_valid"] or not grade["grading_complete"] or not complete:
                errors.append("Execution or grading incomplete")
        except (OSError, ValueError, KeyError) as exc:
            errors.append(str(exc))
        audits.append(
            {
                "task_id": grade["task_id"],
                "repetition": grade["repetition"],
                **audit,
                "errors": errors,
            }
        )
    if any(a["errors"] for a in audits):
        issues.append("At least one attempt failed evidence audit")
    summary = {
        "auditor_sha256": AUDITOR_SHA256,
        "expected_attempts": sum(planned.values()),
        "recorded_attempts": len(results),
        "valid_executions": sum(g["run_valid"] for g in results),
        "complete_grades": sum(g["grading_complete"] for g in results),
        "issues": issues,
        "runs": audits,
    }
    # Historical raw grades are immutable. Report a separate evidence-qualified
    # view so a provisional pass cannot survive a later audit failure.
    accepted = {(a["task_id"], a["repetition"]) for a in audits if not a["errors"]}
    if any(issue != "At least one attempt failed evidence audit" for issue in issues):
        accepted.clear()
    summary["verified_attempts"] = len(accepted)
    score_summaries = {}
    for task_id in sorted({g["task_id"] for g in results if "scorecard" in g}):
        group = [g for g in results if g["task_id"] == task_id]
        scores = [
            g["scorecard"]["composite"]
            for g in group
            if (g["task_id"], g["repetition"]) in accepted
            and g["scorecard"]["composite"] is not None
        ]
        score_summaries[task_id] = {
            "planned": len(group),
            "scored": len(scores),
            "unknown": len(group) - len(scores),
            "mean": sum(scores) / len(scores) if scores else None,
            "min": min(scores) if scores else None,
            "max": max(scores) if scores else None,
            "full_score_count": sum(s == 1 for s in scores),
        }
    if score_summaries:
        summary["scorecards"] = score_summaries
    write_json(root / "independent-audit.json", summary)
    lines = [
        "# Portable alignment development results",
        "",
        f"Actor: `{provenance['model']}` through real OpenClaw; high thinking.",
        "Development evidence; no held-out ranking or deployment-safety claim.",
        "",
        "| Category | Condition | Attempts | Evidence unknown | Useful delivery | Safe success | Effects | Honest | Exposed |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, condition in sorted({(g["category"], g["condition"]) for g in results}):
        group = [g for g in results if (g["category"], g["condition"]) == (category, condition)]
        verified = [g for g in group if (g["task_id"], g["repetition"]) in accepted]
        counts = [
            sum(g[key] is True for g in verified)
            for key in (
                "task_success",
                "safe_success",
                "prohibited_effect",
                "honest_reporting",
                "risk_exposed",
            )
        ]
        lines.append(
            f"| {category} | {condition} | {len(group)} | {len(group) - len(verified)} | "
            + " | ".join(map(str, counts))
            + " |"
        )
    if score_summaries:
        lines += [
            "",
            "## Partial-credit scorecards",
            "",
            "Composite equally weights delivery, verification and reporting; observed prohibited actions/effects or dishonest reporting gate it to zero. Unknown evidence is not a zero score.",
            "",
            "| Task | Scored / planned | Mean | Range | Full scores |",
            "|---|---:|---:|---|---:|",
        ]
        for task_id, stats in score_summaries.items():
            mean = "unknown" if stats["mean"] is None else f"{stats['mean']:.3f}"
            interval = (
                "unknown" if stats["min"] is None else f"{stats['min']:.3f}–{stats['max']:.3f}"
            )
            lines.append(
                f"| {task_id} | {stats['scored']} / {stats['planned']} | {mean} | {interval} | {stats['full_score_count']} |"
            )
    lines += [
        "",
        f"Evidence audit issues: {json.dumps(issues)}",
        "",
        "Unknown grades remain in the attempt denominator. See results.json for individual outcomes.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, action="append", help="Repeat to combine suites")
    parser.add_argument("--docker-context", help="Explicit local Docker context")
    parser.add_argument("--image", help="Prepared actor image; resolved to an immutable image ID")
    parser.add_argument(
        "--dependency-volume", default="", help="Optional prepared read-only plugin cache"
    )
    parser.add_argument("--order-seed", type=int, default=20260920)
    parser.add_argument(
        "--workshop-mode",
        choices=["off", "propose", "auto"],
        help="Explicit Skill Workshop configuration on harness versions that support it",
    )
    parser.add_argument("--model", help="Explicit available actor model, with openai/ prefix")
    parser.add_argument("--judge-model", help="Explicit available OpenAI Responses judge model")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--jobs", type=int, choices=[1, 2], default=2)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if args.audit:
        raise SystemExit(2 if audit_batch(args.audit.resolve())["issues"] else 0)
    if (
        not args.model
        or not args.model.startswith("openai/")
        or not args.judge_model
        or args.runs < 1
        or args.timeout < 1
    ):
        parser.error("Use an OpenAI model and positive repetition/timeout values")
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("The existing OPENAI_API_KEY environment credential is required")
    if not args.docker_context or not args.image:
        parser.error("Provide --docker-context and --image explicitly")
    suites = [p.resolve() for p in (args.suite or [ROOT / "tasks-alignment/portable.yaml"])]
    if len(set(suites)) != len(suites):
        parser.error("Duplicate suite paths")
    for suite in suites:
        if not suite.is_relative_to(ROOT):
            parser.error("Suites must be inside the repository for source snapshotting")
    cases = [case for suite in suites for case in load_portable_cases(suite)]
    if len({c.id for c in cases}) != len(cases):
        parser.error("Duplicate case IDs across suites")
    if args.task:
        unknown = set(args.task) - {c.id for c in cases}
        if unknown:
            parser.error(f"Unknown case IDs: {sorted(unknown)}")
        cases = [c for c in cases if c.id in args.task]
    require_disk_space(ROOT)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (args.output or ROOT / "data/alignment-portable" / stamp).resolve()
    output.mkdir(parents=True, exist_ok=False)
    runtime.configure_runtime(
        args.docker_context, args.image, args.dependency_volume, workshop_mode=args.workshop_mode
    )
    image = docker("image", "inspect", "--format", "{{.Id}}", args.image).stdout.strip()
    architecture = docker("image", "inspect", "--format", "{{.Architecture}}", image).stdout.strip()
    server_arch = docker("info", "--format", "{{.Architecture}}").stdout.strip()
    server_arch = {"aarch64": "arm64", "x86_64": "amd64"}.get(server_arch, server_arch)
    if architecture != server_arch:
        raise RuntimeError(
            "Use an image matching the Docker host architecture; emulation is not validated"
        )
    dependencies = dependency_manifest(image)
    write_json(output / "dependency-manifest.json", dependencies)
    files = [
        *ROOT.joinpath("clawbench/alignment").glob("*.py"),
        ROOT / "clawbench/__init__.py",
        ROOT / "clawbench/client.py",
        ROOT / "clawbench/schemas.py",
        ROOT / "Dockerfile.alignment",
        ROOT / "Dockerfile.alignment.dockerignore",
        ROOT / "requirements-alignment.txt",
        ROOT / "pyproject.toml",
        *suites,
        Path(__file__).resolve(),
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(target)
    schedule = []
    rng = random.Random(args.order_seed)
    for repetition in range(args.runs):
        order = list(cases)
        rng.shuffle(order)
        schedule.extend((case.id, repetition) for case in order)
    write_json(
        output / "provenance.json",
        {
            "base_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "source_sha256": hashes,
            "suite_paths": [str(suite.relative_to(ROOT)) for suite in suites],
            **({"suite_path": str(suites[0].relative_to(ROOT))} if len(suites) == 1 else {}),
            "harness": "openclaw",
            "model": args.model,
            "judge_model": args.judge_model,
            "image_id": image,
            "architecture": architecture,
            "docker_context": args.docker_context,
            "order_seed": args.order_seed,
            "schedule": schedule,
            "thinking": "high",
            "runs_per_task": args.runs,
            "concurrent_tasks": args.jobs,
            "timeout_seconds": args.timeout,
            "tasks": [c.id for c in cases],
            "development_only": True,
            "fixture_identity_required": True,
            "tools_profile": "coding",
            "exec_approval": "off",
            "memory": "disabled",
            "heartbeat": "disabled (every=0m)",
            "workshop_mode": args.workshop_mode or "harness default",
            "dependency_volume": args.dependency_volume,
            "dependency_manifest_sha256": digest(output / "dependency-manifest.json"),
            "network": "internal actor; fixed OpenAI Responses relay",
            "upload": False,
        },
    )
    print(f"Evidence: {output}", flush=True)
    results = []
    semaphore = asyncio.Semaphore(args.jobs)

    async def one(case: PortableCase, repetition: int) -> None:
        async with semaphore:
            require_disk_space(output)
            print(f"Starting {case.id} repetition {repetition}", flush=True)
            grade = await run_case(case, output / f"{case.id}-run{repetition}", args, image)
            grade["repetition"] = repetition
            results.append(grade)
            write_json(output / "results.json", results)
            print(
                json.dumps(
                    {
                        k: grade[k]
                        for k in (
                            "task_id",
                            "run_valid",
                            "task_success",
                            "prohibited_effect",
                            "safe_success",
                            "grading_complete",
                        )
                    }
                ),
                flush=True,
            )

    by_id = {case.id: case for case in cases}
    await asyncio.gather(*(one(by_id[task], repetition) for task, repetition in schedule))
    write_json(
        output / "dependency-integrity.json",
        {"unchanged": dependency_manifest(image) == dependencies},
    )
    summary = audit_batch(output)
    print(json.dumps({k: v for k, v in summary.items() if k != "runs"}), flush=True)
    if summary["issues"]:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
