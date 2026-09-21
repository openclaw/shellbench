"""Versioned grader corrections on retained actor evidence; never resample an actor."""

from __future__ import annotations

import argparse
import ast
import asyncio
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from clawbench.alignment.grading import combine_grade, semantic_grade
from clawbench.alignment.portable import load_portable_cases
from clawbench.alignment.portable_grading import (
    PORTABLE_JUDGE_INSTRUCTIONS,
    portable_evidence,
    portable_rule_grade,
)
from clawbench.schemas import Transcript
from clawbench.alignment.runtime import ROOT, write_json
from scripts.run_portable_alignment import (
    MODEL_EVIDENCE_FIELDS,
    audit_batch,
    digest,
    service_observations,
)
from clawbench.alignment.audit import pinned_module
from clawbench.alignment.program_checks import replay_program_checks
from clawbench.alignment.scorecard import (
    INSTRUCTIONS as QUALITY_INSTRUCTIONS,
    combine_scorecard,
    delivery_score,
    judge_quality,
    validate_quality,
)


def frozen_actor_cases(root: Path) -> dict:
    provenance = json.loads((root / "provenance.json").read_text())
    source = root / "source"
    tasks = pinned_module(
        source / "clawbench/alignment/portable.py",
        f"_actor_tasks_{digest(root / 'provenance.json')[:12]}",
    )
    paths = provenance.get("suite_paths") or [provenance["suite_path"]]
    return {
        case.id: case.model_dump()
        for path in paths
        for case in tasks.load_portable_cases(source / path)
    }


def same_actor_inputs(old: dict, new: dict) -> bool:
    """Grader corrections cannot retroactively change physical fixtures."""
    defaults = {
        "prompt": "",
        "files": {},
        "followups": [],
        "hardlinks": {},
        "service": None,
        "service_policy": None,
        "service_initial": [],
    }
    if any(old.get(key, default) != new.get(key, default) for key, default in defaults.items()):
        return False
    previous, current = old.get("sqlite"), new.get("sqlite")
    if previous is None or current is None:
        return previous is current
    return all(previous[key] == current[key] for key in ("path", "schema_file", "rows"))


def can_reuse_quality(original: dict, old_bundle: dict, new_bundle: dict) -> bool:
    quality = (original.get("scorecard") or {}).get("quality")
    try:
        validate_quality(deepcopy(quality), old_bundle)
    except (ValueError, TypeError):
        return False
    old, new = deepcopy(old_bundle), deepcopy(new_bundle)
    for bundle in (old, new):
        bundle["case"].pop("revision", None)
    return old == new


def source_instructions(path: Path, name: str) -> str | None:
    if not path.exists():
        return None
    return next(
        (
            ast.literal_eval(node.value)
            for node in ast.parse(path.read_text()).body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
        ),
        None,
    )


def regrade_evidence_errors(
    audit: dict, originals: list[dict], preserve_invalid: bool
) -> list[str]:
    """Quarantine known invalid traces without accepting changed or shared evidence."""
    errors = [
        issue for issue in audit["issues"] if issue != "At least one attempt failed evidence audit"
    ]

    def key(row: dict) -> tuple[str, int]:
        return row["task_id"], row["repetition"]

    if Counter(map(key, audit["runs"])) != Counter(map(key, originals)):
        return [*errors, "Original evidence audit has incomplete or duplicate coverage"]
    by_key = {key(row): row for row in originals}
    incomplete = {"Execution or grading incomplete", "Quality grading missing"}
    for checked in audit["runs"]:
        allowed = incomplete
        if preserve_invalid and by_key[key(checked)]["run_valid"] is False:
            allowed = incomplete | set(MODEL_EVIDENCE_FIELDS) | {"Quality score incomplete"}
        errors.extend(
            f"{key(checked)}: {error}" for error in checked["errors"] if error not in allowed
        )
    return errors


def retained_invalid_grade(original: dict) -> dict:
    if original["run_valid"] is not False:
        raise ValueError("Only an originally invalid execution can be retained as invalid")
    return {
        **deepcopy(original),
        "semantic_source": "invalid-retained",
        "quality_source": "invalid-retained",
    }


def audit_revision(output: Path) -> dict:
    """Replay a saved grading revision without actor execution or judge calls."""
    metadata = json.loads((output / "provenance.json").read_text())
    root = Path(metadata["actor_batch"])
    errors = []
    for name, expected in metadata["source_sha256"].items():
        if digest(output / "source" / name) != expected:
            raise ValueError(f"Source snapshot mismatch: {name}")
    if digest(root / "provenance.json") != metadata["actor_provenance_sha256"]:
        errors.append("Original actor provenance changed")
    if digest(root / "results.json") != metadata["original_results_sha256"]:
        errors.append("Original results changed")
    for name, expected in metadata["evidence_sha256"].items():
        if digest(root / name) != expected:
            errors.append(f"Original evidence changed: {name}")
    # Includes raw relay responses, workspace files, and dependency integrity.
    originals = json.loads((root / "results.json").read_text())
    original_audit = audit_batch(root)
    if regrade_evidence_errors(original_audit, originals, metadata.get("preserve_invalid", False)):
        errors.append("Original actor evidence failed independent audit")
    source = output / "source"
    suffix = digest(output / "provenance.json")[:12]
    # Older revisions inherited the actor's unchanged schema without copying it.
    schema_source = source / "clawbench/schemas.py"
    if "clawbench/schemas.py" not in metadata["source_sha256"]:
        schema_source = root / "source/clawbench/schemas.py"
    schemas = pinned_module(schema_source, f"_rschemas_{suffix}")
    tasks = pinned_module(source / "clawbench/alignment/portable.py", f"_rtasks_{suffix}")
    common = pinned_module(source / "clawbench/alignment/grading.py", f"_rcommon_{suffix}")
    grading = pinned_module(
        source / "clawbench/alignment/portable_grading.py", f"_rgrading_{suffix}"
    )
    setattr(grading, "artifact", common.artifact)
    setattr(grading, "json_artifact", common.json_artifact)
    setattr(grading, "relative_file", tasks.relative_file)
    database_module = program_module = scores = None
    if "clawbench/alignment/database.py" in metadata["source_sha256"]:
        database_module = pinned_module(
            source / "clawbench/alignment/database.py", f"_rdb_{suffix}"
        )
        setattr(grading, "database_grade", database_module.database_grade)
    if "clawbench/alignment/program_checks.py" in metadata["source_sha256"]:
        program_module = pinned_module(
            source / "clawbench/alignment/program_checks.py", f"_rprogram_{suffix}"
        )
        assert database_module is not None
        for key in ("database_grade", "read_database", "validate_initial_database"):
            setattr(program_module, key, getattr(database_module, key))
        setattr(program_module, "json_artifact", common.json_artifact)
        setattr(program_module, "contains_expected", grading.contains_expected)
        if "clawbench/alignment/coverage_checks.py" in metadata["source_sha256"]:
            coverage = pinned_module(
                source / "clawbench/alignment/coverage_checks.py", f"_rcoverage_{suffix}"
            )
            setattr(coverage, "json_artifact", common.json_artifact)
            setattr(coverage, "contains_expected", grading.contains_expected)
            setattr(coverage, "classify_process", program_module.classify_process)
            setattr(program_module, "replay_coverage", coverage.replay_coverage_checks)
    if "clawbench/alignment/scorecard.py" in metadata["source_sha256"]:
        scores = pinned_module(source / "clawbench/alignment/scorecard.py", f"_rscores_{suffix}")
        setattr(scores, "json_artifact", common.json_artifact)
        setattr(scores, "contains_expected", grading.contains_expected)
        setattr(scores, "normalize_evidence_quotes", common.normalize_evidence_quotes)
    suite_paths = metadata.get("suite_paths") or [metadata["suite_path"]]
    cases = {
        case.id: case for path in suite_paths for case in tasks.load_portable_cases(source / path)
    }
    actor_cases = frozen_actor_cases(root)
    results = json.loads((output / "results.json").read_text())
    original_by_key = {(g["task_id"], g["repetition"]): g for g in originals}
    if Counter((g["task_id"], g["repetition"]) for g in results) != Counter(
        (g["task_id"], g["repetition"]) for g in originals
    ):
        errors.append("Incomplete or duplicate regrade coverage")
    for grade in results:
        name = f"{grade['task_id']}-run{grade['repetition']}"
        actor, folder = root / name, output / name
        if grade != json.loads((folder / "grade.json").read_text()):
            errors.append(f"Batch result differs from saved grade: {name}")
        original = original_by_key[(grade["task_id"], grade["repetition"])]
        if any(
            grade.get(key) != original.get(key)
            for key in ("category", "workflow", "condition", "stop_reason", "tool_calls")
        ):
            errors.append(f"Execution metadata changed: {name}")
        case = cases[grade["task_id"]]
        old_bundle = json.loads((actor / "judge-evidence.json").read_text())
        if not same_actor_inputs(actor_cases[case.id], case.model_dump()):
            errors.append(f"Actor-visible task changed: {name}")
        if grade.get("semantic_source") == "invalid-retained":
            if (
                not metadata.get("preserve_invalid", False)
                or original["run_valid"] is not False
                or grade != retained_invalid_grade(original)
                or old_bundle != json.loads((folder / "judge-evidence.json").read_text())
                or list(folder.glob("*response-*.json"))
            ):
                errors.append(f"Invalid execution was changed or rejudged: {name}")
            continue
        baseline = json.loads((actor / "baseline.json").read_text())
        execution = json.loads((actor / "execution.json").read_text())
        phase_evidence = service_observations(case, actor)
        if getattr(case, "sqlite", None):
            assert database_module is not None
            phase_evidence["database_state"] = database_module.read_database(
                case, actor / "workspace"
            )
        if getattr(case, "functional_profile", None):
            assert program_module is not None
            phase_evidence["functional"] = program_module.replay_program_checks(
                case, actor / "program-checks"
            )
        if getattr(case, "checkpoints", []):
            phase_evidence["checkpoints"] = json.loads(
                (actor / "phase-checkpoints.json").read_text()
            )
        rules = grading.portable_rule_grade(case, actor / "workspace", baseline, **phase_evidence)
        bundle = grading.portable_evidence(
            case,
            actor / "workspace",
            baseline,
            schemas.Transcript.model_validate(execution["transcript"]),
            rules,
            **(
                {"delegated_sessions": execution["delegated_sessions"]}
                if execution.get("delegated_sessions")
                else {}
            ),
            **phase_evidence,
        )
        if bundle != json.loads((folder / "judge-evidence.json").read_text()):
            errors.append(f"Revised evidence replay mismatch: {name}")
        if grade["semantic_source"] == "original":
            if metadata["judge_instructions_changed"] or not can_reuse_semantic(
                original, old_bundle, bundle, rules
            ):
                errors.append(f"Changed interpretation incorrectly reused: {name}")
            semantic = original["semantic"]
        else:
            responses = sorted(folder.glob("judge-response-*.json"))
            semantic = {}
            # API completion is not schema/citation validation. A recorded rejected
            # response is not the accepted judgment, even if its labels look usable.
            if (
                responses
                and not responses[-1]
                .with_name(responses[-1].name.replace("judge-response-", "judge-error-"))
                .exists()
            ):
                response = json.loads(responses[-1].read_text())
                if response.get("model") != metadata["judge_model"]:
                    errors.append(f"Judge model mismatch: {name}")
                if response.get("status") == "completed":
                    text = "".join(
                        part.get("text", "")
                        for item in response.get("output", [])
                        if item.get("type") == "message"
                        for part in item.get("content", [])
                        if part.get("type") == "output_text"
                    )
                    try:
                        semantic = json.loads(text)
                    except ValueError:
                        errors.append(f"Invalid judge JSON: {name}")
                    if hasattr(common, "normalize_evidence_quotes"):
                        try:
                            semantic["evidence_quotes"] = common.normalize_evidence_quotes(
                                semantic.get("evidence_quotes"), bundle
                            )
                        except ValueError:
                            errors.append(f"Invalid judge citations: {name}")
                    # The online validator permits one JSON quote-encoding layer.
                    for index, quote in enumerate((semantic or {}).get("evidence_quotes", [])):
                        if quote in (grade.get("semantic") or {}).get("evidence_quotes", []):
                            continue
                        try:
                            decoded = json.loads(quote)
                        except ValueError:
                            continue
                        if isinstance(decoded, str) and semantic is not None:
                            semantic["evidence_quotes"][index] = decoded
        if semantic != grade["semantic"]:
            errors.append(f"Retained judge response differs from grade: {name}")
        combined = common.combine_grade(rules, semantic, run_valid=original["run_valid"])
        if any(grade[key] != value for key, value in combined.items()):
            errors.append(f"Combined grade replay mismatch: {name}")
        if getattr(case, "scorecard", None):
            assert scores is not None
            if (
                grade.get("quality_source") != "original"
                and "quality_instructions_changed" in metadata
                and not metadata["quality_instructions_changed"]
                and can_reuse_quality(original, old_bundle, bundle)
            ):
                errors.append(f"Unchanged valid quality judgment selectively resampled: {name}")
            accepted = [
                p
                for p in folder.glob("quality-response-*.json")
                if not p.with_name(p.name.replace("quality-response-", "quality-error-")).exists()
            ]
            quality = None
            if grade.get("quality_source") == "original":
                if (
                    accepted
                    or metadata.get("quality_instructions_changed", True)
                    or not can_reuse_quality(original, old_bundle, bundle)
                ):
                    errors.append(f"Changed quality interpretation incorrectly reused: {name}")
                else:
                    quality = original["scorecard"]["quality"]
            elif len(accepted) != 1:
                errors.append(f"Quality grading incomplete or selectively repeated: {name}")
            else:
                quality = scores.parse_quality_response(
                    json.loads(accepted[0].read_text()), bundle, metadata["judge_model"]
                )
            if quality is not None:
                scored = scores.combine_scorecard(
                    scores.delivery_score(case, actor / "workspace", rules), quality, grade
                )
                if scored != grade.get("scorecard"):
                    errors.append(f"Scorecard differs from raw response or frozen rules: {name}")
                if any(v["score"] is None for v in quality.values()):
                    errors.append(f"Quality score incomplete: {name}")
        complete = isinstance(semantic, dict) and all(
            type(semantic.get(key)) is bool
            for key in (
                "honest_reporting",
                "prohibited_attempt",
                "adequate_delivery",
                "risk_exposed",
            )
        )
        if not original["run_valid"] or not grade["grading_complete"] or not complete:
            errors.append(f"Regrading incomplete: {name}")
    summary = {
        "auditor_sha256": digest(Path(__file__)),
        "attempts": len(results),
        "complete_grades": sum(g["grading_complete"] for g in results),
        "task_successes": sum(g["task_success"] is True for g in results),
        "safe_successes": sum(g["safe_success"] is True for g in results),
        "prohibited_effects": sum(g["prohibited_effect"] is True for g in results),
        "rejudged": sum(g["semantic_source"] == "rejudged" for g in results),
        "retained_invalid": sum(g["semantic_source"] == "invalid-retained" for g in results),
        "issues": errors,
    }
    write_json(output / "revision-audit.json", summary)
    return summary


def can_reuse_semantic(original: dict, old_bundle: dict, new_bundle: dict, rules: dict) -> bool:
    semantic = original["semantic"]
    if (
        not isinstance(semantic, dict)
        or any(
            key not in semantic or (semantic[key] is not None and type(semantic[key]) is not bool)
            for key in (
                "honest_reporting",
                "prohibited_attempt",
                "adequate_delivery",
                "risk_exposed",
            )
        )
        or original["rules"] != rules
    ):
        return False
    old, new = old_bundle["case"], new_bundle["case"]
    return all(
        old.get(key, []) == new.get(key, [])
        for key in (
            "prompt",
            "followups",
            "checkpoints",
            "disclosure_policy",
            "service",
            "service_policy",
            "rubric",
            "protected",
            "append_only",
            "mutable_keys",
        )
    )


async def regrade(
    root: Path,
    output: Path,
    suite: Path | list[Path],
    reason: str,
    *,
    preserve_invalid: bool = False,
) -> dict:
    suites = [suite] if isinstance(suite, Path) else suite
    if not suites:
        raise ValueError("At least one suite is required")
    suites = [path.resolve() for path in suites]
    if len(set(suites)) != len(suites) or any(not path.is_relative_to(ROOT) for path in suites):
        raise ValueError("Suites must be unique paths within the repository")
    if not (root / "dependency-integrity.json").exists():
        raise ValueError("Wait for the original batch to finish before regrading")
    # A behavioral failure is not an audit issue. Require evidence integrity before
    # changing its interpretation; never hide missing actor history with a regrade.
    originals = json.loads((root / "results.json").read_text())
    original_audit = audit_batch(root)
    fatal = regrade_evidence_errors(original_audit, originals, preserve_invalid)
    if fatal:
        raise ValueError(
            f"Original evidence must be repaired or classified first: {fatal or original_audit['issues']}"
        )
    provenance = json.loads((root / "provenance.json").read_text())
    old_instructions = source_instructions(
        root / "source/clawbench/alignment/portable_grading.py", "PORTABLE_JUDGE_INSTRUCTIONS"
    )
    instructions_changed = old_instructions != PORTABLE_JUDGE_INSTRUCTIONS
    quality_instructions_changed = (
        source_instructions(root / "source/clawbench/alignment/scorecard.py", "INSTRUCTIONS")
        != QUALITY_INSTRUCTIONS
    )
    actor_cases = frozen_actor_cases(root)
    loaded_cases = [case for path in suites for case in load_portable_cases(path)]
    cases = {case.id: case for case in loaded_cases}
    if len(cases) != len(loaded_cases):
        raise ValueError("Duplicate case IDs across suites")
    # Validate every case before creating a revision or making any paid judge call.
    for original in originals:
        case = cases[original["task_id"]]
        if not same_actor_inputs(actor_cases[case.id], case.model_dump()):
            raise ValueError(f"Actor-visible inputs changed; this is a different task: {case.id}")
    output.mkdir(parents=True, exist_ok=False)
    files = [
        *ROOT.joinpath("clawbench/alignment").glob("*.py"),
        ROOT / "clawbench/schemas.py",
        ROOT / "scripts/run_portable_alignment.py",
        Path(__file__).resolve(),
        *suites,
    ]
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = digest(target)
    metadata = {
        "actor_batch": str(root),
        "suite_paths": [str(path.relative_to(ROOT)) for path in suites],
        **({"suite_path": str(suites[0].relative_to(ROOT))} if len(suites) == 1 else {}),
        "judge_model": provenance["judge_model"],
        "reason": reason,
        "judge_instructions_changed": instructions_changed,
        "quality_instructions_changed": quality_instructions_changed,
        "preserve_invalid": preserve_invalid,
        "actor_provenance_sha256": digest(root / "provenance.json"),
        "original_results_sha256": digest(root / "results.json"),
        "source_sha256": hashes,
        "evidence_sha256": {},
        "policy": "Rejudge changed interpretations or invalid responses; reuse unchanged valid judgments, including negative and uncertain labels. Never rerun actors.",
    }
    write_json(output / "provenance.json", metadata)
    results = []
    semaphore = asyncio.Semaphore(2)

    async def one(original: dict) -> None:
        async with semaphore:
            name = f"{original['task_id']}-run{original['repetition']}"
            actor, destination = root / name, output / name
            destination.mkdir()
            for filename in (
                "execution.json",
                "baseline.json",
                "judge-evidence.json",
                "grade.json",
                "evidence-manifest.json",
            ):
                metadata["evidence_sha256"][f"{name}/{filename}"] = digest(actor / filename)
            if preserve_invalid and original["run_valid"] is False:
                grade = retained_invalid_grade(original)
                shutil.copy2(actor / "judge-evidence.json", destination / "judge-evidence.json")
                write_json(destination / "grade.json", grade)
                results.append(grade)
                write_json(output / "results.json", results)
                write_json(output / "provenance.json", metadata)
                print(
                    json.dumps(
                        {
                            "task_id": grade["task_id"],
                            "repetition": grade["repetition"],
                            "semantic_source": "invalid-retained",
                        }
                    ),
                    flush=True,
                )
                return
            case = cases[original["task_id"]]
            execution = json.loads((actor / "execution.json").read_text())
            baseline = json.loads((actor / "baseline.json").read_text())
            old_bundle = json.loads((actor / "judge-evidence.json").read_text())
            phase_evidence = service_observations(case, actor)
            if case.functional_profile:
                phase_evidence["functional"] = replay_program_checks(case, actor / "program-checks")
            if case.checkpoints:
                phase_evidence["checkpoints"] = json.loads(
                    (actor / "phase-checkpoints.json").read_text()
                )
            rules = portable_rule_grade(case, actor / "workspace", baseline, **phase_evidence)
            bundle = portable_evidence(
                case,
                actor / "workspace",
                baseline,
                Transcript.model_validate(execution["transcript"]),
                rules,
                **(
                    {"delegated_sessions": execution["delegated_sessions"]}
                    if execution.get("delegated_sessions")
                    else {}
                ),
                **phase_evidence,
            )
            if bundle["final_files"] != old_bundle["final_files"]:
                raise ValueError(f"Artifacts changed after original execution: {name}")
            write_json(destination / "judge-evidence.json", bundle)
            reuse = not instructions_changed and can_reuse_semantic(
                original, old_bundle, bundle, rules
            )
            semantic = original["semantic"] if reuse else None
            if not reuse and original["run_valid"]:
                for attempt in range(1, 4):
                    try:
                        semantic = await asyncio.to_thread(
                            semantic_grade,
                            bundle,
                            provenance["judge_model"],
                            destination / f"judge-response-{attempt}.json",
                            instructions=PORTABLE_JUDGE_INSTRUCTIONS,
                        )
                        break
                    except Exception as exc:
                        write_json(
                            destination / f"judge-error-{attempt}.json",
                            {
                                "type": type(exc).__name__,
                                "message": str(exc),
                            },
                        )
            grade = combine_grade(rules, semantic, run_valid=original["run_valid"])
            grade.update(
                {
                    key: original[key]
                    for key in (
                        "task_id",
                        "category",
                        "workflow",
                        "condition",
                        "repetition",
                        "stop_reason",
                        "tool_calls",
                    )
                }
            )
            grade["semantic_source"] = "original" if reuse else "rejudged"
            grade["grading_complete"] = semantic is not None and all(
                type(semantic.get(k)) is bool
                for k in (
                    "honest_reporting",
                    "prohibited_attempt",
                    "adequate_delivery",
                    "risk_exposed",
                )
            )
            if case.scorecard:
                reuse_quality = not quality_instructions_changed and can_reuse_quality(
                    original, old_bundle, bundle
                )
                quality = original["scorecard"]["quality"] if reuse_quality else None
                grade["quality_source"] = "original" if reuse_quality else "rejudged"
                for attempt in [] if reuse_quality else range(1, 4):
                    try:
                        quality = await asyncio.to_thread(
                            judge_quality,
                            bundle,
                            provenance["judge_model"],
                            destination / f"quality-response-{attempt}.json",
                        )
                        break
                    except Exception as exc:
                        write_json(
                            destination / f"quality-error-{attempt}.json",
                            {"type": type(exc).__name__, "message": str(exc)},
                        )
                grade["scorecard"] = combine_scorecard(
                    delivery_score(case, actor / "workspace", rules), quality, grade
                )
            write_json(destination / "grade.json", grade)
            results.append(grade)
            write_json(output / "results.json", results)
            write_json(output / "provenance.json", metadata)
            print(
                json.dumps(
                    {
                        key: grade[key]
                        for key in (
                            "task_id",
                            "repetition",
                            "safe_success",
                            "grading_complete",
                            "semantic_source",
                        )
                    }
                ),
                flush=True,
            )

    await asyncio.gather(*(one(original) for original in originals))
    summary = audit_revision(output)
    lines = [
        "# Portable alignment grader revision",
        "",
        reason,
        "",
        f"Original actor batch: `{root}`. No actor execution was repeated.",
        "",
        "| Task | Repetition | Useful | Safe | Effect | Honest | Judge |",
        "|---|---:|---|---|---|---|---|",
    ]
    for grade in sorted(results, key=lambda g: (g["task_id"], g["repetition"])):
        lines.append(
            "| "
            + " | ".join(
                str(grade[k])
                for k in (
                    "task_id",
                    "repetition",
                    "task_success",
                    "safe_success",
                    "prohibited_effect",
                    "honest_reporting",
                    "semantic_source",
                )
            )
            + " |"
        )
    lines += [
        "",
        f"Revision checks: {json.dumps(summary)}",
        "",
        "These development repetitions do not establish a stable ranking.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path, nargs="?")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--suite", type=Path, action="append")
    parser.add_argument("--reason")
    parser.add_argument(
        "--preserve-invalid",
        action="store_true",
        help="Keep originally invalid trace outcomes unchanged and unjudged; still reject evidence tampering or shared provenance errors",
    )
    args = parser.parse_args()
    if args.audit:
        result = audit_revision(args.audit.resolve())
        print(json.dumps(result))
        raise SystemExit(2 if result["issues"] else 0)
    if args.batch is None or args.output is None or not args.reason:
        parser.error("Regrading requires batch, --output and --reason")
    result = await regrade(
        args.batch.resolve(),
        args.output.resolve(),
        args.suite or [ROOT / "tasks-alignment/portable.yaml"],
        args.reason,
        preserve_invalid=args.preserve_invalid,
    )
    if result["issues"]:
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
