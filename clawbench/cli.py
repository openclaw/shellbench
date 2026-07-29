"""CLI entry point for ClawBench."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click

from clawbench.client import GatewayConfig
from clawbench.harness import BenchmarkHarness, KNOWN_ADAPTERS
from clawbench.schemas import ScenarioDomain

SCENARIO_CHOICES = [scenario.value for scenario in ScenarioDomain]


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@cli.command()
@click.option("--model", "-m", required=True, help="Model to benchmark")
@click.option(
    "--adapter",
    type=click.Choice(KNOWN_ADAPTERS),
    default="openclaw",
    show_default=True,
    help="Agent harness adapter. OpenClaw is executable today; other adapters are tracked targets.",
)
@click.option("--gateway-token", envvar="OPENCLAW_GATEWAY_TOKEN", default="", help="Gateway auth token")
@click.option(
    "--judge-model",
    envvar="CLAWBENCH_JUDGE_MODEL",
    default="",
    help="Optional advisory LLM judge model (does not affect official score)",
)
@click.option(
    "--judge-affects-score",
    is_flag=True,
    envvar="CLAWBENCH_JUDGE_AFFECTS_SCORE",
    help="Opt in to experimental judge-weighted scoring. Official scoring keeps judge advisory.",
)
@click.option("--runs", "-n", default=3, show_default=True, help="Runs per task (reliability uses all runs)")
@click.option("--tier", type=click.Choice(["tier1", "tier2", "tier3", "tier4", "tier5"]), help="Filter tier")
@click.option("--scenario", type=click.Choice(SCENARIO_CHOICES), help="Filter query scenario")
@click.option("--artifact-type", type=click.Choice(["file", "information", "operation", "code", "external_action", "memory", "automation", "mixed"]), help="Filter expected artifact type")
@click.option("--prompt-variant", type=click.Choice(["clear", "ambiguous"]), default="clear", show_default=True, help="Prompt variant to run")
@click.option("--pool", type=click.Choice(["public_dev", "official_hidden"]), help="Filter task pool")
@click.option("--subset", multiple=True, type=click.Choice(["consensus", "hard"]), help="Filter task subset")
@click.option(
    "--capability",
    multiple=True,
    type=click.Choice(
        [
            "bugfix",
            "refactor",
            "test_authoring",
            "multifile_reasoning",
            "browser_debugging",
            "structured_output",
            "memory_continuation",
            "delegation",
            "tool_composition",
            "research_synthesis",
            "graceful_refusal",
            "spec_revision",
            "cross_repo_change",
            "automation",
        ]
    ),
    help="Filter by capability tag",
)
@click.option("--official-only", is_flag=True, help="Only run tasks marked official")
@click.option("--task", "-t", multiple=True, help="Specific task IDs to run")
@click.option("--output", "-o", type=click.Path(), help="Output JSON file path")
@click.option("--no-randomize", is_flag=True, help="Run tasks in definition order")
@click.option("--upload", is_flag=True, help="Upload results to HF Dataset")
@click.option(
    "--concurrency",
    "-c",
    default=1,
    show_default=True,
    type=int,
    envvar="CLAWBENCH_CONCURRENCY",
    help="Number of (task, run) work items to execute in parallel against the gateway. "
         "Set to 4-8 for dramatic speedup. Browser tasks are still serialized.",
)
@click.option(
    "--browser-concurrency",
    default=1,
    show_default=True,
    type=int,
    help="Maximum browser tasks to run concurrently. Should normally stay 1 — "
         "Chromium uses a fixed port that does not parallelize.",
)
@click.option(
    "--profile",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Optional Plugin Profile YAML. When provided, after the benchmark run "
         "completes the v0.5 Configuration Diagnostic Report is generated and "
         "the run is recorded in the historical profile database.",
)
@click.option(
    "--insights-dir",
    type=click.Path(path_type=Path),
    default=Path(".clawbench/insights"),
    show_default=True,
    help="Where to write ecosystem insight files after a --profile run.",
)
@click.option(
    "--dynamics",
    is_flag=True,
    help="Run quick post-benchmark dynamics analysis. Prefer dynamics-report for offline cache/archive analysis.",
)
def run(
    model: str,
    adapter: str,
    gateway_token: str,
    judge_model: str,
    judge_affects_score: bool,
    runs: int,
    tier: str | None,
    scenario: str | None,
    artifact_type: str | None,
    prompt_variant: str,
    pool: str | None,
    subset: tuple[str, ...],
    capability: tuple[str, ...],
    official_only: bool,
    task: tuple[str, ...],
    output: str | None,
    no_randomize: bool,
    upload: bool,
    concurrency: int,
    browser_concurrency: int,
    profile: Path | None,
    insights_dir: Path,
    dynamics: bool,
) -> None:
    gateway_config = GatewayConfig(token=gateway_token)
    harness = BenchmarkHarness(
        gateway_config=gateway_config,
        model=model,
        adapter=adapter,
        judge_model=judge_model,
        judge_affects_score=judge_affects_score,
        runs_per_task=runs,
        tier=tier,
        scenario=scenario,
        artifact_type=artifact_type,
        prompt_variant=prompt_variant,
        pool=pool,
        subsets=list(subset),
        capabilities=list(capability),
        official_only=official_only,
        task_ids=list(task) if task else None,
        randomize_order=not no_randomize,
        concurrency=concurrency,
        browser_concurrency=browser_concurrency,
    )

    result = asyncio.run(harness.run())
    out_path = output or f"results/{result.submission_id}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result.model_dump(), handle, indent=2)
    click.echo(f"\nResults saved to {out_path}")

    if dynamics:
        _run_dynamics_analysis(harness.last_task_runs, out_path)

    if profile is not None:
        _run_v05_diagnostic(
            profile_path=profile,
            result=result,
            task_runs=harness.last_task_runs,
            runs_per_task=runs,
            insights_dir=insights_dir,
        )

    if upload:
        from clawbench.upload import upload_result

        asyncio.run(upload_result(result))


@cli.command("dynamics-report")
@click.option(
    "--archive-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to a run cache/archive root or a single model cache directory.",
)
@click.option(
    "--model",
    default=None,
    help="Model id to select when the archive root contains multiple model directories.",
)
@click.option("--tier", type=click.Choice(["tier1", "tier2", "tier3", "tier4", "tier5"]))
@click.option("--task", "task_ids", multiple=True, help="Specific task IDs to include from the archive.")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results/offline_dynamics"),
    show_default=True,
    help="Directory where dynamics.json and plots will be written.",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Write only dynamics.json and skip plot rendering.",
)
def dynamics_report(
    archive_dir: Path,
    model: str | None,
    tier: str | None,
    task_ids: tuple[str, ...],
    output_dir: Path,
    no_plots: bool,
) -> None:
    """Generate dynamics plots and a JSON report from cached TaskRunResult archives."""
    from clawbench.dynamics_archive import load_task_runs_archive

    try:
        task_runs = load_task_runs_archive(
            archive_dir=archive_dir,
            model=model,
            task_ids=task_ids,
            tier=tier,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if not task_runs:
        raise click.ClickException(f"No cached runs found under {archive_dir}")

    report_path, plots, n_runs = _write_dynamics_report(
        task_runs,
        output_dir,
        generate_plots=not no_plots,
    )
    click.echo(f"Loaded {n_runs} cached runs across {len(task_runs)} tasks")
    click.echo(f"Dynamics report saved to {report_path}")
    click.echo(f"Saved {len(plots)} plots to {output_dir}/")


@cli.command("task-analysis")
@click.option(
    "--summaries-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing native aggregate_results.csv and per_task_results.csv.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("results/task_analysis"),
    show_default=True,
    help="Directory where task diagnostics, Markdown, and plots will be written.",
)
@click.option(
    "--no-plots",
    is_flag=True,
    help="Write CSV, JSON, and Markdown outputs without rendering SVG plots.",
)
def task_analysis(summaries_dir: Path, output_dir: Path, no_plots: bool) -> None:
    """Analyze task variance across harness, model, and reasoning cells."""
    from clawbench.task_analysis import analyze_task_matrix

    try:
        report = analyze_task_matrix(
            summaries_dir,
            output_dir,
            generate_plots=not no_plots,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Analyzed {report['source']['eligible_run_count']} eligible runs "
        f"across {len(report['datasets'])} task dataset(s)"
    )
    click.echo(f"Task analysis saved to {output_dir}")


@cli.command("trace-upload")
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Native run directory containing raw/, summaries/, and manifests/.",
)
@click.option(
    "--bucket",
    envvar="SHELLBENCH_TRACE_BUCKET",
    default="",
    help="Private destination bucket. May be set with SHELLBENCH_TRACE_BUCKET.",
)
@click.option(
    "--prefix",
    default=None,
    help="Object prefix. Defaults to the run directory name.",
)
@click.option(
    "--workers",
    type=click.IntRange(1, 32),
    default=4,
    show_default=True,
    help="Number of files to upload concurrently.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the bundle plan without authenticating or uploading.",
)
def trace_upload(
    run_dir: Path,
    bucket: str,
    prefix: str | None,
    workers: int,
    dry_run: bool,
) -> None:
    """Publish native trace archives to private S3 with hash verification."""
    from clawbench.trace_upload import trace_bundle_plan, upload_trace_bundle

    try:
        if dry_run:
            plan = trace_bundle_plan(run_dir)
            click.echo(json.dumps(plan, indent=2, sort_keys=True))
            return
        if not bucket:
            raise ValueError(
                "set --bucket or SHELLBENCH_TRACE_BUCKET before uploading"
            )
        report = upload_trace_bundle(
            run_dir,
            bucket=bucket,
            prefix=prefix,
            workers=workers,
            progress=click.echo,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(
        f"Verified {report['file_count']} trace objects "
        f"({report['uploaded_count']} uploaded, {report['skipped_count']} unchanged)"
    )
    click.echo(f"Upload manifest saved to {report['manifest_path']}")


def _write_dynamics_report(
    task_runs: dict[str, list],
    output_dir: Path,
    *,
    generate_plots: bool = True,
) -> tuple[Path, list[Path], int]:
    from clawbench.dynamics_archive import write_dynamics_report

    report_path, plots = write_dynamics_report(
        task_runs,
        output_dir,
        generate_plots=generate_plots,
    )
    n_runs = sum(len(runs) for runs in task_runs.values())
    return report_path, plots, n_runs


def _run_v05_diagnostic(
    *,
    profile_path: Path,
    result,
    task_runs: dict[str, list] | None,
    runs_per_task: int,
    insights_dir: Path,
) -> None:
    """Post-benchmark v0.5 diagnostic: fingerprint + predict + record + publish."""
    from clawbench.diagnose_cli import (
        DEFAULT_DB_PATH,
        DEFAULT_MANIFEST_DIR,
        DEFAULT_SUBMISSIONS_DIR,
        ensure_data_dirs,
        infer_registration_traces_from_manifests,
        load_manifests,
        write_submission_record,
    )
    from clawbench.diagnostic import submit_run
    from clawbench.insights import publish_insights
    from clawbench.prediction import HistoricalDatabase
    from clawbench.profile import PluginProfile

    ensure_data_dirs()

    plugin_profile = PluginProfile.from_yaml_file(profile_path)
    plugin_ids = [e.id for e in plugin_profile.plugins]
    manifests = load_manifests(DEFAULT_MANIFEST_DIR, plugin_ids)
    traces = infer_registration_traces_from_manifests(plugin_profile, manifests)
    db = HistoricalDatabase(path=DEFAULT_DB_PATH)

    # Extract per-task scores + tier map from the BenchmarkResult
    actual_per_task: dict[str, float] = {}
    tier_of: dict[str, str] = {}
    for task_stats in result.task_results:
        actual_per_task[task_stats.task_id] = float(task_stats.mean_task_score)
        if getattr(task_stats, "tier", ""):
            tier_of[task_stats.task_id] = task_stats.tier

    transcripts = _merge_task_transcripts_from_runs(task_runs or {})

    diagnostic = submit_run(
        profile=plugin_profile,
        manifests=manifests,
        db=db,
        actual_overall_score=float(result.overall_score),
        actual_per_task_scores=actual_per_task,
        traces=traces,
        transcripts=transcripts,
        tier_of=tier_of or None,
        n_runs_contributing=runs_per_task,
    )

    write_submission_record(
        DEFAULT_SUBMISSIONS_DIR,
        diagnostic.fingerprint_hash,
        diagnostic.to_dict(),
    )
    publish_insights(
        db, insights_dir, factor_report=diagnostic.factor_analysis
    )

    click.echo("")
    click.echo(diagnostic.render_text())
    click.echo(
        f"\nv0.5 diagnostic recorded for profile '{plugin_profile.name}' "
        f"(fingerprint {diagnostic.fingerprint_hash}). "
        f"Insights published to {insights_dir}."
    )


def _merge_task_transcripts_from_runs(task_runs: dict[str, list]):
    """Merge all run transcripts per task for the v0.5 utilization audit."""
    if not task_runs:
        return None
    from clawbench.schemas import Transcript

    merged: dict[str, Transcript] = {}
    for task_id, runs in task_runs.items():
        transcript = Transcript()
        for run in runs:
            transcript.messages.extend(getattr(run.transcript, "messages", []))
        if transcript.messages:
            merged[task_id] = transcript
    return merged or None


@cli.command()
@click.argument("profile", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--results",
    type=click.Path(path_type=Path),
    default=None,
    help="Optional v0.4 BenchmarkResult JSON; enables post-run analysis.",
)
@click.option(
    "--manifests",
    type=click.Path(path_type=Path),
    default=Path(".clawbench/manifests"),
    show_default=True,
    help="Directory of plugin manifest JSON files.",
)
@click.option(
    "--db",
    type=click.Path(path_type=Path),
    default=Path(".clawbench/historical/profile_runs.json"),
    show_default=True,
    help="Path to the historical profile database.",
)
@click.option(
    "--insights-dir",
    type=click.Path(path_type=Path),
    default=Path(".clawbench/insights"),
    show_default=True,
)
@click.option("--json-out", is_flag=True, help="Print diagnostic as JSON")
def diagnose(
    profile: Path,
    results: Path | None,
    manifests: Path,
    db: Path,
    insights_dir: Path,
    json_out: bool,
) -> None:
    """Run the ClawBench v0.5 Configuration Diagnostic for a plugin profile."""
    from clawbench.diagnose_cli import (
        DEFAULT_SUBMISSIONS_DIR,
        ensure_data_dirs,
        load_manifests,
        write_submission_record,
    )
    from clawbench.diagnostic import build_diagnostic, submit_run
    from clawbench.insights import publish_insights
    from clawbench.prediction import HistoricalDatabase
    from clawbench.profile import PluginProfile
    from clawbench.schemas import BenchmarkResult

    ensure_data_dirs()

    plugin_profile = PluginProfile.from_yaml_file(profile)
    plugin_ids = [e.id for e in plugin_profile.plugins]
    manifest_map = load_manifests(manifests, plugin_ids)
    database = HistoricalDatabase(path=db)

    actual_overall: float | None = None
    actual_per_task: dict[str, float] | None = None
    tier_of: dict[str, str] | None = None

    if results is not None:
        with open(results, encoding="utf-8") as handle:
            raw = json.load(handle)
        br = BenchmarkResult(**raw)
        actual_overall = float(br.overall_score)
        actual_per_task = {
            ts.task_id: float(ts.mean_task_score) for ts in br.task_results
        }
        tier_of = {
            ts.task_id: ts.tier for ts in br.task_results if getattr(ts, "tier", "")
        }

    if results is not None and actual_per_task is not None and actual_overall is not None:
        report = submit_run(
            profile=plugin_profile,
            manifests=manifest_map,
            db=database,
            actual_overall_score=actual_overall,
            actual_per_task_scores=actual_per_task,
            tier_of=tier_of,
        )
        publish_insights(database, insights_dir, factor_report=report.factor_analysis)
    else:
        report = build_diagnostic(
            profile=plugin_profile,
            manifests=manifest_map,
            db=database,
            actual_overall_score=actual_overall,
            actual_per_task_scores=actual_per_task,
            tier_of=tier_of,
        )

    write_submission_record(
        DEFAULT_SUBMISSIONS_DIR, report.fingerprint_hash, report.to_dict()
    )

    if json_out:
        click.echo(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        click.echo(report.render_text())


@cli.command()
@click.option("--release-id", required=True, help="Identifier for the hidden release snapshot")
@click.option("--tasks-dir", type=click.Path(exists=True), help="Optional source tasks directory")
@click.option("--tier", type=click.Choice(["tier1", "tier2", "tier3", "tier4", "tier5"]), help="Filter tier")
@click.option("--scenario", type=click.Choice(SCENARIO_CHOICES), help="Filter query scenario")
@click.option("--artifact-type", type=click.Choice(["file", "information", "operation", "code", "external_action", "memory", "automation", "mixed"]), help="Filter expected artifact type")
@click.option("--prompt-variant", type=click.Choice(["clear", "ambiguous"]), default="clear", show_default=True, help="Filter prompt variant support")
@click.option("--subset", multiple=True, type=click.Choice(["consensus", "hard"]), help="Filter task subset")
@click.option(
    "--capability",
    multiple=True,
    type=click.Choice(
        [
            "bugfix",
            "refactor",
            "test_authoring",
            "multifile_reasoning",
            "browser_debugging",
            "structured_output",
            "memory_continuation",
            "delegation",
            "tool_composition",
            "research_synthesis",
            "graceful_refusal",
            "spec_revision",
            "cross_repo_change",
            "automation",
        ]
    ),
    help="Filter by capability tag",
)
@click.option("--task", "-t", multiple=True, help="Specific source task IDs to include")
@click.option("--max-tasks", type=int, default=0, show_default=True, help="Limit the snapshot to the first N matching tasks")
@click.option(
    "--private-tasks-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the private release root directory",
)
@click.option(
    "--active-release-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override where the active hidden-release manifest is written",
)
@click.option("--activate/--no-activate", default=True, show_default=True, help="Set the new hidden release as active")
def build_release(
    release_id: str,
    tasks_dir: str | None,
    tier: str | None,
    scenario: str | None,
    artifact_type: str | None,
    prompt_variant: str,
    subset: tuple[str, ...],
    capability: tuple[str, ...],
    task: tuple[str, ...],
    max_tasks: int,
    private_tasks_dir: Path | None,
    active_release_path: Path | None,
    activate: bool,
) -> None:
    from clawbench.releases import build_hidden_release
    from clawbench.tasks import load_all_tasks

    tasks = load_all_tasks(
        tasks_dir=Path(tasks_dir) if tasks_dir else None,
        tier=tier,
        task_ids=list(task) if task else None,
        scenario=scenario,
        artifact_type=artifact_type,
        prompt_variant=prompt_variant,
        pool="public_dev",
        subsets=list(subset),
        capabilities=list(capability),
    )
    if not tasks:
        raise click.ClickException("No public tasks matched the requested filters.")
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    manifest = build_hidden_release(
        tasks=tasks,
        release_id=release_id,
        private_tasks_root=private_tasks_dir,
        activate=activate,
        active_release_path=active_release_path,
    )
    click.echo(
        f"Built hidden release '{manifest.release_id}' with {len(manifest.task_ids)} task(s) at "
        f"{manifest.hidden_tasks_dir}"
    )
    click.echo(f"Snapshot fingerprint: {manifest.task_snapshot_fingerprint}")
    if activate:
        click.echo("Active hidden release manifest updated.")


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path), help="JSON or JSONL file of raw trace records")
@click.option(
    "--source-kind",
    required=True,
    type=click.Choice(["hf_open_trace", "partner_trace", "internal_run", "synthetic"]),
    help="Origin of the traces being ingested",
)
@click.option(
    "--privacy-tier",
    default="public",
    show_default=True,
    type=click.Choice(["public", "private", "partner_restricted"]),
    help="Privacy level for the ingested traces",
)
@click.option("--partner-name", default="", help="Optional partner/source label")
@click.option(
    "--factory-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the local task-factory registry root",
)
@click.option("--emit-templates/--no-emit-templates", default=True, show_default=True, help="Also derive reusable task templates from the normalized seeds")
def ingest_traces(
    input_path: Path,
    source_kind: str,
    privacy_tier: str,
    partner_name: str,
    factory_root: Path | None,
    emit_templates: bool,
) -> None:
    from clawbench.task_factory import ingest_trace_file

    traces, seeds, templates = ingest_trace_file(
        input_path=input_path,
        source_kind=source_kind,
        privacy_tier=privacy_tier,
        partner_name=partner_name,
        factory_root=factory_root,
        emit_templates=emit_templates,
    )
    click.echo(
        f"Ingested {len(traces)} trace(s) -> {len(seeds)} seed(s)"
        + (f" -> {len(templates)} template(s)" if emit_templates else "")
    )
    if seeds:
        click.echo(f"First seed: {seeds[0].seed_id}  family={seeds[0].family}  scenario={seeds[0].scenario}")


@cli.command()
@click.option(
    "--kind",
    default="seeds",
    show_default=True,
    type=click.Choice(["traces", "seeds", "templates"]),
    help="Registry slice to inspect",
)
@click.option(
    "--factory-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the local task-factory registry root",
)
def list_factory(kind: str, factory_root: Path | None) -> None:
    from clawbench.task_factory import ensure_task_factory_dirs

    dirs = ensure_task_factory_dirs(factory_root)
    files = sorted(dirs[kind].glob("*.json"))
    click.echo(f"{kind}: {len(files)} file(s)")
    for path in files[:50]:
        click.echo(f"  {path.name}")


@cli.command()
@click.option("--threshold", type=float, default=0.72, show_default=True, help="Similarity threshold for reporting findings")
@click.option(
    "--factory-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the local task-factory registry root",
)
@click.option("--include-public/--no-include-public", default=True, show_default=True, help="Compare templates against public tasks")
@click.option("--include-hidden/--no-include-hidden", default=True, show_default=True, help="Compare templates against the active hidden release")
def audit_contamination(
    threshold: float,
    factory_root: Path | None,
    include_public: bool,
    include_hidden: bool,
) -> None:
    from clawbench.task_factory import audit_contamination as run_audit

    report = run_audit(
        threshold=threshold,
        factory_root=factory_root,
        include_public_tasks=include_public,
        include_hidden_tasks=include_hidden,
    )
    click.echo(
        f"Audit complete: {len(report.findings)} finding(s) at threshold >= {report.threshold:.2f} "
        f"(templates={report.template_count}, public={report.public_task_count}, hidden={report.hidden_task_count})"
    )
    click.echo(f"Report: {report.report_path}")
    for finding in report.findings[:10]:
        click.echo(
            f"  {finding.score:.2f}  {finding.left_kind}:{finding.left_id}  ~  "
            f"{finding.right_kind}:{finding.right_id}"
        )


@cli.command()
@click.option("--release-id", required=True, help="Identifier for the hidden release built from templates")
@click.option("--template-id", multiple=True, help="Specific template IDs to promote")
@click.option("--max-templates", type=int, default=0, show_default=True, help="Limit promotion to the first N matching templates")
@click.option(
    "--factory-root",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the local task-factory registry root",
)
@click.option(
    "--private-tasks-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Override the private release root directory",
)
@click.option(
    "--active-release-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Override where the active hidden-release manifest is written",
)
@click.option("--activate/--no-activate", default=True, show_default=True, help="Set the new hidden release as active")
def promote_templates(
    release_id: str,
    template_id: tuple[str, ...],
    max_templates: int,
    factory_root: Path | None,
    private_tasks_dir: Path | None,
    active_release_path: Path | None,
    activate: bool,
) -> None:
    from clawbench.task_factory import build_hidden_release_from_templates

    manifest, tasks = build_hidden_release_from_templates(
        release_id=release_id,
        template_ids=list(template_id) if template_id else None,
        max_templates=max_templates,
        factory_root=factory_root,
        private_tasks_root=private_tasks_dir,
        active_release_path=active_release_path,
        activate=activate,
    )
    click.echo(
        f"Promoted {len(tasks)} template-derived task(s) into hidden release '{manifest.release_id}' at "
        f"{manifest.hidden_tasks_dir}"
    )
    click.echo(f"Snapshot fingerprint: {manifest.task_snapshot_fingerprint}")
    if tasks:
        click.echo(f"First promoted task: {tasks[0].id}  template={tasks[0].template_id}")
    if activate:
        click.echo("Active hidden release manifest updated.")


@cli.command()
@click.option("--tasks-dir", type=click.Path(exists=True), help="Custom tasks directory")
@click.option("--scenario", type=click.Choice(SCENARIO_CHOICES), help="Filter query scenario")
@click.option("--prompt-variant", type=click.Choice(["clear", "ambiguous"]), help="Filter prompt variant support")
@click.option("--pool", type=click.Choice(["public_dev", "official_hidden"]), help="Filter task pool")
@click.option("--subset", multiple=True, type=click.Choice(["consensus", "hard"]), help="Filter task subset")
def list_tasks(tasks_dir: str | None, scenario: str | None, prompt_variant: str | None, pool: str | None, subset: tuple[str, ...]) -> None:
    from clawbench.tasks import load_all_tasks

    tasks = load_all_tasks(
        tasks_dir=Path(tasks_dir) if tasks_dir else None,
        scenario=scenario,
        prompt_variant=prompt_variant,
        pool=pool,
        subsets=list(subset),
    )
    click.echo(f"\n{'ID':<34} {'Tier':<7} {'Scene':<24} {'Prompt':<10} {'Pool':<15} {'Family':<12}")
    click.echo("-" * 116)
    for task in tasks:
        click.echo(
            f"  {task.id:<32} {task.tier.value:<7} "
            f"{(task.scenario.value if task.scenario else '-'): <24} "
            f"{'/'.join(variant.value for variant in task.prompt_variants):<10} "
            f"{task.pool.value:<15} {task.family.value:<12}"
        )


@cli.command()
@click.argument("result_file", type=click.Path(exists=True))
def show(result_file: str) -> None:
    from rich.console import Console
    from clawbench.schemas import BenchmarkResult

    with open(result_file, encoding="utf-8") as handle:
        data = json.load(handle)
    result = BenchmarkResult(**data)

    console = Console()
    console.print(f"\n[bold]Model:[/] {result.model}")
    console.print(
        f"[bold]Score:[/] {result.overall_score:.3f} "
        f"(CI: {result.overall_ci_lower:.3f}-{result.overall_ci_upper:.3f})"
    )
    console.print(
        f"  [green]Completion: {result.overall_completion:.3f}[/]  "
        f"[blue]Trajectory: {result.overall_trajectory:.3f}[/]  "
        f"[yellow]Behavior: {result.overall_behavior:.3f}[/]  "
        f"[magenta]Reliability: {result.overall_reliability:.3f}[/]"
    )
    if result.judge_model:
        console.print(
            f"  [magenta]Judge: {result.overall_judge_score:.3f}[/]  "
            f"Confidence: {result.overall_judge_confidence:.3f}  "
            f"Pass rate: {result.overall_judge_pass_rate:.0%}  "
            f"Coverage: {result.judge_task_coverage:.0%}"
        )
    console.print(
        f"  Weighted query: {result.overall_weighted_query_score:.3f}  "
        f"Clear prompt: {result.clear_prompt_score:.3f}  "
        f"Ambiguous prompt: {result.ambiguous_prompt_score:.3f}"
    )
    console.print(
        f"  Latency p50={result.overall_median_latency_ms:.0f}ms "
        f"p95={result.overall_p95_latency_ms:.0f}ms  "
        f"Tokens/pass={result.overall_tokens_per_pass:.0f}  "
        f"Cost/pass=${result.overall_cost_per_pass:.4f}"
    )
    console.print(
        f"  Hard subset: {result.hard_subset_score:.3f}  "
        f"Consensus subset: {result.consensus_subset_score:.3f}"
    )
    console.print(f"  [bold]pass^k reliability: {result.overall_pass_hat_k:.0%}[/]\n")

    for task in result.task_results:
        color = "green" if task.mean_task_score >= 0.7 else "yellow" if task.mean_task_score >= 0.4 else "red"
        top_failure = max(task.failure_mode_counts.items(), key=lambda item: item[1])[0] if task.failure_mode_counts else "-"
        judge_value = f"{task.mean_judge_score:.2f}" if task.judged_runs > 0 else "-"
        console.print(
            f"  [{color}]{task.mean_task_score:.3f}[/]  {task.task_id}  "
            f"scene={task.scenario or '-'} prompt={task.prompt_variant} "
            f"run={task.mean_run_score:.2f} comp={task.mean_completion_score:.2f} "
            f"traj={task.mean_trajectory_score:.2f} beh={task.mean_behavior_score:.2f} "
            f"judge={judge_value} "
            f"rel={task.reliability_score:.2f} delivery={task.delivery_outcome_counts} "
            f"tok/pass={task.tokens_per_pass:.0f} p50={task.median_duration_ms:.0f}ms fail={top_failure}"
        )


def _run_dynamics_analysis(
    task_runs: dict[str, list],
    result_path: str,
) -> None:
    """Compute stratified dynamics from raw TaskRunResult objects."""
    run_stem = Path(result_path).stem
    dyn_dir = Path(result_path).parent / f"{run_stem}_dynamics"
    try:
        dyn_path, plots, n_runs = _write_dynamics_report(task_runs, dyn_dir)
    except ValueError as exc:
        click.echo(str(exc))
        return

    click.echo(f"\n[dynamics] Analysed {n_runs} cached runs")
    click.echo(f"  Dynamics report saved to {dyn_path}")
    click.echo(f"  Saved {len(plots)} plots to {dyn_dir}/")


def main() -> None:
    cli()
