"""Benchmark harness for ClawBench."""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from clawbench import __version__
from clawbench.ablation import build_ablation_profile
from clawbench.client import GatewayClient, GatewayConfig
from clawbench.judge import build_task_review_evidence
from clawbench.releases import compute_task_snapshot_fingerprint, load_active_release
from clawbench.review_report import persist_run_review
from clawbench.run_review import RUBRIC_VERSION, unreviewed_result
from clawbench.schemas import (
    BenchmarkResult,
    DeliveryOutcome,
    ScenarioResult,
    TaskDefinition,
    TaskRunResult,
    TaskStats,
    TierResult,
    Transcript,
)
from clawbench.scorer import classify_error_failure_mode, score_task_run
from clawbench.session_labels import unique_session_label
from clawbench.services import build_runtime_values, start_background_services, stop_background_services
from clawbench.simulated_user import UserSimulator
from clawbench.stats import bootstrap_ci, summarize_task_runs
from clawbench.tasks import get_assets_dir, load_all_tasks

logger = logging.getLogger(__name__)
console = Console()

KNOWN_ADAPTERS = ("openclaw", "hermes", "codex", "claude-code")
EXECUTABLE_ADAPTERS = {"openclaw"}
RUN_CACHE_SCHEMA_VERSION = 3


class _NullCtx:
    """A no-op async context manager used to skip the browser semaphore
    for non-browser tasks without branching the call site twice.
    """

    async def __aenter__(self) -> "_NullCtx":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class BenchmarkHarness:
    def __init__(
        self,
        *,
        gateway_config: GatewayConfig,
        model: str,
        provider: str = "",
        runs_per_task: int = 5,
        tier: str | None = None,
        task_ids: list[str] | None = None,
        scenario: str | None = None,
        artifact_type: str | None = None,
        prompt_variant: str = "clear",
        judge_model: str = "",
        pool: str | None = None,
        subsets: list[str] | None = None,
        capabilities: list[str] | None = None,
        official_only: bool = False,
        randomize_order: bool = True,
        tasks_dir: Path | None = None,
        prepare_run: Callable[[TaskDefinition, int], Awaitable[None]] | None = None,
        progress_callback: Callable[[TaskDefinition, int], Awaitable[None]] | None = None,
        print_report: bool = True,
        quiet: bool = False,
        concurrency: int = 1,
        browser_concurrency: int = 1,
        adapter: str = "openclaw",
        judge_affects_score: bool = False,
        tool_profile_name: str | None = None,
        enabled_toolsets: list[str] | None = None,
        disabled_toolsets: list[str] | None = None,
    ) -> None:
        self.gateway_config = gateway_config
        self.model = model
        self.provider = provider or (model.split("/")[0] if "/" in model else "")
        self.runs_per_task = max(1, runs_per_task)
        self.tier = tier
        self.task_ids = task_ids
        self.scenario = scenario
        self.artifact_type = artifact_type
        self.prompt_variant = prompt_variant
        self.judge_model = judge_model
        self.judge_affects_score = judge_affects_score
        self.pool = pool
        self.subsets = subsets or []
        self.capabilities = capabilities or []
        self.official_only = official_only
        self.randomize_order = randomize_order
        self.tasks_dir = tasks_dir
        self.prepare_run = prepare_run
        self.progress_callback = progress_callback
        self.print_report = print_report
        self.quiet = quiet
        self.concurrency = max(1, int(concurrency))
        self.browser_concurrency = max(1, int(browser_concurrency))
        self.adapter = adapter
        self.tool_profile_name = tool_profile_name
        self.enabled_toolsets = enabled_toolsets or []
        self.disabled_toolsets = disabled_toolsets or []
        self.repo_root = Path(__file__).parent.parent
        self.last_task_runs: dict[str, list[TaskRunResult]] = {}

    async def run(self) -> BenchmarkResult:
        if self.adapter not in KNOWN_ADAPTERS:
            raise ValueError(
                f"Unknown adapter '{self.adapter}'. Known adapters: {', '.join(KNOWN_ADAPTERS)}"
            )
        if self.adapter not in EXECUTABLE_ADAPTERS:
            raise ValueError(
                f"Adapter '{self.adapter}' is registered as a target but is not yet wired "
                "into the end-to-end scoring harness. Use 'openclaw' for executable runs."
            )

        tasks = load_all_tasks(
            tasks_dir=self.tasks_dir,
            tier=self.tier,
            task_ids=self.task_ids,
            scenario=self.scenario,
            artifact_type=self.artifact_type,
            prompt_variant=self.prompt_variant,
            pool=self.pool,
            subsets=self.subsets,
            capabilities=self.capabilities,
            official_only=self.official_only,
        )
        if not tasks:
            raise ValueError("No tasks to run")

        if self.randomize_order:
            import random

            random.shuffle(tasks)

        if not self.quiet:
            console.print(f"\n[bold]ClawBench v{__version__}[/bold] — {len(tasks)} tasks x {self.runs_per_task} runs")
            console.print(f"Model: [cyan]{self.model}[/cyan]")
            console.print(f"Adapter: [cyan]{self.adapter}[/cyan]")
            if self.judge_model:
                console.print(f"Advisory judge: [magenta]{self.judge_model}[/magenta]")
            mode = "serial" if self.concurrency == 1 else f"parallel(concurrency={self.concurrency}, browser={self.browser_concurrency})"
            console.print(f"Execution: [bright_blue]{mode}[/]")
            console.print(
                "Axes: [green]Completion[/] + [blue]Trajectory[/] + [yellow]Behavior[/] + [magenta]Reliability[/]\n"
            )

        wall_start = time.monotonic()
        all_results = await self._execute_runs(tasks)
        wall_seconds = time.monotonic() - wall_start

        if not self.quiet:
            total_runs = sum(len(runs) for runs in all_results.values())
            mean_run = (wall_seconds / total_runs) if total_runs else 0.0
            console.print(
                f"\n[dim]Wall time: {wall_seconds:.1f}s across {total_runs} runs "
                f"({mean_run:.1f}s avg, concurrency={self.concurrency})[/dim]"
            )

        self.last_task_runs = all_results
        return self._aggregate(tasks, all_results)

    async def _execute_runs(
        self,
        tasks: list[TaskDefinition],
    ) -> dict[str, list[TaskRunResult]]:
        """Run every (task, run_index) work item, serial or parallel.

        Browser tasks are gated by a separate semaphore so the Chromium
        port collision can't ever occur, regardless of concurrency level.
        Non-browser tasks share the global semaphore.
        """
        global_sem = asyncio.Semaphore(self.concurrency)
        browser_sem = asyncio.Semaphore(self.browser_concurrency)
        print_lock = asyncio.Lock()

        # Build the flat work list. Browser tasks float to the front so they
        # don't end up sitting in the queue while non-browser slots churn.
        work_items: list[tuple[TaskDefinition, int]] = []
        browser_items: list[tuple[TaskDefinition, int]] = []
        non_browser_items: list[tuple[TaskDefinition, int]] = []
        for task in tasks:
            for run_index in range(self.runs_per_task):
                item = (task, run_index)
                if task.family.value == "browser":
                    browser_items.append(item)
                else:
                    non_browser_items.append(item)
        work_items = browser_items + non_browser_items

        results_by_task: dict[str, list[TaskRunResult | None]] = {
            task.id: [None] * self.runs_per_task for task in tasks
        }
        completed = 0
        total = len(work_items)

        async def run_one(task: TaskDefinition, run_index: int) -> None:
            nonlocal completed
            is_browser = task.family.value == "browser"

            async with global_sem:
                # Browser tasks additionally need the browser-only semaphore
                # so a parallel non-browser run can never collide with the
                # Chromium-using run on the gateway's fixed browser port.
                browser_ctx = browser_sem if is_browser else _NullCtx()
                async with browser_ctx:
                    if self.prepare_run is not None:
                        await self.prepare_run(task, run_index)
                    if self.progress_callback is not None:
                        await self.progress_callback(task, run_index)
                    result = await self._run_single(task, run_index)
                    results_by_task[task.id][run_index] = result

                    completed += 1
                    if not self.quiet:
                        async with print_lock:
                            self._print_run_result(task, run_index, result, completed, total)

        await asyncio.gather(*(run_one(task, idx) for task, idx in work_items))

        # Convert from list-with-Nones to plain list, preserving run order
        return {
            task.id: [r for r in results_by_task[task.id] if r is not None]
            for task in tasks
        }

    def _print_run_result(
        self,
        task: TaskDefinition,
        run_index: int,
        result: TaskRunResult,
        completed: int,
        total: int,
    ) -> None:
        passed = self._is_passing_run(task, result)
        marker = "[green]+" if passed else "[yellow]~" if result.run_score >= 0.4 else "[red]-"
        failure_suffix = f" [red]{result.failure_mode.value}[/]" if result.failure_mode else ""
        console.print(
            f"[dim][{completed}/{total}][/dim] [bold]{task.id}[/bold] "
            f"({task.tier.value}/{task.family.value}) run {run_index + 1}: "
            f"{marker} {result.run_score:.2f}[/] "
            f"[green]C={result.completion_result.score:.2f}[/] "
            f"[blue]T={result.trajectory_result.score:.2f}[/] "
            f"[yellow]B={result.behavior_result.score:.2f}[/]"
            f"{f' [magenta]J={result.judge_result.score:.2f}[/]' if result.judge_result.enabled and not result.judge_result.error else ''}"
            f"{failure_suffix}"
        )
        if result.judge_result.error:
            console.print(f"    [yellow]? judge unavailable: {result.judge_result.error}[/]")
        for failure in result.completion_result.failed_assertions[:2]:
            console.print(f"    [red]! {failure}[/]")
        for failure in result.trajectory_result.forbidden_violations[:2]:
            console.print(f"    [red]! {failure}[/]")

    async def _run_single(self, task: TaskDefinition, run_index: int) -> TaskRunResult:
        # Per-turn timeout cap: prevents a single send_and_wait from burning the entire task
        # timeout (often 300-600s). Default 180s is enough for any reasonable single-turn
        # response and fails fast on stuck models. Override with env var if needed.
        per_turn_cap = float(os.environ.get("CLAWBENCH_PER_TURN_TIMEOUT_SECONDS", "180"))
        # Per-run hard budget: total wall time a single (task, run) is allowed to consume.
        # Default 300s (5 min) bounds the worst case to 5min * 120 = 10h/model if fully
        # serial, and <3h/model at lanes=4. Env override available for longer slower models.
        per_run_budget = float(os.environ.get("CLAWBENCH_PER_RUN_BUDGET_SECONDS", "300"))

        # Per-run result cache: allows a failed job to resume from previously completed
        # (task, run) pairs on resubmit. Keyed by model + task + run_index so the same
        # model's runs are reused, but different models stay isolated. The cache is
        # written AFTER successful score_task_run and read at the start of this method.
        # Set CLAWBENCH_RUN_CACHE_DIR="" to disable.
        cache_dir_env = os.environ.get("CLAWBENCH_RUN_CACHE_DIR", "/data/run_cache")
        cache_path: Path | None = None
        if cache_dir_env:
            cache_path = self._run_cache_path(Path(cache_dir_env), task, run_index)
            if cache_path.exists():
                try:
                    cached = TaskRunResult.model_validate_json(cache_path.read_text(encoding="utf-8"))
                    cached.run_index = run_index
                    logger.info(
                        "TIMING %s/run%s total=cached score=%.2f C=%.2f T=%.2f B=%.2f J=%.2f  (resumed from %s)",
                        task.id, run_index,
                        cached.run_score,
                        cached.completion_result.score,
                        cached.trajectory_result.score,
                        cached.behavior_result.score,
                        cached.judge_result.score if cached.judge_result.enabled else 0.0,
                        cache_path,
                    )
                    return cached
                except Exception as exc:
                    logger.warning("Cache load failed for %s/run%s: %s (will re-run)", task.id, run_index, exc)

        workspace = self._create_run_workspace(task, run_index)
        services = []
        session_keys: list[str] = []
        agent_id: str | None = None
        transcript = Transcript()
        user_turns: list[dict[str, Any]] = []
        review_notes: list[str] = []
        execution_status = "unknown"

        # Per-phase timings so we can see where slow runs are spending their wall time.
        timings: dict[str, float] = {}

        def _tick(label: str, since: float) -> float:
            now = time.monotonic()
            timings[label] = round(now - since, 2)
            return now

        t_run_start = time.monotonic()
        try:
            t_phase = t_run_start
            self._setup_workspace(task, workspace)
            t_phase = _tick("workspace_setup", t_phase)

            runtime_values = build_runtime_values(
                workspace=workspace,
                repo_root=self.repo_root,
                extra={"task_id": task.id, "model": self.model, "prompt_variant": self.prompt_variant},
            )
            services, runtime_values = await start_background_services(
                task.setup.background_services,
                workspace=workspace,
                repo_root=self.repo_root,
                runtime_values=runtime_values,
            )
            t_phase = _tick("bg_services_start", t_phase)

            start_ms = _now_ms()

            async with GatewayClient(self.gateway_config) as client:
                t_phase = _tick("gateway_connect", t_phase)
                agent_id = await self._create_run_agent(
                    client,
                    task=task,
                    workspace=workspace,
                    run_index=run_index,
                )
                t_phase = _tick("agent_create", t_phase)
                for phase_index, phase in enumerate(task.normalized_phases()):
                    phase_message_start = len(transcript.messages)
                    session_key = await client.create_session(
                        model=self.model,
                        agent_id=agent_id,
                        label=unique_session_label(
                            f"clawbench-{task.id}-run{run_index}-phase{phase_index}"
                        ),
                    )
                    session_keys.append(session_key)
                    await client.subscribe(session_key)
                    if task.family.value == "browser":
                        await self._assert_browser_support(client, session_key)
                    t_phase = _tick(f"phase{phase_index}_session_setup", t_phase)

                    simulator = UserSimulator(
                        phase.user,
                        runtime_values,
                        prompt_variant=self.prompt_variant,
                    )
                    turn_index = 0
                    phase_raw_timeout = float(phase.timeout_seconds or task.timeout_seconds)
                    turn_timeout = min(phase_raw_timeout, per_turn_cap)
                    while not simulator.is_done:
                        # Enforce per-run budget: if we've already burned our whole budget
                        # on previous turns of this run, bail out and score whatever we have.
                        elapsed = time.monotonic() - t_run_start
                        if elapsed >= per_run_budget:
                            execution_status = "incomplete"
                            transcript.stop_reason = "timeout"
                            review_notes.append("The run budget ended before the user simulator completed.")
                            logger.warning(
                                "Run %s/%s hit per-run budget (%.0fs); stopping user simulator",
                                task.id,
                                run_index,
                                per_run_budget,
                            )
                            break
                        remaining_budget = per_run_budget - elapsed
                        effective_timeout = min(turn_timeout, remaining_budget)

                        user_message = await simulator.next_message(transcript)
                        if user_message is None:
                            break
                        user_turns.append({
                            "before_message_index": len(transcript.messages),
                            "phase_index": phase_index,
                            "turn_index": turn_index,
                            "text": user_message,
                        })
                        t_turn_start = time.monotonic()
                        try:
                            phase_transcript = await client.send_and_wait(
                                session_key,
                                user_message,
                                timeout=effective_timeout,
                            )
                        except (Exception, asyncio.CancelledError):
                            # Capture live session evidence before the connection/session
                            # is closed. Prior phases and turns remain in the transcript.
                            try:
                                partial = await asyncio.wait_for(
                                    client.get_session_messages(session_key), timeout=10,
                                )
                                previous = transcript.messages[phase_message_start:]
                                if partial[:len(previous)] == previous:
                                    transcript.messages.extend(partial[len(previous):])
                                else:
                                    transcript.messages.extend(partial)
                                    if previous and partial:
                                        review_notes.append(
                                            "Recovered session history may overlap earlier captured turns."
                                        )
                            except Exception as capture_error:
                                review_notes.append(f"Partial session history unavailable: {capture_error}")
                            raise
                        timings[f"phase{phase_index}_turn{turn_index}"] = round(
                            time.monotonic() - t_turn_start, 2
                        )
                        transcript.messages.extend(phase_transcript.messages)
                        if phase_transcript.stop_reason == "completed":
                            if execution_status != "incomplete":
                                execution_status = "completed"
                                transcript.stop_reason = "completed"
                        else:
                            execution_status = "incomplete"
                            transcript.stop_reason = phase_transcript.stop_reason
                            review_notes.append(
                                f"Phase {phase_index} turn {turn_index} stop reason: "
                                f"{phase_transcript.stop_reason}."
                            )
                        turn_index += 1
                    t_phase = _tick(f"phase{phase_index}_total", t_phase)

                duration_ms = _now_ms() - start_ms
                last_session_key = session_keys[-1] if session_keys else ""
                t_score_start = time.monotonic()
                result = await score_task_run(
                    task=task,
                    transcript=transcript,
                    workspace=workspace,
                    client=client,
                    session_key=last_session_key,
                    agent_id=agent_id,
                    duration_ms=duration_ms,
                    runtime_values=runtime_values,
                    judge_model=self.judge_model,
                    judge_affects_score=self.judge_affects_score,
                    run_id=f"{task.id}/{workspace.name}",
                    user_turns=user_turns,
                    execution_status=execution_status,
                    coverage_notes=review_notes,
                )
                timings["score"] = round(time.monotonic() - t_score_start, 2)
                timings["total"] = round(time.monotonic() - t_run_start, 2)
                result.run_index = run_index
                self._persist_review(result, workspace)

                # Write per-run cache so a future resume of this job can skip this run.
                if cache_path is not None:
                    try:
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path = cache_path.with_suffix(".json.tmp")
                        tmp_path.write_text(
                            result.model_dump_json(indent=2), encoding="utf-8"
                        )
                        tmp_path.replace(cache_path)
                    except Exception as exc:
                        logger.warning("Cache write failed for %s/run%s: %s", task.id, run_index, exc)

                logger.info(
                    "TIMING %s/run%s total=%.1fs score=%.2f C=%.2f T=%.2f B=%.2f J=%.2f  %s",
                    task.id,
                    run_index,
                    timings["total"],
                    result.run_score,
                    result.completion_result.score,
                    result.trajectory_result.score,
                    result.behavior_result.score,
                    result.judge_result.score if (result.judge_result.enabled and not result.judge_result.error) else 0.0,
                    " ".join(f"{k}={v}s" for k, v in timings.items() if k != "total"),
                )
                return result
        except (Exception, asyncio.CancelledError) as exc:
            logger.exception("Run %s/%s failed", task.id, run_index)
            cancelled = isinstance(exc, asyncio.CancelledError)
            error_message = str(exc) or ("Run cancelled" if cancelled else type(exc).__name__)
            transcript.stop_reason = "cancelled" if cancelled else "error"
            evidence = build_task_review_evidence(
                task=task,
                transcript=transcript,
                workspace=workspace,
                run_id=f"{task.id}/{workspace.name}",
                user_turns=user_turns,
                execution_status="incomplete",
                coverage_notes=[*review_notes, f"Run execution failed: {error_message}"],
            )
            result = TaskRunResult(
                task_id=task.id,
                tier=task.tier.value,
                family=task.family.value,
                scenario=task.scenario.value if task.scenario else "",
                subscenario=task.subscenario,
                artifact_type=task.artifact_type.value if task.artifact_type else "",
                prompt_variant=self.prompt_variant,
                query_difficulty=task.query_difficulty.value if task.query_difficulty else "",
                query_weight=task.query_weight,
                pool=task.pool.value,
                subsets=[subset.value for subset in task.subsets],
                capabilities=[capability.value for capability in task.capabilities],
                variant_group=task.variant_group,
                variant_id=task.variant_id,
                template_id=task.template_id,
                release_id=task.release_id,
                source_kind=task.source_kind,
                privacy_tier=task.privacy_tier,
                contamination_risk=task.contamination_risk,
                freshness_epoch=task.freshness_epoch,
                similarity_hash=task.similarity_hash,
                official=task.official,
                run_index=run_index,
                run_score=0.0,
                transcript=transcript,
                duration_ms=int((time.monotonic() - t_run_start) * 1000),
                token_usage=transcript.total_usage,
                review_evidence=evidence,
                run_review=unreviewed_result(
                    evidence, model=self.judge_model, error=f"Run execution failed: {error_message}",
                ),
                delivery_outcome=DeliveryOutcome.FAIL,
                failure_mode=classify_error_failure_mode(task, error_message),
                error=error_message,
            )
            self._persist_review(result, workspace)
            if cancelled:
                raise
            return result
        finally:
            await stop_background_services(services)
            if session_keys or agent_id:
                try:
                    async with GatewayClient(self.gateway_config) as cleanup_client:
                        for session_key in session_keys:
                            await cleanup_client.delete_session(session_key)
                        if agent_id:
                            await cleanup_client.delete_agent(agent_id, delete_files=False)
                except Exception as exc:
                    logger.warning("Session cleanup failed: %s", exc)
            if os.environ.get("CLAWBENCH_KEEP_WORKSPACES") != "1":
                shutil.rmtree(workspace, ignore_errors=True)

    def _persist_review(self, result: TaskRunResult, workspace: Path) -> None:
        """Write outside the disposable actor workspace before it is removed."""
        if result.review_evidence is None:
            return
        configured_root = os.environ.get("CLAWBENCH_REVIEW_DIR")
        review_root = Path(configured_root) if configured_root else workspace.parent / "_reviews"
        directory = review_root / workspace.name
        try:
            persist_run_review(directory, result.review_evidence, result.run_review)
            result.review_artifact_dir = str(directory.resolve())
        except Exception as exc:
            # The result still embeds the exact evidence and review for recovery.
            logger.warning("Review persistence failed for %s: %s", result.task_id, exc)

    async def _create_run_agent(
        self,
        client: GatewayClient,
        *,
        task: TaskDefinition,
        workspace: Path,
        run_index: int,
    ) -> str:
        agent_name = f"clawbench-{task.id}-run-{run_index}-{uuid.uuid4().hex[:6]}"
        return await client.create_agent(name=agent_name, workspace=str(workspace))

    def _create_run_workspace(self, task: TaskDefinition, run_index: int) -> Path:
        state_dir = Path(os.environ.get("OPENCLAW_STATE_DIR", os.path.expanduser("~/.openclaw")))
        workspace_root = state_dir / "workspace-clawbench" / task.id
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace = workspace_root / f"run-{run_index}-{uuid.uuid4().hex[:8]}"
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def _setup_workspace(self, task: TaskDefinition, workspace: Path) -> None:
        assets_dir = get_assets_dir()

        for pack in task.setup.asset_packs:
            source = assets_dir / pack
            if not source.exists():
                raise FileNotFoundError(f"Missing asset pack {pack}")
            self._copy_into_workspace(source, workspace)

        for rel_path in task.setup.workspace_files:
            source = assets_dir / rel_path
            if not source.exists():
                raise FileNotFoundError(f"Missing workspace asset {rel_path}")
            target = workspace / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def _copy_into_workspace(self, source: Path, workspace: Path) -> None:
        if source.is_file():
            target = workspace / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            return
        for item in source.rglob("*"):
            relative = item.relative_to(source)
            target = workspace / relative
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)

    def _run_cache_path(self, cache_root: Path, task: TaskDefinition, run_index: int) -> Path:
        identity = {
            "schema": RUN_CACHE_SCHEMA_VERSION,
            "model": self.model,
            "adapter": self.adapter,
            "prompt_variant": self.prompt_variant,
            "judge_model": self.judge_model,
            "review_api_url": os.environ.get("AGENT_JUDGE_API_URL", "").strip(),
            "review_rubric_version": RUBRIC_VERSION,
            "judge_affects_score": self.judge_affects_score,
            "tool_profile_name": self.tool_profile_name,
            "enabled_toolsets": self.enabled_toolsets,
            "disabled_toolsets": self.disabled_toolsets,
            "benchmark_version": __version__,
            "task_fingerprint": _task_definition_fingerprint(task),
        }
        scope = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        ).hexdigest()[:16]
        return (
            cache_root
            / _safe_cache_component(self.model)
            / f"v{RUN_CACHE_SCHEMA_VERSION}-{scope}"
            / _safe_cache_component(task.id)
            / f"run{run_index}.json"
        )

    async def _assert_browser_support(self, client: GatewayClient, session_key: str) -> None:
        inventory = await client.get_effective_tools(session_key)
        tool_ids = {
            str(tool.get("id", ""))
            for group in inventory.get("groups", [])
            for tool in group.get("tools", [])
        }
        if "browser" not in tool_ids:
            raise RuntimeError("Browser tasks require the browser tool, but it is not available in this gateway.")

    def _aggregate(
        self,
        tasks: list[TaskDefinition],
        all_results: dict[str, list[TaskRunResult]],
    ) -> BenchmarkResult:
        task_stats: list[TaskStats] = []
        for task in tasks:
            runs = all_results.get(task.id, [])
            run_scores = [result.run_score for result in runs]
            completion_scores = [result.completion_result.score for result in runs]
            trajectory_scores = [result.trajectory_result.score for result in runs]
            behavior_scores = [result.behavior_result.score for result in runs]
            judged_runs = [
                result
                for result in runs
                if result.judge_result.enabled and not result.judge_result.error
            ]
            judge_scores = [result.judge_result.score for result in judged_runs]
            judge_confidences = [result.judge_result.confidence for result in judged_runs]
            durations = [result.duration_ms for result in runs if result.duration_ms > 0]
            input_tokens = [result.efficiency_result.input_tokens for result in runs]
            output_tokens = [result.efficiency_result.output_tokens for result in runs]
            reasoning_tokens = [result.efficiency_result.reasoning_tokens for result in runs]
            total_tokens = [result.efficiency_result.total_tokens for result in runs]
            cost_values = [result.efficiency_result.estimated_cost_usd for result in runs]
            pass_flags = [self._is_passing_run(task, result) for result in runs]
            passing_runs = [result for result, passed in zip(runs, pass_flags, strict=False) if passed]
            failure_mode_counts = _count_values(
                result.failure_mode.value for result in runs if result.failure_mode is not None
            )
            delivery_outcome_counts = _count_values(result.delivery_outcome.value for result in runs)
            judge_error_count = sum(1 for result in runs if result.judge_result.enabled and result.judge_result.error)

            summary = summarize_task_runs(
                run_scores,
                pass_threshold=task.pass_threshold,
                pass_flags=pass_flags,
            )
            task_stats.append(
                TaskStats(
                    task_id=task.id,
                    tier=task.tier.value,
                    family=task.family.value,
                    scenario=task.scenario.value if task.scenario else "",
                    subscenario=task.subscenario,
                    artifact_type=task.artifact_type.value if task.artifact_type else "",
                    prompt_variant=self.prompt_variant,
                    query_difficulty=task.query_difficulty.value if task.query_difficulty else "",
                    query_weight=task.query_weight,
                    pool=task.pool.value,
                    subsets=[subset.value for subset in task.subsets],
                    capabilities=[capability.value for capability in task.capabilities],
                    variant_group=task.variant_group,
                    variant_id=task.variant_id,
                    template_id=task.template_id,
                    release_id=task.release_id,
                    source_kind=task.source_kind,
                    privacy_tier=task.privacy_tier,
                    contamination_risk=task.contamination_risk,
                    freshness_epoch=task.freshness_epoch,
                    similarity_hash=task.similarity_hash,
                    official=task.official,
                    runs=len(runs),
                    mean_completion_score=_mean(completion_scores),
                    mean_trajectory_score=_mean(trajectory_scores),
                    mean_behavior_score=_mean(behavior_scores),
                    mean_judge_score=_mean(judge_scores),
                    mean_judge_confidence=_mean(judge_confidences),
                    judge_pass_rate=(
                        sum(1 for result in judged_runs if result.judge_result.passed) / len(judged_runs)
                        if judged_runs
                        else 0.0
                    ),
                    judged_runs=len(judged_runs),
                    judge_error_count=judge_error_count,
                    mean_run_score=summary.mean,
                    reliability_score=summary.reliability_score,
                    variance_score=summary.variance_score,
                    mean_task_score=summary.task_score,
                    stddev=summary.stddev,
                    min_score=summary.min_score,
                    max_score=summary.max_score,
                    pass_at_1=summary.pass_at_1,
                    pass_rate=summary.pass_rate,
                    pass_hat_k=summary.pass_hat_k,
                    scores=run_scores,
                    mean_duration_ms=_mean(durations),
                    median_duration_ms=_percentile(durations, 50),
                    p95_duration_ms=_percentile(durations, 95),
                    mean_input_tokens=_mean(input_tokens),
                    mean_output_tokens=_mean(output_tokens),
                    mean_reasoning_tokens=_mean(reasoning_tokens),
                    mean_total_tokens=_mean(total_tokens),
                    mean_cost_usd=_mean(cost_values),
                    tokens_per_pass=(
                        sum(run.efficiency_result.total_tokens for run in passing_runs) / len(passing_runs)
                        if passing_runs
                        else 0.0
                    ),
                    cost_per_pass=(
                        sum(run.efficiency_result.estimated_cost_usd for run in passing_runs) / len(passing_runs)
                        if passing_runs
                        else 0.0
                    ),
                    worst_of_n=summary.worst_of_n,
                    delivery_outcome_counts=delivery_outcome_counts,
                    failure_mode_counts=failure_mode_counts,
                    high_variance=summary.high_variance,
                )
            )

        return self.compose_result_from_task_stats(task_stats, tasks=tasks)

    def compose_result_from_task_stats(
        self,
        task_stats: list[TaskStats],
        *,
        tasks: list[TaskDefinition],
        environment_extra: dict[str, Any] | None = None,
        print_report: bool | None = None,
    ) -> BenchmarkResult:
        tier_results: list[TierResult] = []
        for tier in sorted({task.tier.value for task in tasks}):
            current = [stat for stat in task_stats if stat.tier == tier]
            ci = bootstrap_ci([stat.mean_task_score for stat in current])
            tier_results.append(
                TierResult(
                    tier=tier,
                    mean_task_score=ci.mean,
                    mean_completion=_mean([stat.mean_completion_score for stat in current]),
                    mean_trajectory=_mean([stat.mean_trajectory_score for stat in current]),
                    mean_behavior=_mean([stat.mean_behavior_score for stat in current]),
                    mean_judge=_mean([stat.mean_judge_score for stat in current if stat.judged_runs > 0]),
                    mean_reliability=_mean([stat.reliability_score for stat in current]),
                    ci_lower=ci.lower,
                    ci_upper=ci.upper,
                    pass_hat_k_rate=_mean([1.0 if stat.pass_hat_k else 0.0 for stat in current]),
                    task_stats=current,
                )
            )

        scenario_results: list[ScenarioResult] = []
        for scenario in sorted({stat.scenario for stat in task_stats if stat.scenario}):
            current = [stat for stat in task_stats if stat.scenario == scenario]
            total_weight = sum(stat.query_weight for stat in current)
            weighted_score = (
                sum(stat.mean_task_score * stat.query_weight for stat in current) / total_weight
                if total_weight
                else _mean([stat.mean_task_score for stat in current])
            )
            scenario_results.append(
                ScenarioResult(
                    scenario=scenario,
                    mean_task_score=_mean([stat.mean_task_score for stat in current]),
                    weighted_score=weighted_score,
                    mean_completion=_mean([stat.mean_completion_score for stat in current]),
                    mean_trajectory=_mean([stat.mean_trajectory_score for stat in current]),
                    mean_behavior=_mean([stat.mean_behavior_score for stat in current]),
                    mean_judge=_mean([stat.mean_judge_score for stat in current if stat.judged_runs > 0]),
                    mean_reliability=_mean([stat.reliability_score for stat in current]),
                    pass_hat_k_rate=_mean([1.0 if stat.pass_hat_k else 0.0 for stat in current]),
                    total_weight=total_weight,
                    task_stats=current,
                )
            )

        overall_ci = bootstrap_ci([stat.mean_task_score for stat in task_stats])
        total_weight = sum(stat.query_weight for stat in task_stats)
        overall_failure_mode_counts = _count_values(
            failure_mode
            for stat in task_stats
            for failure_mode, count in stat.failure_mode_counts.items()
            for _ in range(count)
        )
        overall_delivery_outcome_counts = _count_values(
            outcome
            for stat in task_stats
            for outcome, count in stat.delivery_outcome_counts.items()
            for _ in range(count)
        )
        active_release = load_active_release()
        ablation_profile = build_ablation_profile(
            model=self.model,
            adapter=self.adapter,
            prompt_profile=self.prompt_variant,
            harness_version=__version__,
            tool_profile_name=self.tool_profile_name,
            enabled_toolsets=self.enabled_toolsets,
            disabled_toolsets=self.disabled_toolsets,
        )
        result = BenchmarkResult(
            submission_id=str(uuid.uuid4()),
            model=self.model,
            provider=self.provider,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            benchmark_release_id=active_release.benchmark_release_id if active_release else "",
            public_release_id=active_release.public_release_id if active_release else "public",
            hidden_release_id=active_release.hidden_release_id if active_release else "",
            environment={
                "task_count": len(tasks),
                "pool": self.pool or "all",
                "scenario": self.scenario or "all",
                "artifact_type": self.artifact_type or "all",
                "prompt_variant": self.prompt_variant,
                "judge_model": self.judge_model,
                "judge_affects_score": self.judge_affects_score,
                "adapter": self.adapter,
                "ablation_profile": ablation_profile.model_dump(),
                "known_adapters": list(KNOWN_ADAPTERS),
                "executable_adapters": sorted(EXECUTABLE_ADAPTERS),
                "subsets": self.subsets,
                "capabilities": self.capabilities,
                "official_only": self.official_only,
                **(environment_extra or {}),
            },
            overall_score=overall_ci.mean,
            overall_completion=_mean([stat.mean_completion_score for stat in task_stats]),
            overall_trajectory=_mean([stat.mean_trajectory_score for stat in task_stats]),
            overall_behavior=_mean([stat.mean_behavior_score for stat in task_stats]),
            judge_model=self.judge_model,
            overall_judge_score=_mean([stat.mean_judge_score for stat in task_stats if stat.judged_runs > 0]),
            overall_judge_confidence=_mean(
                [stat.mean_judge_confidence for stat in task_stats if stat.judged_runs > 0]
            ),
            overall_judge_pass_rate=_mean([stat.judge_pass_rate for stat in task_stats if stat.judged_runs > 0]),
            judge_task_coverage=(
                sum(1 for stat in task_stats if stat.judged_runs > 0) / len(task_stats)
                if task_stats
                else 0.0
            ),
            judge_error_count=sum(stat.judge_error_count for stat in task_stats),
            overall_reliability=_mean([stat.reliability_score for stat in task_stats]),
            overall_weighted_query_score=(
                sum(stat.mean_task_score * stat.query_weight for stat in task_stats) / total_weight
                if total_weight
                else 0.0
            ),
            overall_median_latency_ms=_mean([stat.median_duration_ms for stat in task_stats]),
            overall_p95_latency_ms=_mean([stat.p95_duration_ms for stat in task_stats]),
            overall_input_tokens=_mean([stat.mean_input_tokens for stat in task_stats]),
            overall_output_tokens=_mean([stat.mean_output_tokens for stat in task_stats]),
            overall_reasoning_tokens=_mean([stat.mean_reasoning_tokens for stat in task_stats]),
            overall_total_tokens=_mean([stat.mean_total_tokens for stat in task_stats]),
            overall_cost_usd=_mean([stat.mean_cost_usd for stat in task_stats]),
            overall_tokens_per_pass=_mean([stat.tokens_per_pass for stat in task_stats]),
            overall_cost_per_pass=_mean([stat.cost_per_pass for stat in task_stats]),
            overall_worst_of_n=_mean([stat.worst_of_n for stat in task_stats]),
            public_dev_score=_mean([stat.mean_task_score for stat in task_stats if stat.pool == "public_dev"]),
            official_hidden_score=_mean(
                [stat.mean_task_score for stat in task_stats if stat.pool == "official_hidden"]
            ),
            clear_prompt_score=_mean(
                [stat.mean_task_score for stat in task_stats if stat.prompt_variant == "clear"]
            ),
            ambiguous_prompt_score=_mean(
                [stat.mean_task_score for stat in task_stats if stat.prompt_variant == "ambiguous"]
            ),
            consensus_subset_score=_mean(
                [stat.mean_task_score for stat in task_stats if "consensus" in stat.subsets]
            ),
            hard_subset_score=_mean([stat.mean_task_score for stat in task_stats if "hard" in stat.subsets]),
            overall_delivery_outcome_counts=overall_delivery_outcome_counts,
            overall_failure_mode_counts=overall_failure_mode_counts,
            overall_ci_lower=overall_ci.lower,
            overall_ci_upper=overall_ci.upper,
            overall_pass_hat_k=_mean([1.0 if stat.pass_hat_k else 0.0 for stat in task_stats]),
            tier_results=tier_results,
            scenario_results=scenario_results,
            task_results=task_stats,
            environment_checksum=self._benchmark_checksum(tasks),
            task_snapshot_fingerprint=compute_task_snapshot_fingerprint(tasks),
        )
        if print_report is None:
            should_print_report = self.print_report and not self.quiet
        else:
            should_print_report = print_report
        if should_print_report:
            self._print_report(result)
        return result

    def _is_passing_run(self, task: TaskDefinition, result: TaskRunResult) -> bool:
        completion = result.completion_result
        if completion.total_assertions > 0:
            completion_passed = completion.passed_assertions >= completion.total_assertions
        else:
            completion_passed = completion.score >= 0.9999
        return completion_passed and result.run_score >= task.pass_threshold

    def _print_report(self, result: BenchmarkResult) -> None:
        console.print(f"\n[bold]{'=' * 60}[/]")
        console.print(f"[bold]Results — {result.model}[/]")
        console.print(f"[bold]{'=' * 60}[/]")
        console.print(
            f"\nScore: [bold cyan]{result.overall_score:.3f}[/] "
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
                f"Pass rate={result.overall_judge_pass_rate:.0%}  "
                f"Coverage={result.judge_task_coverage:.0%}  "
                f"Errors={result.judge_error_count}"
            )
        console.print(
            f"  Prompt variant={self.prompt_variant}  "
            f"Weighted query score={result.overall_weighted_query_score:.3f}"
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

        table = Table(title="Task Breakdown")
        table.add_column("Task", style="bold")
        table.add_column("Tier", justify="center")
        table.add_column("Scene", justify="center")
        table.add_column("Pool", justify="center")
        table.add_column("Task Score", justify="right")
        table.add_column("Run", justify="right")
        table.add_column("Comp", justify="right")
        table.add_column("Traj", justify="right")
        table.add_column("Behav", justify="right")
        table.add_column("Judge", justify="right")
        table.add_column("Reliab", justify="right")
        table.add_column("p50 ms", justify="right")
        table.add_column("Tok/pass", justify="right")
        table.add_column("Failure", justify="left")

        for stat in result.task_results:
            color = "green" if stat.mean_task_score >= 0.7 else "yellow" if stat.mean_task_score >= 0.4 else "red"
            top_failure = max(stat.failure_mode_counts.items(), key=lambda item: item[1])[0] if stat.failure_mode_counts else "-"
            table.add_row(
                stat.task_id,
                stat.tier,
                stat.scenario or "-",
                stat.pool,
                f"[{color}]{stat.mean_task_score:.3f}[/]",
                f"{stat.mean_run_score:.2f}",
                f"{stat.mean_completion_score:.2f}",
                f"{stat.mean_trajectory_score:.2f}",
                f"{stat.mean_behavior_score:.2f}",
                f"{stat.mean_judge_score:.2f}" if stat.judged_runs > 0 else "-",
                f"{stat.reliability_score:.2f}",
                f"{stat.median_duration_ms:.0f}",
                f"{stat.tokens_per_pass:.0f}",
                top_failure,
            )

        console.print(table)

    def _benchmark_checksum(self, tasks: list[TaskDefinition]) -> str:
        payload = "|".join(
            sorted(f"{task.id}:{task.pool.value}:{task.variant_id}:{task.release_id}" for task in tasks)
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    index = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def _safe_cache_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value.strip())
    return cleaned.strip("._-") or "unknown"


def _task_definition_fingerprint(task: TaskDefinition) -> str:
    payload = task.model_dump(mode="json")
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _now_ms() -> int:
    return int(time.monotonic() * 1000)
