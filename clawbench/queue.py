"""Job queue backed by HF Dataset for persistent state.

Architecture:
- Submissions stored as rows in HF Dataset (requests split)
- Status: PENDING → EVALUATING → FINISHED | FAILED
- Results stored in a separate split (results)
- Queue worker polls for PENDING jobs and evaluates them

This runs inside the HF Space container — no external infra needed.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import tempfile
import threading
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from clawbench.hub import dataset_repo_files, ensure_dataset_repo, resolve_dataset_repo

logger = logging.getLogger(__name__)

HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Local fallback when HF is unavailable
def _resolve_local_queue_dir() -> Path:
    override = os.environ.get("CLAWBENCH_LOCAL_QUEUE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return Path("/data/queue") if Path("/data").exists() else Path("data/queue")


LOCAL_QUEUE_DIR = _resolve_local_queue_dir()


class JobStatus(str, Enum):
    PENDING = "pending"
    EVALUATING = "evaluating"
    FINISHED = "finished"
    FAILED = "failed"


ACTIVE_JOB_STATUSES = {JobStatus.PENDING, JobStatus.EVALUATING}


class SubmissionRequest(BaseModel):
    model: str  # e.g. "anthropic/claude-sonnet-4-6"
    provider: str = ""  # e.g. "anthropic"
    api_key_env: str = ""  # Env var name holding the API key (NOT the key itself)
    judge_model: str = ""
    judge_affects_score: bool = False
    runs_per_task: int = Field(default=3, ge=1, le=10)
    max_parallel_lanes: int = Field(default=1, ge=1, le=8)
    tier: str | None = None  # Filter to a specific tier
    task_ids: list[str] = Field(default_factory=list)
    scenario: str | None = None
    prompt_variant: str = "clear"
    submitter: str = ""  # HF username
    notes: str = ""

    def active_fingerprint(self) -> str:
        """Stable key for deduping equivalent queued/evaluating jobs."""
        payload = {
            "model": self.model.strip(),
            "provider": self.provider.strip(),
            "judge_model": self.judge_model.strip(),
            "judge_affects_score": self.judge_affects_score,
            "runs_per_task": self.runs_per_task,
            "max_parallel_lanes": self.max_parallel_lanes,
            "tier": self.tier or "",
            "task_ids": sorted({task_id.strip() for task_id in self.task_ids if task_id.strip()}),
            "scenario": self.scenario or "",
            "prompt_variant": self.prompt_variant,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


class Job(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.PENDING
    request: SubmissionRequest
    submitted_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    result_id: str | None = None  # Links to BenchmarkResult.submission_id
    attempt_count: int = 0
    stale_requeues: int = 0
    last_progress_at: str | None = None
    current_task_id: str | None = None
    current_run_index: int | None = None
    current_run_total: int | None = None
    progress_message: str | None = None


class JobQueue:
    """Manages the evaluation queue with HF Dataset persistence."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        # Gradio calls asyncio.run(queue.submit()) on a fresh loop while the
        # worker keeps its own loop. asyncio.Lock is loop-bound and can hang.
        self._lock = threading.Lock()
        self._dataset_repo = resolve_dataset_repo(HF_TOKEN)
        LOCAL_QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        self._load_local()
        self._load_hub()

    def _load_local(self) -> None:
        """Load queue state from local disk."""
        jobs_file = LOCAL_QUEUE_DIR / "jobs.json"
        if jobs_file.exists():
            try:
                data = json.loads(jobs_file.read_text())
                for item in data:
                    job = Job(**item)
                    self._jobs[job.job_id] = job
                logger.info("Loaded %d jobs from local queue", len(self._jobs))
            except Exception as e:
                logger.error("Failed to load local queue: %s", e)

    def _load_hub(self) -> None:
        """Best-effort queue rehydrate from HF Dataset."""
        if not HF_TOKEN:
            return
        try:
            from huggingface_hub import HfApi, hf_hub_download

            api = HfApi(token=HF_TOKEN)
            ensure_dataset_repo(api, self._dataset_repo)
            if "queue/jobs.json" not in dataset_repo_files(api, self._dataset_repo):
                return

            jobs_path = hf_hub_download(
                repo_id=self._dataset_repo,
                repo_type="dataset",
                filename="queue/jobs.json",
                token=HF_TOKEN,
            )
            data = json.loads(Path(jobs_path).read_text())
            merged = 0
            for item in data:
                job = Job(**item)
                self._jobs[job.job_id] = job
                merged += 1
            if merged:
                self._save_local()
                logger.info("Loaded %d jobs from HF queue dataset", merged)
        except Exception as e:
            logger.info("HF queue bootstrap unavailable: %s", e)

    def _save_local(self) -> None:
        """Persist queue state to local disk."""
        jobs_file = LOCAL_QUEUE_DIR / "jobs.json"
        data = [job.model_dump() for job in self._jobs.values()]
        payload = json.dumps(data, indent=2) + "\n"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=LOCAL_QUEUE_DIR,
                prefix="jobs.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                tmp_file.write(payload)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())
                tmp_path = Path(tmp_file.name)
            tmp_path.replace(jobs_file)
        finally:
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()

    async def submit(self, request: SubmissionRequest) -> Job:
        """Submit a new evaluation job."""
        import uuid
        with self._lock:
            max_runs = _env_int("CLAWBENCH_MAX_RUNS_PER_SUBMISSION", 3, minimum=1, maximum=100)
            if request.runs_per_task > max_runs:
                raise ValueError(
                    f"Requested runs_per_task={request.runs_per_task}, but this deployment allows at most {max_runs}."
                )

            max_lanes = _env_int("CLAWBENCH_MAX_LANES_PER_SUBMISSION", 4, minimum=1, maximum=32)
            if request.max_parallel_lanes > max_lanes:
                raise ValueError(
                    f"Requested max_parallel_lanes={request.max_parallel_lanes}, but this deployment allows at most {max_lanes}."
                )

            active_jobs = [
                job for job in self._jobs.values() if job.status in ACTIVE_JOB_STATUSES
            ]
            fingerprint = request.active_fingerprint()
            for job in active_jobs:
                if job.request.active_fingerprint() == fingerprint:
                    logger.info(
                        "Deduped submission for model %s onto active job %s",
                        request.model,
                        job.job_id,
                    )
                    return job

            max_active_jobs = _env_int("CLAWBENCH_MAX_ACTIVE_QUEUE_JOBS", 25, minimum=1, maximum=1000)
            if len(active_jobs) >= max_active_jobs:
                raise ValueError(
                    f"Queue is at capacity ({len(active_jobs)}/{max_active_jobs} active jobs). "
                    "Try again after current evaluations finish."
                )

            max_per_submitter = _env_int("CLAWBENCH_MAX_ACTIVE_JOBS_PER_SUBMITTER", 3, minimum=0, maximum=1000)
            if max_per_submitter:
                submitter_key = _submitter_key(request)
                active_for_submitter = sum(
                    1 for job in active_jobs if _submitter_key(job.request) == submitter_key
                )
                if active_for_submitter >= max_per_submitter:
                    raise ValueError(
                        f"Submitter '{submitter_key}' already has {active_for_submitter} active job(s); "
                        f"limit is {max_per_submitter}."
                    )

            job = Job(
                job_id=str(uuid.uuid4())[:8],
                request=request,
                submitted_at=_now_iso(),
            )
            self._jobs[job.job_id] = job
            self._save_local()
        await self._sync_to_hub()
        logger.info("Job %s submitted for model %s", job.job_id, request.model)
        return job

    async def get_status(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def list_jobs(self, limit: int = 50) -> list[Job]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.submitted_at, reverse=True)
        return jobs[:limit]

    async def list_pending(self) -> list[Job]:
        return [j for j in self._jobs.values() if j.status == JobStatus.PENDING]

    async def claim_pending(self, limit: int = 1) -> list[Job]:
        """Atomically claim up to ``limit`` pending jobs for evaluation."""
        if limit <= 0:
            return []
        with self._lock:
            claimed: list[Job] = []
            pending = sorted(
                (job for job in self._jobs.values() if job.status == JobStatus.PENDING),
                key=lambda job: job.submitted_at,
            )
            now_iso = _now_iso()
            for job in pending[:limit]:
                job.status = JobStatus.EVALUATING
                job.started_at = now_iso
                job.last_progress_at = now_iso
                job.finished_at = None
                job.error = None
                job.result_id = None
                job.current_task_id = None
                job.current_run_index = None
                job.current_run_total = None
                job.progress_message = "Queued for evaluation"
                job.attempt_count += 1
                claimed.append(job)
            if claimed:
                self._save_local()
        if claimed:
            await self._sync_to_hub()
        return claimed

    async def update_progress(
        self,
        job_id: str,
        *,
        current_task_id: str | None,
        current_run_index: int | None,
        current_run_total: int | None,
        progress_message: str | None,
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status != JobStatus.EVALUATING:
                return
            job.last_progress_at = _now_iso()
            job.current_task_id = current_task_id
            job.current_run_index = current_run_index
            job.current_run_total = current_run_total
            job.progress_message = progress_message
            self._save_local()
        await self._sync_to_hub()

    async def reclaim_stale_jobs(self, stale_after_seconds: int) -> list[Job]:
        """Return evaluating jobs to pending when their heartbeat is stale."""
        if stale_after_seconds <= 0:
            return []
        with self._lock:
            reclaimed: list[Job] = []
            cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=stale_after_seconds)
            now_iso = _now_iso()
            for job in self._jobs.values():
                if job.status != JobStatus.EVALUATING:
                    continue
                last_seen = _parse_iso(job.last_progress_at or job.started_at)
                if last_seen is None or last_seen > cutoff:
                    continue
                stale_label = (job.last_progress_at or job.started_at or "")[:19]
                job.status = JobStatus.PENDING
                job.started_at = None
                job.finished_at = None
                job.error = None
                job.result_id = None
                job.last_progress_at = now_iso
                job.current_task_id = None
                job.current_run_index = None
                job.current_run_total = None
                job.progress_message = (
                    "Auto-requeued after stale evaluation lease"
                    + (f" ({stale_label})" if stale_label else "")
                )
                job.stale_requeues += 1
                reclaimed.append(job)
            if reclaimed:
                self._save_local()
                logger.warning("Reclaimed %d stale evaluating jobs", len(reclaimed))
        if reclaimed:
            await self._sync_to_hub()
        return reclaimed

    async def mark_evaluating(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.EVALUATING
            now_iso = _now_iso()
            if job.started_at is None:
                job.attempt_count += 1
            job.started_at = now_iso
            job.last_progress_at = now_iso
            job.finished_at = None
            job.error = None
            job.result_id = None
            job.current_task_id = None
            job.current_run_index = None
            job.current_run_total = None
            job.progress_message = "Queued for evaluation"
            self._save_local()
        await self._sync_to_hub()

    async def mark_finished(self, job_id: str, result_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.FINISHED
            job.finished_at = _now_iso()
            job.last_progress_at = job.finished_at
            job.result_id = result_id
            job.current_task_id = None
            job.current_run_index = None
            job.current_run_total = None
            job.progress_message = "Finished"
            self._save_local()
        await self._sync_to_hub()

    async def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = JobStatus.FAILED
            job.finished_at = _now_iso()
            job.last_progress_at = job.finished_at
            job.error = error
            job.current_task_id = None
            job.current_run_index = None
            job.current_run_total = None
            job.progress_message = "Failed"
            self._save_local()
        await self._sync_to_hub()

    async def _sync_to_hub(self) -> None:
        """Push queue state to HF Dataset for persistence across restarts."""
        await asyncio.to_thread(self._sync_to_hub_blocking)

    def _sync_to_hub_blocking(self) -> None:
        """Blocking Hub upload implementation, kept off the event loop."""
        if not HF_TOKEN:
            return
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=HF_TOKEN)
            ensure_dataset_repo(api, self._dataset_repo)

            # Upload jobs.json to the dataset repo
            local_path = LOCAL_QUEUE_DIR / "jobs.json"
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo="queue/jobs.json",
                repo_id=self._dataset_repo,
                repo_type="dataset",
            )
        except Exception as e:
            logger.warning("Failed to sync queue to Hub: %s", e)


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %d", name, raw, default)
        return default
    return max(minimum, min(maximum, value))


def _submitter_key(request: SubmissionRequest) -> str:
    submitter = request.submitter.strip().lower()
    return submitter or "anonymous"


def _parse_iso(value: str | None) -> datetime.datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)
