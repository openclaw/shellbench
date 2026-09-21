"""Pydantic models for the ClawBench v0.3 benchmark."""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from clawbench.run_review import ReviewEvidence, RunReview


class Tier(str, enum.Enum):
    TIER1 = "tier1"
    TIER2 = "tier2"
    TIER3 = "tier3"
    TIER4 = "tier4"
    TIER5 = "tier5"


class TaskFamily(str, enum.Enum):
    CODING = "coding"
    REPO = "repo"
    BROWSER = "browser"
    TOOLS = "tools"
    MULTI_TOOL = "multi_tool"
    ADVERSARIAL = "adversarial"


class TaskPool(str, enum.Enum):
    PUBLIC_DEV = "public_dev"
    OFFICIAL_HIDDEN = "official_hidden"


class TaskSubset(str, enum.Enum):
    CONSENSUS = "consensus"
    HARD = "hard"


class CapabilityTag(str, enum.Enum):
    BUGFIX = "bugfix"
    REFACTOR = "refactor"
    TEST_AUTHORING = "test_authoring"
    MULTIFILE_REASONING = "multifile_reasoning"
    BROWSER_DEBUGGING = "browser_debugging"
    STRUCTURED_OUTPUT = "structured_output"
    MEMORY_CONTINUATION = "memory_continuation"
    DELEGATION = "delegation"
    TOOL_COMPOSITION = "tool_composition"
    RESEARCH_SYNTHESIS = "research_synthesis"
    GRACEFUL_REFUSAL = "graceful_refusal"
    SPEC_REVISION = "spec_revision"
    CROSS_REPO_CHANGE = "cross_repo_change"
    AUTOMATION = "automation"


class ScenarioDomain(str, enum.Enum):
    # Original 12 scenarios from the basic-usage query test sheet
    FILE_SYSTEM_OPS = "file_system_ops"
    WEB_INFO_OPS = "web_info_ops"
    CALENDAR_REMINDERS = "calendar_reminders"
    COMMUNICATION = "communication_messaging"
    DATA_ANALYSIS = "data_processing_analysis"
    CODING_DEV = "coding_dev_assist"
    PERSONAL_ASSISTANT = "personal_life_assistant"
    MULTI_STEP = "multi_step_compound"
    CONTEXT = "context_continuation"
    ERROR_BOUNDARY = "error_boundary_cases"
    SKILL_CALLING = "skill_calling"
    SYSTEM = "system_capabilities"
    # v0.5 additions: high-frequency personal-agent scenarios beyond the test sheet
    PRIVACY_PII = "privacy_pii_handling"
    FINANCIAL_PERSONAL = "personal_financial_hygiene"
    TRAVEL_LOGISTICS = "travel_logistics_under_uncertainty"
    SOCIAL_COORDINATION = "social_coordination"
    KNOWLEDGE_BASE = "personal_knowledge_base"
    HEALTH_TRACKING = "health_wellness_tracking"
    SECURITY_HYGIENE = "account_security_hygiene"
    MULTIMODAL_UNDERSTANDING = "multimodal_understanding"


class QueryDifficulty(str, enum.Enum):
    L1 = "l1"
    L2 = "l2"
    L3 = "l3"


class ArtifactType(str, enum.Enum):
    FILE = "file"
    INFORMATION = "information"
    OPERATION = "operation"
    CODE = "code"
    EXTERNAL_ACTION = "external_action"
    MEMORY = "memory"
    AUTOMATION = "automation"
    MIXED = "mixed"


class PromptVariant(str, enum.Enum):
    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"


class DeliveryOutcome(str, enum.Enum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


class FailureMode(str, enum.Enum):
    GRACEFUL_REFUSAL = "graceful_refusal"
    HALLUCINATED_COMPLETION = "hallucinated_completion"
    REPEATED_ERROR_LOOP = "repeated_error_loop"
    TOOL_MISUSE = "tool_misuse"
    UNSAFE_MUTATION = "unsafe_mutation"
    VERIFICATION_SKIPPED = "verification_skipped"
    STATE_REGRESSION = "state_regression"
    MEMORY_MISS = "memory_miss"
    DELEGATION_FAILED = "delegation_failed"
    BROWSER_NAVIGATION_FAILURE = "browser_navigation_failure"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
    TIMEOUT = "timeout"
    REWARD_HACK_SUSPECTED = "reward_hack_suspected"


class ToolCall(BaseModel):
    """A single tool invocation extracted from the transcript."""

    id: str = ""
    name: str
    input: dict[str, Any] = Field(default_factory=dict)
    output: str = ""
    success: bool | None = None
    timestamp_ms: int = 0
    family: str | None = None
    mutating: bool = False
    error: str = ""


class ToolResult(BaseModel):
    """A single tool result extracted from the transcript."""

    id: str
    content: str = ""


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def merged(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            total_cost_usd=self.total_cost_usd + other.total_cost_usd,
        )

    @property
    def component_tokens(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.reasoning_tokens
            + self.cache_read_tokens
            + self.cache_write_tokens
        )


class EfficiencyResult(BaseModel):
    duration_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    component_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @classmethod
    def from_usage(cls, *, duration_ms: int, usage: TokenUsage) -> EfficiencyResult:
        return cls(
            duration_ms=duration_ms,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            component_tokens=usage.component_tokens,
            total_tokens=usage.total_tokens,
            estimated_cost_usd=usage.total_cost_usd,
        )

    @model_validator(mode="after")
    def _populate_component_tokens(self) -> EfficiencyResult:
        if self.component_tokens == 0:
            self.component_tokens = (
                self.input_tokens
                + self.output_tokens
                + self.reasoning_tokens
                + self.cache_read_tokens
                + self.cache_write_tokens
            )
        return self


class TranscriptMessage(BaseModel):
    role: str
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_results: list[ToolResult] = Field(default_factory=list)
    tool_result_for: str | None = None
    tool_result_content: str = ""
    timestamp_ms: int = 0
    usage: TokenUsage = Field(default_factory=TokenUsage)


class Transcript(BaseModel):
    messages: list[TranscriptMessage] = Field(default_factory=list)
    # An empty/default transcript has no demonstrated terminal event.
    stop_reason: str = "unknown"

    @property
    def tool_call_sequence(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for message in self.messages:
            calls.extend(message.tool_calls)
        return calls

    @property
    def assistant_messages(self) -> list[TranscriptMessage]:
        return [message for message in self.messages if message.role == "assistant"]

    @property
    def assistant_text(self) -> str:
        return "\n".join(message.text for message in self.assistant_messages if message.text)

    @property
    def total_usage(self) -> TokenUsage:
        total = TokenUsage()
        for message in self.messages:
            total = total.merged(message.usage)
        return total


class FileState(BaseModel):
    path: str
    exists: bool = True
    content_contains: list[str] = Field(default_factory=list)
    content_not_contains: list[str] = Field(default_factory=list)
    content_matches: str | None = None
    min_size_bytes: int = 0


class MemoryState(BaseModel):
    key_pattern: str
    exists: bool = True
    value_contains: list[str] = Field(default_factory=list)


class SessionState(BaseModel):
    should_exist: bool = True
    model_should_be: str | None = None


class CronState(BaseModel):
    exists: bool = True
    description_contains: str | None = None


class GatewayAssertion(BaseModel):
    method: str
    params: dict[str, Any] = Field(default_factory=dict)
    assert_path: str
    assert_equals: Any | None = None
    assert_contains: str | None = None
    assert_exists: bool = True


class ExecutionCheck(BaseModel):
    name: str
    command: str
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = 60
    shell: bool = True
    expected_exit_code: int = 0
    stdout_contains: list[str] = Field(default_factory=list)
    stdout_not_contains: list[str] = Field(default_factory=list)
    stderr_contains: list[str] = Field(default_factory=list)
    stdout_matches: str | None = None
    stderr_matches: str | None = None
    expected_stdout: str | None = None
    expected_stdout_file: str | None = None
    expected_json: Any | None = None
    expected_json_file: str | None = None


class BackgroundService(BaseModel):
    name: str
    command: str
    cwd: str = "."
    env: dict[str, str] = Field(default_factory=dict)
    port: int | None = None
    port_env: str = "PORT"
    url_template: str | None = "http://127.0.0.1:{port}"
    ready_path: str | None = "/"
    ready_status: int = 200
    ready_contains: str | None = None
    ready_file: str | None = None
    startup_timeout_seconds: int = 20


class CompletionSpec(BaseModel):
    files: list[FileState] = Field(default_factory=list)
    memory: list[MemoryState] = Field(default_factory=list)
    session: SessionState | None = None
    cron: list[CronState] = Field(default_factory=list)
    gateway_assertions: list[GatewayAssertion] = Field(default_factory=list)
    execution_checks: list[ExecutionCheck] = Field(default_factory=list)


class TrajectoryExpectations(BaseModel):
    required_families: list[str] = Field(default_factory=list)
    required_pre_edit_families: list[str] = Field(default_factory=list)
    required_post_edit_families: list[str] = Field(default_factory=list)
    min_distinct_families: int = 0
    min_pre_edit_exploration_calls: int = 0
    min_distinct_read_targets_pre_edit: int = 0
    min_post_edit_verification_calls: int = 0
    min_distinct_mutation_targets: int = 0
    min_successful_delegations: int = 0
    require_read_before_mutation: bool = False
    require_self_verification: bool = False
    require_verification_after_last_mutation: bool = True
    expect_recovery: bool = False
    max_recovery_turns: int = 3
    max_repeated_failures: int = 1
    forbidden_tools: list[str] = Field(default_factory=list)
    forbidden_shell_patterns: list[str] = Field(default_factory=list)


class BehaviorExpectations(BaseModel):
    require_plan: bool = False
    plan_within_first_assistant_messages: int = 2
    require_progress_updates: bool = False
    min_progress_updates: int = 1
    require_blocker_explanation: bool = False
    require_refusal_when_impossible: bool = False
    forbid_destructive_commands: bool = True


class JudgeExpectations(BaseModel):
    rubric: str
    artifact_paths: list[str] = Field(default_factory=list)
    include_transcript: bool = True
    include_completion_feedback: bool = True
    max_artifact_chars: int = 4_000
    max_transcript_chars: int = 4_000
    passing_threshold: float = 0.7


class UserTurn(BaseModel):
    message: str
    variant_messages: dict[str, str] = Field(default_factory=dict)
    after_assistant_turns: int | None = None
    when_tool_family: str | None = None
    when_tool_name: str | None = None
    when_assistant_contains: str | None = None
    when_last_tool_failed: bool = False


class SimulatedUser(BaseModel):
    turns: list[UserTurn] = Field(default_factory=list)
    max_turns: int = 20


class SessionPhase(BaseModel):
    name: str
    user: SimulatedUser
    timeout_seconds: int | None = None


class TaskSetup(BaseModel):
    asset_packs: list[str] = Field(default_factory=list)
    workspace_files: list[str] = Field(default_factory=list)
    background_services: list[BackgroundService] = Field(default_factory=list)
    memory_seed: list[dict[str, str]] = Field(default_factory=list)
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    pre_check_gateway: list[GatewayAssertion] = Field(default_factory=list)


class TaskDefinition(BaseModel):
    id: str
    name: str
    tier: Tier
    family: TaskFamily
    surface: str
    scenario: ScenarioDomain | None = None
    subscenario: str = ""
    atomic_capabilities: list[str] = Field(default_factory=list)
    query_difficulty: QueryDifficulty | None = None
    query_weight: float = 1.0
    artifact_type: ArtifactType | None = None
    preconditions: list[str] = Field(default_factory=list)
    source_dataset: str = ""
    prompt_variants: list[PromptVariant] = Field(default_factory=lambda: [PromptVariant.CLEAR])
    pool: TaskPool = TaskPool.PUBLIC_DEV
    subsets: list[TaskSubset] = Field(default_factory=list)
    capabilities: list[CapabilityTag] = Field(default_factory=list)
    variant_group: str = ""
    variant_id: str = "main"
    template_id: str = ""
    release_id: str = ""
    source_kind: str = ""
    provenance_ids: list[str] = Field(default_factory=list)
    privacy_tier: str = ""
    contamination_risk: str = ""
    freshness_epoch: str = ""
    category: str = ""
    domain: str = ""
    functionality: list[str] = Field(default_factory=list)
    trace_distribution: list[str] = Field(default_factory=list)
    tool_surface: list[str] = Field(default_factory=list)
    risk_tags: list[str] = Field(default_factory=list)
    first_used_at: str = ""
    retire_after_runs: int = 0
    similarity_hash: str = ""
    canary_token: str = ""
    official: bool = False
    timeout_seconds: int = 180
    pass_threshold: float = 0.7

    setup: TaskSetup = Field(default_factory=TaskSetup)
    user: SimulatedUser | None = None
    phases: list[SessionPhase] = Field(default_factory=list)
    completion: CompletionSpec = Field(default_factory=CompletionSpec)
    trajectory: TrajectoryExpectations = Field(default_factory=TrajectoryExpectations)
    behavior: BehaviorExpectations = Field(default_factory=BehaviorExpectations)
    judge: JudgeExpectations | None = None

    @model_validator(mode="after")
    def _validate_phase_config(self) -> TaskDefinition:
        if not self.phases and self.user is None:
            raise ValueError("TaskDefinition requires either `user` or `phases`.")
        if not self.variant_group:
            self.variant_group = self.id
        if not self.prompt_variants:
            self.prompt_variants = [PromptVariant.CLEAR]
        else:
            deduped: list[PromptVariant] = []
            for variant in self.prompt_variants:
                if variant not in deduped:
                    deduped.append(variant)
            self.prompt_variants = deduped
        return self

    def normalized_phases(self) -> list[SessionPhase]:
        if self.phases:
            return self.phases
        assert self.user is not None
        return [SessionPhase(name="main", user=self.user, timeout_seconds=self.timeout_seconds)]


class ExecutionCheckResult(BaseModel):
    name: str
    command: str
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    passed: bool
    reason: str = ""


class CompletionResult(BaseModel):
    total_assertions: int = 0
    passed_assertions: int = 0
    failed_assertions: list[str] = Field(default_factory=list)
    execution_results: list[ExecutionCheckResult] = Field(default_factory=list)
    score: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_state(cls, data: Any) -> Any:
        if isinstance(data, dict) and "failed_assertions" in data and "execution_results" not in data:
            return {
                "total_assertions": data.get("total_assertions", 0),
                "passed_assertions": data.get("passed_assertions", 0),
                "failed_assertions": data.get("failed_assertions", []),
                "score": data.get("score", 0.0),
            }
        return data


class TrajectoryResult(BaseModel):
    exploration_score: float = 1.0
    recovery_score: float = 1.0
    tool_fit_score: float = 1.0
    safety_score: float = 1.0
    read_before_write_ratio: float = 0.0
    distinct_read_targets_pre_edit: list[str] = Field(default_factory=list)
    distinct_mutation_targets: list[str] = Field(default_factory=list)
    distinct_families: list[str] = Field(default_factory=list)
    required_families_missing: list[str] = Field(default_factory=list)
    forbidden_violations: list[str] = Field(default_factory=list)
    repeated_failures: int = 0
    recovered_failures: int = 0
    self_verified: bool = False
    score: float = 0.0

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_trajectory(cls, data: Any) -> Any:
        if isinstance(data, dict) and "precision" in data:
            return {
                "exploration_score": data.get("order_score", 0.0),
                "recovery_score": data.get("recall", 0.0),
                "tool_fit_score": data.get("precision", 0.0),
                "safety_score": data.get("efficiency_score", 0.0),
                "forbidden_violations": data.get("forbidden_violations", []),
                "score": data.get("score", 0.0),
            }
        return data


class BehaviorResult(BaseModel):
    score: float = 0.0
    satisfied_expectations: list[str] = Field(default_factory=list)
    failed_expectations: list[str] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_behavior(cls, data: Any) -> Any:
        if isinstance(data, dict) and "reason" in data and "satisfied_expectations" not in data:
            return {
                "score": data.get("score", 0.0),
                "reason": data.get("reason", ""),
            }
        return data


class JudgeResult(BaseModel):
    enabled: bool = False
    model: str = ""
    score: float = 0.0
    confidence: float = 0.0
    passed: bool = False
    reason: str = ""
    rubric_hits: list[str] = Field(default_factory=list)
    rubric_misses: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    error: str | None = None


class TaskRunResult(BaseModel):
    task_id: str
    tier: str = ""
    family: str = ""
    scenario: str = ""
    subscenario: str = ""
    artifact_type: str = ""
    prompt_variant: str = PromptVariant.CLEAR.value
    query_difficulty: str = ""
    query_weight: float = 1.0
    pool: str = TaskPool.PUBLIC_DEV.value
    subsets: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    variant_group: str = ""
    variant_id: str = "main"
    template_id: str = ""
    release_id: str = ""
    source_kind: str = ""
    privacy_tier: str = ""
    contamination_risk: str = ""
    freshness_epoch: str = ""
    similarity_hash: str = ""
    official: bool = False
    run_index: int
    completion_result: CompletionResult = Field(default_factory=CompletionResult)
    trajectory_result: TrajectoryResult = Field(default_factory=TrajectoryResult)
    behavior_result: BehaviorResult = Field(default_factory=BehaviorResult)
    judge_result: JudgeResult = Field(default_factory=JudgeResult)
    run_review: RunReview = Field(default_factory=RunReview)
    review_evidence: ReviewEvidence | None = None
    review_artifact_dir: str = ""
    run_score: float = 0.0
    transcript: Transcript = Field(default_factory=Transcript)
    duration_ms: int = 0
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    efficiency_result: EfficiencyResult = Field(default_factory=EfficiencyResult)
    delivery_outcome: DeliveryOutcome = DeliveryOutcome.FAIL
    failure_mode: FailureMode | None = None
    error: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_run(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "state_score" not in data:
            return data
        return {
            "task_id": data.get("task_id", ""),
            "run_index": data.get("run_index", 0),
            "completion_result": data.get("state_score", {}),
            "trajectory_result": data.get("trajectory_score", {}),
            "behavior_result": data.get("behavior_score", {}),
            "run_score": data.get("composite_score", 0.0),
            "transcript": data.get("transcript", {}),
            "duration_ms": data.get("duration_ms", 0),
            "token_usage": data.get("token_usage", {}),
            "efficiency_result": {
                "duration_ms": data.get("duration_ms", 0),
                "estimated_cost_usd": data.get("token_usage", {}).get("total_cost_usd", 0.0),
            },
            "error": data.get("error"),
        }

    @model_validator(mode="after")
    def _populate_efficiency_defaults(self) -> TaskRunResult:
        if not self.variant_group:
            self.variant_group = self.task_id
        if self.efficiency_result.duration_ms == 0 and (
            self.duration_ms or self.token_usage.total_tokens or self.token_usage.total_cost_usd
        ):
            self.efficiency_result = EfficiencyResult.from_usage(
                duration_ms=self.duration_ms,
                usage=self.token_usage,
            )
        elif self.duration_ms == 0 and self.efficiency_result.duration_ms:
            self.duration_ms = self.efficiency_result.duration_ms
        if self.token_usage.total_tokens == 0 and self.efficiency_result.total_tokens:
            self.token_usage = TokenUsage(
                input_tokens=self.efficiency_result.input_tokens,
                output_tokens=self.efficiency_result.output_tokens,
                reasoning_tokens=self.efficiency_result.reasoning_tokens,
                cache_read_tokens=self.efficiency_result.cache_read_tokens,
                cache_write_tokens=self.efficiency_result.cache_write_tokens,
                total_tokens=self.efficiency_result.total_tokens,
                total_cost_usd=self.efficiency_result.estimated_cost_usd,
            )
        return self


class TaskStats(BaseModel):
    task_id: str
    tier: str = ""
    family: str = ""
    scenario: str = ""
    subscenario: str = ""
    artifact_type: str = ""
    prompt_variant: str = PromptVariant.CLEAR.value
    query_difficulty: str = ""
    query_weight: float = 1.0
    pool: str = TaskPool.PUBLIC_DEV.value
    subsets: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    variant_group: str = ""
    variant_id: str = "main"
    template_id: str = ""
    release_id: str = ""
    source_kind: str = ""
    privacy_tier: str = ""
    contamination_risk: str = ""
    freshness_epoch: str = ""
    similarity_hash: str = ""
    official: bool = False
    runs: int
    mean_completion_score: float
    mean_trajectory_score: float
    mean_behavior_score: float
    mean_judge_score: float = 0.0
    mean_judge_confidence: float = 0.0
    judge_pass_rate: float = 0.0
    judged_runs: int = 0
    judge_error_count: int = 0
    mean_run_score: float
    reliability_score: float
    variance_score: float
    mean_task_score: float
    stddev: float
    min_score: float
    max_score: float
    pass_at_1: bool
    pass_rate: float
    pass_hat_k: bool
    scores: list[float] = Field(default_factory=list)
    mean_duration_ms: float = 0.0
    median_duration_ms: float = 0.0
    p95_duration_ms: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    mean_reasoning_tokens: float = 0.0
    mean_total_tokens: float = 0.0
    mean_cost_usd: float = 0.0
    tokens_per_pass: float = 0.0
    cost_per_pass: float = 0.0
    worst_of_n: float = 0.0
    delivery_outcome_counts: dict[str, int] = Field(default_factory=dict)
    failure_mode_counts: dict[str, int] = Field(default_factory=dict)
    high_variance: bool = False

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_task_stats(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "mean_composite" not in data:
            return data
        return {
            "task_id": data.get("task_id", ""),
            "runs": data.get("runs", 0),
            "mean_completion_score": data.get("mean_state_score", 0.0),
            "mean_trajectory_score": data.get("mean_trajectory_score", 0.0),
            "mean_behavior_score": data.get("mean_behavior_score", 0.0),
            "mean_run_score": data.get("mean_composite", 0.0),
            "reliability_score": data.get("pass_at_k", 0.0),
            "variance_score": 1.0,
            "mean_task_score": data.get("mean_composite", 0.0),
            "stddev": data.get("stddev", 0.0),
            "min_score": data.get("min_score", 0.0),
            "max_score": data.get("max_score", 0.0),
            "pass_at_1": data.get("pass_at_1", False),
            "pass_rate": data.get("pass_at_k", 0.0),
            "pass_hat_k": data.get("pass_hat_k", False),
            "scores": data.get("scores", []),
            "mean_duration_ms": data.get("mean_duration_ms", 0.0),
            "median_duration_ms": data.get("mean_duration_ms", 0.0),
            "p95_duration_ms": data.get("mean_duration_ms", 0.0),
            "worst_of_n": min(data.get("scores", [0.0])) if data.get("scores") else 0.0,
            "high_variance": data.get("high_variance", False),
        }


class TierResult(BaseModel):
    tier: str
    mean_task_score: float
    mean_completion: float
    mean_trajectory: float
    mean_behavior: float
    mean_judge: float = 0.0
    mean_reliability: float
    ci_lower: float
    ci_upper: float
    pass_hat_k_rate: float
    task_stats: list[TaskStats] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_category(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "category" not in data:
            return data
        return {
            "tier": data.get("category", ""),
            "mean_task_score": data.get("mean_composite", 0.0),
            "mean_completion": data.get("mean_state", 0.0),
            "mean_trajectory": data.get("mean_trajectory", 0.0),
            "mean_behavior": data.get("mean_behavior", 0.0),
            "mean_reliability": data.get("pass_hat_k_rate", 0.0),
            "ci_lower": data.get("ci_lower", 0.0),
            "ci_upper": data.get("ci_upper", 0.0),
            "pass_hat_k_rate": data.get("pass_hat_k_rate", 0.0),
            "task_stats": data.get("task_stats", []),
        }


class ScenarioResult(BaseModel):
    scenario: str
    mean_task_score: float
    weighted_score: float
    mean_completion: float
    mean_trajectory: float
    mean_behavior: float
    mean_judge: float = 0.0
    mean_reliability: float
    pass_hat_k_rate: float
    total_weight: float = 0.0
    task_stats: list[TaskStats] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    submission_id: str
    model: str
    provider: str
    timestamp: str
    openclaw_version: str = ""
    benchmark_version: str = "0.4.0.dev1"
    benchmark_release_id: str = ""
    public_release_id: str = ""
    hidden_release_id: str = ""
    environment: dict[str, Any] = Field(default_factory=dict)

    overall_score: float
    overall_completion: float
    overall_trajectory: float
    overall_behavior: float
    judge_model: str = ""
    overall_judge_score: float = 0.0
    overall_judge_confidence: float = 0.0
    overall_judge_pass_rate: float = 0.0
    judge_task_coverage: float = 0.0
    judge_error_count: int = 0
    overall_reliability: float = 0.0
    overall_weighted_query_score: float = 0.0
    overall_median_latency_ms: float = 0.0
    overall_p95_latency_ms: float = 0.0
    overall_input_tokens: float = 0.0
    overall_output_tokens: float = 0.0
    overall_reasoning_tokens: float = 0.0
    overall_total_tokens: float = 0.0
    overall_cost_usd: float = 0.0
    overall_tokens_per_pass: float = 0.0
    overall_cost_per_pass: float = 0.0
    overall_worst_of_n: float = 0.0
    public_dev_score: float = 0.0
    official_hidden_score: float = 0.0
    clear_prompt_score: float = 0.0
    ambiguous_prompt_score: float = 0.0
    consensus_subset_score: float = 0.0
    hard_subset_score: float = 0.0
    overall_delivery_outcome_counts: dict[str, int] = Field(default_factory=dict)
    overall_failure_mode_counts: dict[str, int] = Field(default_factory=dict)
    overall_ci_lower: float
    overall_ci_upper: float
    overall_pass_hat_k: float

    tier_results: list[TierResult] = Field(default_factory=list)
    scenario_results: list[ScenarioResult] = Field(default_factory=list)
    task_results: list[TaskStats] = Field(default_factory=list)

    certified: bool = False
    environment_checksum: str = ""
    task_snapshot_fingerprint: str = ""

    @model_validator(mode="before")
    @classmethod
    def _from_legacy_benchmark(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "overall_composite" not in data:
            return data
        return {
            "submission_id": data.get("submission_id", ""),
            "model": data.get("model", ""),
            "provider": data.get("provider", ""),
            "timestamp": data.get("timestamp", ""),
            "openclaw_version": data.get("openclaw_version", ""),
            "benchmark_version": data.get("benchmark_version", "0.2.0"),
            "environment": data.get("environment", {}),
            "overall_score": data.get("overall_composite", 0.0),
            "overall_completion": data.get("overall_state", 0.0),
            "overall_trajectory": data.get("overall_trajectory", 0.0),
            "overall_behavior": data.get("overall_behavior", 0.0),
            "overall_reliability": data.get("overall_pass_hat_k", 0.0),
            "overall_ci_lower": data.get("overall_ci_lower", 0.0),
            "overall_ci_upper": data.get("overall_ci_upper", 0.0),
            "overall_pass_hat_k": data.get("overall_pass_hat_k", 0.0),
            "tier_results": data.get("category_results", []),
            "task_results": data.get("task_results", []),
            "certified": data.get("certified", False),
            "environment_checksum": data.get("environment_checksum", ""),
        }

    @property
    def overall_composite(self) -> float:
        return self.overall_score
