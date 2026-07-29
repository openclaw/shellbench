from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Iterable


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    friendly_name: str
    provider: str
    provider_model_id: str
    proxy_model_name: str


@dataclass(frozen=True)
class HarnessSpec:
    name: str
    version: str


@dataclass(frozen=True)
class RunSpec:
    run_label: str
    harness: str
    harness_version: str
    model_slug: str
    model_id: str
    provider: str
    proxy_model_name: str
    repetition: int
    expected_task_count: int
    run_date: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec("gpt55", "gpt-5.5", "openai", "gpt-5.5", "gpt-5.5"),
    ModelSpec(
        "gpt56-sol",
        "gpt-5.6-sol",
        "openai",
        "gpt-5.6-sol",
        "gpt-5.6-sol",
    ),
    ModelSpec(
        "gpt56-luna",
        "gpt-5.6-luna",
        "openai",
        "gpt-5.6-luna",
        "gpt-5.6-luna",
    ),
    ModelSpec(
        "gpt56-terra",
        "gpt-5.6-terra",
        "openai",
        "gpt-5.6-terra",
        "gpt-5.6-terra",
    ),
    ModelSpec(
        "fable5",
        "fable-5",
        "anthropic",
        "claude-fable-5",
        "claude-fable-5",
    ),
    ModelSpec(
        "opus47",
        "opus-4.7",
        "anthropic",
        "claude-opus-4-7",
        "claude-opus-4-7",
    ),
    ModelSpec(
        "opus48",
        "opus-4.8",
        "anthropic",
        "claude-opus-4-8",
        "claude-opus-4-8",
    ),
    ModelSpec(
        "opus5",
        "opus-5",
        "anthropic",
        "claude-opus-5",
        "claude-opus-5",
    ),
)

HARNESSES: tuple[HarnessSpec, ...] = (
    HarnessSpec("openclaw", "2026.7.1-2"),
    HarnessSpec("hermes", "cb06017b1d6e1b9ae0cb35f99a48ffa6bcbaa828"),
    HarnessSpec("codex", "0.145.0"),
    HarnessSpec("claude-code", "2.1.220"),
)

NODE_VERSION = "22.23.1"
LITELLM_VERSION = "1.93.0"
REAL_TRAJECTORY_HARNESSES = frozenset({"openclaw", "hermes", "codex"})


def model_by_slug(slug: str) -> ModelSpec:
    for model in MODELS:
        if model.slug == slug:
            return model
    raise KeyError(f"Unknown model slug: {slug}")


def harness_by_name(name: str) -> HarnessSpec:
    for harness in HARNESSES:
        if harness.name == name:
            return harness
    raise KeyError(f"Unknown harness: {name}")


def trajectory_mode_for_harness(name: str) -> str:
    if name in REAL_TRAJECTORY_HARNESSES:
        return "real_harness_events"
    return "unsupported"


def build_matrix_plan(
    expected_task_count: int,
    *,
    run_date: str | None = None,
    harnesses: Iterable[HarnessSpec] = HARNESSES,
    models: Iterable[ModelSpec] = MODELS,
    repetitions: Iterable[int] = (1, 2, 3),
    reasoning_effort: str | None = None,
    run_kind: str = "full",
) -> list[RunSpec]:
    stamp = run_date or date.today().strftime("%Y%m%d")
    repetition_values = tuple(repetitions)
    if not repetition_values or any(value < 0 for value in repetition_values):
        raise ValueError("repetitions must contain non-negative integers")
    if len(set(repetition_values)) != len(repetition_values):
        raise ValueError("repetitions must not contain duplicates")
    if not run_kind or "-" in run_kind:
        raise ValueError("run_kind must be a non-empty label segment")
    reasoning_slug = reasoning_effort.replace("_", "-") if reasoning_effort else None
    plan: list[RunSpec] = []
    for harness in harnesses:
        for model in models:
            for repetition in repetition_values:
                reasoning_label = f"-{reasoning_slug}" if reasoning_slug else ""
                label = (
                    f"{harness.name}-{model.slug}{reasoning_label}-{run_kind}-"
                    f"{expected_task_count}-r{repetition}-{stamp}"
                )
                plan.append(
                    RunSpec(
                        run_label=label,
                        harness=harness.name,
                        harness_version=harness.version,
                        model_slug=model.slug,
                        model_id=model.provider_model_id,
                        provider=model.provider,
                        proxy_model_name=model.proxy_model_name,
                        repetition=repetition,
                        expected_task_count=expected_task_count,
                        run_date=stamp,
                    )
                )
    return plan
