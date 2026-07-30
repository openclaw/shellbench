"""Tests for `OpenClawAdapter` — exercised against a stub gateway.

This validates the adapter wiring (lifecycle + state-query resolution)
in isolation, before the harness is rewired through it. The stub
`GatewayClient` records every call and produces canned responses so
the adapter's branches are covered end-to-end without a real gateway.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from clawbench.adapters import get_adapter
from clawbench.adapters.base import AdapterContext, StateQueryResult
from clawbench.adapters.openclaw import OpenClawAdapter, OpenClawAdapterConfig
from clawbench.canonical import (
    AdapterCapability,
    CanonicalTask,
    StateQuery,
)
from clawbench.canonical.convert import from_task_definition
from clawbench.schemas import (
    CompletionSpec,
    ExecutionCheck,
    FileState,
    GatewayAssertion,
    MemoryState,
    SessionState,
    SimulatedUser,
    TaskDefinition,
    TaskFamily,
    TaskSetup,
    Tier,
    Transcript,
    UserTurn,
)


# ---------------------------------------------------------------------------
# Stub GatewayClient
# ---------------------------------------------------------------------------


class _StubGateway:
    """Minimal GatewayClient stand-in for adapter tests.

    Records every `create_agent`, `create_session`, `subscribe`,
    `send_and_wait`, `delete_*` call in `.calls`, and serves canned
    responses for the verification RPCs used by `OpenClawAdapter`.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.rpc_responses: dict[str, dict[str, Any]] = {}
        self.send_transcript = Transcript()

    async def __aenter__(self) -> "_StubGateway":
        self.calls.append(("__aenter__", {}))
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.calls.append(("__aexit__", {}))

    async def create_agent(self, *, name: str, workspace: str) -> str:
        self.calls.append(("create_agent", {"name": name, "workspace": workspace}))
        return "agent-stub"

    async def reconnect(self) -> None:
        self.calls.append(("reconnect", {}))

    async def create_session(self, *, model: str, agent_id: str, label: str) -> str:
        self.calls.append(
            ("create_session", {"model": model, "agent_id": agent_id, "label": label})
        )
        return f"session-{label}"

    def set_session_auth_profile_override(
        self,
        session_key: str,
        *,
        agent_id: str,
        auth_profile_id: str,
        source: str = "user",
    ) -> bool:
        self.calls.append(
            (
                "set_session_auth_profile_override",
                {
                    "session_key": session_key,
                    "agent_id": agent_id,
                    "auth_profile_id": auth_profile_id,
                    "source": source,
                },
            )
        )
        return True

    async def subscribe(self, session_key: str) -> None:
        self.calls.append(("subscribe", {"session_key": session_key}))

    async def send_and_wait(
        self,
        session_key: str,
        message: str,
        *,
        timeout: float,
    ) -> Transcript:
        self.calls.append(
            (
                "send_and_wait",
                {"session_key": session_key, "message": message, "timeout": timeout},
            )
        )
        return self.send_transcript

    async def delete_session(self, session_key: str) -> None:
        self.calls.append(("delete_session", {"session_key": session_key}))

    async def delete_agent(self, agent_id: str, *, delete_files: bool) -> None:
        self.calls.append(
            ("delete_agent", {"agent_id": agent_id, "delete_files": delete_files})
        )

    async def get_effective_tools(self, session_key: str) -> dict[str, Any]:
        self.calls.append(("get_effective_tools", {"session_key": session_key}))
        return self.rpc_responses.get(
            "tools.effective",
            {"groups": [{"tools": [{"id": "bash"}, {"id": "browser"}]}]},
        )

    async def _rpc(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((f"_rpc:{method}", dict(params)))
        if method in self.rpc_responses:
            return self.rpc_responses[method]
        raise RuntimeError(f"stub gateway: no response set for {method}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _coding_task() -> CanonicalTask:
    return from_task_definition(
        TaskDefinition(
            id="oa-adapter-test",
            name="OA adapter test",
            tier=Tier.TIER1,
            family=TaskFamily.CODING,
            surface="coding",
            setup=TaskSetup(),
            user=SimulatedUser(
                max_turns=1,
                turns=[UserTurn(message="Do the task.")],
            ),
            completion=CompletionSpec(
                files=[FileState(path="out.txt", exists=True)],
                execution_checks=[ExecutionCheck(name="ok", command="true")],
            ),
        )
    )


def _mixed_state_task() -> CanonicalTask:
    return from_task_definition(
        TaskDefinition(
            id="oa-adapter-state-test",
            name="OA state test",
            tier=Tier.TIER2,
            family=TaskFamily.MULTI_TOOL,
            surface="tools",
            setup=TaskSetup(
                pre_check_gateway=[
                    GatewayAssertion(
                        method="agents.list",
                        assert_path="$.count",
                        assert_equals=0,
                    ),
                ],
            ),
            user=SimulatedUser(max_turns=1, turns=[UserTurn(message="go")]),
            completion=CompletionSpec(
                memory=[MemoryState(key_pattern="stack", exists=True, value_contains=["React"])],
                session=SessionState(should_exist=True, model_should_be="opus"),
            ),
        )
    )


def _make_adapter_and_gateway() -> tuple[OpenClawAdapter, _StubGateway]:
    gateway = _StubGateway()
    adapter = OpenClawAdapter(OpenClawAdapterConfig(model="test-model"))
    adapter._client_factory = lambda: gateway  # type: ignore[assignment]
    return adapter, gateway


def _make_ctx(task: CanonicalTask, workspace: Path) -> AdapterContext:
    return AdapterContext(
        task=task,
        workspace=workspace,
        runtime_values={},
        run_index=0,
        model="test-model",
        transcript=Transcript(),
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_openclaw_adapter_is_registered() -> None:
    cls = get_adapter("openclaw")
    assert cls is OpenClawAdapter


def test_openclaw_declares_full_capability_set() -> None:
    assert AdapterCapability.FILES in OpenClawAdapter.capabilities
    assert AdapterCapability.EXECUTION in OpenClawAdapter.capabilities
    assert AdapterCapability.MEMORY in OpenClawAdapter.capabilities
    assert AdapterCapability.SESSION in OpenClawAdapter.capabilities
    assert AdapterCapability.CRON in OpenClawAdapter.capabilities
    assert AdapterCapability.GATEWAY_RPC in OpenClawAdapter.capabilities
    assert AdapterCapability.BROWSER in OpenClawAdapter.capabilities


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_setup_realizes_memory_seed_files(tmp_path: Path) -> None:
    task = from_task_definition(
        TaskDefinition(
            id="oa-seeded-memory",
            name="OA seeded memory",
            tier=Tier.TIER2,
            family=TaskFamily.MULTI_TOOL,
            surface="tools",
            setup=TaskSetup(
                memory_seed=[
                    {
                        "key": "event profile",
                        "value": "Vegetarian food, quiet rooms, and no stairs.",
                    }
                ]
            ),
            user=SimulatedUser(max_turns=1, turns=[UserTurn(message="go")]),
        )
    )
    adapter, gateway = _make_adapter_and_gateway()

    async def _go() -> None:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)

    asyncio.run(_go())

    assert (tmp_path / "MEMORY.md").read_text(encoding="utf-8").count("event profile") == 1
    assert "Vegetarian food" in (tmp_path / "memory" / "event_profile.md").read_text(encoding="utf-8")
    assert any(call[0] == "create_agent" for call in gateway.calls)


def test_run_phase_creates_session_subscribes_and_drives_simulator(tmp_path: Path) -> None:
    task = _coding_task()
    adapter, gateway = _make_adapter_and_gateway()

    async def _go() -> None:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            result = await adapter.run_phase(task.phases[0], ctx)
            assert result.error is None
            await adapter.teardown(ctx)

    asyncio.run(_go())

    methods = [name for name, _ in gateway.calls]
    # Ordered sequence we expect:
    assert "create_agent" in methods
    assert "create_session" in methods
    assert "subscribe" in methods
    assert "send_and_wait" in methods
    assert "delete_session" in methods
    assert "delete_agent" in methods
    # The send_and_wait call should use the rendered user turn text.
    send_args = next(args for name, args in gateway.calls if name == "send_and_wait")
    assert send_args["message"] == "Do the task."


def test_run_phase_uses_canonical_per_turn_timeout(tmp_path: Path) -> None:
    task = _coding_task()
    task.budgets.timeout_seconds = 900
    task.budgets.per_turn_timeout_seconds = 600
    task.phases[0].timeout_seconds = None
    adapter, gateway = _make_adapter_and_gateway()

    async def _go() -> None:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            await adapter.run_phase(task.phases[0], ctx)

    asyncio.run(_go())

    send_args = next(args for name, args in gateway.calls if name == "send_and_wait")
    assert send_args["timeout"] == 600


def test_run_phase_honors_explicit_adapter_timeout_cap(tmp_path: Path) -> None:
    task = _coding_task()
    task.budgets.timeout_seconds = 900
    task.budgets.per_turn_timeout_seconds = 600
    task.phases[0].timeout_seconds = None
    gateway = _StubGateway()
    adapter = OpenClawAdapter(
        OpenClawAdapterConfig(model="test-model", turn_timeout_seconds=120)
    )
    adapter._client_factory = lambda: gateway  # type: ignore[assignment]

    async def _go() -> None:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            await adapter.run_phase(task.phases[0], ctx)

    asyncio.run(_go())

    send_args = next(args for name, args in gateway.calls if name == "send_and_wait")
    assert send_args["timeout"] == 120


def test_run_phase_routes_openai_codex_runtime_to_codex_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = _coding_task()
    adapter, gateway = _make_adapter_and_gateway()
    monkeypatch.setenv("CLAWBENCH_OPENCLAW_AGENT_RUNTIME", "codex")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "state"))

    async def _go() -> None:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            ctx.model = "openai/gpt-5.5"
            await adapter.setup(ctx)
            result = await adapter.run_phase(task.phases[0], ctx)
            assert result.error is None

    asyncio.run(_go())

    create_args = next(args for name, args in gateway.calls if name == "create_session")
    assert create_args["model"] == "openai-codex/gpt-5.5"
    auth_args = next(
        args for name, args in gateway.calls if name == "set_session_auth_profile_override"
    )
    assert auth_args["auth_profile_id"] == "openai-codex:clawbench-env"
    auth_store = json.loads(
        (tmp_path / "state" / "agents" / "agent-stub" / "agent" / "auth-profiles.json").read_text(
            encoding="utf-8"
        )
    )
    assert auth_store["profiles"]["openai-codex:clawbench-env"]["keyRef"]["id"] == "OPENAI_API_KEY"
    assert auth_store["profiles"]["openai-codex:clawbench-env"]["key"] == "test-key"


def test_run_phase_fails_fast_without_setup(tmp_path: Path) -> None:
    task = _coding_task()
    adapter, _ = _make_adapter_and_gateway()

    async def _go() -> None:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            # Skip setup() — run_phase should return an error phase.
            result = await adapter.run_phase(task.phases[0], ctx)
            assert result.completed_normally is False
            assert result.error and "agent_id" in result.error

    asyncio.run(_go())


# ---------------------------------------------------------------------------
# State queries
# ---------------------------------------------------------------------------


def test_memory_query_uses_memory_search_primary_path(tmp_path: Path) -> None:
    task = _mixed_state_task()
    adapter, gateway = _make_adapter_and_gateway()
    gateway.rpc_responses["memory.search"] = {
        "payload": {"entries": [{"value": "stack = React, Node, Postgres"}]}
    }

    query = StateQuery(
        kind="memory",
        predicate="exists",
        selector={"key_pattern": "stack"},
        expected={"value_contains": ["React"]},
        required_capability=AdapterCapability.MEMORY,
    )

    async def _go() -> StateQueryResult:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            return await adapter.verify_state_query(query, ctx)

    result = asyncio.run(_go())
    assert result.ok is True
    assert result.detail == "OK"


def test_memory_query_falls_back_to_workspace_on_rpc_failure(tmp_path: Path) -> None:
    task = _mixed_state_task()
    adapter, gateway = _make_adapter_and_gateway()
    # No memory.search response → primary path raises, fallback runs.
    # Seed a MEMORY.md file in the workspace so the fallback succeeds.
    (tmp_path / "MEMORY.md").write_text(
        "stack: React, Node, Postgres", encoding="utf-8"
    )

    query = StateQuery(
        kind="memory",
        predicate="exists",
        selector={"key_pattern": "stack"},
        expected={"value_contains": ["React"]},
        required_capability=AdapterCapability.MEMORY,
    )

    async def _go() -> StateQueryResult:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            return await adapter.verify_state_query(query, ctx)

    result = asyncio.run(_go())
    assert result.ok is True


def test_session_query_uses_sessions_resolve(tmp_path: Path) -> None:
    task = _mixed_state_task()
    adapter, gateway = _make_adapter_and_gateway()
    gateway.rpc_responses["sessions.resolve"] = {
        "payload": {"model": "claude-opus-4"}
    }

    query = StateQuery(
        kind="session",
        predicate="exists",
        selector={},
        expected={"model": "opus"},
        required_capability=AdapterCapability.SESSION,
    )

    async def _go() -> StateQueryResult:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            ctx.adapter_state["last_session_key"] = "some-session"
            return await adapter.verify_state_query(query, ctx)

    result = asyncio.run(_go())
    assert result.ok is True


def test_gateway_query_resolves_json_path(tmp_path: Path) -> None:
    task = _mixed_state_task()
    adapter, gateway = _make_adapter_and_gateway()
    gateway.rpc_responses["memory.list"] = {
        "payload": {"count": 3}
    }

    query = StateQuery(
        kind="custom",
        predicate="equals",
        selector={"method": "memory.list", "params": {}, "assert_path": "$.count"},
        expected={"equals": 3, "exists": True},
        required_capability=AdapterCapability.GATEWAY_RPC,
    )

    async def _go() -> StateQueryResult:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            return await adapter.verify_state_query(query, ctx)

    result = asyncio.run(_go())
    assert result.ok is True


def test_cron_query_returns_false_when_no_jobs(tmp_path: Path) -> None:
    task = _mixed_state_task()
    adapter, gateway = _make_adapter_and_gateway()
    gateway.rpc_responses["cron.list"] = {"payload": {"jobs": []}}

    query = StateQuery(
        kind="cron",
        predicate="exists",
        selector={"description_contains": "daily"},
        expected={},
        required_capability=AdapterCapability.CRON,
    )

    async def _go() -> StateQueryResult:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            return await adapter.verify_state_query(query, ctx)

    result = asyncio.run(_go())
    assert result.ok is False


def test_pre_run_queries_evaluated_during_setup(tmp_path: Path) -> None:
    task = _mixed_state_task()
    adapter, gateway = _make_adapter_and_gateway()
    # Deliberately return the wrong count to trigger a pre-run failure.
    gateway.rpc_responses["agents.list"] = {"payload": {"count": 99}}

    async def _go() -> list[str]:
        async with adapter:
            ctx = _make_ctx(task, tmp_path)
            await adapter.setup(ctx)
            return ctx.adapter_state.get("pre_run_failures", [])

    failures = asyncio.run(_go())
    assert failures, "pre-run gateway assertion should have failed"


# ---------------------------------------------------------------------------
# Requires-context guard
# ---------------------------------------------------------------------------


def test_client_accessor_errors_when_not_in_context() -> None:
    adapter, _ = _make_adapter_and_gateway()
    with pytest.raises(RuntimeError):
        _ = adapter.client
