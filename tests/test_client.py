from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import InvalidMessage, InvalidStatus
from websockets.http11 import Response

from clawbench.client import GatewayClient, GatewayConfig, _correlate_transcript, _parse_single_message
from clawbench.schemas import EfficiencyResult, TokenUsage, Transcript


def test_gateway_config_defaults():
    cfg = GatewayConfig()
    # Defaults raised from 15s/60s -- see GatewayConfig docstring for
    # the rationale; 15s used to race gateway cold-start and produce
    # spurious empty_response failures.
    assert cfg.connect_timeout == 30.0
    assert cfg.request_timeout == 60.0


@pytest.mark.asyncio
async def test_gateway_client_connects_and_creates_session_over_websocket(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    requests = []
    origins = []

    async def gateway(websocket):
        origins.append(websocket.request.headers["Origin"])
        await websocket.send(
            json.dumps({"type": "event", "event": "connect.challenge", "payload": {"nonce": ""}})
        )
        async for raw in websocket:
            request = json.loads(raw)
            requests.append(request)
            payload = (
                {"type": "hello-ok", "protocol": 3}
                if request["method"] == "connect"
                else {"sessionKey": "loopback-session"}
            )
            await websocket.send(
                json.dumps({"type": "res", "id": request["id"], "ok": True, "payload": payload})
            )

    async with asyncio.timeout(10):
        async with serve(gateway, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            config = GatewayConfig(
                url=f"ws://127.0.0.1:{port}", connect_timeout=2, request_timeout=2
            )
            async with GatewayClient(config) as client:
                session = await client.create_session(model="test/model", label="dependency-smoke")
            assert client._ws is None

    assert session == "loopback-session"
    assert origins == [f"http://127.0.0.1:{port}"]
    assert [request["method"] for request in requests] == ["connect", "sessions.create"]
    assert requests[0]["params"]["minProtocol"] == 3
    assert requests[0]["params"]["maxProtocol"] == 4
    assert requests[1]["params"] == {"model": "test/model", "label": "dependency-smoke"}


def test_set_session_auth_profile_override_patches_local_store(tmp_path: Path, monkeypatch):
    state_dir = tmp_path / "state"
    store_dir = state_dir / "agents" / "agent-stub" / "sessions"
    store_dir.mkdir(parents=True)
    store_path = store_dir / "sessions.json"
    store_path.write_text(
        json.dumps({"session-1": {"sessionId": "session-1"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(state_dir))

    ok = GatewayClient().set_session_auth_profile_override(
        "session-1",
        agent_id="agent-stub",
        auth_profile_id="openai-codex:clawbench-env",
    )

    assert ok is True
    entry = json.loads(store_path.read_text(encoding="utf-8"))["session-1"]
    assert entry["authProfileOverride"] == "openai-codex:clawbench-env"
    assert entry["authProfileOverrideSource"] == "user"
    assert "authProfileOverrideCompactionCount" not in entry


def test_gateway_config_env_overrides(monkeypatch):
    monkeypatch.setenv("CLAWBENCH_CONNECT_TIMEOUT", "45")
    monkeypatch.setenv("CLAWBENCH_REQUEST_TIMEOUT", "120")
    cfg = GatewayConfig()
    assert cfg.connect_timeout == 45.0
    assert cfg.request_timeout == 120.0


@pytest.mark.parametrize("raw", ["not-a-number", "nan", "inf", "0", "-1"])
def test_gateway_config_invalid_env_falls_back_to_default(monkeypatch, caplog, raw):
    monkeypatch.setenv("CLAWBENCH_CONNECT_TIMEOUT", raw)
    with caplog.at_level("WARNING"):
        cfg = GatewayConfig()
    assert cfg.connect_timeout == 30.0
    assert any("CLAWBENCH_CONNECT_TIMEOUT" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_gateway_client_disables_websocket_keepalive_for_long_rpc(
    monkeypatch: pytest.MonkeyPatch,
):
    connect_kwargs: dict[str, object] = {}
    connect_params: dict[str, object] = {}

    class FakeWebSocket:
        async def close(self) -> None:
            return None

    async def fake_connect(*args, **kwargs):
        connect_kwargs.update(kwargs)
        return FakeWebSocket()

    async def fake_wait_event(self, event_name: str, *, timeout: float):
        return {"payload": {"nonce": ""}}

    async def fake_rpc(self, method: str, params=None, **kwargs):
        connect_params.update(params or {})
        return {"payload": {"type": "hello-ok", "protocol": 3}}

    async def fake_listener(self):
        await asyncio.sleep(60)

    monkeypatch.setattr("clawbench.client.websockets.connect", fake_connect)
    monkeypatch.setattr(GatewayClient, "_wait_event", fake_wait_event)
    monkeypatch.setattr(GatewayClient, "_rpc", fake_rpc)
    monkeypatch.setattr(GatewayClient, "_listener", fake_listener)

    client = GatewayClient(GatewayConfig(connect_timeout=2))
    await client.connect()
    await client.close()

    assert connect_kwargs["ping_interval"] is None
    assert connect_kwargs["ping_timeout"] is None
    assert connect_params["minProtocol"] == 3
    assert connect_params["maxProtocol"] == 4


def test_tool_results_are_correlated_back_to_tool_calls():
    tool_message = _parse_single_message(
        {
            "role": "assistant",
            "content": [
                {"type": "toolCall", "id": "call-1", "name": "exec", "arguments": {"command": "pytest -q"}},
            ],
        }
    )
    result_message = _parse_single_message(
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": "ERROR failed test"},
            ],
        }
    )

    transcript = _correlate_transcript(Transcript(messages=[tool_message, result_message]))  # type: ignore[arg-type]
    call = transcript.tool_call_sequence[0]

    assert call.output == "ERROR failed test"
    assert call.success is False
    assert call.error == "ERROR failed test"


def test_parser_accepts_codex_tool_search_output_shape():
    tool_message = _parse_single_message(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_search_call",
                    "call_id": "search-1",
                    "name": "tool_search",
                    "arguments": {"query": "message"},
                },
                {
                    "type": "functionCall",
                    "callId": "call-1",
                    "name": "message",
                    "arguments": '{"text":"hello"}',
                },
            ],
        }
    )
    result_message = _parse_single_message(
        {
            "role": "toolResult",
            "toolCallId": "call-1",
            "content": [
                {
                    "type": "toolSearchOutput",
                    "callId": "search-1",
                    "output": [{"text": "message: send a message"}],
                },
                {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": "sent",
                },
            ],
        }
    )

    transcript = _correlate_transcript(Transcript(messages=[tool_message, result_message]))  # type: ignore[arg-type]

    assert [call.name for call in transcript.tool_call_sequence] == ["tool_search", "message"]
    assert transcript.tool_call_sequence[0].output == "message: send a message"
    assert transcript.tool_call_sequence[1].output == "sent"
    assert transcript.tool_call_sequence[1].success is True


def test_parser_accepts_kebab_case_codex_tool_search_blocks():
    tool_message = _parse_single_message(
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool-search-call",
                    "call_id": "search-2",
                    "title": "tool_search",
                    "parameters": {"query": "calendar"},
                },
                {
                    "type": "tool-call",
                    "tool_call_id": "call-2",
                    "tool": "message",
                    "args": {"text": "ok"},
                },
            ],
        }
    )
    result_message = _parse_single_message(
        {
            "role": "tool",
            "content": [
                {
                    "type": "tool-search-output",
                    "call_id": "search-2",
                    "content": [{"content": "message: available"}],
                },
                {
                    "type": "tool-call-output",
                    "tool_call_id": "call-2",
                    "text": "delivered",
                },
            ],
        }
    )

    transcript = _correlate_transcript(Transcript(messages=[tool_message, result_message]))  # type: ignore[arg-type]

    assert [call.id for call in transcript.tool_call_sequence] == ["search-2", "call-2"]
    assert [call.name for call in transcript.tool_call_sequence] == ["tool_search", "message"]
    assert transcript.tool_call_sequence[0].input == {"query": "calendar"}
    assert transcript.tool_call_sequence[0].output == "message: available"
    assert transcript.tool_call_sequence[1].input == {"text": "ok"}
    assert transcript.tool_call_sequence[1].output == "delivered"


def test_parser_correlates_plain_top_level_tool_result_message():
    tool_message = _parse_single_message(
        {
            "role": "assistant",
            "content": [{"type": "toolUse", "id": "call-1", "name": "read", "input": {}}],
        }
    )
    result_message = _parse_single_message(
        {
            "role": "toolResult",
            "toolUseId": "call-1",
            "content": "file contents",
        }
    )

    transcript = _correlate_transcript(Transcript(messages=[tool_message, result_message]))  # type: ignore[arg-type]

    assert transcript.tool_call_sequence[0].output == "file contents"
    assert transcript.tool_call_sequence[0].success is True


def test_message_usage_is_parsed_into_transcript_usage():
    message = _parse_single_message(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Done."}],
            "usage": {
                "input": 10,
                "output": 20,
                "reasoning": 5,
                "cacheRead": 3,
                "cacheWrite": 2,
                "totalTokens": 40,
                "cost": {"total": 0.0125},
            },
        }
    )

    assert message is not None
    assert message.usage.input_tokens == 10
    assert message.usage.output_tokens == 20
    assert message.usage.reasoning_tokens == 5
    assert message.usage.total_tokens == 40
    assert message.usage.total_cost_usd == 0.0125


def test_efficiency_component_tokens_stay_separate_from_total_snapshot():
    usage = TokenUsage(input_tokens=10, output_tokens=5, cache_read_tokens=3, total_tokens=10_000)

    result = EfficiencyResult.from_usage(duration_ms=123, usage=usage)

    assert result.component_tokens == 18
    assert result.total_tokens == 10_000


@pytest.mark.asyncio
async def test_gateway_client_retries_transient_drain_errors(monkeypatch: pytest.MonkeyPatch):
    attempts = 0

    class FakeWebSocket:
        async def close(self) -> None:
            return None

    async def fake_connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InvalidStatus(Response(503, "Service Unavailable", Headers()))
        return FakeWebSocket()

    async def fake_wait_event(self, event_name: str, *, timeout: float):
        return {"payload": {"nonce": ""}}

    async def fake_rpc(self, method: str, params=None, **kwargs):
        return {"payload": {"type": "hello-ok", "protocol": 3}}

    async def fake_listener(self):
        await asyncio.sleep(60)

    monkeypatch.setattr("clawbench.client.websockets.connect", fake_connect)
    monkeypatch.setattr(GatewayClient, "_wait_event", fake_wait_event)
    monkeypatch.setattr(GatewayClient, "_rpc", fake_rpc)
    monkeypatch.setattr(GatewayClient, "_listener", fake_listener)

    client = GatewayClient(GatewayConfig(connect_timeout=2))
    await client.connect()
    assert attempts == 2
    await client.close()


@pytest.mark.asyncio
async def test_gateway_client_retries_half_closed_handshake_errors(
    monkeypatch: pytest.MonkeyPatch,
):
    attempts = 0

    class FakeWebSocket:
        async def close(self) -> None:
            return None

    async def fake_connect(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise InvalidMessage("did not receive a valid HTTP response")
        return FakeWebSocket()

    async def fake_wait_event(self, event_name: str, *, timeout: float):
        return {"payload": {"nonce": ""}}

    async def fake_rpc(self, method: str, params=None, **kwargs):
        return {"payload": {"type": "hello-ok", "protocol": 3}}

    async def fake_listener(self):
        await asyncio.sleep(60)

    monkeypatch.setattr("clawbench.client.websockets.connect", fake_connect)
    monkeypatch.setattr(GatewayClient, "_wait_event", fake_wait_event)
    monkeypatch.setattr(GatewayClient, "_rpc", fake_rpc)
    monkeypatch.setattr(GatewayClient, "_listener", fake_listener)

    client = GatewayClient(GatewayConfig(connect_timeout=2))
    await client.connect()
    assert attempts == 2
    await client.close()


@pytest.mark.asyncio
async def test_send_and_wait_collects_messages_that_arrive_after_final_state():
    client = GatewayClient(GatewayConfig(request_timeout=1))
    session_key = "session-1"

    async def fake_rpc(method: str, params=None):
        assert method == "sessions.send"

        async def emit() -> None:
            await asyncio.sleep(0.01)
            await client._event_queues[f"chat:{session_key}"].put({"payload": {"state": "final"}})
            await asyncio.sleep(0.2)
            await client._event_queues[f"session.message:{session_key}"].put(
                {
                    "payload": {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Late but valid."}],
                            "usage": {"input": 1, "output": 2, "totalTokens": 3},
                        }
                    }
                }
            )

        asyncio.create_task(emit())
        return {"ok": True, "payload": {}}

    client._rpc = fake_rpc  # type: ignore[method-assign]

    transcript = await client.send_and_wait(session_key, "hello", timeout=1.0)

    assert [message.text for message in transcript.assistant_messages] == ["Late but valid."]


@pytest.mark.asyncio
async def test_send_and_wait_passes_gateway_timeout_and_waits_for_run():
    client = GatewayClient(GatewayConfig(request_timeout=1))
    session_key = "session-1"
    calls: list[tuple[str, dict | None, dict]] = []

    async def fake_rpc(method: str, params=None, **kwargs):
        calls.append((method, params, kwargs))
        if method == "sessions.send":
            return {"ok": True, "payload": {"runId": "run-1"}}
        if method == "agent.wait":
            return {"ok": True, "payload": {"runId": "run-1", "status": "completed"}}
        if method == "sessions.get":
            return {
                "ok": True,
                "payload": {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": "Done."}],
                        }
                    ]
                },
            }
        return {"ok": True, "payload": {}}

    client._rpc = fake_rpc  # type: ignore[method-assign]

    transcript = await client.send_and_wait(session_key, "hello", timeout=1.5)

    send_call = next(call for call in calls if call[0] == "sessions.send")
    assert send_call[1] == {
        "key": session_key,
        "message": "hello",
        "idempotencyKey": send_call[1]["idempotencyKey"],
        "timeoutMs": 1500,
    }
    wait_call = next(call for call in calls if call[0] == "agent.wait")
    assert wait_call[1] == {"runId": "run-1", "timeoutMs": 1500}
    assert wait_call[2]["timeout"] == 11.5
    assert [message.text for message in transcript.assistant_messages] == ["Done."]


@pytest.mark.asyncio
async def test_send_and_wait_aborts_run_when_no_terminal_state_arrives():
    client = GatewayClient(GatewayConfig(request_timeout=1))
    session_key = "session-1"
    calls: list[tuple[str, dict | None, dict]] = []

    async def fake_rpc(method: str, params=None, **kwargs):
        calls.append((method, params, kwargs))
        if method == "sessions.send":
            return {"ok": True, "payload": {"runId": "run-timeout"}}
        if method == "agent.wait":
            await asyncio.sleep(60)
        if method == "sessions.abort":
            return {"ok": True, "payload": {"status": "aborted"}}
        if method == "sessions.get":
            return {"ok": True, "payload": {"messages": []}}
        return {"ok": True, "payload": {}}

    client._rpc = fake_rpc  # type: ignore[method-assign]

    await client.send_and_wait(session_key, "hello", timeout=0.01)

    assert ("sessions.abort", {"key": session_key, "runId": "run-timeout"}, {"timeout": 1}) in calls
