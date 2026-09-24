from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import pytest
from websockets.asyncio.server import serve

from clawbench.client import GatewayClient, GatewayConfig


@pytest.fixture
def gateway_client(monkeypatch):
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("CLAWBENCH_DISABLE_GATEWAY_DEVICE_IDENTITY", "1")

    @asynccontextmanager
    async def connect(respond, *, request_timeout=2):
        async def gateway(websocket):
            await websocket.send(json.dumps({
                "type": "event", "event": "connect.challenge", "payload": {"nonce": "test-nonce"},
            }))
            async for raw in websocket:
                request = json.loads(raw)
                response = respond(request)
                if response is None:
                    continue
                await websocket.send(json.dumps({"type": "res", "id": request["id"], **response}))

        async with asyncio.timeout(10):
            async with serve(gateway, "127.0.0.1", 0) as server:
                port = server.sockets[0].getsockname()[1]
                config = GatewayConfig(
                    url=f"ws://127.0.0.1:{port}", connect_timeout=2,
                    request_timeout=request_timeout,
                )
                async with GatewayClient(config) as client:
                    yield client

    return connect


def success(payload):
    return {"ok": True, "payload": payload}


def failure(code="INVALID_REQUEST", message='Unknown agent id "new-agent"'):
    return {"ok": False, "error": {"code": code, "message": message}}


def _default_response(request):
    if request["method"] == "connect":
        return success({"type": "hello-ok", "protocol": 4})
    if request["method"] == "agents.create":
        return success({"agentId": "new-agent"})
    return success({})



@pytest.mark.asyncio
async def test_cli_handshake_avoids_control_ui_build_check(gateway_client, monkeypatch):
    signed = {}
    requests = []

    def build_device(**kwargs):
        signed.update(kwargs)
        return {"id": "synthetic-device"}

    monkeypatch.setattr("clawbench.client._build_connect_device", build_device)

    def respond(request):
        requests.append(request)
        if request["params"]["client"]["id"] == "openclaw-control-ui":
            return failure("UNAVAILABLE", "protocol mismatch: Control UI updated; reload this page to continue")
        return _default_response(request)

    async with gateway_client(respond):
        pass

    params = requests[0]["params"]
    assert params["client"]["id"] == signed["client_id"] == "cli"
    assert params["client"]["mode"] == signed["client_mode"] == "cli"
    assert "buildId" not in params["client"]
    assert params["device"] == {"id": "synthetic-device"}
    assert "operator.admin" in params["scopes"] == signed["scopes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["NOT_PAIRED", "FORBIDDEN", "UNAVAILABLE"])
async def test_connect_preserves_pairing_and_scope_errors(gateway_client, code):
    requests = []

    def respond(request):
        requests.append(request)
        return failure(code, "explicit device approval required")

    with pytest.raises(RuntimeError, match=code):
        async with gateway_client(respond):
            pytest.fail("Unapproved client must not connect")
    assert len(requests) == 1
    assert "device" not in requests[0]["params"]


@pytest.mark.asyncio
async def test_connect_retries_gateway_starting_response(gateway_client):
    attempts = 0

    def respond(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return failure("UNAVAILABLE", "gateway starting; retry shortly")
        return _default_response(request)

    async with gateway_client(respond):
        pass

    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("reconnect", [False, True])
async def test_session_waits_for_created_agent_registration(gateway_client, reconnect):
    sessions = []

    def respond(request):
        if request["method"] == "sessions.create":
            sessions.append(request)
            if len(sessions) < 3:
                return failure()
            return success({"sessionKey": "registered-session"})
        return _default_response(request)

    async with gateway_client(respond) as client:
        agent_id = await client.create_agent(name="Test", workspace="/synthetic")
        if reconnect:
            await client.reconnect()
        session = await client.create_session(agent_id=agent_id, model="test/model", label="test")

    assert session == "registered-session"
    assert len(sessions) == 3
    assert all(r["params"] == {"agentId": "new-agent", "model": "test/model", "label": "test"} for r in sessions)
    assert len({r["id"] for r in sessions}) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("code,message,created", [
    ("FORBIDDEN", 'Unknown agent id "new-agent"', True),
    ("INVALID_REQUEST", "invalid model", True),
    ("INVALID_REQUEST", 'Unknown agent id "other-agent"', True),
    ("INVALID_REQUEST", 'Unknown agent id "new-agent"', False),
])
async def test_session_does_not_retry_unrelated_errors(gateway_client, code, message, created):
    attempts = 0

    def respond(request):
        nonlocal attempts
        if request["method"] == "sessions.create":
            attempts += 1
            return failure(code, message)
        return _default_response(request)

    async with gateway_client(respond) as client:
        if created:
            await client.create_agent(name="Test", workspace="/synthetic")
        with pytest.raises(RuntimeError) as raised:
            await client.create_session(agent_id="new-agent")

    assert str(raised.value) == f"RPC sessions.create failed: {code} - {message}"
    assert attempts == 1


@pytest.mark.asyncio
async def test_session_registration_has_a_deadline(gateway_client):
    attempts = 0

    def respond(request):
        nonlocal attempts
        if request["method"] == "sessions.create":
            attempts += 1
            return failure()
        return _default_response(request)

    async with gateway_client(respond, request_timeout=0.3) as client:
        await client.create_agent(name="Test", workspace="/synthetic")
        async with asyncio.timeout(1):
            with pytest.raises(RuntimeError, match="Unknown agent id"):
                await client.create_session(agent_id="new-agent")

    assert attempts >= 2


@pytest.mark.asyncio
@pytest.mark.parametrize("settled", ["session", "deleted", "expired"])
async def test_registration_retry_window_is_not_reopened(gateway_client, settled):
    attempts = 0

    def respond(request):
        nonlocal attempts
        if request["method"] == "sessions.create":
            attempts += 1
            if settled == "session" and attempts == 1:
                return success({"sessionKey": "first-session"})
            return failure()
        return _default_response(request)

    async with gateway_client(respond, request_timeout=0.1) as client:
        await client.create_agent(name="Test", workspace="/synthetic")
        if settled == "session":
            assert await client.create_session(agent_id="new-agent") == "first-session"
        elif settled == "deleted":
            await client.delete_agent("new-agent")
        else:
            await asyncio.sleep(0.11)
        with pytest.raises(RuntimeError, match="Unknown agent id"):
            await client.create_session(agent_id="new-agent")

    assert attempts == (2 if settled == "session" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_registration_wait_releases_pending_rpc(gateway_client, cancel):
    requested = asyncio.Event()
    attempts = 0

    def respond(request):
        nonlocal attempts
        if request["method"] == "sessions.create":
            attempts += 1
            requested.set()
            return None
        return _default_response(request)

    async with gateway_client(respond, request_timeout=0.1) as client:
        await client.create_agent(name="Test", workspace="/synthetic")
        session = asyncio.create_task(client.create_session(agent_id="new-agent"))
        await requested.wait()
        if cancel:
            session.cancel()
        with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
            await session
        assert not client._pending

    assert attempts == 1
