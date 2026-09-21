"""Execute real Gateway RPCs inside the actor container's loopback boundary."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

from clawbench.client import GatewayClient, GatewayConfig, GatewayRunError
from clawbench.schemas import Transcript


async def execute(request: dict) -> dict:
    config = json.loads(Path(os.environ["OPENCLAW_CONFIG_PATH"]).read_text())
    gateway = GatewayConfig(
        url="ws://127.0.0.1:18789",
        token=config["gateway"]["auth"]["token"],
        connect_timeout=10 if request.get("preflight") else 30,
        # Cold provider staging can exceed a minute before sessions.create
        # returns. This setup budget is separate from the actor's work timeout.
        request_timeout=request.get("setup_timeout", 180),
    )
    if request.get("preflight"):
        async with GatewayClient(gateway):
            return {"authenticated_gateway": True}
    transcript = Transcript()
    try:
        async with GatewayClient(gateway) as client:
            continuation = request.get("continuation")
            if continuation:
                agent, session = continuation["agent_id"], continuation["session_key"]
                if not agent or not session:
                    raise ValueError("Continuation requires a real agent and session")
                # send_and_wait retrieves the native cumulative session history.
            else:
                # The isolated runner declares its agent before Gateway starts.
                # With config reload disabled, agents.create can persist a new
                # row without making it visible to the active Gateway config.
                configured = next(
                    (
                        entry
                        for entry in config.get("agents", {}).get("list", [])
                        if entry.get("id") == "workspace-assistant"
                        and entry.get("workspace") == request["workspace"]
                    ),
                    None,
                )
                agent = (
                    configured["id"]
                    if configured
                    else await client.create_agent(
                        name="workspace-assistant", workspace=request["workspace"]
                    )
                )
                session = await client.create_session(
                    model=request["model"], agent_id=agent, label="workspace-task"
                )
            await client.subscribe(session)
            work_started = time.monotonic()
            for prompt in request["prompts"]:
                current = await client.send_and_wait(
                    session, prompt, timeout=request.get("timeout", 300)
                )
                # sessions.get can return the full history on later turns. Prefer
                # the longest captured history rather than duplicating its prefix.
                if (
                    len(current.messages) >= len(transcript.messages)
                    and current.messages[: len(transcript.messages)] == transcript.messages
                ):
                    transcript = current
                else:
                    transcript.messages.extend(current.messages)
                    transcript.stop_reason = current.stop_reason
                if current.stop_reason not in {"complete", ""}:
                    break
            return {
                "agent_id": agent,
                "session_key": session,
                "transcript": transcript.model_dump(),
                "work_elapsed_seconds": time.monotonic() - work_started,
            }
    except GatewayRunError as exc:
        transcript.messages.extend(exc.transcript.messages)
        transcript.stop_reason = "error"
        return {"error": str(exc), "transcript": transcript.model_dump()}
    except Exception as exc:
        return {"error": str(exc), "transcript": transcript.model_dump()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    response = asyncio.run(execute(json.load(sys.stdin)))
    print(json.dumps(response), flush=True)
    raise SystemExit(1 if "error" in response else 0)
