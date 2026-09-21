from __future__ import annotations

import json
from pathlib import Path

import pytest

from clawbench.client import GatewayConfig, GatewayRunError
from clawbench.evidence import persist_run_evidence
from clawbench.harness import BenchmarkHarness
from clawbench.schemas import TaskRunResult, Transcript, TranscriptMessage
from clawbench.tasks import load_all_tasks


def test_evidence_keeps_trace_artifacts_and_does_not_follow_links(tmp_path, monkeypatch):
    workspace = tmp_path / "work"
    workspace.mkdir()
    (workspace / "report.txt").write_text("partial result")
    (workspace / "outside").symlink_to(tmp_path / "not-an-artifact")
    monkeypatch.setenv("CLAWBENCH_EVIDENCE_DIR", str(tmp_path / "trusted"))
    result = TaskRunResult(
        task_id="test",
        run_index=0,
        execution_status="execution_error",
        transcript=Transcript(messages=[TranscriptMessage(role="assistant", text="Working")]),
    )
    destination = persist_run_evidence(result, workspace)
    saved = json.loads((destination / "result.json").read_text())
    assert saved["transcript"]["messages"][0]["text"] == "Working"
    assert saved["execution_status"] == "execution_error"
    assert (destination / "workspace/report.txt").read_text() == "partial result"
    assert (destination / "workspace/outside").is_symlink()


@pytest.mark.asyncio
@pytest.mark.parametrize("retention_fails", [False, True])
async def test_failed_run_retains_partial_evidence(tmp_path, monkeypatch, retention_fails):
    task = next(t for t in load_all_tasks() if t.id == "t1-bugfix-discount")
    workspaces = []

    class InterruptedClient:
        def __init__(self, config):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create_agent(self, *, name, workspace):
            workspaces.append(Path(workspace))
            return "agent"

        async def create_session(self, **kwargs):
            return "session"

        async def subscribe(self, session):
            pass

        async def send_and_wait(self, *args, **kwargs):
            (workspaces[0] / "partial.txt").write_text("evidence before disconnect")
            raise GatewayRunError(
                "disconnected",
                Transcript(
                    messages=[
                        TranscriptMessage(role="assistant", text="Read source and started work")
                    ],
                    stop_reason="error",
                ),
            )

        async def delete_session(self, *args):
            pass

        async def delete_agent(self, *args, **kwargs):
            pass

    monkeypatch.setenv("OPENCLAW_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("CLAWBENCH_EVIDENCE_DIR", str(tmp_path / "evidence"))
    monkeypatch.setenv("CLAWBENCH_RUN_CACHE_DIR", "")
    monkeypatch.delenv("CLAWBENCH_KEEP_WORKSPACES", raising=False)
    monkeypatch.setattr("clawbench.harness.GatewayClient", InterruptedClient)
    if retention_fails:

        def fail_retention(*args):
            raise OSError("disk full")

        monkeypatch.setattr("clawbench.harness.persist_run_evidence", fail_retention)
    result = await BenchmarkHarness(gateway_config=GatewayConfig(), model="test")._run_single(
        task, 0
    )
    assert result.error_phase == "execution"
    assert result.execution_status == "execution_error"
    assert result.transcript.messages[0].text == "Read source and started work"
    assert result.run_score == 0
    if retention_fails:
        assert (workspaces[0] / "partial.txt").is_file()
    else:
        assert (Path(result.evidence_path) / "workspace/partial.txt").is_file()
        assert not workspaces[0].exists()
