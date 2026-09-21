from __future__ import annotations

import json
import subprocess

import pytest

from clawbench.alignment.tasks import load_cases
from scripts.run_alignment_pilot import functional_check, require_disk_space
from scripts.summarize_alignment_pilot import model_audit


@pytest.mark.parametrize("exit_code", [125, 126, 127])
def test_functional_evaluator_startup_error_is_unknown(tmp_path, monkeypatch, exit_code):
    commands = []

    def docker(*args, **kwargs):
        commands.append(args)
        return subprocess.CompletedProcess(args, exit_code, "", "unable to start container")

    monkeypatch.setattr("scripts.run_alignment_pilot.docker", docker)
    result = functional_check(load_cases()[0], tmp_path, "image", tmp_path)
    assert result["passed"] is None
    assert commands[-1][:2] == ("rm", "-f")


def test_timed_out_candidate_container_is_removed(tmp_path, monkeypatch):
    commands = []

    def docker(*args, **kwargs):
        commands.append(args)
        if args[0] == "run":
            raise subprocess.TimeoutExpired(args, 30)
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr("scripts.run_alignment_pilot.docker", docker)
    result = functional_check(load_cases()[0], tmp_path, "image", tmp_path)
    assert result["passed"] is False
    assert commands[-1][:2] == ("rm", "-f")


def test_low_space_refuses_new_runs(tmp_path, monkeypatch):
    from collections import namedtuple

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("scripts.run_alignment_pilot.shutil.disk_usage", lambda _: usage(10, 9, 1))
    with pytest.raises(RuntimeError, match="GiB free"):
        require_disk_space(tmp_path)


def test_independent_model_audit_detects_modified_history(tmp_path):
    audit_dir = tmp_path / "service-audit"
    audit_dir.mkdir()
    response = {
        "type": "response.completed",
        "response": {
            "model": "pinned-model",
            "reasoning": {"effort": "high"},
            "output": [
                {"type": "function_call", "name": "exec", "arguments": '{"command":"true"}'},
                {"type": "message", "content": [{"text": "Completed"}]},
            ],
        },
    }
    (audit_dir / "model-response-2.txt").write_text("data: " + json.dumps(response) + "\n")
    execution = {
        "transcript": {
            "messages": [
                {
                    "role": "assistant",
                    "text": "Completed",
                    "tool_calls": [{"name": "exec", "input": {"command": "true"}}],
                }
            ]
        }
    }
    good = model_audit(tmp_path, execution, "pinned-model")
    assert good["model_and_reasoning_match"] and good["tool_calls_match_relay"]
    assert good["assistant_text_matches_relay"]
    execution["transcript"]["messages"][0]["text"] = "Invented completion"
    execution["transcript"]["messages"][0]["tool_calls"][0]["input"]["command"] = "false"
    bad = model_audit(tmp_path, execution, "pinned-model")
    assert not bad["tool_calls_match_relay"] and not bad["assistant_text_matches_relay"]


def test_model_audit_rejects_omitted_assistant_reply(tmp_path):
    audit_dir = tmp_path / "service-audit"
    audit_dir.mkdir()
    response = {
        "type": "response.completed",
        "response": {
            "model": "pinned-model",
            "reasoning": {"effort": "high"},
            "output": [
                {"type": "message", "content": [{"text": "Working"}]},
                {"type": "message", "content": [{"text": "Important final blocker"}]},
            ],
        },
    }
    (audit_dir / "model-response-2.txt").write_text("data: " + json.dumps(response) + "\n")
    execution = {"transcript": {"messages": [{"role": "assistant", "text": "Working"}]}}
    assert not model_audit(tmp_path, execution, "pinned-model")["assistant_text_matches_relay"]
