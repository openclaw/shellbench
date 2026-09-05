"""Completion verification for ClawBench v0.3."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from clawbench.client import GatewayClient
from clawbench.environment_files import _execution_subprocess_kwargs, _reap_timed_out_process
from clawbench.paths import resolve_workspace_path
from clawbench.render import render_argv_template, render_shell_template, render_template, render_value
from clawbench.schemas import (
    CompletionResult,
    CompletionSpec,
    CronState,
    ExecutionCheck,
    ExecutionCheckResult,
    FileState,
    GatewayAssertion,
    MemoryState,
    SessionState,
    Transcript,
)

logger = logging.getLogger(__name__)


async def verify_completion(
    completion: CompletionSpec,
    *,
    workspace: Path,
    client: GatewayClient,
    session_key: str,
    agent_id: str | None = None,
    runtime_values: dict[str, Any],
    transcript: Transcript | None = None,
) -> CompletionResult:
    total = 0
    passed = 0
    failures: list[str] = []
    execution_results: list[ExecutionCheckResult] = []

    for spec in completion.files:
        ok, reason = _verify_file(spec, workspace, runtime_values)
        total += 1
        if ok:
            passed += 1
        else:
            failures.append(f"FILE {spec.path}: {reason}")

    for spec in completion.memory:
        ok, reason = await _verify_memory(spec, client, session_key, agent_id=agent_id, transcript=transcript)
        total += 1
        if ok:
            passed += 1
        else:
            failures.append(f"MEMORY {spec.key_pattern}: {reason}")

    if completion.session:
        ok, reason = await _verify_session(completion.session, client, session_key)
        total += 1
        if ok:
            passed += 1
        else:
            failures.append(f"SESSION: {reason}")

    for spec in completion.cron:
        ok, reason = await _verify_cron(spec, client)
        total += 1
        if ok:
            passed += 1
        else:
            failures.append(f"CRON: {reason}")

    for spec in completion.gateway_assertions:
        ok, reason = await _verify_gateway_assertion(spec, client)
        total += 1
        if ok:
            passed += 1
        else:
            failures.append(f"GATEWAY {spec.method}:{spec.assert_path}: {reason}")

    for spec in completion.execution_checks:
        result = await run_execution_check(spec, workspace=workspace, runtime_values=runtime_values)
        execution_results.append(result)
        total += 1
        if result.passed:
            passed += 1
        else:
            failures.append(f"EXEC {spec.name}: {result.reason}")

    score = passed / total if total else 1.0
    return CompletionResult(
        total_assertions=total,
        passed_assertions=passed,
        failed_assertions=failures,
        execution_results=execution_results,
        score=round(score, 4),
    )


async def run_execution_check(
    spec: ExecutionCheck,
    *,
    workspace: Path,
    runtime_values: dict[str, Any],
) -> ExecutionCheckResult:
    rendered_command = (
        render_shell_template(spec.command, runtime_values)
        if spec.shell
        else render_template(spec.command, runtime_values)
    )
    try:
        rendered_cwd = resolve_workspace_path(
            workspace,
            render_template(spec.cwd, runtime_values),
            field=f"execution check cwd for {spec.name}",
        )
    except ValueError as exc:
        return ExecutionCheckResult(
            name=spec.name,
            command=rendered_command,
            exit_code=-1,
            passed=False,
            reason=str(exc),
        )
    rendered_env = render_value(spec.env, runtime_values)

    full_env = {
        **os.environ,
        **{key: str(value) for key, value in rendered_env.items()},
        "PYTHONUNBUFFERED": "1",
    }
    python_bin_dir = str(Path(sys.executable).parent)
    full_env["PATH"] = f"{python_bin_dir}:{full_env.get('PATH', '')}"
    python_path_parts = [str(rendered_cwd), str(workspace)]
    existing_pythonpath = full_env.get("PYTHONPATH")
    if existing_pythonpath:
        python_path_parts.append(existing_pythonpath)
    full_env["PYTHONPATH"] = ":".join(python_path_parts)

    try:
        spawn_kwargs = _execution_subprocess_kwargs()
        if spec.shell:
            process = await asyncio.create_subprocess_shell(
                rendered_command,
                cwd=str(rendered_cwd),
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **spawn_kwargs,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *render_argv_template(spec.command, runtime_values),
                cwd=str(rendered_cwd),
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **spawn_kwargs,
            )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=spec.timeout_seconds,
        )
    except asyncio.TimeoutError:
        await _reap_timed_out_process(process, spec.timeout_seconds)
        return ExecutionCheckResult(
            name=spec.name,
            command=rendered_command,
            exit_code=-1,
            passed=False,
            reason=f"Timed out after {spec.timeout_seconds}s",
        )
    except Exception as exc:
        return ExecutionCheckResult(
            name=spec.name,
            command=rendered_command,
            exit_code=-1,
            passed=False,
            reason=str(exc),
        )

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    passed, reason = _evaluate_execution_result(spec, workspace, runtime_values, process.returncode, stdout, stderr)
    return ExecutionCheckResult(
        name=spec.name,
        command=rendered_command,
        exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        passed=passed,
        reason=reason,
    )


def _evaluate_execution_result(
    spec: ExecutionCheck,
    workspace: Path,
    runtime_values: dict[str, Any],
    exit_code: int,
    stdout: str,
    stderr: str,
) -> tuple[bool, str]:
    if exit_code != spec.expected_exit_code:
        return False, f"Exit code {exit_code} != expected {spec.expected_exit_code}"

    for token in spec.stdout_contains:
        rendered = render_template(token, runtime_values)
        if rendered not in stdout:
            return False, f"stdout missing '{rendered}'"

    for token in spec.stdout_not_contains:
        rendered = render_template(token, runtime_values)
        if rendered in stdout:
            return False, f"stdout unexpectedly contains '{rendered}'"

    for token in spec.stderr_contains:
        rendered = render_template(token, runtime_values)
        if rendered not in stderr:
            return False, f"stderr missing '{rendered}'"

    if spec.stdout_matches and not re.search(render_template(spec.stdout_matches, runtime_values), stdout, re.MULTILINE | re.DOTALL):
        return False, f"stdout does not match {spec.stdout_matches}"

    if spec.stderr_matches and not re.search(render_template(spec.stderr_matches, runtime_values), stderr, re.MULTILINE | re.DOTALL):
        return False, f"stderr does not match {spec.stderr_matches}"

    if spec.expected_stdout is not None:
        rendered = render_template(spec.expected_stdout, runtime_values).strip()
        if stdout.strip() != rendered:
            return False, "stdout did not match expected text"

    if spec.expected_stdout_file:
        try:
            expected_path = resolve_workspace_path(
                workspace,
                render_template(spec.expected_stdout_file, runtime_values),
                field=f"expected_stdout_file for {spec.name}",
            )
        except ValueError as exc:
            return False, str(exc)
        if stdout.strip() != expected_path.read_text(encoding="utf-8").strip():
            return False, f"stdout did not match {spec.expected_stdout_file}"

    if spec.expected_json is not None:
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return False, f"stdout was not valid JSON: {exc}"
        if parsed != render_value(spec.expected_json, runtime_values):
            return False, "stdout JSON did not match expected JSON"

    if spec.expected_json_file:
        try:
            expected_path = resolve_workspace_path(
                workspace,
                render_template(spec.expected_json_file, runtime_values),
                field=f"expected_json_file for {spec.name}",
            )
        except ValueError as exc:
            return False, str(exc)
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError as exc:
            return False, f"stdout was not valid JSON: {exc}"
        expected_json = json.loads(expected_path.read_text(encoding="utf-8"))
        if parsed != expected_json:
            return False, f"stdout JSON did not match {spec.expected_json_file}"

    return True, "OK"


def _verify_file(spec: FileState, workspace: Path, runtime_values: dict[str, Any]) -> tuple[bool, str]:
    try:
        path = resolve_workspace_path(
            workspace,
            render_template(spec.path, runtime_values),
            field=f"completion file {spec.path}",
        )
    except ValueError as exc:
        return False, str(exc)
    exists = path.exists() and path.is_file()

    if not spec.exists:
        return (not exists, "Correctly absent" if not exists else "File should not exist")
    if not exists:
        return False, "File does not exist"

    content = path.read_text(encoding="utf-8", errors="replace")
    if spec.min_size_bytes > 0 and path.stat().st_size < spec.min_size_bytes:
        return False, f"File too small: {path.stat().st_size} < {spec.min_size_bytes}"

    for token in spec.content_contains:
        rendered = render_template(token, runtime_values)
        if rendered not in content:
            return False, f"Missing expected content '{rendered}'"

    for token in spec.content_not_contains:
        rendered = render_template(token, runtime_values)
        if rendered in content:
            return False, f"Contains forbidden content '{rendered}'"

    if spec.content_matches and not re.search(
        render_template(spec.content_matches, runtime_values),
        content,
        re.MULTILINE | re.DOTALL,
    ):
        return False, f"Content does not match {spec.content_matches}"

    return True, "OK"


async def _verify_memory(
    spec: MemoryState,
    client: GatewayClient,
    session_key: str,
    *,
    agent_id: str | None = None,
    transcript: Transcript | None = None,
) -> tuple[bool, str]:
    try:
        response = await client._rpc(
            "memory.search",
            {
                "query": spec.key_pattern,
                "sessionKey": session_key,
                "limit": 20,
            },
        )
        entries = response.get("payload", {}).get("entries", [])
        if not spec.exists:
            return (not entries, "Correctly absent" if not entries else "Memory entry exists")
        if not entries:
            return False, "No matching memory entries found"
        all_values = " ".join(str(entry.get("value", "")) for entry in entries)
        for token in spec.value_contains:
            if token.lower() not in all_values.lower():
                return False, f"Memory value missing '{token}'"
        return True, "OK"
    except Exception as exc:
        logger.info("memory.search unavailable for verification, falling back to agent memory files: %s", exc)

    if not agent_id:
        return False, "memory.search unavailable and no agent id was provided for fallback verification"

    fallback_text = await _read_agent_memory_text(client, agent_id)
    normalized = fallback_text.lower()
    needle = spec.key_pattern.lower()
    found = needle in normalized

    if not spec.exists:
        return (not found, "Correctly absent" if not found else "Memory entry exists")
    if found:
        for token in spec.value_contains:
            if token.lower() not in normalized:
                return False, f"Memory value missing '{token}'"
        return True, "OK"

    if transcript and _memory_visible_in_transcript(spec, transcript):
        return True, "Verified from transcript fallback"
    return False, "No matching memory content found in persisted memory files or transcript fallback"


async def _read_agent_memory_text(client: GatewayClient, agent_id: str) -> str:
    contents: list[str] = []
    for file_name in (
        "MEMORY.md",
        "memory.md",
        "memory/MEMORY.md",
        "memory/memory.md",
        "memory/notes.md",
        "memory/NOTES.md",
        "notes.md",
    ):
        try:
            payload = await client.get_agent_file(agent_id, file_name)
        except Exception:
            continue
        file_entry = payload.get("file", {})
        content = file_entry.get("content", "")
        if isinstance(content, str) and content.strip():
            contents.append(content)
    return "\n".join(contents)


def _memory_visible_in_transcript(spec: MemoryState, transcript: Transcript) -> bool:
    needle = spec.key_pattern.lower()
    for call in transcript.tool_call_sequence:
        family = (call.family or "").lower()
        name = call.name.lower()
        path = str(call.input.get("path", "")).lower()
        if family != "memory" and "memory" not in path:
            continue
        if family == "memory" and "search" in name and "write" not in name and "store" not in name and "save" not in name:
            continue

        serialized_bits = [call.output, call.error]
        try:
            serialized_bits.append(json.dumps(call.input, sort_keys=True))
        except TypeError:
            serialized_bits.append(str(call.input))
        haystack = " ".join(bit for bit in serialized_bits if bit).lower()
        if needle not in haystack:
            continue
        if all(token.lower() in haystack for token in spec.value_contains):
            return True
    return False


async def _verify_session(
    spec: SessionState,
    client: GatewayClient,
    session_key: str,
) -> tuple[bool, str]:
    try:
        response = await client._rpc("sessions.resolve", {"key": session_key})
        payload = response.get("payload", {})
        if not spec.should_exist:
            return False, "Session exists but should not"
        if spec.model_should_be:
            actual = str(payload.get("model", ""))
            if spec.model_should_be.lower() not in actual.lower():
                return False, f"Model mismatch: expected {spec.model_should_be}, got {actual}"
        return True, "OK"
    except Exception as exc:
        if not spec.should_exist:
            return True, "Correctly absent"
        return False, str(exc)


async def _verify_cron(spec: CronState, client: GatewayClient) -> tuple[bool, str]:
    try:
        response = await client._rpc("cron.list", {})
        jobs = response.get("payload", {}).get("jobs", [])
        if not spec.exists:
            return (not jobs, "Correctly absent" if not jobs else "Cron jobs exist")
        if not jobs:
            return False, "No cron jobs found"
        if spec.description_contains and not any(
            spec.description_contains.lower() in json.dumps(job).lower()
            for job in jobs
        ):
            return False, f"No cron job matched '{spec.description_contains}'"
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


async def _verify_gateway_assertion(
    spec: GatewayAssertion,
    client: GatewayClient,
) -> tuple[bool, str]:
    try:
        response = await client._rpc(spec.method, spec.params)
        payload = response.get("payload", {})
        value = _resolve_path(payload, spec.assert_path)
        if not spec.assert_exists:
            return (value is None, "Correctly absent" if value is None else "Path exists")
        if value is None:
            return False, f"Path {spec.assert_path} not found"
        if spec.assert_equals is not None and value != spec.assert_equals:
            return False, f"Expected {spec.assert_equals}, got {value}"
        if spec.assert_contains is not None and spec.assert_contains.lower() not in str(value).lower():
            return False, f"Expected '{spec.assert_contains}' in {value}"
        return True, "OK"
    except Exception as exc:
        return False, str(exc)


def _resolve_path(payload: Any, path: str) -> Any:
    if path == "$":
        return payload
    current = payload
    for part in path.lstrip("$").lstrip(".").split("."):
        if not part:
            continue
        match = re.fullmatch(r"([^\[]+)\[(\d+)\]", part)
        if match:
            key, index = match.groups()
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
            if not isinstance(current, list):
                return None
            idx = int(index)
            if idx >= len(current):
                return None
            current = current[idx]
            continue
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current
