"""Agent-agnostic workspace verification primitives.

This is the half of `environment.py` that does not touch the OpenClaw
gateway: file-state checks, execution-check subprocessing, stdout/JSON
assertions, JSON path resolution, and the filesystem/transcript-based
memory fallback readers.

Adapters (OpenClaw, Hermes, future) consume these primitives directly.
`environment.py` re-exports them for back-compat so existing callers
keep working while the gateway-tied halves (`_verify_memory` primary
path, `_verify_session`, `_verify_cron`, `_verify_gateway_assertion`)
stay where they are and move to `adapters/openclaw.py` in a later step.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from clawbench.paths import resolve_workspace_path
from clawbench.render import render_argv_template, render_shell_template, render_template, render_value
from clawbench.schemas import (
    ExecutionCheck,
    ExecutionCheckResult,
    FileState,
    MemoryState,
    Transcript,
)

logger = logging.getLogger(__name__)


def _normalize_line_endings(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


# ---------------------------------------------------------------------------
# File-state verification
# ---------------------------------------------------------------------------


def verify_file_state(
    spec: FileState,
    workspace: Path,
    runtime_values: dict[str, Any],
) -> tuple[bool, str]:
    """Verify a single `FileState` against the workspace filesystem."""

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


# ---------------------------------------------------------------------------
# Execution checks
# ---------------------------------------------------------------------------


async def run_execution_check(
    spec: ExecutionCheck,
    *,
    workspace: Path,
    runtime_values: dict[str, Any],
) -> ExecutionCheckResult:
    """Run a single `ExecutionCheck` subprocess and evaluate its output."""

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
        if spec.shell:
            process = await asyncio.create_subprocess_shell(
                rendered_command,
                cwd=str(rendered_cwd),
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            process = await asyncio.create_subprocess_exec(
                *render_argv_template(spec.command, runtime_values),
                cwd=str(rendered_cwd),
                env=full_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(),
            timeout=spec.timeout_seconds,
        )
    except asyncio.TimeoutError:
        process.kill()
        await process.communicate()
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
    passed, reason = evaluate_execution_result(
        spec, workspace, runtime_values, process.returncode, stdout, stderr
    )
    return ExecutionCheckResult(
        name=spec.name,
        command=rendered_command,
        exit_code=process.returncode,
        stdout=stdout,
        stderr=stderr,
        passed=passed,
        reason=reason,
    )


def evaluate_execution_result(
    spec: ExecutionCheck,
    workspace: Path,
    runtime_values: dict[str, Any],
    exit_code: int,
    stdout: str,
    stderr: str,
) -> tuple[bool, str]:
    """Apply every assertion declared on an `ExecutionCheck`."""

    if exit_code != spec.expected_exit_code:
        return False, f"Exit code {exit_code} != expected {spec.expected_exit_code}"

    normalized_stdout = _normalize_line_endings(stdout)
    normalized_stderr = _normalize_line_endings(stderr)

    for token in spec.stdout_contains:
        rendered = _normalize_line_endings(render_template(token, runtime_values))
        if rendered not in normalized_stdout:
            return False, f"stdout missing '{rendered}'"

    for token in spec.stdout_not_contains:
        rendered = _normalize_line_endings(render_template(token, runtime_values))
        if rendered in normalized_stdout:
            return False, f"stdout unexpectedly contains '{rendered}'"

    for token in spec.stderr_contains:
        rendered = _normalize_line_endings(render_template(token, runtime_values))
        if rendered not in normalized_stderr:
            return False, f"stderr missing '{rendered}'"

    if spec.stdout_matches and not re.search(
        _normalize_line_endings(render_template(spec.stdout_matches, runtime_values)),
        normalized_stdout,
        re.MULTILINE | re.DOTALL,
    ):
        return False, f"stdout does not match {spec.stdout_matches}"

    if spec.stderr_matches and not re.search(
        _normalize_line_endings(render_template(spec.stderr_matches, runtime_values)),
        normalized_stderr,
        re.MULTILINE | re.DOTALL,
    ):
        return False, f"stderr does not match {spec.stderr_matches}"

    if spec.expected_stdout is not None:
        rendered = _normalize_line_endings(
            render_template(spec.expected_stdout, runtime_values).strip()
        )
        if normalized_stdout.strip() != rendered:
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
        expected = _normalize_line_endings(
            expected_path.read_text(encoding="utf-8").strip()
        )
        if normalized_stdout.strip() != expected:
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


# ---------------------------------------------------------------------------
# Memory fallback: read well-known files from the workspace directly.
# ---------------------------------------------------------------------------


MEMORY_FILE_CANDIDATES: tuple[str, ...] = (
    "MEMORY.md",
    "memory.md",
    "memory/MEMORY.md",
    "memory/memory.md",
    "memory/notes.md",
    "memory/NOTES.md",
    "notes.md",
)


def read_workspace_memory_text(workspace: Path) -> str:
    """Read concatenated memory-file contents straight from the workspace.

    This is the adapter-free equivalent of
    `environment._read_agent_memory_text`, which reads the same files via
    `GatewayClient.get_agent_file`. Use this from any adapter whose agent
    runs directly in the ClawBench workspace (Hermes, Claude Code, Codex).
    """

    contents: list[str] = []
    for name in MEMORY_FILE_CANDIDATES:
        path = workspace / name
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                if text.strip():
                    contents.append(text)
        except Exception:
            continue
    return "\n".join(contents)


def memory_visible_in_transcript(spec: MemoryState, transcript: Transcript) -> bool:
    """Return True if the transcript shows a memory *write* matching `spec`.

    Same heuristic as `environment._memory_visible_in_transcript` — kept
    agent-agnostic: it reads `ToolCall.family`, `call.name`, `call.input`,
    `call.output`, `call.error`, all of which are canonical.
    """

    needle = spec.key_pattern.lower()
    for call in transcript.tool_call_sequence:
        family = (call.family or "").lower()
        name = call.name.lower()
        path = str(call.input.get("path", "")).lower()
        if family != "memory" and "memory" not in path:
            continue
        if (
            family == "memory"
            and "search" in name
            and "write" not in name
            and "store" not in name
            and "save" not in name
        ):
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


def verify_memory_fallback(
    spec: MemoryState,
    workspace: Path,
    *,
    transcript: Transcript | None = None,
    extra_memory_text: str = "",
) -> tuple[bool, str]:
    """Resolve a `MemoryState` assertion using workspace files + transcript.

    Used by any adapter that doesn't expose an OpenClaw-style
    `memory.search` RPC. The lookup strategy is deliberately permissive
    (matches the existing fallback path in `environment._verify_memory`):

    1. Concatenate every known memory file in the workspace.
    2. Optionally add any adapter-supplied text (e.g. OpenClaw's
       `_read_agent_memory_text`) via `extra_memory_text`.
    3. If the key_pattern appears (case-insensitive), check every
       `value_contains` token.
    4. If that fails, fall back to scanning the transcript for a memory
       write that matches.
    """

    memory_text = (read_workspace_memory_text(workspace) + "\n" + extra_memory_text).lower()
    needle = spec.key_pattern.lower()
    found = needle in memory_text

    if not spec.exists:
        return (not found, "Correctly absent" if not found else "Memory entry exists")

    if found:
        for token in spec.value_contains:
            if token.lower() not in memory_text:
                return False, f"Memory value missing '{token}'"
        return True, "OK"

    if transcript is not None and memory_visible_in_transcript(spec, transcript):
        return True, "Verified from transcript fallback"
    return (
        False,
        "No matching memory content found in persisted memory files or transcript fallback",
    )


# ---------------------------------------------------------------------------
# JSON-path resolver (pure function over dict/list payloads)
# ---------------------------------------------------------------------------


def resolve_json_path(payload: Any, path: str) -> Any:
    """Resolve a dotted `$.foo.bar[0].baz` path into `payload`.

    Returns None if any part of the path is missing or the type is
    wrong. Handles index syntax via `foo[3]`.
    """

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


__all__ = [
    "MEMORY_FILE_CANDIDATES",
    "evaluate_execution_result",
    "memory_visible_in_transcript",
    "read_workspace_memory_text",
    "resolve_json_path",
    "run_execution_check",
    "verify_file_state",
    "verify_memory_fallback",
]
