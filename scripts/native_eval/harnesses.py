from __future__ import annotations

import json
import secrets
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from scripts.native_eval.models import RunSpec
from scripts.native_eval.tasks import McpServer


TOOLCHAIN_ROOT = PurePosixPath("/opt/shellbench-native")
NODE_BIN = TOOLCHAIN_ROOT / "node" / "bin"
NPM_BIN = TOOLCHAIN_ROOT / "npm-packages" / "node_modules" / ".bin"
HERMES_BIN = TOOLCHAIN_ROOT / "home" / ".local" / "bin"

_OPENCLAW_CHILD_EXPORTS_READY = """\
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
if not path.is_file():
    sys.exit(1)
ready = False
spawned = {}
exported = {}
for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        continue
    if not isinstance(event, dict):
        continue
    if event.get("type") == "audit_ready":
        ready = True
        continue
    run_id = event.get("runId")
    session_key = event.get("sessionKey")
    if (
        not isinstance(run_id, str)
        or not run_id.strip()
        or not isinstance(session_key, str)
        or not session_key.strip()
    ):
        continue
    run_id = run_id.strip()
    session_key = session_key.strip()
    if event.get("type") == "subagent_spawned":
        spawned[run_id] = session_key
    elif event.get("type") == "subagent_exported":
        exported[run_id] = event
if not ready:
    sys.exit(1)
failed = [
    run_id
    for run_id in sorted(spawned)
    if run_id in exported and exported[run_id].get("exportOk") is not True
]
if failed:
    for run_id in failed:
        print(
            f"child export failed: {spawned[run_id]} ({run_id})",
            file=sys.stderr,
        )
    sys.exit(2)
if not spawned.keys() <= exported.keys():
    sys.exit(1)
for run_id in sorted(spawned):
    output = exported[run_id].get("exportOutput")
    if not isinstance(output, str) or not output.strip():
        sys.exit(2)
    print(output.strip())
"""

_OPENCLAW_EXPORT_READY = """\
import json
import pathlib
import sys

bundle = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
scope = sys.argv[3]
try:
    manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (bundle / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
except (OSError, json.JSONDecodeError):
    sys.exit(1)
if (
    manifest.get("traceSchema") != "openclaw-trajectory"
    or manifest.get("schemaVersion") != 1
    or len(events) != manifest.get("eventCount")
    or sum(event.get("source") == "runtime" for event in events)
    != manifest.get("runtimeEventCount")
    or sum(event.get("source") == "transcript" for event in events)
    != manifest.get("transcriptEventCount")
):
    sys.exit(1)
if any(
    event.get("traceId") != manifest.get("traceId")
    or event.get("sessionId") != manifest.get("sessionId")
    or event.get("sessionKey") != manifest.get("sessionKey")
    for event in events
):
    sys.exit(1)
if any(
    isinstance(warning, dict)
    and warning.get("code") in {"cyclic-session-branch", "incomplete-session-branch"}
    for warning in manifest.get("warnings", [])
):
    sys.exit(1)
runtime = [event for event in events if event.get("source") == "runtime"]
terminal = next(
    (event for event in reversed(runtime) if event.get("type") == "session.ended"),
    None,
)
if terminal is None:
    sys.exit(1)
run_id = terminal.get("runId")
terminal_data = terminal.get("data")
terminal_status = (
    terminal_data.get("status") if isinstance(terminal_data, dict) else None
)
if scope == "root" and terminal_status != "success":
    sys.exit(1)
completion = next(
    (
        event
        for event in reversed(runtime)
        if event.get("type") == "model.completed"
        and (not isinstance(run_id, str) or event.get("runId") == run_id)
    ),
    None,
)
if completion is None:
    if scope != "child" or terminal_status not in {"error", "interrupted"}:
        sys.exit(1)
if mode == "code" and completion is not None:
    completion_data = completion.get("data")
    snapshot = (
        completion_data.get("messagesSnapshot")
        if isinstance(completion_data, dict)
        else None
    )
    if (
        not isinstance(completion_data, dict)
        or completion_data.get("truncated") is True
        or not isinstance(snapshot, list)
        or not snapshot
        or not all(isinstance(message, dict) for message in snapshot)
        or not {"user", "assistant"} <= {
            message.get("role")
            for message in snapshot
            if isinstance(message.get("role"), str)
        }
    ):
        sys.exit(1)
    context = next(
        (
            event
            for event in reversed(runtime)
            if event.get("type") == "context.compiled"
            and (not isinstance(run_id, str) or event.get("runId") == run_id)
        ),
        None,
    )
    data = context.get("data") if isinstance(context, dict) else None
    if not isinstance(data, dict) or "providerVisibleTools" not in data:
        sys.exit(1)
    tools = data.get("providerVisibleTools")
    names = sorted(
        tool.get("name")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ) if isinstance(tools, list) else []
    name_set = set(names)
    if (
        not {"exec", "wait"} <= name_set
        or name_set
        & {"tool_search_code", "tool_search", "tool_describe", "tool_call"}
    ):
        sys.exit(1)
"""

_OPENCLAW_GATEWAY_PROBE = """\
import json
import sys
import urllib.request

request = urllib.request.Request(
    "http://127.0.0.1:18789/readyz",
    headers={"Authorization": f"Bearer {sys.argv[1]}"},
)
try:
    with urllib.request.urlopen(request, timeout=1) as response:
        payload = json.load(response)
        sys.exit(
            0
            if response.status == 200
            and isinstance(payload, dict)
            and payload.get("ready") is True
            else 1
        )
except Exception:
    sys.exit(1)
"""

_OPENCLAW_AUDIT_PLUGIN = """\
const crypto = require("node:crypto");
const childProcess = require("node:child_process");
const fs = require("node:fs");
const path = require("node:path");

const auditRoot = path.join(process.env.HOME, ".openclaw", "shellbench-audit");
const eventPath = path.join(auditRoot, "sessions.jsonl");
const workspace = process.env.SHELLBENCH_WORKSPACE || process.cwd();
const exportsByRun = new Map();

function append(event) {
  fs.mkdirSync(auditRoot, { recursive: true });
  fs.appendFileSync(eventPath, `${JSON.stringify(event)}\\n`, { mode: 0o600 });
}

async function exportRun(runId, sessionKey) {
  const cacheKey = `${runId}\\0${sessionKey}`;
  const cached = exportsByRun.get(cacheKey);
  if (cached) {
    return await cached;
  }
  const digest = crypto.createHash("sha256").update(cacheKey).digest("hex").slice(0, 16);
  const output = `shellbench-child-${digest}`;
  const pending = new Promise((resolve) => {
    childProcess.execFile(
      "openclaw",
      [
        "sessions",
        "export-trajectory",
        "--session-key",
        sessionKey,
        "--workspace",
        workspace,
        "--output",
        output,
        "--json",
      ],
      {
        cwd: workspace,
        encoding: "utf8",
        env: process.env,
        timeout: 120000,
      },
      (error, _stdout, stderr) => {
        resolve({
          exportOk: !error,
          exportOutput: output,
          exportStatus:
            typeof error?.code === "number" ? error.code : error ? null : 0,
          exportError: error ? String(error.message || error) : undefined,
          exportStderr:
            typeof stderr === "string" && stderr.trim()
              ? stderr.trim().slice(-4000)
              : undefined,
        });
      },
    );
  });
  exportsByRun.set(cacheKey, pending);
  return await pending;
}

module.exports = {
  id: "shellbench-audit",
  register(api) {
    append({ type: "audit_ready" });
    api.on("subagent_spawned", (event, ctx) => {
      append({
        type: "subagent_spawned",
        sessionKey: event.childSessionKey,
        spawnedBy: ctx.requesterSessionKey,
        runId: event.runId,
      });
    });
    api.on("subagent_progress", async (event, ctx) => {
      if (event.phase !== "ended") {
        return;
      }
      const exported = await exportRun(event.runId, event.childSessionKey);
      append({
        type: "subagent_exported",
        sessionKey: event.childSessionKey,
        spawnedBy: ctx.requesterSessionKey,
        runId: event.runId,
        status: event.outcome,
        ...exported,
      });
    });
    api.on("subagent_ended", (event, ctx) => {
      append({
        type: "subagent_ended",
        sessionKey: event.targetSessionKey,
        spawnedBy: ctx.requesterSessionKey,
        runId: event.runId,
        status: event.outcome,
        reason: event.reason,
      });
    });
  },
};
"""

_OPENCLAW_AUDIT_PLUGIN_MANIFEST = {
    "id": "shellbench-audit",
    "configSchema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {},
    },
}


@dataclass(frozen=True)
class HarnessCommand:
    setup_command: str
    run_command: str
    cleanup_command: str
    env: dict[str, str]


def build_harness_command(
    run: RunSpec,
    *,
    proxy_url: str,
    proxy_key: str,
    mcp_servers: tuple[McpServer, ...],
    agent_timeout_sec: float | None = None,
) -> HarnessCommand:
    builders = {
        "openclaw": _openclaw,
        "hermes": _hermes,
        "codex": _codex,
        "claude-code": _claude_code,
    }
    try:
        builder = builders[run.harness]
    except KeyError as exc:
        raise ValueError(f"Unsupported harness: {run.harness}") from exc
    if run.harness == "openclaw":
        return _openclaw(
            run,
            proxy_url,
            proxy_key,
            mcp_servers,
            agent_timeout_sec=agent_timeout_sec,
        )
    return builder(run, proxy_url, proxy_key, mcp_servers)


def _base_path() -> str:
    return f"{NODE_BIN}:{NPM_BIN}:{HERMES_BIN}:$PATH"


def _openclaw(
    run: RunSpec,
    proxy_url: str,
    proxy_key: str,
    mcp_servers: tuple[McpServer, ...],
    *,
    agent_timeout_sec: float | None,
) -> HarnessCommand:
    provider = "openai"
    model = f"{provider}/{run.model_id}"
    thinking = run.reasoning_effort or "off"
    tool_mode = run.openclaw_tool_mode or "direct"
    home = "/tmp/shellbench-openclaw"
    audit_plugin_root = f"{home}/.openclaw/shellbench-audit"
    gateway_token = secrets.token_urlsafe(32)
    servers: dict[str, dict[str, object]] = {}
    for server in mcp_servers:
        if server.transport == "stdio":
            servers[server.name] = {
                "command": server.command,
                "args": list(server.args),
            }
        else:
            servers[server.name] = {
                "url": server.url,
                "transport": server.transport,
            }
    config = {
        "agents": {
            "defaults": {
                "workspace": ".",
                "skipBootstrap": True,
                "model": {"primary": model},
                "thinkingDefault": thinking,
                "subagents": {"model": model, "thinking": thinking},
                **(
                    {"timeoutSeconds": max(1, int(agent_timeout_sec))}
                    if agent_timeout_sec is not None
                    else {}
                ),
            }
        },
        "gateway": {
            "mode": "local",
            "auth": {"mode": "token", "token": gateway_token},
        },
        "models": {
            "providers": {
                provider: {
                    "baseUrl": f"{proxy_url.rstrip('/')}/v1",
                    "api": "openai-responses",
                    "apiKey": proxy_key,
                    "agentRuntime": {"id": "openclaw"},
                    "models": [
                        {
                            "id": run.model_id,
                            "name": run.model_id,
                        }
                    ],
                }
            }
        },
        "plugins": {
            "load": {"paths": [audit_plugin_root]},
            "entries": {"shellbench-audit": {"enabled": True}},
        },
        "tools": {
            "deny": ["message", "computer"],
            "codeMode": run.openclaw_tool_mode == "code",
            "toolSearch": (
                {"enabled": True, "mode": "directory"}
                if run.openclaw_tool_mode == "directory"
                else False
            ),
        },
    }
    if servers:
        config["mcp"] = {"servers": servers}
    config_json = shlex.quote(json.dumps(config, separators=(",", ":")))
    child_exports_ready = shlex.quote(_OPENCLAW_CHILD_EXPORTS_READY)
    export_ready = shlex.quote(_OPENCLAW_EXPORT_READY)
    gateway_probe = shlex.quote(_OPENCLAW_GATEWAY_PROBE)
    audit_plugin = shlex.quote(_OPENCLAW_AUDIT_PLUGIN)
    audit_manifest = shlex.quote(json.dumps(_OPENCLAW_AUDIT_PLUGIN_MANIFEST, separators=(",", ":")))
    setup = (
        f"set -eu; export PATH={_base_path()}; export HOME={home}; "
        'rm -rf "$HOME"; mkdir -p "$HOME/.openclaw/shellbench-audit"; '
        f'printf %s {audit_plugin} > "$HOME/.openclaw/shellbench-audit/index.cjs"; '
        f"printf %s {audit_manifest} "
        '> "$HOME/.openclaw/shellbench-audit/openclaw.plugin.json"; '
        f'printf %s {config_json} > "$HOME/.openclaw/openclaw.json"; '
        "openclaw setup --baseline --workspace . "
        ">/logs/agent/setup.log 2>&1; "
        "rm -f AGENTS.md BOOTSTRAP.md HEARTBEAT.md IDENTITY.md "
        "SOUL.md TOOLS.md USER.md; "
        f'printf %s {config_json} > "$HOME/.openclaw/openclaw.json"'
    )
    run_command = (
        f"export PATH={_base_path()}; export HOME={home}; "
        f"export OPENCLAW_GATEWAY_TOKEN={shlex.quote(gateway_token)}; "
        "export SHELLBENCH_WORKSPACE=\"$PWD\"; "
        "log=/logs/agent/openclaw.txt; "
        "gateway_log=/logs/agent/openclaw-gateway.txt; "
        'openclaw gateway --port 18789 >"$gateway_log" 2>&1 & gateway_pid=$!; '
        "ready=0; for _ in $(seq 1 60); do "
        f'if python3 -c {gateway_probe} "$OPENCLAW_GATEWAY_TOKEN"; then '
        "ready=1; break; fi; sleep 1; done; "
        'if [ "$ready" -ne 1 ]; then '
        'kill "$gateway_pid" 2>/dev/null || true; '
        'wait "$gateway_pid" 2>/dev/null || true; '
        'cat "$gateway_log" >&2; exit 70; fi; '
        "openclaw agent --json --agent main "
        f"--thinking {shlex.quote(thinking)} "
        f"--model {shlex.quote(model)} "
        '--message "$(cat /tmp/shellbench-instruction.md)" '
        '>"$log" 2>&1 </dev/null; status=$?; '
        'if [ "$status" -eq 0 ]; then '
        "export_ok=0; for _ in $(seq 1 10); do "
        'rm -rf .openclaw/trajectory-exports/shellbench-root; '
        "if openclaw sessions export-trajectory "
        '--session-key "agent:main:main" --workspace . '
        '--output shellbench-root --json >>"$log" 2>&1 '
        f"&& python3 -c {export_ready} "
        '".openclaw/trajectory-exports/shellbench-root" '
        f"{shlex.quote(tool_mode)} root; then "
        "export_ok=1; break; fi; sleep 1; done; "
        'if [ "$export_ok" -ne 1 ]; then '
        "echo 'OpenClaw root trajectory export failed' >>\"$log\"; status=71; fi; "
        "fi; "
        'if [ "$status" -eq 0 ]; then '
        "child_wait=0; while true; do "
        f"python3 -c {child_exports_ready} "
        '"$HOME/.openclaw/shellbench-audit/sessions.jsonl" '
        ">/tmp/shellbench-openclaw-child-exports.txt 2>>\"$log\"; "
        "child_state=$?; "
        'if [ "$child_state" -eq 0 ]; then break; fi; '
        'if [ "$child_state" -eq 2 ]; then status=71; break; fi; '
        'if ! kill -0 "$gateway_pid" 2>/dev/null; then '
        "echo 'OpenClaw gateway exited while child exports were pending' "
        '>>"$log"; status=70; break; fi; '
        "child_wait=$((child_wait + 1)); "
        'if [ "$child_wait" -ge 300 ]; then '
        "echo 'OpenClaw child trajectory exports did not settle within 300s' "
        '>>"$log"; status=71; break; fi; '
        "sleep 1; done; fi; "
        'if [ "$status" -eq 0 ]; then '
        "while IFS= read -r output; do "
        '[ -n "$output" ] || continue; '
        f"if ! python3 -c {export_ready} "
        '".openclaw/trajectory-exports/$output" '
        f"{shlex.quote(tool_mode)} child; then "
        'echo "OpenClaw child trajectory validation failed: $output" '
        '>>"$log"; status=71; break; fi; '
        "done </tmp/shellbench-openclaw-child-exports.txt; fi; "
        'kill "$gateway_pid" 2>/dev/null || true; '
        'wait "$gateway_pid" 2>/dev/null || true; '
        'cat "$gateway_log" >>"$log"; cat "$log"; exit "$status"'
    )
    cleanup = (
        "python3 - <<'PY'\n"
        "import pathlib, shutil\n"
        "archive=pathlib.Path('/logs/agent/openclaw.sessions')\n"
        "audit=pathlib.Path('/tmp/shellbench-openclaw/.openclaw/shellbench-audit')\n"
        "exports=pathlib.Path('.openclaw/trajectory-exports')\n"
        "if exports.is_dir():\n"
        " shutil.copytree(exports, archive/'exports', dirs_exist_ok=True)\n"
        "if audit.is_dir():\n"
        " shutil.copytree(audit, archive/'audit', dirs_exist_ok=True)\n"
        "if archive.is_dir():\n"
        " for item in [archive, *archive.rglob('*')]:\n"
        "  item.chmod(0o755 if item.is_dir() else 0o644)\n"
        "PY"
    )
    return HarnessCommand(
        setup_command=setup,
        run_command=run_command,
        cleanup_command=cleanup,
        env={
            "OPENAI_API_KEY": proxy_key,
            "OPENAI_BASE_URL": f"{proxy_url.rstrip('/')}/v1",
            "OPENCLAW_GATEWAY_TOKEN": gateway_token,
        },
    )


def _hermes(
    run: RunSpec,
    proxy_url: str,
    proxy_key: str,
    mcp_servers: tuple[McpServer, ...],
) -> HarnessCommand:
    home = "/tmp/shellbench-hermes"
    provider_name = "custom:shellbench"
    config: dict[str, object] = {
        "model": {
            "default": run.model_id,
            "provider": provider_name,
        },
        "providers": {
            "shellbench": {
                "name": "shellbench",
                "api": f"{proxy_url.rstrip('/')}/v1",
                "key_env": "SHELLBENCH_PROXY_KEY",
                "transport": "chat_completions",
                "default_model": run.model_id,
                "models": {run.model_id: {}},
            }
        },
        "toolsets": ["hermes-cli"],
        "agent": {"max_turns": 90},
        "memory": {"memory_enabled": False, "user_profile_enabled": False},
        "compression": {"enabled": True, "threshold": 0.85},
        "terminal": {"backend": "local", "timeout": 180},
        "delegation": {
            "max_iterations": 50,
            "provider": provider_name,
            "model": run.model_id,
        },
        "checkpoints": {"enabled": False},
    }
    if mcp_servers:
        config["mcp_servers"] = {
            server.name: (
                {"command": server.command, "args": list(server.args)}
                if server.transport == "stdio"
                else {"url": server.url}
            )
            for server in mcp_servers
        }
    config_json = shlex.quote(json.dumps(config, separators=(",", ":")))
    setup = (
        f"export PATH={_base_path()}; export HERMES_HOME={home}; "
        'rm -rf "$HERMES_HOME"; mkdir -p "$HERMES_HOME"; '
        f'printf %s {config_json} > "$HERMES_HOME/config.yaml"'
    )
    run_command = (
        f"export PATH={_base_path()}; export HERMES_HOME={home}; "
        'export TERMINAL_ENV=local; export TERMINAL_CWD="$PWD"; '
        'hermes --yolo chat -q "$(cat /tmp/shellbench-instruction.md)" -Q '
        f"--model {shlex.quote(run.model_id)} "
        f"--provider {shlex.quote(provider_name)} "
        ">/logs/agent/hermes.txt 2>&1; status=$?; "
        "cat /logs/agent/hermes.txt; "
        'if grep -Eq "Unknown provider|No LLM provider configured" '
        "/logs/agent/hermes.txt; then exit 64; fi; "
        'exit "$status"'
    )
    cleanup = (
        f"export PATH={_base_path()}; export HERMES_HOME={home}; "
        "hermes sessions export /logs/agent/hermes-session.jsonl "
        "--yes --redact"
    )
    return HarnessCommand(
        setup_command=setup,
        run_command=run_command,
        cleanup_command=cleanup,
        env={
            "SHELLBENCH_PROXY_KEY": proxy_key,
        },
    )


def _codex(
    run: RunSpec,
    proxy_url: str,
    proxy_key: str,
    mcp_servers: tuple[McpServer, ...],
) -> HarnessCommand:
    home = "/tmp/shellbench-codex"
    config_lines = [f'openai_base_url = "{proxy_url.rstrip("/")}/v1"']
    for server in mcp_servers:
        config_lines.append(f"[mcp_servers.{server.name}]")
        if server.transport == "stdio":
            command = shlex.join([server.command or "", *server.args])
            config_lines.append(f"command = {json.dumps(command)}")
        else:
            config_lines.append(f"url = {json.dumps(server.url)}")
    config_text = shlex.quote("\n".join(config_lines) + "\n")
    auth_json = shlex.quote(json.dumps({"OPENAI_API_KEY": proxy_key}))
    setup = (
        f"export PATH={_base_path()}; export CODEX_HOME={home}; "
        'rm -rf "$CODEX_HOME"; mkdir -p "$CODEX_HOME"; '
        f'printf %s {auth_json} > "$CODEX_HOME/auth.json"; '
        f'printf %s {config_text} > "$CODEX_HOME/config.toml"'
    )
    run_command = (
        f"export PATH={_base_path()}; export CODEX_HOME={home}; "
        "codex exec --dangerously-bypass-approvals-and-sandbox "
        "--skip-git-repo-check "
        f"--model {shlex.quote(run.model_id)} "
        "--json --enable unified_exec -- "
        '"$(cat /tmp/shellbench-instruction.md)" '
        ">/logs/agent/codex.txt 2>/logs/agent/codex-stderr.txt "
        "</dev/null; status=$?; "
        "cat /logs/agent/codex.txt; "
        'cat /logs/agent/codex-stderr.txt >&2; exit "$status"'
    )
    cleanup = (
        "rm -rf /logs/agent/sessions; "
        f"if [ -d {home}/sessions ]; then cp -R {home}/sessions /logs/agent/sessions; fi"
    )
    return HarnessCommand(
        setup_command=setup,
        run_command=run_command,
        cleanup_command=cleanup,
        env={
            "OPENAI_API_KEY": proxy_key,
            "OPENAI_BASE_URL": f"{proxy_url.rstrip('/')}/v1",
        },
    )


def _claude_code(
    run: RunSpec,
    proxy_url: str,
    proxy_key: str,
    mcp_servers: tuple[McpServer, ...],
) -> HarnessCommand:
    home = "/tmp/shellbench-claude"
    servers: dict[str, dict[str, object]] = {}
    for server in mcp_servers:
        if server.transport == "stdio":
            servers[server.name] = {
                "type": "stdio",
                "command": server.command,
                "args": list(server.args),
            }
        else:
            servers[server.name] = {
                "type": "http" if server.transport == "streamable-http" else server.transport,
                "url": server.url,
            }
    mcp_json = shlex.quote(json.dumps({"mcpServers": servers}, separators=(",", ":")))
    setup = (
        f"export PATH={_base_path()}; export CLAUDE_CONFIG_DIR={home}; "
        'rm -rf "$CLAUDE_CONFIG_DIR"; '
        'mkdir -p "$CLAUDE_CONFIG_DIR/debug" "$CLAUDE_CONFIG_DIR/projects/-app"; '
        f'printf %s {mcp_json} > "$CLAUDE_CONFIG_DIR/.claude.json"'
    )
    run_command = (
        f"export PATH={_base_path()}; export CLAUDE_CONFIG_DIR={home}; "
        "claude --verbose --output-format=stream-json "
        "--permission-mode=bypassPermissions --print "
        f"--model {shlex.quote(run.model_id)} "
        '"$(cat /tmp/shellbench-instruction.md)" '
        ">/logs/agent/claude-code.txt 2>&1 </dev/null; status=$?; "
        'cat /logs/agent/claude-code.txt; exit "$status"'
    )
    return HarnessCommand(
        setup_command=setup,
        run_command=run_command,
        cleanup_command="true",
        env={
            "ANTHROPIC_API_KEY": proxy_key,
            "ANTHROPIC_AUTH_TOKEN": proxy_key,
            "ANTHROPIC_BASE_URL": proxy_url.rstrip("/"),
            "ANTHROPIC_MODEL": run.model_id,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": run.model_id,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": run.model_id,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": run.model_id,
            "CLAUDE_CODE_SUBAGENT_MODEL": run.model_id,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "IS_SANDBOX": "1",
        },
    )
