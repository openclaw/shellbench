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

_OPENCLAW_COMPLETION_PROBE = """\
import json
import pathlib
import sys

try:
    text = pathlib.Path(sys.argv[1]).read_text(
        encoding="utf-8",
        errors="replace",
    ).strip()
except OSError:
    sys.exit(1)

decoder = json.JSONDecoder()
for start in range(len(text) - 1, -1, -1):
    if text[start] != "{":
        continue
    try:
        value, _ = decoder.raw_decode(text[start:])
    except (json.JSONDecodeError, ValueError):
        continue
    if (
        isinstance(value, dict)
        and isinstance(value.get("payloads"), list)
        and isinstance(value.get("meta"), dict)
    ):
        meta = value["meta"]
        liveness = str(meta.get("livenessState") or "").lower()
        if meta.get("yielded") is True or liveness in {
            "active",
            "paused",
            "running",
            "waiting",
        }:
            continue
        completion = meta.get("completion")
        stop_reason = (
            completion.get("stopReason")
            if isinstance(completion, dict)
            else meta.get("stopReason")
        )
        visible_payload = any(
            isinstance(item, dict)
            and isinstance(item.get("text"), str)
            and item["text"].strip()
            and item.get("isReasoning") is not True
            for item in value["payloads"]
        )
        if visible_payload or stop_reason or meta.get("aborted") is True:
            sys.exit(0)
sys.exit(1)
"""

# ShellBench pins OpenClaw 2026.7.1-2, whose runtime session contract is the
# sessions.json registry plus JSONL transcripts under each agent's sessions dir.
_OPENCLAW_SESSION_PROBE = """\
import json
import pathlib
import sys

sessions = pathlib.Path(sys.argv[1])
audit_root = pathlib.Path(sys.argv[2])
try:
    store = json.loads((sessions / "sessions.json").read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    sys.exit(1)
if not isinstance(store, dict):
    sys.exit(1)

audit = {}
audit_path = audit_root / "sessions.jsonl"
if audit_path.is_file():
    for line in audit_path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = event.get("sessionKey") if isinstance(event, dict) else None
        if isinstance(key, str) and key.strip():
            audit.setdefault(key.strip(), {}).update(event)

def session_path(key, entry):
    def contained(root, candidate):
        try:
            return candidate.resolve().is_relative_to(root.resolve())
        except (OSError, RuntimeError):
            return False

    session_file = entry.get("sessionFile")
    if isinstance(session_file, str) and session_file.strip():
        path = pathlib.Path(session_file)
        candidate = path if path.is_absolute() else sessions / path
        if contained(sessions, candidate) and candidate.is_file():
            return candidate
    session_id = entry.get("sessionId")
    active = sessions / f"{session_id}.jsonl" if session_id else None
    if active and active.is_file():
        return active
    transcript = audit.get(key, {}).get("auditTranscript")
    if isinstance(transcript, str) and transcript.strip():
        candidate = audit_root / transcript
        return candidate if contained(audit_root, candidate) else None
    return active

def records_for(key, entry):
    path = session_path(key, entry)
    if path is None or not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records

def content_text(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(
        part.get("text", "")
        for part in content
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    )

def active_records(records):
    canonical_types = {
        "message",
        "thinking_level_change",
        "model_change",
        "compaction",
        "reset",
        "branch_summary",
        "custom",
        "custom_message",
        "label",
        "session_info",
    }
    nodes = {}
    leaf = None
    append_parent = None
    explicit_update = False
    invalid_leaf_ids = set()

    def text(value):
        return value.strip() if isinstance(value, str) and value.strip() else None

    def resolve_parent(parent):
        seen = set()
        while parent is not None:
            if parent in seen:
                return parent
            seen.add(parent)
            node = nodes.get(parent)
            if node is None or not node["is_leaf"]:
                return parent
            parent = node["parent"]
        return None

    for record in records:
        record_type = record.get("type")
        canonical = record_type in canonical_types
        explicit = "parentId" in record
        if record_type == "session" or (not explicit and not canonical):
            continue
        record_id = text(record.get("id"))
        if record_id is None:
            continue
        raw_parent = record.get("parentId") if explicit else leaf
        parent = None if raw_parent is None else text(raw_parent)
        if raw_parent is not None and parent is None:
            continue
        is_leaf = record_type == "leaf"
        if is_leaf:
            raw_target = record.get("targetId")
            target = None if raw_target is None else text(raw_target)
            raw_append = record.get("appendParentId", raw_target)
            next_append = None if raw_append is None else text(raw_append)
            if (
                (raw_target is not None and target is None)
                or (raw_append is not None and next_append is None)
                or record.get("appendMode") not in {None, "side"}
            ):
                continue
            invalid = any(
                ref is not None and (ref not in nodes or ref in invalid_leaf_ids)
                for ref in (target, next_append)
            )
            if invalid:
                invalid_leaf_ids.add(record_id)
                next_leaf = ...
                next_append = append_parent
            else:
                parent = target
                next_leaf = target
        else:
            if (
                explicit
                and parent is not None
                and parent not in nodes
                and leaf is not None
            ):
                parent = leaf
            elif (
                explicit
                and record.get("appendMode") != "side"
                and parent == append_parent
                and leaf != append_parent
            ):
                parent = leaf
            parent = resolve_parent(parent)
            next_leaf = record_id if canonical and record.get("appendMode") != "side" else ...
            next_append = record_id
        node = {
            "record": record,
            "parent": parent,
            "leaf": next_leaf,
            "append": next_append,
            "is_leaf": is_leaf,
        }
        nodes[record_id] = node
        append_parent = next_append
        if next_leaf is not ...:
            leaf = next_leaf
            explicit_update = explicit_update or explicit

    if not explicit_update:
        return records
    if leaf is None:
        return []
    selected = []
    seen = set()
    current = leaf
    while current is not None:
        if current in seen:
            return []
        seen.add(current)
        node = nodes.get(current)
        if node is None:
            break
        if not node["is_leaf"]:
            selected.append(node["record"])
        current = node["parent"]
    selected.reverse()
    return selected

def terminal(entry, records):
    if str(entry.get("status") or "").lower() in {
        "cancelled",
        "deleted",
        "error",
        "failed",
        "killed",
        "reset",
        "timeout",
    }:
        return bool(records)
    for record in reversed(active_records(records)):
        message = record.get("message")
        if (
            record.get("type") != "message"
            or not isinstance(message, dict)
        ):
            continue
        if message.get("role") != "assistant":
            return False
        content = message.get("content")
        parts = content if isinstance(content, list) else []
        text = content_text(content)
        tools = [
            part
            for part in parts
            if isinstance(part, dict) and part.get("type") == "toolCall"
        ]
        return (
            bool(text.strip())
            and not tools
            and str(message.get("stopReason") or "").lower() in {"end_turn", "stop"}
        )
    return False

def result_object(message):
    details = message.get("details")
    if isinstance(details, dict):
        return details
    text = content_text(message.get("content")).strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(text):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return {}
    return value if isinstance(value, dict) else {}

def spawned_children(records):
    pending = set()
    children = []
    for record in active_records(records):
        message = record.get("message")
        if record.get("type") != "message" or not isinstance(message, dict):
            continue
        if message.get("role") == "assistant":
            content = message.get("content")
            if not isinstance(content, list):
                continue
            pending.update(
                str(part.get("id"))
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "toolCall"
                and part.get("name") == "sessions_spawn"
                and part.get("id")
            )
            continue
        if message.get("role") != "toolResult":
            continue
        call_id = str(message.get("toolCallId") or "")
        if call_id not in pending:
            continue
        pending.discard(call_id)
        payload = result_object(message)
        child = payload.get("childSessionKey")
        if (
            str(payload.get("status") or "").lower() == "accepted"
            and isinstance(child, str)
            and child.strip()
        ):
            children.append(child.strip())
    return children

root_key = "agent:main:main"
pending = [(root_key, None)]
seen = set()
while pending:
    key, parent = pending.pop(0)
    if key in seen:
        continue
    seen.add(key)
    entry = store.get(key)
    if not isinstance(entry, dict):
        entry = audit.get(key)
    if not isinstance(entry, dict):
        sys.exit(1)
    if parent and entry.get("spawnedBy") not in {None, "", parent}:
        sys.exit(1)
    records = records_for(key, entry)
    if not terminal(entry, records):
        sys.exit(1)
    pending.extend((child, key) for child in spawned_children(records))
sys.exit(0)
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
const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

const auditRoot = path.join(process.env.HOME, ".openclaw", "shellbench-audit");
const eventPath = path.join(auditRoot, "sessions.jsonl");

function append(event) {
  fs.mkdirSync(auditRoot, { recursive: true });
  fs.appendFileSync(eventPath, `${JSON.stringify(event)}\\n`, { mode: 0o600 });
}

function materializeTranscript(event) {
  const source = event.sessionFile;
  if (typeof source !== "string" || !source || !fs.existsSync(source)) {
    return undefined;
  }
  const transcriptRoot = path.join(auditRoot, "transcripts");
  fs.mkdirSync(transcriptRoot, { recursive: true });
  const destination = path.join(transcriptRoot, `${event.sessionId}.jsonl`);
  try {
    const content = source.endsWith(".zst")
      ? zlib.zstdDecompressSync(fs.readFileSync(source))
      : fs.readFileSync(source);
    fs.writeFileSync(destination, content, { mode: 0o600 });
    return path.relative(auditRoot, destination);
  } catch {
    return undefined;
  }
}

module.exports = {
  id: "shellbench-audit",
  register(api) {
    api.on("subagent_spawned", (event, ctx) => {
      append({
        type: "subagent_spawned",
        sessionKey: event.childSessionKey,
        spawnedBy: ctx.requesterSessionKey,
        runId: event.runId,
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
    api.on("session_start", (event) => {
      append({
        type: "session_start",
        sessionKey: event.sessionKey,
        sessionId: event.sessionId,
      });
    });
    api.on("session_end", (event) => {
      const auditTranscript = materializeTranscript(event);
      append({
        type: "session_end",
        sessionKey: event.sessionKey,
        sessionId: event.sessionId,
        reason: event.reason,
        transcriptArchived: event.transcriptArchived,
        auditTranscript,
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
    return builder(run, proxy_url, proxy_key, mcp_servers)


def _base_path() -> str:
    return f"{NODE_BIN}:{NPM_BIN}:{HERMES_BIN}:$PATH"


def _openclaw(
    run: RunSpec,
    proxy_url: str,
    proxy_key: str,
    mcp_servers: tuple[McpServer, ...],
) -> HarnessCommand:
    provider = "openai"
    model = f"{provider}/{run.model_id}"
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
                "subagents": {"model": model},
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
        "tools": {"deny": ["message"]},
    }
    if servers:
        config["mcp"] = {"servers": servers}
    config_json = shlex.quote(json.dumps(config, separators=(",", ":")))
    completion_probe = shlex.quote(_OPENCLAW_COMPLETION_PROBE)
    session_probe = shlex.quote(_OPENCLAW_SESSION_PROBE)
    gateway_probe = shlex.quote(_OPENCLAW_GATEWAY_PROBE)
    audit_plugin = shlex.quote(_OPENCLAW_AUDIT_PLUGIN)
    audit_manifest = shlex.quote(json.dumps(_OPENCLAW_AUDIT_PLUGIN_MANIFEST, separators=(",", ":")))
    setup = (
        f"export PATH={_base_path()}; export HOME={home}; "
        'rm -rf "$HOME"; mkdir -p "$HOME/.openclaw/shellbench-audit"; '
        f'printf %s {audit_plugin} > "$HOME/.openclaw/shellbench-audit/index.cjs"; '
        f"printf %s {audit_manifest} "
        '> "$HOME/.openclaw/shellbench-audit/openclaw.plugin.json"; '
        f'printf %s {config_json} > "$HOME/.openclaw/openclaw.json"; '
        "openclaw setup --baseline --skip-bootstrap --workspace . "
        ">/logs/agent/setup.log 2>&1; "
        "rm -f AGENTS.md BOOTSTRAP.md HEARTBEAT.md IDENTITY.md "
        "SOUL.md TOOLS.md USER.md; "
        f'printf %s {config_json} > "$HOME/.openclaw/openclaw.json"'
    )
    run_command = (
        f"export PATH={_base_path()}; export HOME={home}; "
        f"export OPENCLAW_GATEWAY_TOKEN={shlex.quote(gateway_token)}; "
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
        "openclaw agent --json --agent main --thinking off "
        f"--model {shlex.quote(model)} "
        '--message "$(cat /tmp/shellbench-instruction.md)" '
        '>"$log" 2>&1 </dev/null & pid=$!; '
        "reaped=0; status=0; "
        'while kill -0 "$pid" 2>/dev/null; do '
        f'if python3 -c {completion_probe} "$log"; then '
        'sleep 1; kill "$pid" 2>/dev/null || true; sleep 1; '
        'kill -KILL "$pid" 2>/dev/null || true; '
        'wait "$pid" 2>/dev/null || true; reaped=1; break; '
        "fi; sleep 1; done; "
        'if [ "$reaped" -ne 1 ]; then wait "$pid"; status=$?; fi; '
        'session_ready=0; if [ "$status" -eq 0 ]; then '
        "for _ in $(seq 1 60); do "
        f"if python3 -c {session_probe} "
        '"$HOME/.openclaw/agents/main/sessions" '
        '"$HOME/.openclaw/shellbench-audit"; then '
        "session_ready=1; break; fi; "
        'if ! kill -0 "$gateway_pid" 2>/dev/null; then status=70; break; fi; '
        "sleep 1; done; fi; "
        'if [ "$status" -eq 0 ] && [ "$session_ready" -ne 1 ]; then '
        "echo 'OpenClaw terminal session evidence did not stabilize within 60 seconds' "
        '>>"$log"; status=71; fi; '
        'kill "$gateway_pid" 2>/dev/null || true; '
        'wait "$gateway_pid" 2>/dev/null || true; '
        'cat "$gateway_log" >>"$log"; cat "$log"; exit "$status"'
    )
    # Keep this archive path aligned with the pinned OpenClaw 2026.7.1-2
    # sessions.json/JSONL contract used by the completion probe above.
    cleanup = (
        "python3 - <<'PY'\n"
        "import json, pathlib, shutil\n"
        "p=pathlib.Path('/logs/agent/openclaw.txt')\n"
        "agents=pathlib.Path('/tmp/shellbench-openclaw/.openclaw/agents')\n"
        "sessions=agents/'main'/'sessions'\n"
        "archive=pathlib.Path('/logs/agent/openclaw.sessions')\n"
        "audit=pathlib.Path('/tmp/shellbench-openclaw/.openclaw/shellbench-audit')\n"
        "if agents.is_dir():\n"
        " for agent in agents.iterdir():\n"
        "  source=agent/'sessions'\n"
        "  if not source.is_dir(): continue\n"
        "  target=archive if agent.name=='main' else archive/'agents'/agent.name\n"
        "  shutil.copytree(source, target, dirs_exist_ok=True)\n"
        "if audit.is_dir():\n"
        " shutil.copytree(audit, archive/'audit', dirs_exist_ok=True)\n"
        "if archive.is_dir():\n"
        " for item in [archive, *archive.rglob('*')]:\n"
        "  item.chmod(0o755 if item.is_dir() else 0o644)\n"
        "sources=[]\n"
        "try:\n"
        " raw=p.read_text(encoding='utf-8', errors='replace').strip()\n"
        " dec=json.JSONDecoder(); d=None\n"
        " for start in range(len(raw)-1, -1, -1):\n"
        "  if raw[start] != '{': continue\n"
        "  try: candidate, _ = dec.raw_decode(raw[start:])\n"
        "  except (json.JSONDecodeError, ValueError): continue\n"
        "  if isinstance(candidate, dict) and "
        "isinstance(candidate.get('meta'), dict):\n"
        "   d=candidate; break\n"
        " if d:\n"
        "  src=((d.get('meta') or {}).get('agentMeta') or {}).get('sessionFile')\n"
        "  if isinstance(src, str) and src: sources.append(src)\n"
        "except Exception:\n"
        " pass\n"
        "if sessions.is_dir():\n"
        " try:\n"
        "  store=json.loads((sessions/'sessions.json').read_text())\n"
        "  entry=store.get('agent:main:main') if isinstance(store, dict) else None\n"
        "  if isinstance(entry, dict):\n"
        "   session_file=entry.get('sessionFile')\n"
        "   if isinstance(session_file, str) and session_file:\n"
        "    sources.append(session_file)\n"
        "   if entry.get('sessionId'):\n"
        "    sources.append(str(sessions/f\"{entry['sessionId']}.jsonl\"))\n"
        " except Exception:\n"
        "  pass\n"
        "source=None\n"
        "for src in sources:\n"
        " candidate=pathlib.Path(src)\n"
        " if not candidate.is_absolute(): candidate=sessions/candidate\n"
        " if candidate.is_file():\n"
        "  source=candidate\n"
        "  break\n"
        "if source and source.is_file():\n"
        " destination=pathlib.Path('/logs/agent/openclaw.session.jsonl')\n"
        " shutil.copy2(source, destination)\n"
        " destination.chmod(0o644)\n"
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
