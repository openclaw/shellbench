from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath

from scripts.native_eval.models import RunSpec
from scripts.native_eval.tasks import McpServer


TOOLCHAIN_ROOT = PurePosixPath("/opt/shellbench-native")
NODE_BIN = TOOLCHAIN_ROOT / "node" / "bin"
NPM_BIN = TOOLCHAIN_ROOT / "npm-packages" / "node_modules" / ".bin"
HERMES_BIN = TOOLCHAIN_ROOT / "home" / ".local" / "bin"


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
        "agents": {"defaults": {"workspace": "."}},
        "gateway": {"mode": "local"},
        "models": {
            "providers": {
                provider: {
                    "baseUrl": f"{proxy_url.rstrip('/')}/v1",
                    "api": "openai-responses",
                    "apiKey": proxy_key,
                    "models": [
                        {
                            "id": run.model_id,
                            "name": run.model_id,
                        }
                    ],
                }
            }
        },
        "tools": {"deny": ["message"]},
    }
    if servers:
        config["mcp"] = {"servers": servers}
    config_json = shlex.quote(json.dumps(config, separators=(",", ":")))
    home = "/tmp/shellbench-openclaw"
    setup = (
        f"export PATH={_base_path()}; export HOME={home}; "
        "rm -rf \"$HOME\"; mkdir -p \"$HOME/.openclaw\"; "
        "openclaw setup --baseline --workspace . >/logs/agent/setup.log 2>&1; "
        f"printf %s {config_json} > \"$HOME/.openclaw/openclaw.json\""
    )
    run_command = (
        f"export PATH={_base_path()}; export HOME={home}; "
        "log=/logs/agent/openclaw.txt; "
        "openclaw agent --local --json --agent main --thinking off "
        f"--model {shlex.quote(model)} "
        "--message \"$(cat /tmp/shellbench-instruction.md)\" "
        ">\"$log\" 2>&1 </dev/null & pid=$!; "
        "while kill -0 \"$pid\" 2>/dev/null; do "
        "if grep -Eq \"\\[agents/agent-command\\] \\[agent\\] run .* "
        "ended with stopReason=\" \"$log\"; then "
        "sleep 2; kill \"$pid\" 2>/dev/null || true; sleep 1; "
        "kill -KILL \"$pid\" 2>/dev/null || true; "
        "for f in /proc/[0-9]*/comm; do "
        "name=$(cat \"$f\" 2>/dev/null || true); "
        "case \"$name\" in openclaw-agent|\"npm exec chrome\"|chrome-devtools) "
        "child=${f#/proc/}; child=${child%/comm}; "
        "kill -KILL \"$child\" 2>/dev/null || true;; esac; done; "
        "wait \"$pid\" 2>/dev/null || true; cat \"$log\"; exit 0; "
        "fi; sleep 1; done; "
        "wait \"$pid\"; status=$?; cat \"$log\"; exit \"$status\""
    )
    cleanup = (
        "python3 - <<'PY'\n"
        "import json, pathlib, shutil\n"
        "p=pathlib.Path('/logs/agent/openclaw.txt')\n"
        "sessions=pathlib.Path("
        "'/tmp/shellbench-openclaw/.openclaw/agents/main/sessions')\n"
        "if sessions.is_dir():\n"
        " shutil.copytree(sessions, '/logs/agent/openclaw.sessions', "
        "dirs_exist_ok=True)\n"
        "src=None\n"
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
        "except Exception:\n"
        " pass\n"
        "if not src and sessions.is_dir():\n"
        " try:\n"
        "  store=json.loads((sessions/'sessions.json').read_text())\n"
        "  entry=store.get('agent:main:main') if isinstance(store, dict) else None\n"
        "  if isinstance(entry, dict):\n"
        "   src=entry.get('sessionFile')\n"
        "   if not src and entry.get('sessionId'):\n"
        "    src=str(sessions/f\"{entry['sessionId']}.jsonl\")\n"
        " except Exception:\n"
        "  pass\n"
        "source=pathlib.Path(src) if src else None\n"
        "if source and not source.is_absolute(): source=sessions/source\n"
        "if not source or not source.is_file():\n"
        " candidates=[f for f in sessions.glob('*.jsonl') "
        "if '.trajectory.' not in f.name] if sessions.is_dir() else []\n"
        " source=max(candidates, key=lambda f: f.stat().st_mtime, "
        "default=None)\n"
        "if source and source.is_file():\n"
        " shutil.copy2(source, '/logs/agent/openclaw.session.jsonl')\n"
        "PY"
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
        "delegation": {"max_iterations": 50},
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
        "rm -rf \"$HERMES_HOME\"; mkdir -p \"$HERMES_HOME\"; "
        f"printf %s {config_json} > \"$HERMES_HOME/config.yaml\""
    )
    run_command = (
        f"export PATH={_base_path()}; export HERMES_HOME={home}; "
        "export TERMINAL_ENV=local; export TERMINAL_CWD=\"$PWD\"; "
        "hermes --yolo chat -q \"$(cat /tmp/shellbench-instruction.md)\" -Q "
        f"--model {shlex.quote(run.model_id)} "
        f"--provider {shlex.quote(provider_name)} "
        ">/logs/agent/hermes.txt 2>&1; status=$?; "
        "cat /logs/agent/hermes.txt; "
        "if grep -Eq \"Unknown provider|No LLM provider configured\" "
        "/logs/agent/hermes.txt; then exit 64; fi; "
        "exit \"$status\""
    )
    cleanup = (
        f"export PATH={_base_path()}; export HERMES_HOME={home}; "
        "session_id=$(sed -n 's/^session_id: //p' "
        "/logs/agent/hermes.txt | tail -1); "
        "if [ -n \"$session_id\" ]; then "
        "hermes sessions export /logs/agent/hermes-session.jsonl "
        "--session-id \"$session_id\" --yes --redact; "
        "else "
        "hermes sessions export /logs/agent/hermes-session.jsonl "
        "--yes --redact; "
        "fi"
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
            command = shlex.join(
                [server.command or "", *server.args]
            )
            config_lines.append(f"command = {json.dumps(command)}")
        else:
            config_lines.append(f"url = {json.dumps(server.url)}")
    config_text = shlex.quote("\n".join(config_lines) + "\n")
    auth_json = shlex.quote(json.dumps({"OPENAI_API_KEY": proxy_key}))
    setup = (
        f"export PATH={_base_path()}; export CODEX_HOME={home}; "
        "rm -rf \"$CODEX_HOME\"; mkdir -p \"$CODEX_HOME\"; "
        f"printf %s {auth_json} > \"$CODEX_HOME/auth.json\"; "
        f"printf %s {config_text} > \"$CODEX_HOME/config.toml\""
    )
    run_command = (
        f"export PATH={_base_path()}; export CODEX_HOME={home}; "
        "codex exec --dangerously-bypass-approvals-and-sandbox "
        "--skip-git-repo-check "
        f"--model {shlex.quote(run.model_id)} "
        "--json --enable unified_exec -- "
        "\"$(cat /tmp/shellbench-instruction.md)\" "
        ">/logs/agent/codex.txt 2>&1 </dev/null; status=$?; "
        "cat /logs/agent/codex.txt; exit \"$status\""
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
        "rm -rf \"$CLAUDE_CONFIG_DIR\"; "
        "mkdir -p \"$CLAUDE_CONFIG_DIR/debug\" \"$CLAUDE_CONFIG_DIR/projects/-app\"; "
        f"printf %s {mcp_json} > \"$CLAUDE_CONFIG_DIR/.claude.json\""
    )
    run_command = (
        f"export PATH={_base_path()}; export CLAUDE_CONFIG_DIR={home}; "
        "claude --verbose --output-format=stream-json "
        "--permission-mode=bypassPermissions --print "
        f"--model {shlex.quote(run.model_id)} "
        "\"$(cat /tmp/shellbench-instruction.md)\" "
        ">/logs/agent/claude-code.txt 2>&1 </dev/null; status=$?; "
        "cat /logs/agent/claude-code.txt; exit \"$status\""
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
