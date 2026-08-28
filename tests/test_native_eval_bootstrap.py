from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


BOOTSTRAP = Path(__file__).resolve().parents[1] / "scripts/native_eval/bootstrap_beast.sh"


@pytest.fixture
def bootstrap_env(tmp_path: Path) -> dict[str, str]:
    if shutil.which("jq") is None:
        pytest.skip("bootstrap manifest checks require jq")
    root = tmp_path / "toolchain"
    for relative, output in (
        ("node/bin/node", "v22.23.1"),
        ("litellm-venv/bin/python", "1.93.0"),
    ):
        executable = root / relative
        executable.parent.mkdir(parents=True, exist_ok=True)
        executable.write_text(f"#!/bin/sh\nprintf '%s\\n' '{output}'\n")
        executable.chmod(0o755)
    uv = root / "bin/uv"
    uv.parent.mkdir(parents=True)
    uv.write_text("""#!/bin/sh
test "$UV_PYTHON_INSTALL_DIR" = "$TOOLCHAIN_ROOT/uv-python" || exit 1
test "$UV_CACHE_DIR" = "$TOOLCHAIN_ROOT/uv-cache" || exit 1
printf 'uv %s\\n' "$*" >> "$BOOTSTRAP_CALLS"
""")
    uv.chmod(0o755)
    stubs = tmp_path / "commands.sh"
    stubs.write_text("""
record() { printf '%s\\n' "$*" >> "$BOOTSTRAP_CALLS"; }
id() { printf '0\\n'; }
apt-get() { record "apt-get $*"; }
rm() { :; }
chmod() { :; }
docker() { printf 'docker-test\\n'; }
curl() { record "curl $*"; }
bash() { record "hermes-install $*"; }
npm() { record "npm $*"; }
openclaw() { record 'probe openclaw'; printf 'openclaw-test\\n'; }
codex() { record 'probe codex'; printf 'codex-test\\n'; }
claude() { record 'probe claude-code'; printf 'claude-code-test\\n'; }
hermes() { record 'probe hermes'; printf 'hermes-test\\n'; }
""")
    return {
        **os.environ,
        "BASH_ENV": str(stubs),
        "TOOLCHAIN_ROOT": str(root),
        "BOOTSTRAP_CALLS": str(tmp_path / "calls.txt"),
        "NODE_VERSION": "22.23.1",
        "OPENCLAW_VERSION": "2026.7.1-2",
        "CODEX_VERSION": "0.145.0",
        "CLAUDE_CODE_VERSION": "2.1.220",
        "HERMES_COMMIT": "test-hermes-commit",
    }


@pytest.mark.parametrize("harness,package", (
    ("openclaw", "openclaw@2026.7.1-2"),
    ("codex", "@openai/codex@0.145.0"),
    ("claude-code", "@anthropic-ai/claude-code@2.1.220"),
    ("hermes", None),
))
def test_bootstrap_installs_and_probes_only_selected_harness(
    bootstrap_env: dict[str, str], harness: str, package: str | None,
) -> None:
    result = subprocess.run(
        ["/bin/bash", str(BOOTSTRAP), harness], env=bootstrap_env,
        capture_output=True, text=True, check=True,
    )
    calls = Path(bootstrap_env["BOOTSTRAP_CALLS"]).read_text().splitlines()
    npm_calls = [line for line in calls if line.startswith("npm ")]
    if package:
        assert len(npm_calls) == 1
        assert npm_calls[0].split()[4:] == [package]
    else:
        assert npm_calls == []
    assert sum(line.startswith("hermes-install ") for line in calls) == (harness == "hermes")
    assert [line for line in calls if line.startswith("probe ")] == [f"probe {harness}"]
    assert sum(line.startswith("uv ") for line in calls) == 2
    manifest = json.loads(result.stdout[result.stdout.index("{"):])
    key = harness.replace("-", "_")
    assert manifest[key] == f"{harness}-test"
    assert manifest["harness"] == harness
    assert manifest["node"] == "v22.23.1"
    assert manifest["litellm"] == "1.93.0"
    assert set(manifest) == {
        "created_at_utc", "node", "litellm", "harness", key,
    } | ({"hermes_commit"} if harness == "hermes" else set())


@pytest.mark.parametrize("arguments", [[], ["unknown"], ["openclaw", "codex"]])
def test_bootstrap_rejects_invalid_selection_before_installing(
    bootstrap_env: dict[str, str], arguments: list[str],
) -> None:
    result = subprocess.run(
        ["/bin/bash", str(BOOTSTRAP), *arguments], env=bootstrap_env,
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr
    assert not Path(bootstrap_env["BOOTSTRAP_CALLS"]).exists()
