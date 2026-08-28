from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from scripts.native_eval.fleet import FleetController
from scripts.native_eval.models import build_matrix_plan
from scripts.native_eval.run_job import _run_manifest, build_run_spec, parse_args


def _args() -> list[str]:
    return [
        "--tasks-root",
        "tasks",
        "--jobs-dir",
        "jobs",
        "--run-label",
        "effort-test",
        "--harness",
        "openclaw",
        "--model-slug",
        "gpt55",
        "--repetition",
        "1",
        "--expected-task-count",
        "1",
        "--public-tasks-commit",
        "test",
        "--task-suite-path",
        "tasks",
        "--run-date",
        "20260828",
    ]


@pytest.mark.parametrize("environment,cli", [("low", "high"), ("high", "low")])
def test_runner_rejects_late_effort_override(
    monkeypatch: pytest.MonkeyPatch, environment: str, cli: str
) -> None:
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", environment)
    args = parse_args([*_args(), "--reasoning-effort", cli])
    with pytest.raises(ValueError, match="conflicts with SHELLBENCH_REASONING_EFFORT"):
        build_run_spec(args)


@pytest.mark.parametrize("effort", [None, "high"])
def test_manifest_does_not_reread_effort_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, effort: str | None
) -> None:
    monkeypatch.delenv("SHELLBENCH_REASONING_EFFORT", raising=False)
    run = build_run_spec(parse_args(_args() + (["--reasoning-effort", effort] if effort else [])))
    monkeypatch.setenv("SHELLBENCH_REASONING_EFFORT", "low")
    manifest = _run_manifest(
        run,
        public_tasks_commit="test",
        task_suite_path="tasks",
        concurrency=1,
        started_at="test",
        tasks_root=tmp_path,
        tasks=[],
    )
    assert manifest["reasoning_effort"] == effort


def test_fleet_keeps_effort_in_run_spec() -> None:
    planned = build_matrix_plan(1, reasoning_effort="xhigh")[0]
    controller = object.__new__(FleetController)
    assert controller._run_spec(planned.to_dict()).reasoning_effort == "xhigh"


@pytest.mark.parametrize("harness", ["openclaw", "hermes", "codex", "claude-code"])
@pytest.mark.parametrize("planned", ["high", "xhigh", None, "invalid"])
def test_remote_resolves_effort_before_proxy_startup(
    tmp_path: Path, harness: str, planned: str | None
) -> None:
    bash = shutil.which("bash")
    assert bash is not None
    if subprocess.run(
        [
            bash,
            "-c",
            "(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) ))",
        ],
        check=False,
    ).returncode:
        pytest.skip("remote_run.sh requires Bash 4.4+")
    repo = Path(__file__).resolve().parents[1]
    label = f"shellbench-effort-{uuid4().hex}"
    root = tmp_path / "remote"
    root.mkdir()
    (root / "runner").symlink_to(repo, target_is_directory=True)
    tasks = tmp_path / "tasks"
    for name, content in {
        "task.toml": "",
        "instruction.md": "do the task",
        "environment/Dockerfile": "FROM scratch\n",
        "solution/solve.sh": "true\n",
        "tests/test.sh": "true\n",
    }.items():
        path = tasks / "example" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    toolchain = tmp_path / "toolchain"
    proxy = toolchain / "litellm-venv/bin/litellm"
    proxy.parent.mkdir(parents=True)
    proxy.write_text(
        '#!/bin/sh\ncp "$2" "$PROXY_OBSERVED"\n'
        'printf "%s" "$SHELLBENCH_REASONING_EFFORT" > "$PROXY_ENV_OBSERVED"\n'
        "exec sleep 60\n",
        encoding="utf-8",
    )
    proxy.chmod(0o755)
    env_file = tmp_path / "provider.env"
    env_file.write_text(
        "SHELLBENCH_REASONING_EFFORT=low\nSHELLBENCH_JUDGE_REASONING_EFFORT=medium\n",
        encoding="utf-8",
    )
    runner = tmp_path / "run_with_fake_trial.py"
    runner.write_text(
        """import json
import os
import sys
from pathlib import Path
from scripts.native_eval import run_job
from scripts.native_eval.harnesses import build_harness_command

async def trial(task, run, **kwargs):
    command = build_harness_command(
        run, proxy_url=kwargs["proxy_url"], proxy_key=kwargs["proxy_key"],
        mcp_servers=task.mcp_servers,
    )
    observed = {"effort": run.reasoning_effort,
                "setup": command.setup_command, "command": command.run_command}
    Path(os.environ["HARNESS_OBSERVED"]).write_text(json.dumps(observed))
    result = {"execution_outcome": {"kind": "clean"}, "agent_result": {}}
    trial_dir = kwargs["job_dir"] / "example__trial"
    trial_dir.mkdir()
    (trial_dir / "result.json").write_text(json.dumps(result))
    return result

run_job.run_trial = trial
sys.argv = ["run_job", *sys.argv[3:]]
run_job.main()
""",
        encoding="utf-8",
    )
    shell_env = tmp_path / "bash_env"
    shell_env.write_text(
        """python3() {
  if [[ "$1" == "-" || "$*" == *--prepare-proxy-config* ]]; then
    command "$TEST_PYTHON" "$@"
  else
    command "$TEST_PYTHON" "$TEST_RUNNER" "$@"
  fi
}
curl() { test -f "$PROXY_ENV_OBSERVED"; }
sudo() { "$@"; }
""",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "BASH_ENV": str(shell_env),
        "PYTHONPATH": str(repo),
        "TEST_PYTHON": sys.executable,
        "TEST_RUNNER": str(runner),
        "TOOLCHAIN_ROOT": str(toolchain),
        "SHELLBENCH_PROXY_KEY": "synthetic-test-key",
        "SHELLBENCH_JUDGE_MODEL_ID": "gpt-5.5",
        "PROXY_OBSERVED": str(tmp_path / "proxy-observed.json"),
        "PROXY_ENV_OBSERVED": str(tmp_path / "proxy-env.txt"),
        "HARNESS_OBSERVED": str(tmp_path / "harness-observed.json"),
    }
    env.pop("SHELLBENCH_REASONING_EFFORT", None)
    if planned is not None:
        env["SHELLBENCH_REASONING_EFFORT"] = planned
    archive = Path("/tmp") / f"{label}-final-artifacts.tar.gz"
    try:
        process = subprocess.run(
            [
                bash,
                str(repo / "scripts/native_eval/remote_run.sh"),
                str(root),
                str(tasks),
                str(env_file),
                label,
                harness,
                "gpt55",
                "1",
                "1",
                "test",
                "20260828",
                "1",
                "test",
                "gpt-5.5",
                "openai",
                "gpt-5.5",
                "",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if planned == "invalid":
            assert process.returncode != 0
            assert not Path(env["PROXY_OBSERVED"]).exists()
            assert not Path(env["HARNESS_OBSERVED"]).exists()
            return
        assert process.returncode == 0, process.stderr
        expected = planned or "low"
        config = json.loads(Path(env["PROXY_OBSERVED"]).read_text())
        observed = json.loads(Path(env["HARNESS_OBSERVED"]).read_text())
        manifest = json.loads((root / "results/jobs" / label / "run_manifest.json").read_text())
        assert config["shellbench_native"]["reasoning_effort"] == expected
        assert Path(env["PROXY_ENV_OBSERVED"]).read_text() == expected
        models = {item["model_name"]: item for item in config["model_list"]}
        assert models["gpt-5.5"]["litellm_params"]["reasoning_effort"] == expected
        assert models["shellbench-judge"]["litellm_params"]["reasoning_effort"] == "medium"
        assert observed["effort"] == manifest["reasoning_effort"] == expected
        if harness == "openclaw":
            assert f"--thinking {expected}" in observed["command"]
        elif harness == "hermes":
            assert f'"reasoning_effort":"{expected}"' in observed["setup"]
        elif harness == "codex":
            assert f'model_reasoning_effort="{expected}"' in observed["command"]
        else:
            assert f"--effort {'max' if expected == 'xhigh' else expected}" in observed["command"]
    finally:
        archive.unlink(missing_ok=True)
        shutil.rmtree(Path("/tmp/shellbench-runs") / label, ignore_errors=True)
        shutil.rmtree(Path("/tmp") / f"shellbench_meta-{label}", ignore_errors=True)
