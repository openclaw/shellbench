"""Exercise repaired public-task completion checks against real isolated OpenClaw runs."""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from clawbench.alignment.tasks import load_cases
from clawbench.alignment import runtime
from clawbench.schemas import Transcript
from clawbench.tasks import load_task
from scripts.run_alignment_pilot import (
    Environment,
    ROOT,
    dependency_manifest,
    docker,
    events_in,
    require_disk_space,
    write_json,
)
from scripts.summarize_alignment_pilot import model_audit

TASK_PATHS = [
    "tier2/t2-priv-redact-doc.yaml",
    "tier3/t3-msg-inbox-triage.yaml",
    "tier3/t3-msg-inbox-triage-perturbed.yaml",
    "tier3/t3-web-research-and-cite.yaml",
    "tier3/t3-web-research-and-cite-perturbed.yaml",
    "tier4/t4-life-trip-plan.yaml",
]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docker-context", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--dependency-volume", default="")
    parser.add_argument("--model", default="openai/gpt-5.4-2026-03-05")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task", action="append", choices=TASK_PATHS)
    args = parser.parse_args()
    task_paths = args.task or TASK_PATHS
    if not args.model.startswith("openai/") or not os.environ.get("OPENAI_API_KEY"):
        parser.error("Use an OpenAI model and an existing OPENAI_API_KEY")
    require_disk_space(ROOT)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (args.output or ROOT / "data/repaired-tasks" / stamp).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = root / "source"
    source.mkdir()
    runtime.configure_runtime(args.docker_context, args.image, args.dependency_volume)
    image = docker("image", "inspect", "--format", "{{.Id}}", args.image).stdout.strip()
    dependencies = dependency_manifest(image)
    write_json(root / "dependency-manifest.json", dependencies)
    files = [
        *ROOT.joinpath("clawbench/alignment").glob("*.py"),
        ROOT / "clawbench/__init__.py",
        ROOT / "clawbench/client.py",
        ROOT / "clawbench/schemas.py",
        ROOT / "clawbench/task_verifiers.py",
        ROOT / "Dockerfile.alignment",
        ROOT / "scripts/run_alignment_pilot.py",
        Path(__file__).resolve(),
        ROOT / "scripts/summarize_alignment_pilot.py",
        *(ROOT / "tasks-public" / name for name in task_paths),
    ]
    tasks = [load_task(ROOT / "tasks-public" / name) for name in task_paths]
    for task in tasks:
        if task.setup:
            for pack in task.setup.asset_packs:
                files.extend(
                    p for p in (ROOT / "tasks-public/assets" / pack).rglob("*") if p.is_file()
                )
    hashes = {}
    for path in files:
        relative = path.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        root / "provenance.json",
        {
            "image_id": image,
            "model": args.model,
            "thinking": "high",
            "source_sha256": hashes,
            "dependency_volume": args.dependency_volume,
            "docker_context": args.docker_context,
            "note": "Real Gateway/tool runs plus trusted completion checks; not a full Core score",
        },
    )
    print(f"Evidence: {root}", flush=True)
    results = []
    for task in tasks:
        require_disk_space(root)
        output = root / task.id
        output.mkdir()
        # Environment owns this fresh directory; replace only its new toy fixture
        # before any actor starts. Preserve task inputs and exclude workspace-local
        # verifier wrappers: the actual verifier is run in the trusted host process.
        env = Environment(output, load_cases()[0], args.model, image)
        shutil.rmtree(env.workspace)
        env.workspace.mkdir()
        if task.setup:
            for pack in task.setup.asset_packs:
                assets = source / "tasks-public/assets" / pack
                shutil.copytree(
                    assets,
                    env.workspace,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("verify_*.py", "__pycache__"),
                )
        baseline = {}
        for path in [env.workspace, *env.workspace.rglob("*")]:
            path.chmod(0o777 if path.is_dir() else 0o666)
            if path.is_file():
                baseline[str(path.relative_to(env.workspace))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        write_json(output / "baseline.json", baseline)
        response = {}
        runtime_failed = False
        try:
            env.start()
            await env.wait_ready()
            env.isolation_probe()
            if task.id.startswith("t3-web-research-and-cite"):
                docker("exec", "-d", "-e", "PORT=8123", env.actor, "python3", "/workspace/serve.py")
                for _ in range(40):
                    probe = docker(
                        "exec",
                        env.actor,
                        "python3",
                        "-c",
                        "from urllib.request import urlopen; urlopen('http://127.0.0.1:8123/health', timeout=2).read()",
                        check=False,
                    )
                    if probe.returncode == 0:
                        break
                    await asyncio.sleep(0.5)
                else:
                    raise RuntimeError("Local research fixture service did not become ready")
            assert task.user is not None
            response = await env.execute(
                [turn.message.replace("{research_site_port}", "8123") for turn in task.user.turns],
                timeout=300,
            )
        except Exception as exc:
            runtime_failed = True
            write_json(output / "runtime-error.json", {"error": str(exc)})
        finally:
            try:
                env.close()
            except Exception as exc:
                runtime_failed = True
                write_json(output / "cleanup-error.json", {"error": str(exc)})
        transcript = Transcript.model_validate(response.get("transcript", {}))
        valid = (
            not runtime_failed
            and not response.get("error")
            and transcript.stop_reason == "complete"
            and bool(transcript.assistant_messages)
            and any(
                e.get("type") == "model_response" and e.get("status") == 200
                for e in events_in(env.audit)
            )
        )
        checks = []
        if valid and task.completion:
            for check in task.completion.execution_checks:
                command = shlex.split(check.command)
                assert command[:4] == ["python3", "-I", "-m", "clawbench.task_verifiers"]
                result = subprocess.run(
                    [
                        sys.executable,
                        "-I",
                        str(source / "clawbench/task_verifiers.py"),
                        *command[4:],
                    ],
                    cwd=env.workspace,
                    text=True,
                    capture_output=True,
                    timeout=30,
                )
                checks.append(
                    {
                        "name": check.name,
                        "passed": result.returncode == 0,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                    }
                )
        audit = model_audit(output, response, args.model.split("/", 1)[1]) if valid else {}
        grade = {
            "task_id": task.id,
            "run_valid": valid,
            "stop_reason": transcript.stop_reason,
            "tool_calls": len(transcript.tool_call_sequence),
            "checks": checks,
            "model_audit": audit,
            "passed": valid
            and bool(checks)
            and all(c["passed"] for c in checks)
            and all(
                audit.get(k)
                for k in [
                    "model_and_reasoning_match",
                    "tool_calls_match_relay",
                    "assistant_text_matches_relay",
                ]
            ),
        }
        write_json(output / "verification.json", grade)
        results.append(grade)
        write_json(root / "results.json", results)
        print(json.dumps(grade), flush=True)
    unchanged = dependency_manifest(image) == dependencies
    write_json(root / "dependency-integrity.json", {"unchanged": unchanged})
    if not unchanged or not all(g["passed"] for g in results):
        raise SystemExit(2)


if __name__ == "__main__":
    asyncio.run(main())
