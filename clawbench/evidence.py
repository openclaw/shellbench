"""Persist run evidence before workspace cleanup.

For adversarial runs the caller must enforce an OS/container permission boundary
between this directory and the actor. Retention alone is not trusted isolation.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from clawbench.schemas import TaskRunResult


def persist_run_evidence(result: TaskRunResult, workspace: Path) -> Path:
    configured = os.environ.get("CLAWBENCH_EVIDENCE_DIR", "").strip()
    root = Path(configured) if configured else workspace.parent / "_evidence"
    destination = root / workspace.name
    destination.mkdir(parents=True, exist_ok=True)
    result.evidence_path = str(destination)
    # Save the trace first; an artifact-copy error must not erase it.
    temporary = destination / "result.json.tmp"
    temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination / "result.json")
    if workspace.exists():
        # Preserve links as links, never follow them into unrelated host files.
        shutil.copytree(workspace, destination / "workspace", symlinks=True, dirs_exist_ok=True)
    (destination / "retention.json").write_text(
        json.dumps({"workspace": str(workspace), "artifact_copy_complete": True}) + "\n",
        encoding="utf-8",
    )
    return destination
