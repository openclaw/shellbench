"""Create a deterministic existing checkout before the actor receives its task.

The runtime executes this module's source inside its isolated actor container.
Task helpers are configured only after all index/commit operations have finished.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def initialize(fixture: dict, workspace: Path) -> dict:
    repository = workspace / fixture["directory"]
    metadata = repository / ".git"
    if metadata.exists() or metadata.is_symlink():
        raise ValueError("Git fixture requires an uninitialized repository")

    def regular(name: str) -> Path:
        path = workspace / name
        if not path.is_relative_to(workspace) or ".." in Path(name).parts:
            raise ValueError("Git fixture path escapes the workspace")
        if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
            raise ValueError("Git fixture inputs must be regular files without symlink parents")
        return path

    tracked = {name: regular(f"{fixture['directory']}/{name}") for name in fixture["tracked"]}
    originals = {name: path.read_bytes() for name, path in tracked.items()}
    replacements = {
        name: regular(source).read_bytes() for name, source in fixture["baseline_sources"].items()
    }
    if not replacements.keys() <= originals.keys():
        raise ValueError("Git baseline replacements must identify tracked paths")
    env = dict(
        os.environ,
        GIT_AUTHOR_DATE="2026-01-01T00:00:00+00:00",
        GIT_COMMITTER_DATE="2026-01-01T00:00:00+00:00",
    )

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
        ).stdout

    try:
        for name, content in replacements.items():
            tracked[name].write_bytes(content)
        git("init", "-b", "main")
        git("config", "user.name", "Synthetic Fixture Author")
        git("config", "user.email", "fixture@example.invalid")
        git("add", "--", *tracked)
        git("commit", "-m", "Initial synthetic fixture")
        baseline_commit = git("rev-parse", "HEAD").strip()
        baseline_entries = git("ls-files", "--stage", "-z").split("\x00")
    finally:
        for name, content in originals.items():
            tracked[name].write_bytes(content)
    git("add", "--", *tracked)
    staged_entries = git("ls-files", "--stage", "-z").split("\x00")
    version = git("--version").strip()
    # No command that refreshes the index runs after enabling local helpers.
    for name, value in fixture["local_config"].items():
        git("config", "--local", name, value)
    configuration = {
        name: git("config", "--local", "--get", name).strip() for name in fixture["local_config"]
    }
    return {
        "baseline_commit": baseline_commit,
        "git_version": version,
        "directory": fixture["directory"],
        "local_config": configuration,
        "baseline_entries": [entry for entry in baseline_entries if entry],
        "staged_entries": [entry for entry in staged_entries if entry],
        "seeded_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in tracked.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(initialize(json.loads(sys.argv[1]), Path("/workspace"))))
