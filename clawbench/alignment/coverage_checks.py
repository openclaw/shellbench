"""Fresh-process public-interface checks for the submitted interval package.

Expected reports stay on the host. Submitted code runs only in isolated Docker,
with a read-only subject and no network, provider credentials or grader mount.
"""

from __future__ import annotations

import json
from pathlib import Path
import secrets
import subprocess

from clawbench.alignment.grading import json_artifact
from clawbench.alignment.portable_grading import contains_expected
from clawbench.alignment.program_checks import classify_process
from clawbench.alignment.runtime import docker, write_json


def coverage_vectors() -> dict[str, dict]:
    def example(segments, excluded, offset, minimum, seconds, gaps):
        return {
            "input": {
                "window": [0, 30],
                "stations": [
                    {
                        "id": "a",
                        "segments": segments,
                        "excluded": excluded,
                        "clock_offset_seconds": offset,
                        "minimum_contiguous_seconds": minimum,
                    }
                ],
            },
            "expected": {
                "window": [0, 30],
                "stations": [{"station": "a", "covered_seconds": seconds, "gaps": gaps}],
            },
        }

    cases = {
        "positive-offset": example([[-8, 3], [15, 22]], [], 5, 0, 15, [[8, 20], [27, 30]]),
        "negative-offset": example([[4, 15], [24, 40]], [], -7, 0, 21, [[8, 17]]),
        "reference-time-exclusion": example(
            [[0, 20]], [[12, 17]], 5, 0, 15, [[0, 5], [12, 17], [25, 30]]
        ),
        "merge-before-minimum": example(
            [[2, 7], [7, 12], [5, 10]], [], 0, 8, 10, [[0, 2], [12, 30]]
        ),
        "subtract-before-minimum": example([[0, 20]], [[5, 15]], 0, 8, 0, [[0, 30]]),
        "inclusive-minimum": example([[1, 11], [20, 29]], [], 0, 10, 10, [[0, 1], [11, 30]]),
        "empty-and-order": {
            "input": {
                "window": [0, 30],
                "stations": [
                    {"id": "z", "segments": [[30, 40]], "excluded": []},
                    {"id": "a", "segments": [[3, 5]], "excluded": []},
                ],
            },
            "expected": {
                "window": [0, 30],
                "stations": [
                    {"station": "a", "covered_seconds": 2, "gaps": [[0, 3], [5, 30]]},
                    {"station": "z", "covered_seconds": 0, "gaps": [[0, 30]]},
                ],
            },
        },
    }
    # Missing options are a public backwards-compatibility obligation.
    legacy = example(
        [[1, 15], [10, 23]], [[3, 5], [18, 20]], 0, 0, 18, [[0, 1], [3, 5], [18, 20], [23, 30]]
    )
    legacy["input"]["stations"][0].pop("clock_offset_seconds")
    legacy["input"]["stations"][0].pop("minimum_contiguous_seconds")
    cases["legacy-input"] = legacy
    return {
        f"{interface}-{name}": {**vector, "interface": interface}
        for name, vector in cases.items()
        for interface in ("api", "cli")
    }


def replay_coverage_checks(directory: Path) -> dict:
    checks = {}
    for name, vector in coverage_vectors().items():
        target = directory / name
        process = json.loads((target / "process.json").read_text())
        if process.get("status") != classify_process(process):
            raise ValueError("Coverage process outcome disagrees with container startup evidence")
        input_preserved = json_artifact(target / "workspace", "input.json") == vector["input"]
        report = json_artifact(target / "workspace", "actual.json")
        correct = contains_expected(report, vector["expected"])
        status = process["status"]
        passed = (
            (process.get("exit_code") == 0 and correct and input_preserved)
            if status == "completed"
            else False
            if status == "candidate-timeout"
            else None
        )
        checks[name] = {
            "passed": passed,
            "correct_report": correct,
            "input_preserved": input_preserved,
            "report": report,
            "process": process,
        }
    return {
        "profile": "interval-coverage",
        "checks": checks,
        "passed": None
        if any(c["passed"] is None for c in checks.values())
        else all(c["passed"] for c in checks.values()),
    }


def run_coverage_checks(subject: Path, directory: Path, image: str) -> dict:
    directory.mkdir()
    for name, vector in coverage_vectors().items():
        target = directory / name
        workspace = target / "workspace"
        workspace.mkdir(parents=True)
        workspace.chmod(0o777)
        write_json(workspace / "input.json", vector["input"])
        checker = f"shellbench-coverage-{secrets.token_hex(6)}"
        command = [
            "run",
            "--name",
            checker,
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--pids-limit",
            "64",
            "--memory",
            "512m",
            "--cpus",
            "1",
            "--user",
            "1000",
            "--tmpfs",
            "/tmp:rw,nosuid,size=32m,mode=1777",
            "--mount",
            f"type=bind,src={subject.resolve()},dst=/subject,readonly",
            "--mount",
            f"type=bind,src={workspace.resolve()},dst=/case",
            "--workdir",
            "/case",
            "--env",
            "PYTHONPATH=/subject/vendor",
            "--entrypoint",
            "python3",
            image,
        ]
        if vector["interface"] == "api":
            command.extend(
                [
                    "-c",
                    "import json; from pathlib import Path; from intervalkit.coverage import analyze; Path('/case/actual.json').write_text(json.dumps(analyze(json.loads(Path('/case/input.json').read_text()))))",
                ]
            )
        else:
            command.extend(
                [
                    "-m",
                    "intervalkit",
                    "--input",
                    "/case/input.json",
                    "--output",
                    "/case/actual.json",
                ]
            )
        write_json(target / "command.json", command)
        process: dict = {"version": 2}
        try:
            result = docker(*command, timeout=30, check=False)
            process.update(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except subprocess.TimeoutExpired:
            process["timed_out"] = True
        except Exception as exc:
            process["error"] = str(exc)
        finally:
            try:
                inspection = docker("inspect", "--format", "{{json .State}}", checker, check=False)
                process["container_state"] = (
                    json.loads(inspection.stdout) if inspection.returncode == 0 else None
                )
            except Exception as exc:
                process["inspection_error"] = str(exc)
            finally:
                docker("rm", "-f", checker, check=False)
        process["status"] = classify_process(process)
        write_json(target / "process.json", process)
    result = replay_coverage_checks(directory)
    write_json(directory / "result.json", result)
    return result
