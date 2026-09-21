"""Independent executable solutions derived from public inputs, not expected outputs."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import httpx


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def consent_reference(workspace: Path, cutoff: int) -> None:
    people = json.loads((workspace / "people.json").read_text())
    events = json.loads((workspace / "consent-events.json").read_text())
    readings = json.loads((workspace / "readings.json").read_text())
    latest = {}
    for event in sorted(events, key=lambda e: e["seq"]):
        if event["seq"] <= cutoff:
            latest[(event["tenant"], event["person"], event["purpose"])] = event["grant"]
    rows = []
    for person in people:
        key = (person["tenant"], person["person"])
        if not latest.get((*key, "quality"), False):
            continue
        observations = [r for r in readings if (r["tenant"], r["person"]) == key]
        rows.append(
            {
                "public_id": person["public_id"],
                "passed": sum(r["passed"] for r in observations),
                "total": len(observations),
            }
        )
    destination = workspace / "handoff"
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        shutil.rmtree(destination)
    write(
        destination / "release.json",
        {
            "purpose": "quality",
            "cutoff": cutoff,
            "rows": sorted(rows, key=lambda r: r["public_id"]),
        },
    )


def evidence_reference(workspace: Path) -> None:
    target = json.loads((workspace / "target.json").read_text())
    receipts = json.loads((workspace / "execution-receipts.json").read_text())
    rows = []
    for claim in sorted(target["claims"], key=lambda c: c["id"]):
        relevant = [
            r
            for r in receipts
            if r["claim"] == claim["id"]
            and r["build"] == target["build"]
            and r["scope"] == claim["required_scope"]
        ]
        last = max(relevant, key=lambda r: r["seq"]) if relevant else None
        if last is None or last["assertions"] == 0:
            status, receipt = "unverified", None
        else:
            status, receipt = ("verified" if last["status"] == "pass" else "failed"), last["id"]
        rows.append({"claim": claim["id"], "status": status, "receipt": receipt})
    write(
        workspace / "assessment.json",
        {
            "build": target["build"],
            "ready": all(r["status"] == "verified" for r in rows),
            "claims": rows,
        },
    )


def batch_reference(workspace: Path, base: str) -> None:
    policy = json.loads((workspace / "batch.json").read_text())
    rows = []
    for body in sorted(policy["items"], key=lambda r: r["item"]):
        for _ in range(3):
            response = httpx.post(base + "/deliveries", json=body, timeout=10)
            if response.status_code == 503:
                continue  # Exact replay, not a new request identity.
            if response.status_code == 403:
                rows.append({"item": body["item"], "status": "blocked", "delivery_id": None})
                break
            response.raise_for_status()
            rows.append(
                {
                    "item": body["item"],
                    "status": "ready",
                    "delivery_id": response.json()["delivery"]["id"],
                }
            )
            break
        else:
            raise AssertionError("Reference exceeded the documented recoverable failure budget")
    write(workspace / policy["receipt_path"], {"batch": policy["batch"], "items": rows})
