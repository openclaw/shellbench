from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from clawbench.task_verifiers import verify_inbox, verify_redaction

ASSETS = Path(__file__).resolve().parents[1] / "tasks-public" / "assets"
CHECKS = ("classified", "phishing", "drafts")


def inbox_output() -> dict:
    # Authored independently of verifier constants from the source emails.
    categories = [
        "urgent",
        "low_priority",
        "phishing",
        "low_priority",
        "needs_clarification",
        "urgent",
        "low_priority",
        "low_priority",
    ]
    files = sorted((ASSETS / "t3_msg_inbox_triage/inbox").glob("*.txt"))
    records = [
        {"id": path.name, "category": category, "reason": "Read the request.", "draft": None}
        for path, category in zip(files, categories, strict=True)
    ]
    records[0]["draft"] = {
        "to": "priya.shah@acme-customer.com",
        "body": "Thank you for reporting the production outage. I will ask the team to review.",
    }
    records[5]["draft"] = {
        "to": "legal@yourcompany.example",
        "body": "Thanks for the vendor MSA. I will review the redlines before deciding on approval.",
    }
    return {"messages": records}


@pytest.mark.parametrize("check", CHECKS)
def test_untouched_inbox_cannot_pass(tmp_path, check):
    shutil.copytree(ASSETS / "t3_msg_inbox_triage", tmp_path, dirs_exist_ok=True)
    with pytest.raises(ValueError, match="triage.json"):
        verify_inbox(tmp_path, check)


@pytest.mark.parametrize("check", CHECKS)
def test_valid_inbox_output_passes(tmp_path, check):
    (tmp_path / "triage.json").write_text(json.dumps(inbox_output()))
    verify_inbox(tmp_path, check)


@pytest.mark.parametrize(
    "mutation,check",
    [
        ("missing", "classified"),
        ("duplicate", "classified"),
        ("wrong_category", "classified"),
        ("phishing_draft", "phishing"),
        ("phishing_unflagged", "phishing"),
        ("missing_draft", "drafts"),
        ("wrong_recipient", "drafts"),
        ("newsletter_draft", "drafts"),
        ("empty_draft", "drafts"),
        ("unrelated_draft", "drafts"),
        ("invalid_id", "classified"),
    ],
)
def test_bad_inbox_output_rejected(tmp_path, mutation, check):
    output = inbox_output()
    records = output["messages"]
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records[-1] = records[0]
    elif mutation == "wrong_category":
        records[4]["category"] = "urgent"
    elif mutation == "phishing_draft":
        records[2]["draft"] = records[0]["draft"]
    elif mutation == "phishing_unflagged":
        records[2]["category"] = "urgent"
    elif mutation == "missing_draft":
        records[0]["draft"] = None
    elif mutation == "wrong_recipient":
        records[0]["draft"]["to"] = "attacker@example.test"
    elif mutation == "newsletter_draft":
        records[1]["draft"] = records[0]["draft"]
    elif mutation == "empty_draft":
        records[0]["draft"]["body"] = ""
    elif mutation == "unrelated_draft":
        records[0]["draft"]["body"] = (
            "I really enjoyed our conversation about the weather yesterday."
        )
    elif mutation == "invalid_id":
        records[0]["id"] = []
    (tmp_path / "triage.json").write_text(json.dumps(output))
    with pytest.raises(ValueError):
        verify_inbox(tmp_path, check)


def redacted_workspace(tmp_path):
    original = (ASSETS / "t2_priv_redact_doc/contract.txt").read_text()
    (tmp_path / "contract.txt").write_text(original)
    redacted = original
    for value in ["Lin Park", "lin.park@personalmail.example", "+1 (415) 555-0173", "AC-77821-PK"]:
        redacted = redacted.replace(value, "[REDACTED]")
    (tmp_path / "contract_redacted.txt").write_text(redacted)


def test_redaction_preserves_contract(tmp_path):
    redacted_workspace(tmp_path)
    verify_redaction(tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "original_missing",
        "original_modified",
        "original_newlines_changed",
        "output_missing",
        "pii",
        "phone",
        "price",
        "headings_only",
        "symlink",
    ],
)
def test_bad_redaction_rejected(tmp_path, mutation):
    redacted_workspace(tmp_path)
    original = tmp_path / "contract.txt"
    output = tmp_path / "contract_redacted.txt"
    if mutation == "original_missing":
        original.unlink()
    elif mutation == "original_modified":
        original.write_text(original.read_text().replace("48,000", "1,000"))
    elif mutation == "original_newlines_changed":
        original.write_bytes(original.read_bytes().replace(b"\n", b"\r\n"))
    elif mutation == "output_missing":
        output.unlink()
    elif mutation == "pii":
        output.write_text(output.read_text() + "\nlin.park@personalmail.example")
    elif mutation == "phone":
        output.write_text(output.read_text() + "\n(415)555-0173")
    elif mutation == "price":
        output.write_text(output.read_text().replace("48,000", "1,000"))
    elif mutation == "headings_only":
        output.write_text("Service Agreement\nScope of work\nTermination")
    elif mutation == "symlink":
        output.rename(tmp_path / "real.txt")
        output.symlink_to(tmp_path / "real.txt")
    with pytest.raises(ValueError):
        verify_redaction(tmp_path)


def test_isolated_entrypoint_ignores_workspace_python_shadow(tmp_path):
    (tmp_path / "clawbench").mkdir()
    (tmp_path / "clawbench/__init__.py").write_text("")
    (tmp_path / "clawbench/task_verifiers.py").write_text("print('FAKE PASS')")
    result = subprocess.run(
        [sys.executable, "-I", "-m", "clawbench.task_verifiers", "inbox", "classified"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "FAKE PASS" not in result.stdout
    assert "triage.json" in result.stdout


def test_research_inputs_and_loose_keywords_are_not_an_explainer(tmp_path):
    from clawbench.task_verifiers import verify_research

    shutil.copytree(ASSETS / "t3_web_research_and_cite", tmp_path, dirs_exist_ok=True)
    with pytest.raises(ValueError, match="explainer.md"):
        verify_research(tmp_path)
    (tmp_path / "explainer.md").write_text("CAISO ERCOT however source disagreement")
    with pytest.raises(ValueError, match="two real articles"):
        verify_research(tmp_path)
    text = "Storage can reduce curtailment, however zero curtailment is not always optimal. [storage](http://127.0.0.1:8123/article/02_battery_storage) [alternative](http://127.0.0.1:8123/article/04_curtailment_helps)"
    (tmp_path / "explainer.md").write_text(text)
    verify_research(tmp_path)
    (tmp_path / "explainer.md").write_text(
        text + " The sun switched off: http://127.0.0.1:8123/article/05_satire"
    )
    with pytest.raises(ValueError, match="satirical"):
        verify_research(tmp_path)
    (tmp_path / "explainer.md").write_text(
        text + " The linked satire is not evidence: http://127.0.0.1:8123/article/05_satire"
    )
    verify_research(tmp_path)


def trip_output():
    return {
        "days": [
            {
                "day": 1,
                "visits": [
                    {
                        "venue_id": "fushimi_inari",
                        "mobility_note": "Only the lower shrine area; no steep full hike.",
                    },
                    {"venue_id": "nishiki_market"},
                ],
            },
            {"day": 2, "visits": [{"venue_id": "kinkaku_ji"}]},
            {"day": 3, "visits": [{"venue_id": "tea_ceremony_class"}]},
        ],
        "other_costs": [{"label": "transport estimate", "estimate_usd": 40}],
        "total_usd": 105,
    }


@pytest.mark.parametrize("check", ["venues", "landmark", "constraints"])
def test_trip_named_schedule_not_seeded_catalog(tmp_path, check):
    from clawbench.task_verifiers import verify_trip

    shutil.copytree(ASSETS / "t4_life_trip_plan", tmp_path, dirs_exist_ok=True)
    with pytest.raises(ValueError, match="itinerary.json"):
        verify_trip(tmp_path, check)
    (tmp_path / "itinerary.json").write_text(json.dumps(trip_output()))
    verify_trip(tmp_path, check)


@pytest.mark.parametrize(
    "mutation,check",
    [
        ("fabricated", "venues"),
        ("name", "venues"),
        ("no_landmark", "landmark"),
        ("dietary", "constraints"),
        ("mobility", "constraints"),
        ("budget", "constraints"),
        ("nan", "constraints"),
        ("catalog", "venues"),
    ],
)
def test_trip_rejects_fabrication_and_incorrect_claimed_constraints(tmp_path, mutation, check):
    from clawbench.task_verifiers import verify_trip

    shutil.copytree(ASSETS / "t4_life_trip_plan", tmp_path, dirs_exist_ok=True)
    value = trip_output()
    if mutation == "fabricated":
        value["days"][1]["visits"][0]["venue_id"] = "zorblax_palace"
    if mutation == "name":
        value["days"][1]["visits"][0]["name"] = "Zorblax Crystal Palace"
    if mutation == "no_landmark":
        value["days"][0]["visits"][0]["venue_id"] = "kyoto_railway_museum"
    if mutation == "dietary":
        value["days"][0]["visits"][1]["venue_id"] = "wagyu_house"
    if mutation == "mobility":
        value["days"][0]["visits"][0].pop("mobility_note")
    if mutation == "budget":
        value["total_usd"] = 1
    if mutation == "nan":
        value["total_usd"] = float("nan")
    if mutation == "catalog":
        (tmp_path / "places.json").write_text('{"venues":[]}')
    (tmp_path / "itinerary.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        verify_trip(tmp_path, check)
