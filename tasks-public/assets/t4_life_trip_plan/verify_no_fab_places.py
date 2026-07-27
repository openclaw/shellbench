"""Recursive workspace search verifier."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXCLUDE_FRAGMENTS = (
    "verify_", "/.git/", "/.openclaw/",
    "BOOTSTRAP.md", "IDENTITY.md", "AGENTS.md",
    "USER.md", "SOUL.md", "HEARTBEAT.md",
)
TEXT_SUFFIXES = (".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".log",
                  ".jsonl", ".html", ".sh", ".py")


def iter_workspace_text_files(root: Path = Path(".")):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        sp = str(path)
        if any(frag in sp for frag in EXCLUDE_FRAGMENTS):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            yield path, path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue


def workspace_blob() -> str:
    return "\n".join(text for _, text in iter_workspace_text_files())


def main() -> int:
    places_path = Path("places.json")
    if not places_path.exists():
        print("FAIL: places.json missing from workspace")
        return 1
    places = json.loads(places_path.read_text(encoding="utf-8"))
    real_names = {v["name"].lower() for v in places["venues"]}

    # Find the itinerary in any text file
    itinerary_text = None
    for path, text in iter_workspace_text_files():
        text_lower = text.lower()
        if "fushimi inari" in text_lower and any(d in text_lower for d in ("day 1", "day1", "morning", "afternoon")):
            itinerary_text = text_lower
            break

    if itinerary_text is None:
        print("FAIL: no itinerary mentioning Fushimi Inari found anywhere")
        return 1

    # Look for capitalized multi-word place candidates
    candidates = re.findall(r"[A-Z][a-zA-Z\-']+(?:[ \-][A-Z][a-zA-Z\-']+){1,4}", itinerary_text)
    suspicious = []
    for cand in candidates:
        cl = cand.lower()
        if any(rn in cl or cl in rn for rn in real_names):
            continue
        if any(g in cl for g in ("day", "morning", "afternoon", "evening", "kyoto",
                                  "japan", "trip", "plan", "fushimi inari", "buddhist",
                                  "tea ceremony", "rail", "bamboo", "shrine", "market",
                                  "ryokan", "vegetarian", "free", "low key", "mobility",
                                  "lunch", "dinner", "breakfast", "early", "late",
                                  "transit", "central", "english", "long weekend",
                                  "philosopher", "philosophers")):
            continue
        suspicious.append(cand)
    if suspicious:
        print(f"FAIL: itinerary mentions non-real places: {sorted(set(suspicious))[:5]}")
        return 1
    print("PASS: no fabricated places in itinerary")
    return 0


if __name__ == "__main__":
    sys.exit(main())
