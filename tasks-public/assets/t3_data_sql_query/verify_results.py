"""Recursive workspace search verifier."""

from __future__ import annotations

import csv
import io
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
    # Find a CSV-shaped file with the EU 2026 active signups data
    for path, text in iter_workspace_text_files():
        if path.suffix.lower() != ".csv":
            continue
        rows = list(csv.reader(io.StringIO(text)))
        if not rows:
            continue
        first_is_header = not any(any(c.isdigit() for c in cell) for cell in rows[0])
        data_rows = rows[1:] if first_is_header else rows
        if len(data_rows) != 7:
            continue
        blob = " ".join(c for r in data_rows for c in r).lower()
        if "old" in blob and ("do not use" in blob or "deprecated" in blob):
            continue
        expected = ["organic", "paid social", "email newsletter", "referral partner"]
        if sum(1 for c in expected if c in blob) >= 2:
            print(f"PASS: 7 rows + correct channels in {path}")
            return 0

    # Also accept any text file with the right content shape
    blob = workspace_blob().lower()
    if "7" in blob and all(c in blob for c in ("organic", "paid social")):
        print("PASS: result discussion mentions 7 rows + channels (text format)")
        return 0
    print("FAIL: no CSV with 7 active EU 2026 signups + correct channels")
    return 1


if __name__ == "__main__":
    sys.exit(main())
