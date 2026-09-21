"""Compatibility entry point; benchmark uses the trusted installed verifier."""

from clawbench.task_verifiers import main

if __name__ == "__main__":
    raise SystemExit(main(["trip", "landmark"]))
