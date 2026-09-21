"""Compatibility entrypoint; official checks use the trusted module directly."""

from clawbench.task_verifiers import main

if __name__ == "__main__":
    raise SystemExit(main(["redaction"]))
