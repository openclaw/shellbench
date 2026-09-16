# Contributing to ClawBench

Thank you for your interest in contributing. This document explains how to get
set up, what kinds of contributions are welcome, and how the review process
works.

---

## Getting started

**Requirements:** Python 3.11+, Docker (for full end-to-end runs).

```bash
git clone https://github.com/openclaw/clawbench.git
cd clawbench
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the test suite to confirm everything is working:

```bash
python -m pytest -q
python -m ruff check clawbench app.py scripts tests
```

The full local suite should pass before you make any changes.

---

## What we welcome

| Type | Notes |
|------|-------|
| **Bug fixes** | Include a test that reproduces the bug before the fix |
| **New tasks** | See [Adding tasks](#adding-tasks) below |
| **Scoring improvements** | Changes to `trajectory.py`, `scorer.py`, or `judge.py` must include updated tests and a clear rationale |
| **Documentation** | Fixes to README, spec docs, or inline comments |
| **Tooling / CI** | Workflow improvements, linting, dependency updates |

We are unlikely to merge:
- Large architectural rewrites without prior discussion in an issue
- New dependencies without justification
- Changes that reduce test coverage

---

## Making a change

1. **Open an issue first** for anything non-trivial. This lets us align on
   approach before you invest time writing code.

2. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/short-description
   ```
   Branch names: `fix/`, `feat/`, `docs/`, `chore/` prefixes.

3. **Write tests.** Bug fixes must include a test that fails before the fix
   and passes after. New features must include tests covering the new
   behaviour.

4. **Run the test suite:**
   ```bash
   python -m pytest -q
   ```

5. **Open a pull request** against `main`. Fill in the PR template.

---

## Adding tasks

Public tasks live in `tasks-public/tier{1-5}/` as YAML files. Domain and
partner tasks live under `tasks-domain/`. Each task needs:

- A unique `id` and descriptive `name`
- The correct `tier` (1 = simple single-tool, 5 = adversarial/multi-step)
- `completion` checks — at least one deterministic verifier (`execution_checks`,
  `file_equality`, or a gateway assertion)
- `trajectory` expectations that reflect how a competent agent should approach
  the task
- A `judge` rubric for semantic tasks

Before submitting a new task, run it against at least one agent to verify the
completion checks fire correctly.

Native Docker tasks may run as a non-root user. Their log mounts remain writable
by that user; the host trial directory is restricted to its owner (`0700`) so
unrelated host users cannot traverse to or modify the logs.

---

## Commit style

```
type: short imperative summary (≤72 chars)

Optional longer explanation. Wrap at 72 chars. Explain *why*, not what —
the diff shows what changed.
```

Types: `fix`, `feat`, `docs`, `test`, `chore`, `refactor`.

---

## Code style

The project uses Ruff and pre-commit for local guardrails. Please follow the
style of the surrounding code: 4-space indentation, descriptive variable names,
and comments only where the logic is not self-evident.

```bash
python -m ruff check clawbench app.py scripts tests
pre-commit run --files <changed files>
```

---

## Reporting bugs

Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md). Include:
- The command you ran
- The full error output or unexpected behaviour
- The Python version and OS

---

## Questions

Open an issue for questions that are not bug reports or feature requests.
