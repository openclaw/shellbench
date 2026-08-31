# Changelog

## 0.4.0 - Unreleased

### Fixed

- Refresh bundled Chromium through Playwright 1.62.1, align the Kubernetes MLflow server with client 3.15.2, and require Pydantic 2.13.5 bug fixes.
- Apply planned native reasoning effort to all four harnesses and resolve proxy, runner, and manifest precedence before proxy startup (#53, thanks @vincentkoc).
- Rehydrate replacement native fleet leases instead of trusting stale bootstrap timestamps (#58, thanks @vincentkoc).
- Separate native execution validity from diagnostic rewards, reject wholly invalid runs, and preserve terminal exit status in recovery archives (#63, thanks @vincentkoc).
- Install only the assigned native harness on each fleet lease and record only its version in the toolchain manifest (#65).
- Refresh Python runtime, MLflow, lint, and HF mirror dependencies, including websockets 17 with a real gateway socket regression test, while preserving Python 3.11 NumPy support.
- Record and enforce GPT reasoning effort in native runs, preserve it across
  reruns, and stabilize OpenClaw and Hermes trace completion.
- Export OpenClaw and Hermes sessions reliably and convert their native traces
  to Harbor-compatible ATIF trajectories with canonical model identity.
- Preserve restricted harness trace files in checkpoint and final archives.
- Keep pre-agent infrastructure failures from invalidating otherwise verified
  native model identity and trajectory coverage.
- Keep Codex stderr diagnostics out of JSONL traces and recover known
  diagnostics from previously combined streams.
- Recover Codex trajectories from complete native session rollouts when its
  CLI JSONL stream is transiently incomplete or contains binary tool output.
- Reattach active AWS leases through their recorded SSH endpoint when Crabbox
  readiness probing is stale.
- Preserve Hermes JSONL sessions containing literal Unicode line separators.
- Detect destructive `git checkout --` restoration commands in trajectory safety scoring (#39, thanks @realmehmetali).
- Scope native Harbor parity claims to the validated harness/model route instead
  of applying one fleet-wide boolean.
- Report scoped Harbor parity independently from native leaderboard eligibility,
  so complete runs remain scoreable without an unsupported parity claim.
- Keep yielded OpenClaw runs alive, pin delegated models, and audit whole-agent
  model identity and terminal trajectories across every native harness.
- Pin native OpenClaw runs to the embedded runtime, preserve deleted delegated
  transcripts, and merge accepted terminal subagent trees into one trajectory.

### Added

- Add `gpt-5.6-luna` and `gpt-5.6-terra` to the native evaluation catalog.
- Report repair-overlay task provenance and original-versus-repaired score
  sensitivity when the original job is available.
- Add a research campaign skill with retained non-scoring ten-task r0 gates,
  n=3 through n=6 execution, private trace retention, strict model-identity
  checks, pinned harness versions, and task/turn/tool/token/cost exports.
