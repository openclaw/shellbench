# Changelog

## 0.4.0 - Unreleased

### Fixed

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
