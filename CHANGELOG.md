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
- Detect destructive `git checkout --` restoration commands in trajectory safety scoring (#39, thanks @realmehmetali).

### Added

- Add `gpt-5.6-luna` and `gpt-5.6-terra` to the native evaluation catalog.
