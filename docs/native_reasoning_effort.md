# Native reasoning effort

The native fleet's planned `reasoning_effort` controls both the provider proxy
and the agent harness. Supported values are `low`, `medium`, `high`, and `xhigh`.
OpenClaw receives `--thinking`, Hermes receives `agent.reasoning_effort`, Codex
receives `model_reasoning_effort`, and Claude Code receives `--effort` (`max` for
the canonical `xhigh` value). The run manifest records the resolved run value,
not a later reading of the environment.

`remote_run.sh` preserves the effort supplied by fleet dispatch before sourcing
the provider environment file. That planned value takes precedence over the
file's default. If dispatch supplies no effort, the environment file supplies
the default instead. Judge reasoning effort remains independent.

Before starting LiteLLM, the launcher invokes `run_job` with the complete run
arguments and `--prepare-proxy-config PATH`. This preparation mode lets an
explicit `--reasoning-effort` override the environment, validates the result,
writes the proxy configuration, and prints the canonical effort to standard
output. The launcher exports that value as `SHELLBENCH_REASONING_EFFORT` before
starting either the proxy or the job. Invalid values stop startup before the
proxy is launched.

For manual launches, use the same preparation step before starting the proxy
and export its returned value for the runner. An ordinary `run_job` invocation
rejects conflicting CLI and environment efforts: it cannot safely override a
proxy that is already running. Without either value, direct harness construction
retains each harness's defaults; the managed proxy still requires an explicit
valid effort, as before.
