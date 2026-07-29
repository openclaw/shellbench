---
name: shellbench-research-runbook
description: Plan, smoke-test, execute, checkpoint, publish, audit, and reproduce full ShellBench native benchmark campaigns across OpenClaw, Hermes, Codex, and Claude Code, including model and reasoning identity, pinned harness versions, n=3 qualification through n=6 research runs, S3 trace retention, and task-turn-tool-token-cost exports.
---

# ShellBench Research Runbook

Use this skill for a real benchmark campaign, not a one-off local score.

Read [references/runbook.md](references/runbook.md) before provisioning machines.
It is the normative campaign contract and contains the commands, gates, artifact
schema, and recovery rules.

## Non-negotiable gates

1. Use remote Crabbox AWS beasts for benchmark execution. Never run scored
   trials on the operator laptop.
2. Pin one public-task commit, runner commit or patch hash, provider model ID,
   harness version, reasoning level, and judge route for the whole campaign.
3. Run three-task smokes for every distinct harness, model, and reasoning route.
   Do not start full-suite jobs until model identity, real traces, tools, usage,
   judge routing, and artifact export pass.
4. Qualify with independent repetitions `r1` through `r3`. After a clean audit,
   add `r4` through `r6`; the research result is six total repetitions.
5. Run every provider-supported non-maximum reasoning level. Never label a
   reasoning level as tested unless the route applies it and the trace or proxy
   evidence proves it. Record unsupported levels instead of fabricating them.
6. Use `gpt-5.6-sol` at `high` as the default judge. Keep the judge alias,
   credentials, logs, and identity audit separate from the agent route.
7. Start checkpointing after the first completed trial and continue at least
   every ten minutes or ten new results. Verify each local archive before it
   counts.
8. Upload every verified checkpoint and final archive to the private S3 prefix
   from `SHELLBENCH_TRACE_S3_URI`. Never put bucket names or credentials in git,
   PR text, public logs, or generated reports.
9. A run is not research-clean when traces are missing, observed model identity
   differs from the request, reasoning is unproven, coverage is incomplete, or
   infrastructure failures dominate.

## Required commands

Generate reasoning-specific plans with unique labels:

```sh
python -m scripts.native_eval.plan \
  --tasks-root "$TASKS_ROOT" \
  --output "$CAMPAIGN/manifests/run-index-high.json" \
  --public-tasks-commit "$PUBLIC_TASKS_COMMIT" \
  --run-date "$RUN_DATE" \
  --reasoning-effort high \
  --judge-model-id gpt-5.6-sol \
  --judge-reasoning-effort high \
  --repetitions 3
```

Use repeatable `--harness` and `--model` filters for smoke or phased plans.
Set `--repetitions 6` only after the first three repetitions pass qualification.

After extraction, produce the research tables and strict identity report:

```sh
python -m scripts.native_eval.research_audit \
  --run-index "$CAMPAIGN/manifests/run-index.json" \
  --extracted-root "$CAMPAIGN/extracted" \
  --output-dir "$CAMPAIGN/summaries/research"
```

Treat any false row in `model_identity_audit.csv` as a blocker. The exporter
does not invent prices: missing exact spend remains explicitly unavailable.

## Stop conditions

- Stop a route after any smoke contains the wrong or multiple model IDs.
- Stop when the judge's observed provider model cannot be proven.
- Stop when a harness version differs from its campaign pin.
- Preserve and exclude infra-dominated runs, then rerun the same repetition
  with a suffix and lower concurrency.
- Never delete, overwrite, or silently replace an artifact or repetition.
