# Full Native ShellBench Research Campaign

## 2026-07-29 Revision Appendix

This replaces the original full-matrix prompt with stronger research gates:
ten-task non-scoring `r0` family/harness qualification runs, verified private S3
retention, strict observed-model audits, requested-versus-installed harness
versions, n=3 qualification followed by n=6 total repetitions, every
provider-supported non-maximum reasoning level, `gpt-5.6-sol` at `high` as
judge, and task/turn/tool/token/cost exports with explicit evidence provenance.

## Objective

Run the current ShellBench combined public-task suite natively on remote
Crabbox AWS beasts across the selected harnesses, provider model IDs, and
reasoning levels. Preserve enough evidence to reproduce every score and audit
which model, harness build, tools, tokens, costs, judge, task commit, and runner
code produced it.

The campaign progresses through four gates:

1. route qualification: ten representative tasks at `r0`
2. initial research: full suite, independent repetitions `r1` through `r3`
3. audit: verify coverage, model identity, versions, traces, usage, and infra
4. research completion: add `r4` through `r6` only after the audit is clean

`n=6` means six independent jobs and writable environments. It does not mean
rerunning one job directory or replaying saved agent state. `r0` is not part of
`n=6`.

## Scope

Harnesses:

- `openclaw`
- `hermes`
- `codex`
- `claude-code`

The model list is campaign input. Resolve every friendly name to an actual
provider model ID from the live catalog before planning. Never infer or silently
substitute an ID.

Reasoning policy:

- enumerate every level the provider and route actually support
- run all supported non-maximum levels
- exclude provider levels named or documented as `max` or `ultra`
- the default OpenAI research set is `low`, `medium`, and `high`
- do not generate low/medium/high variants for a provider whose adapter ignores
  the setting
- record `unsupported` or `unverified` in the campaign manifest instead

Default judge:

```text
provider model ID: gpt-5.6-sol
reasoning: high
```

The judge must use a dedicated proxy alias. Agent and judge requests must be
distinguishable in proxy/provider logs.

## Current Toolchain Pins

The code source of truth is `scripts/native_eval/models.py` and
`scripts/native_eval/bootstrap_beast.sh`. At this runbook revision the pins are:

| Component | Pin |
|---|---|
| OpenClaw | `2026.7.1-2` |
| Hermes | `cb06017b1d6e1b9ae0cb35f99a48ffa6bcbaa828` |
| Codex | `0.145.0` |
| Claude Code | `2.1.220` |
| Node | `22.23.1` |
| LiteLLM | `1.93.0` |

Before every campaign:

1. compare this table with the code pins
2. record requested pins in the campaign manifest
3. record installed versions from `/opt/shellbench-native/manifest.json`
4. fail r0 when requested and installed versions differ

Do not update a pin during a campaign. Start a new campaign ID when a harness,
runner, task, provider route, or judge version changes.

For a trusted, unpublished OpenClaw build, pass its npm package tarball to the
fleet controller:

```sh
python -m scripts.native_eval.fleet \
  ... \
  --openclaw-package-tarball /path/to/openclaw.tgz
```

Candidate campaigns must contain only OpenClaw runs. The controller validates
the npm package name and version, stages it as
`manifests/openclaw-candidate.tgz`, and pins its SHA-256 in campaign, toolchain,
and run provenance. Resume with the same tarball identity; a missing or changed
candidate is rejected before leasing. Registry installation remains the
default when the option is absent.

## Campaign Identity

Use a stable campaign ID:

```text
shellbench-full-<task_count>-<public_tasks_short_sha>-<YYYYMMDD>
```

Reasoning-specific run labels are mandatory:

```text
<harness>-<model_slug>-<reasoning>-full-<task_count>-r<rep>-<date>
```

Examples:

```text
openclaw-gpt56-sol-low-full-115-r1-20260729
hermes-gpt56-sol-high-full-115-r6-20260729
```

Family/harness qualification uses:

```text
<harness>-<model_slug>-<reasoning>-smoke-10-r0-<date>
```

Retries append a suffix and never replace the original:

```text
-rerun1
-c8
-infra-timeout
```

## Preflight

Run locally without printing secrets:

```sh
crabbox whoami
crabbox --version
test -f .env
```

If auth is expired:

```sh
crabbox login
```

Check only whether required environment variables are present. Do not echo
their values.

Fetch and pin the task suite:

```sh
git -C work/public-tasks fetch origin
PUBLIC_TASKS_COMMIT="$(git -C work/public-tasks rev-parse origin/main)"
TASK_SUITE_PATH="combined tasks/tasks"
EXPECTED_TASK_COUNT="$(
  git -C work/public-tasks ls-tree -d --name-only \
    "$PUBLIC_TASKS_COMMIT:$TASK_SUITE_PATH" | wc -l | tr -d ' '
)"
test "$EXPECTED_TASK_COUNT" -gt 0
```

Validate every immediate task directory with
`scripts.native_eval.tasks.validate_suite`. The count must be derived from the
pinned commit, never copied from an older campaign.

Create and verify the immutable suite archive:

```sh
git -C work/public-tasks archive "$PUBLIC_TASKS_COMMIT" -- "$TASK_SUITE_PATH" \
  > public-tasks-main-combined.tar
gzip -f public-tasks-main-combined.tar

ARCHIVE_TASK_COUNT="$(
  tar -tzf public-tasks-main-combined.tar.gz |
    awk -F/ '$1=="combined tasks" && $2=="tasks" && $3!="" {print $3}' |
    sort -u | wc -l | tr -d ' '
)"
test "$ARCHIVE_TASK_COUNT" = "$EXPECTED_TASK_COUNT"
```

Record:

- public-tasks commit and suite path
- expected task count and archive SHA-256
- ShellBench runner commit and dirty patch hash
- every harness requested pin and installed version
- Crabbox CLI version
- lease ID, slug, instance type, IP, region, and timestamps
- friendly model slug, provider, requested provider ID, and proxy alias
- requested reasoning and whether the adapter can prove it
- judge provider ID, reasoning, proxy alias, and observed identity evidence

## Phase 1: Ten-Task r0 Qualification

Select exactly ten tasks from the pinned suite. The set should cover:

1. simple shell and filesystem work
2. code editing and test execution
3. browser, app, or multi-container work
4. long-running or stateful tool use
5. judge-backed or semantically verified work
6. representative easy, medium, and difficult tasks

Record task names and checksums. Do not hard-code task names in the runbook
because the public suite changes.

Run one r0 for every distinct harness and model-family route with:

- one fresh beast or isolated job
- repetition `0`
- exactly one harness and one representative provider model
- a recorded `qualification_family`
- concurrency `1` or `2`
- the same proxy and judge configuration intended for the full campaign
- checkpointing enabled immediately

r0 jobs may run in parallel across beasts. Do not increase per-r0 concurrency;
the purpose is routing and evidence validation, not throughput. Family
qualification proves the shared harness and routing path; every exact provider
model ID is still audited again during `r1` through `r3`.

Treat a different reasoning transport, proxy parameter path, or provider adapter
as a different route and give it its own r0. Do not assume a high-reasoning r0
proves low or medium when those settings travel through different code.

Generate one r0 plan per harness/family:

```sh
python -m scripts.native_eval.plan \
  --tasks-root "$TASKS_ROOT" \
  --output "$CAMPAIGN/manifests/r0-$HARNESS-$FAMILY.json" \
  --public-tasks-commit "$PUBLIC_TASKS_COMMIT" \
  --run-date "$RUN_DATE" \
  --phase r0 \
  --qualification-family "$FAMILY" \
  --harness "$HARNESS" \
  --model "$REPRESENTATIVE_MODEL_SLUG" \
  --reasoning-effort "$REASONING" \
  --judge-model-id gpt-5.6-sol \
  --judge-reasoning-effort high \
  --task "$TASK_01" \
  --task "$TASK_02" \
  --task "$TASK_03" \
  --task "$TASK_04" \
  --task "$TASK_05" \
  --task "$TASK_06" \
  --task "$TASK_07" \
  --task "$TASK_08" \
  --task "$TASK_09" \
  --task "$TASK_10"
```

The planner enforces ten tasks, repetition zero, one harness, one model, and
automatic leaderboard exclusion.

For every r0, verify:

- all ten task results exist
- `trajectory_status` is `real` for supported harnesses
- raw harness events and normalized `trajectory.json` are both retained
- observed agent model IDs equal exactly the requested provider model ID
- no hidden fallback, mixed model, or alias substitution appears
- requested reasoning is present in proxy request evidence
- installed harness version equals the pin
- tool calls and observations are represented in the trace
- task-level token totals exist or are explicitly unavailable
- exact provider cost exists or is explicitly unavailable
- judge requests use only `gpt-5.6-sol` at `high`
- checkpoint and final archives pass local `tar -tzf`
- verified archives upload to S3 and can be read back or headed

The agent trace alone cannot prove judge identity. Preserve proxy/provider
request logs and audit the dedicated judge alias during r0.

Any wrong, mixed, or unobserved model ID blocks that route. Fix the route and
repeat r0 under a new suffixed label.

Retain every r0 checkpoint, final archive, trace, log, and audit row. Discard r0
only from scoring: its manifest must set `leaderboard_eligible=false` and
`exclusion_reason=r0_non_scoring_qualification`.

## Phase 2: Initial Full-Suite Runs At n=3

After every required r0 passes, generate one full plan per reasoning level:

```sh
python -m scripts.native_eval.plan \
  --tasks-root "$TASKS_ROOT" \
  --output "$CAMPAIGN/manifests/run-index-$REASONING.json" \
  --public-tasks-commit "$PUBLIC_TASKS_COMMIT" \
  --run-date "$RUN_DATE" \
  --reasoning-effort "$REASONING" \
  --judge-model-id gpt-5.6-sol \
  --judge-reasoning-effort high \
  --repetitions 3
```

Use repeatable filters to phase the fleet:

```sh
--harness openclaw --harness hermes
--model gpt56-sol
```

Each repetition must have:

- a unique job directory
- a fresh writable task environment
- no reused agent session
- no copied response cache
- independent checkpoints and final archive

Prefer one AWS beast per run when quota allows. Otherwise run waves. Start
browser and app-heavy routes at task concurrency `16`; lower to `8` after
startup pressure, or raise a proven stable route to `32`. Do not use `96+`
except as a named infrastructure stress experiment.

## Phase 3: Audit r1 Through r3

Do not schedule `r4` through `r6` until every exact harness/model/reasoning route
has three complete full-suite runs and the audit confirms:

- complete task coverage
- exact requested agent model identity
- requested reasoning evidence
- installed harness versions match campaign pins
- judge routing is proven
- traces and provider/proxy logs are retained locally and in S3
- infrastructure failures are low enough for a fair comparison

Fix or rerun only failed routes and repetitions, then repeat the audit.

## Phase 4: Research Completion At n=6

Only after all three qualification repetitions are complete and clean:

1. regenerate or extend the plan to repetitions `r1` through `r6`
2. preserve the completed `r1` through `r3` entries and artifacts
3. schedule only new `r4` through `r6` jobs
4. audit all six together

Do not treat three clean runs plus three replacement runs as six independent
clean repetitions unless all six original run identities and artifacts remain.

## Checkpoint And Final Retention

For every active run:

1. pull after the first completed trial
2. pull at least every ten minutes or ten new completed trials
3. pull immediately on stalled progress, repeated infra errors, degraded lease
   health, or operator disconnect
4. make a final pull after the process exits, including failures

Names are monotonic:

```text
<run_label>-checkpoint-0001-artifacts.tar.gz
<run_label>-checkpoint-0002-artifacts.tar.gz
<run_label>-final-artifacts.tar.gz
```

After every local copy:

```sh
tar -tzf "$LOCAL_ARCHIVE" >/dev/null
RESULT_COUNT="$(
  tar -tzf "$LOCAL_ARCHIVE" |
    awk '/\/result\.json$/ {count++} END {print count+0}'
)"
shasum -a 256 "$LOCAL_ARCHIVE"
```

Keep all verified checkpoints even after a successful final export.

The archive must contain:

- run and trial `result.json`
- `run_manifest.json`, `config.json`, and lock/state metadata
- raw harness sessions and event streams
- normalized trajectories
- stdout, stderr, setup, proxy, and verifier logs
- provider/proxy usage and spend records
- task checksums and suite commit
- harness/toolchain installed-version manifest

## Private S3 Publication

The destination comes only from:

```text
SHELLBENCH_TRACE_S3_URI
```

It is a private `s3://` prefix supplied outside git. Do not place its value in
the run index, generated Markdown, PR text, shell history, or logs.

Upload only locally verified archives. Store:

```text
<private-prefix>/<campaign_id>/raw/<archive>
<private-prefix>/<campaign_id>/manifests/<archive>.sha256
<private-prefix>/<campaign_id>/manifests/upload-index.json
```

When `clawbench traces upload` is available, prefer it because it records
SHA-256 metadata and verifies the remote object. Otherwise use AWS CLI with
server-side encryption, write a checksum sidecar, and verify the remote object
metadata or read-back before marking the upload complete.

Never delete the local verified copy merely because S3 upload succeeded.

## Model And Reasoning Audit

After extracting artifacts:

```sh
python -m scripts.native_eval.research_audit \
  --run-index "$CAMPAIGN/manifests/run-index.json" \
  --extracted-root "$CAMPAIGN/extracted" \
  --output-dir "$CAMPAIGN/summaries/research"
```

The command writes:

- `trace_inventory.csv`: one row per task result
- `model_identity_audit.csv`: one strict identity summary per run
- `turn_usage.csv`: one row per normalized trace step
- `tool_calls.csv`: one row per tool call
- `research_audit.json`: counts and audit status

The tables retain r0 rows with `phase=r0` and
`leaderboard_eligible=false`. Aggregate score outputs must exclude them.

Identity passes only when every recovered task trace observes exactly the
requested model ID. Missing traces and missing observed identity fail the
audit; they are not treated as neutral.

Also audit proxy/provider logs:

- agent alias resolves only to the requested provider model ID
- judge alias resolves only to `gpt-5.6-sol`
- requested reasoning is present on each applicable request
- no fallback or retry changes the provider model ID
- request IDs can be joined to run, task, and turn where available

## Tokens, Tools, And Cost

Task-level token totals can be recovered when the harness emits them in
`agent_result` or ATIF `final_metrics`. Tool calls can be recovered per turn
from `steps[].tool_calls`.

Per-turn token and cost analysis has three evidence levels:

| Level | Requirement | Report as |
|---|---|---|
| Exact | per-request usage/spend in trace or proxy/provider log | `exact_*` |
| Estimated | tokens plus pinned provider ID and dated pricing snapshot | `estimated` |
| Missing | neither exact spend nor sufficient pricing evidence | `unavailable_*` |

Never call a pricing reconstruction exact. Pin the pricing snapshot date,
currency, input/cache/output/reasoning rates, and source beside any estimate.

For exact research accounting, preserve a request-level spend record containing:

- request ID
- run label and task name
- harness and provider model ID
- requested reasoning
- input, cached input, output, and reasoning tokens
- provider-reported cost
- started and finished timestamps
- agent versus judge role

Per-tool monetary cost is not an LLM trace field. Report tool count, duration,
and failures from traces; calculate tool infrastructure cost only from separate
runtime, API billing, or machine accounting evidence.

## Failure Policy

Classify:

- `infra`: environment, Docker, setup, gateway, or provider connectivity
- `agent_exit`: nonzero agent exit not caused by infrastructure
- `verifier_missing_reward`: missing reward file
- `clean_fail`: zero reward without exception
- `partial`: reward between zero and one
- `pass`: reward at least one

Preserve every failure. Infra-dominated or identity-invalid runs are excluded,
not discarded. Rerun the same repetition under a suffixed label after fixing
the cause.

## Deliverables

Local campaign layout:

```text
runs-full-YYYYMMDD/
  raw/
  extracted/
  logs/
  manifests/
    campaign_manifest.json
    run_index.json
    upload-index.json
  summaries/
    aggregate_results.csv
    aggregate_results.json
    per_task_results.csv
    infra_failures.csv
    cleaned_leaderboard.md
    research/
      trace_inventory.csv
      model_identity_audit.csv
      turn_usage.csv
      tool_calls.csv
      research_audit.json
```

The audit note must state:

- retained r0 qualification status by harness and model family
- clean, excluded, missing, and rerun-required repetitions
- full task coverage against the pinned task commit
- requested and observed model IDs
- requested and proven reasoning levels
- requested and installed harness versions
- judge identity evidence
- exact, estimated, and unavailable usage/cost coverage
- local and S3 artifact counts plus checksum verification status

Do not call the comparison fair unless coverage is full, identity is proven,
the same task commit and judge contract were used, and infrastructure failures
are low.
