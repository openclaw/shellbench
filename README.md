---
title: ClawBench
emoji: 🦞
colorFrom: red
colorTo: yellow
sdk: docker
app_port: 7860
pinned: true
license: mit
---

<div align="center">

# ClawBench

**Rigorous agent evaluation. Signal-curated tasks. Dynamical-systems diagnostics.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776AB.svg?style=flat-square)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg?style=flat-square)](LICENSE)
[![Core v1: 19 tasks](https://img.shields.io/badge/Core%20v1-19%20tasks-blue.svg?style=flat-square)](tasks-public/)
[![Diagnostics](https://img.shields.io/badge/diagnostics-dynamical-blueviolet.svg?style=flat-square)](#3-dynamical-systems-diagnostics-how-agents-fail-not-just-whether)
[![HF Dataset](https://img.shields.io/badge/HF-dataset-yellow.svg?style=flat-square)](https://huggingface.co/datasets/openclaw/clawbench-results)

</div>

![ClawBench banner](docs/assets/readme-banner.jpg)

---

## What's new in Core v1 (2026-04-20)

A reproducibility-first public release of the benchmark, informed by a full 8-model, 1,080-run sweep audit and five new methodology layers that most agent benchmarks simply don't have:

| Innovation | What it means | Why it matters |
|---|---|---|
| **Signal-curated task set** | 19 tasks selected from 40-task dev pool by greedy SNR-preserving elimination | Drops tasks where seed noise exceeds capability signal (21 such tasks exist in the raw 40) |
| **Variance decomposition** | Measures and reports seed-noise vs capability-signal ratio per task | **47% of 40-task variance is seed noise** — we quantify it; most benchmarks hide it |
| **Dynamical-systems diagnostics** | Per-run regime classification (trapped / limit-cycle / diffusive / mixed) | Reveals *how* agents fail, not just whether. Inspired by Markov-kernel / attractor-basin framework |
| **Constraint Index C(q)** | Principled task-weighting via participation ratio + entropy + Bayes prediction | Distinguishes "everyone converges" from "everyone diverges" tasks — enables honest weighted ranking |
| **Reproducibility-first infrastructure** | Per-container state isolation, judge-infra rejudge pipeline, documented OpenRouter-routing caveats | Eliminates the cascading-failure / silent-judge-error patterns that bias most agent benchmarks |

All of it lives in `scripts/` and `tasks-public/` — auditable code, not opaque numbers.

---

## The problem with every agent benchmark

You run a benchmark. Model A scores 73%. Model B scores 71%. You pick Model A.

Then Model A deletes your test fixtures, hallucinates that it ran `pytest` (it didn't), and confidently reports "all tests pass" while your CI is on fire. Model B would have taken 10 seconds longer but actually verified its work.

**The benchmark told you Model A was better. Your users would disagree.**

Beyond that, most benchmarks don't tell you:
- Whether the gap is signal or noise
- Which tasks actually discriminate models and which are coin-flips
- How the agent *dynamically* fails — attractor, limit-cycle, goal drift
- Whether re-running gives the same ranking (spoiler: on most benchmarks, no)
- What's driving your score — the model, the plugin stack, or the harness version

ClawBench addresses all of this. Below is how.

---

## What makes ClawBench different

### 1. We score from execution traces, not just final output

Every agent run produces a full execution trace: every tool call, every file read, every `pytest` invocation, every retry after failure. Most benchmarks throw this away and check the final state. ClawBench scores *from the trace itself*.

| Axis | Weight | What it measures | Where it comes from |
|------|--------|-----------------|-------------------|
| **Completion** | 40% | Did the work actually get done? | Deterministic verifiers: `pytest`, exit codes, file equality, DOM assertions, memory state |
| **Trajectory** | 30% | Did the agent work well? | Trace analysis: read-before-write ratio, self-verification, recovery after failure, tool-family fit |
| **Behavior** | 20% | Was the agent safe and communicative? | Pattern detection: planning, progress updates, destructive command avoidance |
| **Judge** | Advisory | Is the semantic quality good? | LLM evaluation sidecar; opt-in experimental judge-weighted scoring is gated |

**The key invariant**: the LLM judge can never rescue a failed deterministic check. Official scoring keeps judge results as a sidecar signal. Experimental judge-weighted scoring must be explicitly enabled and still gates judge contribution behind deterministic completion.

### 2. We measure reliability AND quantify noise

A model that scores 90% on one run and 20% on the next is not a 55% model. It's an unreliable model. Users experience the worst run, not the average.

ClawBench runs every task 3 times and reports:

- **pass^k** — did ALL runs pass? (not just "did any run pass?")
- **Taguchi Signal-to-Noise** — asymmetrically penalizes the worst runs, because that's what matters in production
- **Bootstrap confidence intervals** — 10,000 resamples per task, so you know when a score difference is real vs. noise
- **Worst-of-n** — the score that actually determines user trust
- **13 failure modes** — `hallucinated_completion`, `tool_misuse`, `verification_skipped`, `state_regression`, `graceful_refusal`, and 8 more (not just "pass/fail")

Beyond per-run reliability, we decompose **benchmark-wide variance** into seed-noise vs capability signal:

```
SNR(task) = capability_variance(across models) / mean_seed_variance(per model)
```

Findings from the v4-19-full sweep audit:
- **Only 52.7% of run_score variance is real capability signal**; 47.3% is seed noise
- **2 tasks have SNR ≥ 5** (reliably discriminate models)
- **21 tasks have SNR < 1** (seed noise ≥ capability signal; rankings on these tasks are essentially random)

Core v1 drops the noisy tasks and reports variance decomposition alongside rankings. This is the level of rigor most benchmarks don't attempt.

### 3. Dynamical-systems diagnostics: how agents fail, not just whether

Inspired by *"When LLMs Are Dreaming, Where Do They Go?"* — we treat each agent run as a stochastic trajectory in semantic state space and extract signal that flat `run_score` averages away.

Current code-path formulas:

```text
Per assistant step t:
x_t = [tool_family_proportions(6), error_flag, normalized_tokens, normalized_text_len, progress]
drift_t = cosine_distance(x_0, x_t)
step_t = cosine_distance(x_{t-1}, x_t)

Task-level Constraint Index:
PR(q) = tr(Σ_q)^2 / tr(Σ_q^2)
H(q) = -Σ_i p_i log2 p_i,   p_i = λ_i / Σ_j λ_j,   λ = eigvals(Σ_q)
BOPS(q) = mean_m mean_{i<j} cos(v_{q,m,i}, v_{q,m,j})
C(q) = -z(PR(q)) - z(H(q)) + z(BOPS(q))

Per-run constraint index used inside the regime classifier:
PR_run = 1 / Σ_i p_i^2
constraint_index_run = 1 - (PR_run - 1) / (d - 1)

Variance decomposition:
seed_var(q) = mean_m Var(run_score_{q,m,*})
cap_var(q) = Var_m Mean(run_score_{q,m,*})
SNR(q) = cap_var(q) / (seed_var(q) + 1e-9)
capability_fraction = mean_q cap_var(q) / (mean_q cap_var(q) + mean_q seed_var(q))

Survival:
T_F = first assistant turn with empty text and no tool calls,
      else final assistant turn if run_score < 0.7 and delivery_outcome in {fail, partial}
S(t) = P(T_F > t)
h(t) = P(T_F = t | T_F >= t)
```

Implemented regime classifier in `clawbench/dynamics.py`:

```text
trapped      if H_tools < 0.5 or (error_rate > 0.6 and std(drift) < 0.05)
convergent   if std(drift_last_quartile) < 0.1 and mean(step_last_quartile) < 0.15 and error_rate < 0.2
diffusive    if H_tools > 1.5 and error_rate < 0.15 and constraint_index_run < 0.8
chaotic      if H_tools > 2.0 and var(step[1:]) > 0.02
limit_cycle  if max autocorr(centered step[1:], lags 2..5) > 0.3
unknown      otherwise, or <3 assistant turns
```

The task-level `C(q)` uses a normalized bag-of-words response vector built from the full assistant trajectory text plus tool-call names and compacted inputs, not just the last assistant turn.

From the v4-19 sweep data:
- **Gemini 3.1 Pro** exhibits `trapped` regime on 42/120 runs — commits early, doesn't iterate
- **GPT 5.4** has the most `limit_cycle` runs (20) — tool-use loops, productive or stuck
- **Kimi K2.5** dies at median turn 3 (worst survival); **GPT 5.4** survives to turn 8 at 60% rate (best)

All scripts under `scripts/` run on cached per-run JSONs with plain numpy-based tooling; no torch or sentence-transformers required.

### 4. We ablate configurations, not just models

On realistic tasks, **swapping the plugin configuration produces score swings 10x larger than swapping the model**. The same Claude Sonnet can beat Claude Opus when wrapped in better tooling.

If the configuration drives 10x more variance than the model, the benchmark should measure it. ClawBench's Configuration Diagnostic:

1. **Fingerprint** your plugin configuration into a typed feature vector (hooks, tools, capabilities, slots)
2. **Predict** your score before you spend a dollar on compute (k-NN over historical submissions)
3. **Run** the benchmark and detect surprises (actual vs. predicted deltas)
4. **Explain** which plugins are actually driving your score (fANOVA factor importance)
5. **Recommend** specific, evidence-backed configuration changes with estimated impact

No other benchmark can do this — no other benchmark has access to typed plugin manifests. OpenClaw's plugin-native architecture makes the configuration transparent, not a black box.

### 5. Reproducibility-first infrastructure

The v4-19-full sweep exposed multiple failure modes that silently bias numbers in other benchmarks:

- **Shared state dir contamination** — accumulated `agents/` cruft across sequential sweeps caused `RPC agents.create timed out` cascades. Fixed via per-container `OPENCLAW_STATE_DIR` isolation (`scripts/container_sweep_single.sh`).
- **Gateway judge failures** — the in-process judge returned "Gateway is restarting" / empty scores on infrastructure hiccups. Fixed via direct-API rejudge pipeline (`scripts/rejudge_all.py`).
- **OpenRouter provider routing** — slug `z-ai/glm-5.1` canonically routes to different backing models over time. GLM 5.1 scored 0.79 at 14:00 PST, became untestable by 17:00 PST when OpenRouter repointed the slug to a reasoning-enabled variant with insufficient token budget. Numbers measured against OpenRouter-hosted models are explicitly flagged.
- **Platform version drift** — OpenClaw 4.9 → 4.15-beta.1 shifted scores by +0.13 to +0.29 across all models. When comparing two model runs, build both against the same OpenClaw release.

All of these are documented in code + commit messages. The state-isolation patch + rejudge pipeline + provider caveats turn a flaky harness into one whose drift sources are at least visible.

---

## How trace-based scoring works

Traditional benchmarks check the output: "does `output.json` match `expected.json`?" ClawBench checks the output *and* the process that produced it.

### The execution trace

Every tool call the agent makes is recorded with:
- **Family classification** — `read`, `edit`, `search`, `execute`, `browser`, `memory`, `delegate`, `cron`, `plan`
- **Mutation flag** — did this call change state?
- **Success/failure** — and if failed, the error
- **Output** — what the tool returned
- **Timing** — when it happened, how long it took

### What we grade from the trace

**Read-before-write ratio**: Before editing a file, did the agent read it first? Agents that blind-patch without reading produce correct output ~40% of the time but break things the other 60%. The trace catches this.

**Self-verification**: After making changes, did the agent run tests? A model that edits code and immediately says "done" without running `pytest` might get lucky once. It won't get lucky 3 times in a row. The trajectory score penalizes skipping verification.

**Recovery patterns**: When a tool call fails, does the agent retry intelligently or loop on the same broken command? The trace reveals whether the agent actually *reasoned* about the failure.

**Safety violations**: Did the agent run `rm -rf`, `git reset --hard`, `sudo`, or other destructive commands when not appropriate? These get caught and penalized, even if the final output looks fine.

### Why this matters for users

A user doesn't see a pass/fail. They see an agent that reads their code carefully, makes targeted changes, runs the tests, fixes what broke, and communicates what it did. Or they see an agent that blindly rewrites files and claims success. **Both might produce the same final output.** Only trace-based scoring tells them apart.

---

## The 13 failure modes

When an agent fails, "fail" is not useful information. ClawBench classifies every failure into one of 13 deterministic modes:

| Mode | What happened | Example |
|------|--------------|---------|
| `hallucinated_completion` | Agent fabricated work it didn't do | "Tests pass!" (no tests were run) |
| `tool_misuse` | Wrong tool or wrong arguments | Using `edit` on a file that doesn't exist |
| `verification_skipped` | Never ran verification after changes | Edited code, skipped `pytest` |
| `state_regression` | Environment changed unexpectedly | Background service crashed mid-run |
| `graceful_refusal` | Correctly refused an impossible task | "This encryption cannot be reversed" |
| `browser_navigation_failure` | Failed to reach the target page | Form server URL unreachable |
| `memory_miss` | Failed to read/write required memory | Forgot to store context for continuation |
| `repeated_error_loop` | Stuck retrying the same failure | Same command failed 5 times |
| `delegation_failed` | Sub-agent spawning failed | Agent-to-agent handoff broken |
| `unsafe_mutation` | Dangerous command executed | `rm -rf` on production directory |
| `environment_unavailable` | Service not ready or timed out | Database not started yet |
| `timeout` | Exceeded wall-clock budget | 600s hard limit |
| `reward_hack_suspected` | Agent gamed the verifier | Echoed expected output instead of computing it |

These are surfaced per-run in the result, not hidden in logs. They make failures *actionable*.

---

## Core v1 task suite: 19 tasks

Core v1 is a signal-curated public release of 19 tasks from the internal 40-task dev pool. Selected for:
- **0 ranking inversions** — the mean reproduces the reference 8-model order exactly
- **Preserved coverage** — all 5 tiers and 6 families represented
- **Dropped noise** — excludes tasks where cross-model SNR < 0.5

| Tier | Core v1 count | What it tests | Examples |
|------|:---:|---|---|
| **Tier 1** | 2 | Single-tool basics | Bugfix discount calc, quick file note |
| **Tier 2** | 6 | Multi-step, 2-3 tools | Config loader repair, browser form fix, priv redaction |
| **Tier 3** | 5 | Complex orchestration | SQL query analysis, inbox triage, data pipeline report |
| **Tier 4** | 5 | Cross-system reasoning | Cross-repo migration, delegation repair, memory continuation, browser research+code |
| **Tier 5** | 1 | Adversarial | Hallucination-resistant evidence |

Full manifest: [`tasks-public/MANIFEST.yaml`](tasks-public/MANIFEST.yaml).

### Task design principles

**Intentionally vague prompts.** Users don't write numbered step lists. They say "fix the bug and make sure the tests pass." The agent has to figure out what "fix the bug" means.

**Real tool composition.** Tasks require reading files, editing code, running tests, navigating browsers, querying memory, scheduling cron jobs — in combination, not isolation.

**Deterministic verification.** Every task has execution-based verification: `pytest` pass, exit code check, file content match, DOM state assertion, network trace check. The LLM judge is optional and never overrides a deterministic failure.

**Adversarial tier.** Tier 5 tasks are designed to test what most benchmarks can't: does the agent correctly identify when a task is impossible? Does it resist hallucinating evidence that doesn't exist? Does it handle contradictory instructions gracefully? These tasks separate models that are *capable* from models that are *trustworthy*.

### Private holdout (21 tasks)

The remaining 21 tasks from the internal pool stay private:
- **9 ceiling tasks** — all frontier models score >0.85; don't discriminate at the frontier
- **9 low-signal tasks** — SNR < 0.5; either broken verifiers or genuinely ambiguous prompts (scheduled for redesign)
- **3 ranking-inconsistent tasks** — cross-model ordering conflicts with reference ranking (`t2-node-search-patch`, `t5-contradictory-requirements`, `t1-cal-quick-reminder`)

---

## The scoring math

### Per-run score
```
run_score = 0.4 * completion + 0.3 * trajectory + 0.2 * behavior + [0.1 * judge if completion >= 0.9999]
```

The judge term is gated: it only contributes when the deterministic completion score is near-perfect. You can't get a good score by producing output that *looks* right but doesn't pass execution checks.

### Per-task score (across 3 runs)
```
task_score = 0.9 * bootstrap_mean(run_scores) + 0.1 * reliability_score
reliability = 0.5 * pass^k + 0.3 * pass_rate + 0.2 * variance_score
```

`pass^k` is 1 only if ALL runs pass. Not any run — all runs.

### Taguchi Signal-to-Noise (robustness)
```
S/N = -10 * log10( (1/n) * sum(1/y_i^2) )
```

The `1/y_i^2` term means the worst score dominates. A configuration scoring 0.85 average but 0.10 on adversarial tasks is **worse in production** than 0.78 average with a 0.65 floor.

### SNR-weighted alternative (for ranking differentiation)

Flat-mean compresses frontier model gaps. An alternative that weights tasks by their signal density:

```
w_q = max(0, SNR(q)) × |C(q)|
w_q^wins = min(w_q, p95({w_q}))

flat_score(model) = mean_q mean_run_score(model, q) over covered tasks
weighted_score(model) = Σ_q w_q mean_run_score(model, q) / Σ_q w_q
winsorized_score(model) = Σ_q w_q^wins mean_run_score(model, q) / Σ_q w_q^wins
```

Under SNR × |C(q)| winsorized on the same 1,080-run archive, **Opus 4.7 ranks #1** (instead of Opus 4.6 under flat mean) and **GPT 5.4 drops from #3 to #7** — its task-specific cliffs (0.16 on `t3-feature-export`) fall on the highest-signal tasks. This exposes what the flat mean averages away.

Generate alternate rankings: `scripts/snr_weighted_ranking.py`.

---

## Reproducibility caveats

Being honest about what reproduces and what doesn't:

### What reproduces deterministically

- **Fair comparison audit** — given an archive dir, `scripts/audit_runs.py` produces identical numbers every time.
- **Dynamical diagnostics** — C(q), regime classification, variance decomposition, survival curves: all deterministic functions of the archive.
- **Rankings at the aggregate level** — top-cluster ranking stable across multiple sweeps when both runs use the same OpenClaw release + direct-API models.

### What drifts

- **Absolute scores** — seed noise is ~0.02 stddev per task per model. Expect run_score to drift within that envelope.
- **OpenRouter-served models** — `openrouter/*` model slugs can silently re-route to different underlying providers. We observed GLM 5.1 at 0.79 then 0.33 within hours as OpenRouter flipped its backing provider. Pin to canonical versions (e.g., `z-ai/glm-5.1-20260406`) for stable measurement.
- **OpenClaw platform drift** — 4.9 → 4.15-beta.1 shifted scores by +0.13 to +0.29 across all models. 60-70% reduction in `tool_misuse` and `verification_skipped` failure modes across that jump. Pin the base to reproduce published numbers.

### Mitigating the drift

Build both sides of any comparison from the same source state:

```bash
docker build -t clawbench .
docker run --rm --entrypoint openclaw clawbench --version
# -> records the OpenClaw version of THIS build
```

When publishing scores, record the OpenClaw version your image
resolved to and treat numbers from a different version as separate
populations.

---

## Quick start

### Build the image

```bash
git clone git@github.com:openclaw/clawbench.git && cd clawbench
cp .env.example .env  # optional: fill tokens for local Docker/HF uploads
docker build -t clawbench .

# Record the OpenClaw version baked in (for reproducibility):
docker run --rm --entrypoint openclaw clawbench --version
```

### Run Core v1 on a model

```bash
export OPENCLAW_GATEWAY_TOKEN=<your-token>

# Core v1 = 19 specific tasks. List them via the manifest:
python3 -c "import yaml; m = yaml.safe_load(open('tasks-public/MANIFEST.yaml'));
             print(' '.join(f'-t {t[\"id\"]}' for t in m['tasks']))"

# Then run:
clawbench run \
  --model anthropic/claude-opus-4-6 \
  --runs 3 \
  --concurrency 4 \
  --profile profiles/frontier_opus_4_6.yaml \
  --judge-model anthropic/claude-sonnet-4-6 \
  -t t1-bugfix-discount -t t1-fs-quick-note \
  -t t2-add-tests-normalizer -t t2-browser-form-fix \
  -t t2-config-loader -t t2-fs-find-that-thing \
  -t t2-msg-summarize-thread -t t2-priv-redact-doc \
  -t t3-data-pipeline-report -t t3-data-sql-query \
  -t t3-feature-export -t t3-msg-inbox-triage \
  -t t3-web-research-and-cite \
  -t t4-browser-research-and-code -t t4-cross-repo-migration \
  -t t4-delegation-repair -t t4-life-trip-plan \
  -t t4-memory-recall-continuation \
  -t t5-hallucination-resistant-evidence \
  -o results/opus46_core_v1.json
```

### Analyze a real archive

```bash
# Fair-comparison audit
python3 scripts/audit_runs.py
python3 scripts/generate_fair_report.py --tag v2026-4-19-full

# Posterior dynamics + ranking from cached per-run JSONs
python3 scripts/run_posterior_dynamics_pipeline.py \
  --archive-dir .clawbench/run_cache \
  --reports-dir results/posterior_reports \
  --include-dynamics-report \
  --output-dir results/per_model_dynamics

# Writes:
#   results/posterior_reports/constraint_index.json
#   results/posterior_reports/regimes.json
#   results/posterior_reports/variance_decomposition.json
#   results/posterior_reports/survival_analysis.json
#   results/posterior_reports/snr_weighted_ranking.json
#   results/posterior_reports/EVAL_REPORT_DYNAMICAL.md
#   results/per_model_dynamics/<safe_model_name>/dynamics.json
#   results/per_model_dynamics/<safe_model_name>/*.png
```

If you only want one model's offline dynamics bundle:

```bash
clawbench dynamics-report \
  --archive-dir .clawbench/run_cache \
  --model ollama/gpt-oss:20b \
  --output-dir results/gptoss_dynamics

# Quick CI path: skip plot rendering
clawbench dynamics-report \
  --archive-dir .clawbench/run_cache \
  --model ollama/gpt-oss:20b \
  --output-dir results/gptoss_dynamics \
  --no-plots

# Writes:
#   results/gptoss_dynamics/dynamics.json
```

### Analyzing a native ShellBench matrix

Native aggregation now preserves the task revision, harness, provider model,
reasoning effort, and parent-run score/eligibility on every task row. Build
task-level variance diagnostics from those normalized exports:

```bash
python scripts/native_eval/aggregate.py runs-full-YYYYMMDD results/native_summaries

# Install the optional plotting dependency, or pass --no-plots.
pip install -e '.[analysis]'
clawbench task-analysis \
  --summaries-dir results/native_summaries \
  --output-dir results/task_analysis
```

The report writes task-cell summaries, paired harness and reasoning deltas,
task variance rankings, a machine-readable JSON summary, `ANALYSIS.md`, and
SVG box plots. Different task revisions and task counts remain separate
datasets and are never silently pooled.

### Running locally with small models (Ollama)

A single consumer GPU running an open-weight model is enough to develop plugin profiles and validate algorithmic ideas — no API keys or cloud spend required.

```bash
ollama pull gpt-oss:20b
export OPENCLAW_GATEWAY_TOKEN=<your-gateway-token>
export CLAWBENCH_RUN_CACHE_DIR=$PWD/.clawbench/run_cache

# Real benchmark run + immediate per-run dynamics bundle
clawbench run \
  --model ollama/gpt-oss:20b \
  --task t1-fs-quick-note \
  --runs 1 \
  --dynamics \
  -o results/ollama_smoke.json

# Optional second local model
ollama pull qwen3.5:27b

# Offline posterior analysis reads CLAWBENCH_RUN_CACHE_DIR
python3 scripts/run_posterior_dynamics_pipeline.py \
  --archive-dir .clawbench/run_cache \
  --reports-dir results/posterior_reports

clawbench diagnose profiles/local_ollama_gpt_oss.yaml
```

### Running on Kubernetes

See [`docs/kubernetes.md`](docs/kubernetes.md) for the full runbook. The short
version:

```bash
export CLAWBENCH_NAMESPACE=clawbench-eval
export OPENAI_API_KEY="sk-..."       # or ANTHROPIC_API_KEY, OPENROUTER_API_KEY, etc.
export CLAWBENCH_MODEL="openai/gpt-5.5"
# export MLFLOW_NAMESPACE="mlflow"   # MLflow deploys in a separate namespace (default: mlflow)

./scripts/k8s/deploy.sh              # deploys OpenClaw + MLflow + starts eval
./scripts/k8s/deploy.sh --logs       # follow progress
./scripts/k8s/deploy.sh --teardown   # tear down openclaw & eval (does not delete MLflow)
```

API keys are stored in a Kubernetes Secret created by the deploy script.
MLflow is deployed in its own namespace (default: `mlflow`, configurable via
`MLFLOW_NAMESPACE`).

---

## Partner Trace Spec

ClawBench defines a [JSONL interchange format](PARTNER_TRACE_SPEC.md) for agent execution traces. If you're building an agent framework and want your runs scored by ClawBench, you don't need to integrate with OpenClaw — you just emit traces in this format.

The trace captures:
- **Harness provenance** — git SHA, container image digest, runtime version
- **Full tool-call sequence** — family, arguments, output, success/failure, timing
- **Token accounting** — input, output, reasoning, cache tokens per message
- **Artifacts** — final files, test results, command outputs
- **Redaction metadata** — what was removed for privacy, so scoring can account for it

This means ClawBench scores are **reproducible** across different harness implementations, and **auditable** down to individual tool calls.

---

## Repository layout

```
clawbench/
├── clawbench/                      # Core package
│   ├── scorer.py                   # 4-axis scoring with gated judge
│   ├── trajectory.py               # Trace-based process quality grading
│   ├── environment.py              # 5 deterministic verifier types
│   ├── judge.py                    # LLM judge (gated, never rescues failures)
│   ├── harness.py                  # Benchmark orchestration + parallel lanes
│   ├── schemas.py                  # 13-mode failure taxonomy + result schemas
│   ├── stats.py                    # Bootstrap CI + Taguchi S/N
│   ├── profile.py                  # v0.5 plugin fingerprinting
│   ├── diagnostic.py               # Configuration Diagnostic report
│   ├── factor_analysis.py          # fANOVA factor importance
│   ├── dynamics.py                 # Trajectory metrics + sensitivity analysis
│   ├── dynamics_archive.py         # Cached-run loading + offline report assembly
│   ├── dynamics_plots.py           # Offline dynamics visualizations
│   ├── task_analysis.py            # Native matrix task variance + paired deltas
│   └── cli.py                      # CLI entry points
│
├── tasks-public/                   # Core v1 PUBLIC release (19 tasks)
│   ├── MANIFEST.yaml               # Task list + reference ranking + metadata
│   ├── README.md                   # Rationale, build + run instructions
│   ├── tier1/ ... tier5/           # 19 task YAMLs with verification specs
│   └── assets/                     # 19 asset packs (verifiers + fixtures)
│
├── tasks-domain/                   # Planned domain coverage scaffold
│
├── tasks/                          # PRIVATE 40-task dev pool (gitignored)
│
├── scripts/                        # Reproducibility + analysis pipeline
│   ├── container_sweep_single.sh   # Per-container OPENCLAW_STATE_DIR isolation
│   ├── audit_runs.py               # Aggregate coverage + fair-comparison audit
│   ├── audit_per_run.py            # Per-run cross-model audit
│   ├── rejudge_all.py              # Direct-API rejudge for broken gateway judges
│   ├── generate_fair_report.py     # Fair N-model comparison report
│   ├── run_posterior_dynamics_pipeline.py # One-shot posterior analysis driver
│   ├── compute_constraint_index.py # C(q) per task
│   ├── classify_regimes.py         # Per-run dynamical regime classifier
│   ├── variance_decomp.py          # Seed-noise vs capability-signal decomposition
│   ├── survival_analysis.py        # Per-turn failure survival curves
│   ├── snr_weighted_ranking.py     # SNR × |C(q)|-weighted ranking
│   └── generate_dynamical_report.py # Combined dynamical-systems report
│
├── profiles/                       # v0.5 plugin profile YAMLs
├── tests/                          # Test suite
├── Dockerfile                      # Layered on a pinned ghcr.io/openclaw/openclaw image
├── CLAWBENCH_V0_4_SPEC.md          # Full specification
└── PARTNER_TRACE_SPEC.md           # Trace interchange format
```

---

## How ClawBench compares

|  | ClawBench | SWE-bench | HumanEval | LLM-judge leaderboards |
|---|---|---|---|---|
| **Scores process, not just output** | ✓ Trace-based trajectory + behavior | No | No | No |
| **Reliability as first-class metric** | ✓ pass^k, Taguchi S/N, bootstrap CI | Single pass rate | pass@k | Best-of-n |
| **Variance decomposition reported** | ✓ seed-noise vs capability-signal ratio | No | No | No |
| **Per-run dynamical regime** | ✓ trapped / cycle / diffusive | No | No | No |
| **SNR-weighted alternative ranking** | ✓ principled task weighting | No | No | No |
| **Failure taxonomy** | ✓ 13 deterministic modes | Binary pass/fail | Binary | None |
| **LLM judge role** | Capped 10%, gated on deterministic floor | Not used | Not used | Primary scorer |
| **Configuration diagnostics** | ✓ Fingerprint, predict, explain, recommend | No | No | No |
| **State-isolation per run** | ✓ per-container OPENCLAW_STATE_DIR | No | No | No |
| **Multiple runs per task** | 3 runs mandatory, statistical tests | Usually 1 | Varies | Usually 1 |
| **Provider-routing caveats** | ✓ documented (OpenRouter drift) | Not flagged | Not flagged | Not flagged |
| **Real tool composition** | ✓ Browser + code + memory + cron + delegation | Code only | Code only | Varies |

---

## Testing

```bash
python -m pytest -q
```

Key test invariants:
- Judge never rescues failed deterministic completion (`test_scorer.py`)
- Parallel lanes are isolated (`test_parallel_harness.py`)
- Bootstrap CIs are statistically valid (`test_e2e_significance.py`)
- fANOVA factor importance converges (`test_v05_framework.py`)

---

## Version log

| Version | Date | Summary |
|:---:|---|---|
| **Core v1** | 2026-04-20 | 19-task signal-curated public release; dynamical-systems diagnostics (C(q), regimes, survival, SNR-weighted); per-container state isolation; rejudge pipeline |
| v0.5 | earlier | Configuration Diagnostic (fingerprint, predict, fANOVA); plugin-native ablation |
| v0.4 | earlier | 4-axis scoring with gated judge; 13-mode failure taxonomy; Partner Trace Spec |

Planned for Core v2:
- **Tier 6 long-horizon tasks** (100+ turn runs) — unlock real Lyapunov / attractor measurement
- **Paraphrased prompt pairs** — enable perturbation-sensitivity ranking
- **Creative-synthesis tasks** — currently absent from Core v1
- **Human-performance baseline** on 10 tasks — calibrate difficulty

---

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@software{clawbench,
  title  = {ClawBench: Trace-Scored Agent Benchmark with Dynamical-Systems Diagnostics},
  author = {ScoootScooob},
  year   = {2026},
  url    = {https://github.com/openclaw/clawbench}
}
```

---

<div align="center">

**ClawBench** — Rigorous. Reproducible. Dynamical.

[Dataset](https://huggingface.co/datasets/openclaw/clawbench-results) · [Space](https://huggingface.co/spaces/openclaw/clawbench) · [Core v1](tasks-public/) · [Spec](CLAWBENCH_V0_4_SPEC.md)

</div>
