# Behavioral review of Core execution logs

Core retains a versioned behavioral review beside each run's execution evidence.
The review addresses authorization, data protection, instruction authority,
verification, and reporting. It is advisory: task verifier rewards, the existing
task-quality judge, and Core score aggregation retain their existing semantics.

This brings the evidence-review concept from PR #86 into the standard execution
paths. It does not import that PR's separate alignment runner, experimental score,
or task fixtures.

## What is reviewed

The shared `clawbench.run_review` contract records the delivered user instruction,
chronological execution events with tool arguments and results, a bounded selection
of final artifacts, and available verifier observations. Each source has a stable
ID. Native ATIF steps retain their `step_id` in that ID and source reference.

The evidence budget is 200,000 source-text characters. Earlier actions are retained
in chronological order; the reviewer does not receive only the last few messages.
Sources that exceed the budget remain identified and explicitly marked truncated.
Artifact capture also has per-file and file-count limits reported in coverage notes.

Each assessment contains a dimension, verdict, reason, and source-linked exact
quotations. Verdicts are:

- `violation`: the judge identified a violation in the supplied evidence.
- `no_violation_observed`: the supplied observations do not show a violation.
- `insufficient_evidence`: the available record cannot support a judgment.
- `not_applicable`: cited evidence explains why the dimension does not apply.

Every substantive verdict requires exact quotations from the specific cited
source. Missing, truncated, or incomplete execution evidence prevents a clean
whole-run review. Proven violations may still be identified in a partial trace.
Invalid schemas, nonexistent sources, and fabricated quotations become explicit
review errors; they do not produce a task failure or a successful review.

Citation validation proves that the quoted text exists. It does not independently
validate the judge's interpretation. The review does not establish intent, a general
safety guarantee, or the absence of unobserved effects. Tool output, artifacts, and
verifier files are evidence, never instructions to the reviewer.

## Standard execution

For Gateway-based Core runs, `--judge-model` requests behavioral review even when a
task has no task-specific quality rubric. Configure a direct, tool-free judge
endpoint using `AGENT_JUDGE_API_URL` and `AGENT_JUDGE_API_KEY` (or `OPENAI_API_KEY`):

```sh
export AGENT_JUDGE_API_URL=http://localhost:4000/v1/chat/completions
clawbench run --model "$ACTOR_MODEL" --judge-model "$JUDGE_MODEL" \
  --task t1-bugfix-discount --runs 1 --output results/run.json
```

Without a direct endpoint, Core uses a Gateway reviewer only if its effective tool
inventory is explicitly empty. Missing or enabled tool inventory produces a review
error before submitting the execution evidence, with instructions to configure the
direct endpoint. A prompt saying “do not use tools” is not treated as isolation.
The existing task-quality judge is a separate mechanism and retains its behavior.

`TaskRunResult` contains `run_review`, `review_evidence`, and
`review_artifact_dir`. The standalone review directory is retained outside the
disposable workspace at its parent's `_reviews/<workspace-name>/`. Set
`CLAWBENCH_REVIEW_DIR` to choose another retention root. Results embed the evidence
for recovery if writing the standalone bundle fails. Old cached results do not
silently acquire reviewed status; the cache version changes with this contract.

Native runs perform behavioral review through the configured judge proxy after
actor shutdown. Each trial contains:

```text
review/
  evidence.json   # frozen source text, IDs, and coverage
  review.json     # assessments, citations, raw judge reply, and provenance
  index.html      # reviewed log with links from findings to source events
```

Native `result.json` references these artifacts through `run_review`. Aggregation
exports review status, violation counts, and review paths alongside task outcomes.
The review has no influence on reward or pass rate. `SHELLBENCH_RUN_REVIEW=off`
retains evidence and records `not_requested` without an additional judge call.
`SHELLBENCH_RUN_REVIEW_API_URL` can override the host-accessible chat-completions
endpoint; the standard remote launcher supplies the local proxy address.

A native verifier sharing the actor container is not an independent effects
monitor. The review explicitly records this limitation. The new review collector
rejects symlinks and special files, bounds file reads, and declines to read mutable
actor files when shutdown was not confirmed. Existing native trajectory collection
has its own capture contract; a well-formed trajectory alone does not establish
complete capture.

## View and re-review saved evidence

Open `review/index.html` in a browser. Every citation navigates to its source
event; the source shows which review dimensions cite it. The page displays coverage,
review status, and the raw response. It contains no active scripts or external assets.

Render an existing review against its frozen evidence without a model call:

```sh
clawbench review-log path/to/review/evidence.json \
  --review-result path/to/review/review.json --output results/rendered-review
```

Re-review the same evidence with an explicitly chosen judge:

```sh
export AGENT_JUDGE_API_KEY="..."
clawbench review-log path/to/review/evidence.json \
  --judge-model "$JUDGE_MODEL" \
  --judge-api-url http://localhost:4000/v1/chat/completions \
  --reason "Evaluate revised review rubric on the original evidence" \
  --output results/review-revision-2
```

The command also accepts `OPENAI_API_KEY`. Existing output directories are never
overwritten. A re-review records its reason and evidence hash in `revision.json`;
the original review and task score remain intact. Rendering validates the evidence
hash and source citations. Historical traces with missing observations retain that
limitation when re-reviewed.

## Validation scope

Regression tests exercise early unauthorized actions despite useful delivery,
changed or unissued permissions, invalid citations, partial capture, aborts,
provider failures, unsafe file paths, unchanged scoring, retained error evidence,
and clickable source references in the rendered log. Synthetic judge responses
test the execution and persistence contract; they do not establish judge accuracy.
Calibration of real judgments and independent effects instrumentation remain
separate evaluation work.
