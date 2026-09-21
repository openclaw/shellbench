# Native research evidence tables

Run `python -m scripts.native_eval.research_audit` with the existing run index,
extracted archive root, and output directory. It reads artifacts only: it does
not execute agents, query providers, fetch referenced trajectories, or change
leaderboard eligibility. `research_audit.json` reports table schema version 2.

## Nested trajectories and joins

`turn_usage.csv` and `tool_calls.csv` include every unique embedded ATIF node.
`turn_index` remains zero-based **within its node**. Join rows by `run_label`,
`trajectory_path`, `node_ref`, and `turn_index`, not by task name and turn alone.
The original `step_id`, `trajectory_id`, `session_id`, session key, leaf identity,
parent node references, and parent tool-call IDs are retained when present.
`model_evidence_source` distinguishes a step annotation from a trajectory-level
default; neither proves the provider request ledger.

Additional tables:

| Table | Scope |
| --- | --- |
| `trajectory_nodes.csv` | One row per unique embedded node, with its original final metrics and declared metric scope |
| `trajectory_links.csv` | Parent/child and source tool-call references; external or missing references remain `not_embedded` |
| `model_usage_coverage.csv` | Reported metric sums and observed counts grouped by exact model annotation, including the provider prefix |

A trajectory ID is the primary identity. Session key, session ID, and leaf
identity detect conflicting aliases; distinct leaves or session generations
remain distinct. Identical repeated embedded nodes are counted once, with all
parent links retained. Conflicting duplicate identities make capture invalid.
Flat historical trajectories without identifiers receive a local JSON-path
`node_ref`; that is not a runtime session ID. External trajectory paths are
recorded but never followed.

The strict model audit inspects node defaults, step model changes, and legacy
observed-model metadata. It retains the historical provider-prefix normalization
for comparison with the requested model ID. Any observed different model fails.
An unobserved child, conflicting identity, invalid receipt, or incomplete
receipt node coverage cannot produce a `match`. Judge identity and requested
reasoning still need separate request evidence.

## Metric scopes

The existing task token columns prefer harness-reported values, then root
`final_metrics` **only when the harness value is missing**. Explicit zero never
falls through. `usage_sources_json` names the source for each component.
`root_metrics_json`, each node's `node_metrics_json`, and
`receipt_family_metrics_json` are separate evidence. Never add these overlapping
scopes. A root total is not silently promoted to a family total; receipt family
totals can include exported nodes that are not embedded in the trajectory.

Per-model rows aggregate only observed agent-step metrics, not final metrics.
They do not allocate root/family totals to a model or reconstruct provider
requests. Each input/cache/output/cost sum has an observed-step count and an
agent-step denominator. A partial sum remains partial; no observed values is an
empty CSV cell, while an observed zero is `0`. Local mirrors, copied context,
and incomplete traces mean step totals are not a billing or unique-request
accounting ledger.

Runtime costs are `reported_harness`, `reported_trace`, or
`reported_trace_turn`. They are not billing-exact. The deprecated
`exact_task_cost_count` and `task_cost_exact_count` remain available as zero;
use `reported_task_cost_count` and `task_cost_reported_count` for the number of
reported task costs. No price reconstruction or currency conversion is done.

## Independent evidence facets

`trace_inventory.csv` exposes independent statuses. There is deliberately no
single “research-clean” boolean that silently makes one facet stand in for
another.

| Facet | Evidence and limits |
| --- | --- |
| Capture | `complete`, `partial`, `invalid`, `unverified`, or `missing`; bounded to the selected public exported generations |
| Execution | Native `execution_outcome.kind=clean` without an exception is `accepted`; recorded errors are `rejected`; absence is `not_observed` |
| Identity | Existing `model_identity_status`; strict requested-versus-annotated model comparison, not reasoning/judge verification |
| Tokens | `observed_complete`, `partial`, or `not_observed` for input/cache/output on observed agent steps only; see `token_scope` |
| Cost | `reported_unreconciled` or `not_observed`; billing acceptance is unavailable without a joined provider ledger |
| Resources | `not_observed` until a supported resource artifact contract is validated; no host-wide or zero substitute |
| Reward | `observed` for a finite recorded verifier reward, including zero; otherwise `not_observed` |

A clean execution and zero reward remain visible even when capture is partial.
Complete capture does not establish command success, complete token reporting,
full session history, request-level billing, resources, or judge validity.

## Receipt integrity and coverage

An adjacent `receipt.json` is optional for historical traces. For
`openclaw-atif-receipt-v1`, the consumer hashes the **actual trajectory bytes**,
checks the root trajectory/session/key identity, and compares the receipt's
unique session/key/leaf set with the embedded node set. It does not reserialize
JSON before hashing. Missing or malformed receipts never certify capture.

Receipt-only and embedded-only identities are retained in separate JSON
columns. Missing/non-embedded nodes, unstable capture, producer diagnostics,
unresolved relationships, and opaque wrappers remain partial. A digest/root
mismatch, malformed identity set, or conflicting embedded identity is invalid.
Top-level and per-node diagnostics and receipt relationships are retained, so a
deleted source or unavailable generation is not disguised as an empty trace.
Receipt family metrics remain labeled receipt claims even on an invalid row;
consumers must check the integrity and coverage statuses before using them.

These are deterministic archive-consumer checks, not live delegated-lifecycle,
provider, resource-host, archive-restore, or campaign qualification.
