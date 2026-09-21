"""Persist a behavioral review with a standalone, evidence-linked log viewer."""

from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

from clawbench.run_review import (
    ReviewEvidence,
    RunReview,
    evidence_sha256,
    review_evidence_incomplete,
)


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _anchor(source_id: str) -> str:
    return "evidence-" + hashlib.sha256(source_id.encode()).hexdigest()[:20]


def validate_review_evidence(evidence: ReviewEvidence, review: RunReview) -> None:
    """Reject stale evidence and citations before presenting a saved review."""
    if review.evidence_sha256 != evidence_sha256(evidence):
        raise ValueError("Review evidence digest does not match the saved evidence")
    sources = {source.id: source for source in evidence.sources}
    if len(sources) != len(evidence.sources):
        raise ValueError("Evidence source IDs must be unique")
    incomplete = review_evidence_incomplete(evidence)
    for assessment in review.assessments:
        if incomplete and assessment.verdict in {"no_violation_observed", "not_applicable"}:
            raise ValueError("Saved review claims clean coverage against incomplete evidence")
        if review.status == "completed" and assessment.verdict == "insufficient_evidence":
            raise ValueError("Completed review contains an unresolved assessment")
        if (
            review.status in {"error", "not_requested"}
            and assessment.verdict != "insufficient_evidence"
        ):
            raise ValueError("Unreviewed or failed review cannot establish judgments")
        for citation in assessment.citations:
            source = sources.get(citation.source_id)
            if source is None or not citation.quote.strip() or citation.quote not in source.text:
                raise ValueError("Review citation does not match its evidence source")


def render_review_html(evidence: ReviewEvidence, review: RunReview) -> str:
    """Escape all run/model content; citations navigate only to local source anchors."""
    validate_review_evidence(evidence, review)
    cited_by: dict[str, list[str]] = {}
    cards = []
    for assessment in review.assessments:
        citations = []
        for citation in assessment.citations:
            cited_by.setdefault(citation.source_id, []).append(assessment.dimension)
            citations.append(
                f'<blockquote><a href="#{_anchor(citation.source_id)}">'
                f"{_escape(citation.source_id)}</a><pre>{_escape(citation.quote)}</pre></blockquote>"
            )
        cards.append(
            f'<article class="assessment {_escape(assessment.verdict)}">'
            f"<h3>{_escape(assessment.dimension.replace('_', ' ').title())}</h3>"
            f"<strong>{_escape(assessment.verdict.replace('_', ' '))}</strong>"
            f"<p>{_escape(assessment.reason)}</p>{''.join(citations)}</article>"
        )

    sources = []
    for source in evidence.sources:
        labels = ", ".join(dict.fromkeys(cited_by.get(source.id, [])))
        references = (
            json.dumps(source.source_ref, ensure_ascii=False)
            if isinstance(source.source_ref, dict)
            else source.source_ref or ""
        )
        sources.append(
            f'<article class="source" id="{_anchor(source.id)}">'
            f"<h3>{_escape(source.id)} <small>{_escape(source.kind)}</small></h3>"
            f'<p class="muted">{_escape(references)}</p>'
            + (
                f'<p class="cited">Cited in {_escape(labels.replace("_", " "))}</p>'
                if labels
                else ""
            )
            + (
                '<p class="warning">Source truncated; omitted content was not reviewed.</p>'
                if source.truncated
                else ""
            )
            + f"<pre>{_escape(source.text)}</pre></article>"
        )
    coverage = evidence.coverage.model_dump()
    coverage_rows = "".join(
        f"<tr><th>{_escape(key.replace('_', ' '))}</th><td>{_escape(value)}</td></tr>"
        for key, value in coverage.items()
        if key != "notes"
    )
    notes = "".join(f"<li>{_escape(note)}</li>" for note in coverage.get("notes", []))
    error = f'<p class="warning">{_escape(review.error)}</p>' if review.error else ""
    violations = sum(item.verdict == "violation" for item in review.assessments)
    unknowns = sum(item.verdict == "insufficient_evidence" for item in review.assessments)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Run review — {_escape(evidence.task_id)}</title>
<style>
:root {{ color-scheme: light; font: 16px/1.55 system-ui, sans-serif; color: #182b31; background: #f5f4ef; }}
body {{ max-width: 1180px; margin: auto; padding: 32px; }}
header {{ border-bottom: 2px solid #182b31; padding-bottom: 24px; margin-bottom: 28px; }}
h1 {{ font-size: clamp(1.7rem, 4vw, 2.6rem); line-height: 1.15; margin: 8px 0 16px; }}
h2 {{ margin-top: 36px; }} h3 {{ margin: 0 0 8px; }}
a {{ color: #075c77; overflow-wrap: anywhere; }}
.eyebrow, small, .muted {{ color: #526971; }} .eyebrow {{ text-transform: uppercase; letter-spacing: .12em; font-size: .75rem; }}
.notice {{ padding: 14px 18px; background: #e8eef0; border-left: 3px solid #477581; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(300px,1fr)); gap: 16px; }}
.assessment, .source {{ background: #fff; border: 1px solid #d5dedf; padding: 20px; border-radius: 8px; min-width: 0; }}
.violation {{ border-top: 4px solid #b3422f; }}
.insufficient_evidence {{ border-top: 4px solid #a07822; }}
.no_violation_observed {{ border-top: 4px solid #477581; }}
blockquote {{ border-left: 3px solid #d5dedf; margin: 16px 0 0; padding-left: 12px; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; font: 13px/1.65 ui-monospace, monospace; }}
.source {{ margin-top: 16px; scroll-margin-top: 16px; }}
.source:target {{ outline: 3px solid #a07822; background: #fffdf3; }}
.cited {{ color: #075c77; font-size: .85rem; }} .warning {{ color: #923722; }}
table {{ border-collapse: collapse; width: 100%; background: white; }}
th, td {{ text-align: left; padding: 7px 12px; border-bottom: 1px solid #d5dedf; }}
th {{ width: 40%; font-weight: 500; }} details {{ margin-top: 20px; }}
@media(max-width:600px) {{ body {{ padding: 16px; }} .cards {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<header><div class="eyebrow">ShellBench · reviewed execution log</div>
<h1>{_escape(evidence.task_id)}</h1>
<p>Run {_escape(evidence.run_id)} · Review <strong>{_escape(review.status.replace("_", " "))}</strong><br>
{violations} dimensions with violations · {unknowns} unresolved<br>
Requested judge: {_escape(review.model or "not configured")} · Execution: {_escape(evidence.execution_status)}</p>
<p><a href="evidence.json">Evidence JSON</a> · <a href="review.json">Review JSON</a> · <a href="#sources">Source log</a></p>
</header>
<p class="notice">Behavioral review is advisory. Existing task rewards and scores are unchanged.
“No violation observed” applies only to the evidence shown; it does not prove the absence of side effects.
Citations establish where text appears, not that a judge's interpretation is correct.</p>
{error}<h2>Review findings</h2><section class="cards">{"".join(cards)}</section>
<h2>Evidence coverage</h2><table>{coverage_rows}</table><ul>{notes}</ul>
<details><summary>Review provenance</summary><pre>{_escape(json.dumps({"rubric_version": review.rubric_version, "evidence_sha256": review.evidence_sha256, "model": review.model}, indent=2))}</pre></details>
<h2 id="sources">Source log and observations</h2>{"".join(sources)}
<details><summary>Raw judge response</summary><pre>{_escape(review.raw_response or "No response recorded.")}</pre></details>
</body></html>"""


def persist_run_review(directory: Path, evidence: ReviewEvidence, review: RunReview) -> Path:
    """Create one immutable review bundle. Re-reviews must use a new directory."""
    page = render_review_html(evidence, review)
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    for name, content in (
        ("evidence.json", evidence.model_dump_json(indent=2)),
        ("review.json", review.model_dump_json(indent=2)),
        ("index.html", page),
    ):
        with (directory / name).open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.write("\n")
    return directory / "index.html"
