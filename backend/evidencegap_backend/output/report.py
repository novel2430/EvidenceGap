from __future__ import annotations

from typing import Any, Mapping, Sequence

from evidencegap_backend.common import EvidenceGapError


def _text(value: Any, fallback: str = "—") -> str:
    rendered = str(value or "").strip()
    return rendered if rendered else fallback


def _quote_block(value: Any) -> list[str]:
    text = _text(value)
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def _article_label(article: Mapping[str, Any]) -> str:
    title = _text(article.get("display_title") or article.get("title"), "Untitled")
    pmid = article.get("pmid")
    return f"{title} (PMID: {pmid})" if pmid else title


def render_markdown_report(
    presentation_bundle: Mapping[str, Any],
    *,
    run_id: str | None = None,
    execution_summary: Mapping[str, Any] | None = None,
) -> str:
    """Render a deterministic, source-preserving Markdown report."""

    required = {
        "statement",
        "claims",
        "inference_steps",
        "articles",
        "evidence",
        "summary",
        "analysis_context",
    }
    if not required.issubset(presentation_bundle):
        raise EvidenceGapError("Presentation bundle is incomplete for report export")
    claims = presentation_bundle["claims"]
    steps = presentation_bundle["inference_steps"]
    articles = presentation_bundle["articles"]
    evidence = presentation_bundle["evidence"]
    if not all(isinstance(value, Sequence) for value in (claims, steps, articles, evidence)):
        raise EvidenceGapError("Presentation report collections are invalid")

    lines = ["# EvidenceGap Analysis", ""]
    if run_id:
        lines.extend([f"**Run:** `{run_id}`", ""])
    lines.extend(
        [
            f"**Output language:** {_text(presentation_bundle.get('output_language'))}",
            "",
            "## Statement",
            "",
            *_quote_block(presentation_bundle["statement"].get("display_text")),
            "",
            "## Analysis Summary",
            "",
        ]
    )
    summary = presentation_bundle["summary"]
    states = summary.get("evidence_states", {})
    gaps = summary.get("gaps", {})
    lines.extend(
        [
            f"- Claims: {int(summary.get('total_claims', 0))}",
            "- Evidence states: "
            + ", ".join(
                f"{key} {int(value)}" for key, value in states.items()
            ),
            f"- Inference steps: {int(summary.get('total_inference_steps', 0))}",
            "- Gaps: "
            + ", ".join(f"{key} {int(value)}" for key, value in gaps.items()),
            f"- Retrieved articles evaluated: {int(summary.get('articles', 0))}",
            f"- Evidence sentences: {int(summary.get('evidence', 0))}",
            "",
            "## Claims",
            "",
        ]
    )

    articles_by_claim: dict[str, list[Mapping[str, Any]]] = {}
    for article in articles:
        if isinstance(article, Mapping):
            articles_by_claim.setdefault(str(article.get("claim_id") or ""), []).append(article)
    evidence_by_article: dict[str, list[Mapping[str, Any]]] = {}
    for item in evidence:
        if isinstance(item, Mapping):
            evidence_by_article.setdefault(
                str(item.get("article_node_id") or ""), []
            ).append(item)

    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, Mapping):
            continue
        claim_id = str(claim.get("claim_id") or "")
        lines.extend(
            [
                f"### {index}. {_text(claim.get('display_text') or claim.get('canonical_claim_en'))}",
                "",
                f"- Claim ID: `{claim_id}`",
                f"- Evidence state: **{_text(claim.get('evidence_state'))}**",
                f"- Argument role: {_text(claim.get('argument_role'))}",
                f"- Analysis status: {_text(claim.get('analysis_status'))}",
            ]
        )
        if claim.get("display_rationale"):
            lines.extend(["- Rationale:", *_quote_block(claim["display_rationale"])])
        claim_articles = articles_by_claim.get(claim_id, [])
        stance_headings = {
            "support": "Supporting Articles",
            "refute": "Refuting Articles",
            "insufficient": "Insufficient-Evidence Articles",
        }
        for stance in ("support", "refute", "insufficient"):
            selected = [
                article
                for article in claim_articles
                if str(article.get("stance") or "").casefold() == stance
            ]
            if not selected:
                continue
            lines.extend(["", f"#### {stance_headings[stance]}", ""])
            for article in selected:
                node_id = str(article.get("article_node_id") or "")
                lines.extend(
                    [
                        f"- **{_article_label(article)}**",
                        f"  - Rank: {int(article.get('rank', 0))}",
                        f"  - Confidence: {float(article.get('confidence', 0.0)):.4f}",
                        f"  - Rationale: {_text(article.get('display_rationale') or article.get('rationale'))}",
                    ]
                )
                for item in evidence_by_article.get(node_id, []):
                    lines.append(
                        "  - Evidence "
                        f"[{_text(item.get('section'), 'unknown section')}]: "
                        f"{_text(item.get('display_text') or item.get('text'))}"
                    )
        lines.append("")

    lines.extend(["## Inference Gaps", ""])
    any_gap = False
    for step in steps:
        if not isinstance(step, Mapping):
            continue
        for gap in step.get("gaps", []):
            if not isinstance(gap, Mapping):
                continue
            any_gap = True
            impact = step.get("impact") if isinstance(step.get("impact"), Mapping) else {}
            lines.extend(
                [
                    f"### {_text(gap.get('gap_type'))}",
                    "",
                    f"- Inference step: `{_text(step.get('inference_step_id'))}`",
                    f"- Premises: {', '.join(str(value) for value in step.get('premise_claim_ids', []))}",
                    f"- Conclusion: {_text(step.get('conclusion_claim_id'))}",
                    f"- Affects terminal conclusion: {bool(impact.get('affects_terminal_conclusion'))}",
                    f"- Downstream claims: {', '.join(str(value) for value in impact.get('downstream_claim_ids', [])) or 'None'}",
                    "- Reason:",
                    *_quote_block(gap.get("display_reason") or gap.get("reason_en")),
                    "",
                ]
            )
    if not any_gap:
        lines.extend(["No scope or causal gaps were detected.", ""])

    context = presentation_bundle["analysis_context"]
    lines.extend(
        [
            "## Methodological Boundary",
            "",
            f"- Scope: {_text(context.get('scope'))}",
            f"- Systematic review: {bool(context.get('is_systematic_review'))}",
            f"- Clinical recommendation: {bool(context.get('is_clinical_recommendation'))}",
            f"- Final medical truth: {bool(context.get('is_final_medical_truth'))}",
            f"- Aggregation method: {_text(context.get('aggregation_method'))}",
            f"- Retrieved article Top-K: {int(context.get('article_top_k', 0))}",
            "",
        ]
    )
    if execution_summary:
        lines.extend(["## Execution Summary", ""])
        lines.append(
            f"- Total seconds: {float(execution_summary.get('total_seconds', 0.0)):.6f}"
        )
        stages = execution_summary.get("stages")
        if isinstance(stages, Mapping):
            for name, value in stages.items():
                if isinstance(value, Mapping):
                    lines.append(
                        f"- {name}: {float(value.get('seconds', 0.0)):.6f} seconds"
                    )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
