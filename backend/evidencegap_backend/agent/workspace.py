from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Mapping

from evidencegap_backend.agent.schemas import (
    AgentAction,
    ClaimWorkspace,
    EvidenceWorkspace,
    SearchAttempt,
)
from evidencegap_backend.common import EvidenceGapError


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_query(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def initialize_workspace(
    *,
    run_name: str,
    statement: str,
    language: str,
    decomposition: Mapping[str, Any],
    max_steps: int,
    total_search_budget: int,
    per_claim_search_budget: int,
    max_gap_rounds: int = 2,
    gap_remediation_budget: int = 2,
) -> EvidenceWorkspace:
    claims: dict[str, ClaimWorkspace] = {}
    order: list[str] = []
    for raw in decomposition.get("claims", []):
        claim_id = str(raw["claim_id"])
        order.append(claim_id)
        claims[claim_id] = ClaimWorkspace(
            claim_id=claim_id,
            source_text=str(raw["source_text"]),
            source_spans=[dict(x) for x in raw.get("source_spans", [])],
            canonical_claim_en=str(raw["canonical_claim_en"]),
        )
    return EvidenceWorkspace(
        run_name=run_name,
        statement=statement,
        language=language,
        decomposition=dict(decomposition),
        claims=claims,
        claim_order=order,
        max_steps=max_steps,
        initial_search_budget=total_search_budget,
        remaining_search_budget=total_search_budget,
        per_claim_search_budget=per_claim_search_budget,
        max_gap_rounds=max_gap_rounds,
        initial_gap_remediation_budget=gap_remediation_budget,
        remaining_gap_remediation_budget=gap_remediation_budget,
        started_at=utc_now(),
    )


def successful_attempts(claim: ClaimWorkspace) -> list[SearchAttempt]:
    return [attempt for attempt in claim.attempts if attempt.status == "successful"]


def choose_attempt(
    claim: ClaimWorkspace, requested: str | None = None
) -> SearchAttempt:
    attempts = successful_attempts(claim)
    if requested:
        selected = [a for a in attempts if a.attempt_id == requested]
        if not selected:
            raise EvidenceGapError("selected_attempt_id is not a successful attempt")
        return selected[0]
    if not attempts:
        raise EvidenceGapError("claim has no successful search attempt")
    # Higher evidence utility wins; stable earlier attempt breaks ties.
    return max(
        attempts,
        key=lambda a: (
            a.utility_score,
            a.direct_evidence_articles,
            len(a.new_article_ids),
            -attempts.index(a),
        ),
    )


def register_attempt(workspace: EvidenceWorkspace, attempt: SearchAttempt) -> None:
    claim = workspace.claims[attempt.claim_id]
    claim.used_queries.append(attempt.query)
    claim.normalized_queries.append(attempt.normalized_query)
    claim.attempts.append(attempt)
    if attempt.status == "successful":
        claim.seen_article_ids = sorted(
            set(claim.seen_article_ids) | set(attempt.article_ids)
        )
        best = choose_attempt(claim)
        claim.selected_attempt_id = best.attempt_id
        claim.verdict = best.verdict
        claim.last_error = None
    else:
        claim.last_error = attempt.error


def terminal_count(workspace: EvidenceWorkspace) -> int:
    return sum(c.status != "pending" for c in workspace.claims.values())


def compact_summary(workspace: EvidenceWorkspace) -> dict[str, Any]:
    claims = []
    for claim_id in workspace.claim_order:
        claim = workspace.claims[claim_id]
        latest = claim.attempts[-1] if claim.attempts else None
        claims.append(
            {
                "claim_id": claim_id,
                "canonical_claim": claim.canonical_claim_en,
                "status": claim.status,
                "used_queries": claim.used_queries,
                "latest_attempt": None
                if latest is None
                else {
                    "attempt_id": latest.attempt_id,
                    "status": latest.status,
                    "verdict": latest.verdict,
                    "article_counts": latest.article_counts,
                    "new_article_count": len(latest.new_article_ids),
                    "top_article_titles": latest.top_article_titles[:3],
                },
                "remaining_problem": claim.remaining_problem,
                "remaining_claim_budget": max(
                    0, workspace.per_claim_search_budget - len(claim.attempts)
                ),
            }
        )
    return {
        "statement": workspace.statement,
        "claims": claims,
        "step_count": workspace.step_count,
        "max_steps": workspace.max_steps,
        "remaining_search_budget": workspace.remaining_search_budget,
        "evidence_cycle": workspace.evidence_cycle,
        "gap_round": workspace.gap_round,
    }


def increment_action(workspace: EvidenceWorkspace, action: AgentAction) -> None:
    workspace.step_count += 1
    workspace.action_counts[action.value] = (
        workspace.action_counts.get(action.value, 0) + 1
    )
