from __future__ import annotations

from evidencegap_backend.agent.schemas import (
    AgentAction,
    AgentDecision,
    EvidenceWorkspace,
)
from evidencegap_backend.agent.workspace import (
    choose_attempt,
    normalize_query,
    successful_attempts,
)


class InvalidDecision(ValueError):
    pass


def validate_decision(
    workspace: EvidenceWorkspace, decision: AgentDecision
) -> AgentDecision:
    if decision.action is AgentAction.FINISH:
        if any(c.status == "pending" for c in workspace.claims.values()):
            raise InvalidDecision("FINISH requires every claim to be terminal")
        return decision
    claim = workspace.claims.get(str(decision.claim_id))
    if claim is None:
        raise InvalidDecision("claim_id does not exist")
    if claim.status != "pending":
        raise InvalidDecision("action requires a pending claim")
    if decision.action is AgentAction.SEARCH:
        if workspace.remaining_search_budget <= 0:
            raise InvalidDecision("total search budget is exhausted")
        if len(claim.attempts) >= workspace.per_claim_search_budget:
            raise InvalidDecision("per-claim search budget is exhausted")
        normalized = normalize_query(str(decision.query or ""))
        if not normalized:
            raise InvalidDecision("SEARCH query is blank")
        if normalized in claim.normalized_queries:
            raise InvalidDecision("SEARCH query duplicates a previous query")
    elif decision.action is AgentAction.RESOLVE:
        attempts = successful_attempts(claim)
        if not attempts:
            raise InvalidDecision("RESOLVE requires a successful attempt")
        if decision.selected_attempt_id and decision.selected_attempt_id not in {
            a.attempt_id for a in attempts
        }:
            raise InvalidDecision("selected_attempt_id is not a successful attempt")
    elif decision.selected_attempt_id:
        if decision.selected_attempt_id not in {
            attempt.attempt_id for attempt in successful_attempts(claim)
        }:
            raise InvalidDecision("selected_attempt_id is not a successful attempt")
    return decision


def _unused_query(workspace: EvidenceWorkspace, claim_id: str) -> str | None:
    claim = workspace.claims[claim_id]
    candidates = [
        claim.canonical_claim_en,
        f"{claim.canonical_claim_en} clinical trial evidence",
        f"{claim.canonical_claim_en} systematic review",
    ]
    for query in candidates:
        if normalize_query(query) not in claim.normalized_queries:
            return query
    return None


def deterministic_fallback(
    workspace: EvidenceWorkspace, *, reason: str
) -> AgentDecision:
    pending = [
        workspace.claims[cid]
        for cid in workspace.claim_order
        if workspace.claims[cid].status == "pending"
    ]
    if not pending:
        return AgentDecision(action="FINISH", reason="All claims are terminal.")
    claim = pending[0]
    attempts = successful_attempts(claim)
    if attempts:
        return AgentDecision(
            action=AgentAction.RESOLVE,
            claim_id=claim.claim_id,
            selected_attempt_id=choose_attempt(claim).attempt_id,
            reason="Select the best successful attempt deterministically.",
        )
    if (
        workspace.remaining_search_budget > 0
        and len(claim.attempts) < workspace.per_claim_search_budget
    ):
        query = _unused_query(workspace, claim.claim_id)
        if query:
            return AgentDecision(
                action="SEARCH", claim_id=claim.claim_id, query=query, reason=reason
            )
    return AgentDecision(
        action=AgentAction.ABSTAIN,
        claim_id=claim.claim_id,
        remaining_problem="No successful attempt is available.",
        reason="No successful attempt is available and further search cannot proceed.",
    )
