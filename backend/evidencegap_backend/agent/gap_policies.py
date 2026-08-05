from __future__ import annotations

from typing import Any, Mapping

from evidencegap_backend.agent.gap_controller import compact_gap_summary
from evidencegap_backend.agent.schemas import (
    EvidenceWorkspace,
    GapAction,
    GapDecision,
)
from evidencegap_backend.agent.workspace import normalize_query


class InvalidGapDecision(ValueError):
    pass


def validate_gap_decision(
    workspace: EvidenceWorkspace,
    gap_bundle: Mapping[str, Any],
    decision: GapDecision,
) -> GapDecision:
    if decision.action is not GapAction.REQUEST_MORE_EVIDENCE:
        return decision
    summary = compact_gap_summary(workspace, gap_bundle)
    gap_ids = {str(row["gap_id"]) for row in summary["gaps"]}
    if decision.target_gap_id not in gap_ids:
        raise InvalidGapDecision("target_gap_id does not exist")
    claim = workspace.claims.get(str(decision.claim_id))
    if claim is None:
        raise InvalidGapDecision("claim_id does not exist")
    if workspace.remaining_search_budget <= 0:
        raise InvalidGapDecision("total search budget is exhausted")
    if workspace.remaining_gap_remediation_budget <= 0:
        raise InvalidGapDecision("gap remediation budget is exhausted")
    if workspace.gap_round >= workspace.max_gap_rounds:
        raise InvalidGapDecision("maximum gap rounds reached")
    if len(claim.attempts) >= workspace.per_claim_search_budget:
        raise InvalidGapDecision("per-claim search budget is exhausted")
    normalized = normalize_query(str(decision.query or ""))
    if not normalized:
        raise InvalidGapDecision("gap remediation query is blank")
    if normalized in claim.normalized_queries:
        raise InvalidGapDecision("gap remediation query duplicates a previous query")
    return decision


def deterministic_gap_fallback(
    workspace: EvidenceWorkspace, *, reason: str
) -> GapDecision:
    return GapDecision(action="ABSTAIN", reason=reason)
