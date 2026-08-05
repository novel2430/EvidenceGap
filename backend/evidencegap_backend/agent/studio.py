"""Lazy, resource-free Studio graph exposing the production graph topology."""
from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.schemas import AgentDecision, SearchAttempt
from evidencegap_backend.agent.tools import create_search_evidence_tool
from evidencegap_backend.agent.workspace import normalize_query, utc_now


def _controller(workspace, _note=None):
    pending = [
        workspace.claims[c]
        for c in workspace.claim_order
        if workspace.claims[c].status == "pending"
    ]
    if not pending:
        return AgentDecision(action="FINISH", reason="All claims terminal")
    claim = pending[0]
    if claim.attempts:
        return AgentDecision(
            action="RESOLVE",
            claim_id=claim.claim_id,
            reason="Studio default resolves successful attempt",
        )
    return AgentDecision(
        action="SEARCH",
        claim_id=claim.claim_id,
        query=claim.canonical_claim_en,
        reason="Studio default initial search",
    )


def _search(request):
    now = utc_now()
    return SearchAttempt(
        attempt_id=request.attempt_id,
        claim_id=request.claim_id,
        query=request.query,
        normalized_query=normalize_query(request.query),
        verdict="insufficient",
        status="successful",
        started_at=now,
        finished_at=now,
    )


graph = build_agent_graph(
    controller=_controller, search_tool=create_search_evidence_tool(_search)
)
