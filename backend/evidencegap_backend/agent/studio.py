"""Resource-free Studio graph using the production end-to-end topology."""

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from evidencegap_backend.agent.demo_fake import FakeStageExecutor
from evidencegap_backend.agent.gap_controller import compact_gap_summary
from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.schemas import AgentDecision, GapDecision, SearchAttempt
from evidencegap_backend.agent.tools import create_search_evidence_tool
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.agent.workspace import normalize_query, utc_now

_RUN_DIR = Path(tempfile.mkdtemp(prefix="evidencegap-studio-"))
_TRACE = AgentTraceWriter(_RUN_DIR / "agent")
_STAGES = FakeStageExecutor(_RUN_DIR, _TRACE)
_CLAIM_ID = str(_STAGES.decomposition["claims"][0]["claim_id"])


def _controller(workspace: Any, _note: str | None = None) -> AgentDecision:
    claim = workspace.claims[_CLAIM_ID]
    if not claim.attempts:
        return AgentDecision(action="SEARCH", claim_id=_CLAIM_ID, query=claim.canonical_claim_en, reason="Studio initial search")
    if claim.status == "pending":
        return AgentDecision(action="RESOLVE", claim_id=_CLAIM_ID, selected_attempt_id=claim.attempts[-1].attempt_id, reason="Studio selects successful attempt")
    return AgentDecision(action="FINISH", reason="Evidence cycle is terminal")


def _gap_controller(workspace: Any, gap_bundle: Mapping[str, Any], _note: str | None = None) -> GapDecision:
    if workspace.gap_round == 1:
        gap_id = compact_gap_summary(workspace, gap_bundle)["gaps"][0]["gap_id"]
        return GapDecision(action="REQUEST_MORE_EVIDENCE", target_gap_id=gap_id, claim_id=_CLAIM_ID, query="direct clinical outcome evidence", remaining_problem="Studio coverage gap", reason="Exercise the remediation branch")
    return GapDecision(action="ACCEPT_GAPS", reason="Accept the final formal gaps")


def _search(request: Any) -> SearchAttempt:
    now = utc_now()
    directory = _RUN_DIR / "analysis/agent_attempts" / request.claim_id / request.attempt_id
    graph_path = directory / "final_graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps({"nodes": [], "query": request.query}), encoding="utf-8")
    return SearchAttempt(
        attempt_id=request.attempt_id,
        claim_id=request.claim_id,
        query=request.query,
        normalized_query=normalize_query(request.query),
        artifact_dir=str(directory),
        graph_bundle_path=str(graph_path),
        verdict="insufficient",
        status="successful",
        started_at=now,
        finished_at=now,
    )


graph = build_agent_graph(
    controller=_controller,
    gap_controller=_gap_controller,
    search_tool=create_search_evidence_tool(_search),
    stages=_STAGES,
    run_name="studio_demo",
    max_gap_rounds=2,
    gap_remediation_budget=1,
    trace_writer=_TRACE,
)
