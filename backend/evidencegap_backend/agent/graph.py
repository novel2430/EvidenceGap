from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, TypedDict

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from evidencegap_backend.agent.controller import ControllerCallable
from evidencegap_backend.agent.policies import (
    deterministic_fallback,
    validate_decision,
)
from evidencegap_backend.agent.schemas import (
    AgentAction,
    AgentDecision,
    EvidenceWorkspace,
    SearchAttempt,
)
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.agent.workspace import (
    choose_attempt,
    increment_action,
    register_attempt,
    successful_attempts,
    terminal_count,
    utc_now,
)


class GraphState(TypedDict):
    workspace: dict[str, Any]
    node_history: list[str]


def build_agent_graph(
    *,
    controller: ControllerCallable,
    search_tool: StructuredTool,
    controller_retry_count: int = 2,
    trace_writer: AgentTraceWriter | None = None,
    checkpointer: Any = None,
    finalize: Callable[[EvidenceWorkspace], None] | None = None,
    action_callback: Callable[[EvidenceWorkspace, AgentDecision], None] | None = None,
) -> Any:
    def save(ws: EvidenceWorkspace, history: list[str]) -> GraphState:
        return {"workspace": ws.model_dump(mode="json"), "node_history": history}

    def initialize(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        return save(ws, [*state.get("node_history", []), "initialize_workspace"])

    def controller_node(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        history = [*state["node_history"], "controller"]
        decision: AgentDecision | None = None
        note: str | None = None
        source = "controller"
        if ws.step_count >= ws.max_steps:
            source = "deterministic_fallback"
            decision = deterministic_fallback(ws, reason="maximum agent steps reached")
        else:
            for _ in range(controller_retry_count + 1):
                try:
                    candidate = controller(ws, note)
                    decision = validate_decision(
                        ws, AgentDecision.model_validate(candidate)
                    )
                    break
                except Exception as exc:
                    ws.rejected_decisions += 1
                    note = str(exc)[:500]
            if decision is None:
                source = "deterministic_fallback"
                decision = deterministic_fallback(
                    ws, reason=f"controller returned invalid decisions: {note}"
                )
        # Fallback ABSTAIN may intentionally converge a claim with no successful attempt.
        if not (
            source == "deterministic_fallback"
            and decision.action is AgentAction.ABSTAIN
        ):
            validate_decision(ws, decision)
        ws.decision = decision
        ws.decision_source = source
        ws.active_claim_id = decision.claim_id
        if action_callback:
            action_callback(ws, decision)
        return save(ws, history)

    def search(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        decision = ws.decision
        assert decision is not None and decision.action is AgentAction.SEARCH
        claim = ws.claims[str(decision.claim_id)]
        attempt_id = f"attempt_{len(claim.attempts) + 1:03d}"
        ws.remaining_search_budget -= 1
        raw = search_tool.invoke(
            {
                "claim_id": claim.claim_id,
                "canonical_claim": claim.canonical_claim_en,
                "query": decision.query,
                "attempt_id": attempt_id,
            }
        )
        attempt = SearchAttempt.model_validate(raw)
        attempt.new_article_ids = sorted(
            set(attempt.article_ids) - set(claim.seen_article_ids)
        )
        attempt.utility_score = (
            attempt.direct_evidence_articles * 100
            + (20 if attempt.verdict and attempt.verdict != "insufficient" else 0)
            + len(attempt.new_article_ids)
        )
        register_attempt(ws, attempt)
        ws.last_action_result = attempt.model_dump(mode="json")
        increment_action(ws, AgentAction.SEARCH)
        event = {
            "step": ws.step_count,
            "created_at": utc_now(),
            "decision_source": ws.decision_source,
            "action": "SEARCH",
            "claim_id": claim.claim_id,
            "query": decision.query,
            "reason": decision.reason,
            "remaining_search_budget": ws.remaining_search_budget,
            "attempt_id": attempt_id,
            "new_articles": len(attempt.new_article_ids),
            "verdict_after_action": attempt.verdict,
            "status": attempt.status,
        }
        if trace_writer:
            trace_writer.append(event)
        return save(ws, [*state["node_history"], "search_evidence"])

    def terminal(state: GraphState, action: AgentAction) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        decision = ws.decision
        assert decision is not None
        claim = ws.claims[str(decision.claim_id)]
        attempts = successful_attempts(claim)
        if not attempts:
            claim.status = "failed"
            claim.terminal_reason = decision.reason
        else:
            selected = choose_attempt(claim, decision.selected_attempt_id)
            claim.selected_attempt_id = selected.attempt_id
            claim.verdict = selected.verdict
            claim.status = "resolved" if action is AgentAction.RESOLVE else "abstained"
            claim.terminal_reason = decision.reason
            claim.remaining_problem = decision.remaining_problem
        increment_action(ws, action)
        if trace_writer:
            trace_writer.append(
                {
                    "step": ws.step_count,
                    "created_at": utc_now(),
                    "decision_source": ws.decision_source,
                    "action": action.value,
                    "claim_id": claim.claim_id,
                    "selected_attempt_id": claim.selected_attempt_id,
                    "reason": decision.reason,
                    "remaining_search_budget": ws.remaining_search_budget,
                    "completed_units": terminal_count(ws),
                }
            )
        return save(
            ws,
            [
                *state["node_history"],
                "resolve_claim" if action is AgentAction.RESOLVE else "abstain_claim",
            ],
        )

    def resolve(state: GraphState) -> GraphState:
        return terminal(state, AgentAction.RESOLVE)

    def abstain(state: GraphState) -> GraphState:
        return terminal(state, AgentAction.ABSTAIN)

    def finish(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.status = "finished"
        ws.finish_reason = ws.decision.reason if ws.decision else "all claims terminal"
        ws.finished_at = datetime.now(timezone.utc).isoformat()
        increment_action(ws, AgentAction.FINISH)
        if trace_writer:
            trace_writer.append(
                {
                    "step": ws.step_count,
                    "created_at": utc_now(),
                    "decision_source": ws.decision_source,
                    "action": "FINISH",
                    "reason": ws.finish_reason,
                    "remaining_search_budget": ws.remaining_search_budget,
                }
            )
        return save(ws, [*state["node_history"], "finalize_statement_analysis"])

    def tail_node(name: str, *, run_finalize: bool = False):
        def node(state: GraphState) -> GraphState:
            ws = EvidenceWorkspace.model_validate(state["workspace"])
            if run_finalize and finalize:
                finalize(ws)
            return save(ws, [*state["node_history"], name])

        return node

    def route(state: GraphState) -> str:
        return str(state["workspace"]["decision"]["action"]).lower()

    graph = StateGraph(GraphState)
    graph.add_node("initialize_workspace", initialize)
    graph.add_node("controller", controller_node)
    graph.add_node("search_evidence", search)
    graph.add_node("resolve_claim", resolve)
    graph.add_node("abstain_claim", abstain)
    graph.add_node("finalize_statement_analysis", finish)
    graph.add_node("build_statement_bundle", tail_node("build_statement_bundle"))
    graph.add_node("inference_gap_analysis", tail_node("inference_gap_analysis"))
    graph.add_node("generate_output", tail_node("generate_output", run_finalize=True))
    graph.add_edge(START, "initialize_workspace")
    graph.add_edge("initialize_workspace", "controller")
    graph.add_conditional_edges(
        "controller",
        route,
        {
            "search": "search_evidence",
            "resolve": "resolve_claim",
            "abstain": "abstain_claim",
            "finish": "finalize_statement_analysis",
        },
    )
    graph.add_edge("search_evidence", "controller")
    graph.add_edge("resolve_claim", "controller")
    graph.add_edge("abstain_claim", "controller")
    graph.add_edge("finalize_statement_analysis", "build_statement_bundle")
    graph.add_edge("build_statement_bundle", "inference_gap_analysis")
    graph.add_edge("inference_gap_analysis", "generate_output")
    graph.add_edge("generate_output", END)
    return graph.compile(checkpointer=checkpointer)
