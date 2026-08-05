from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Mapping, TypedDict

from langchain_core.tools import StructuredTool
from langgraph.graph import END, START, StateGraph

from evidencegap_backend.agent.controller import ControllerCallable
from evidencegap_backend.agent.gap_controller import (
    GapControllerCallable,
    compact_gap_summary,
)
from evidencegap_backend.agent.gap_policies import (
    deterministic_gap_fallback,
    validate_gap_decision,
)
from evidencegap_backend.agent.policies import deterministic_fallback, validate_decision
from evidencegap_backend.agent.schemas import (
    AgentAction,
    AgentDecision,
    EvidenceWorkspace,
    GapAction,
    GapDecision,
    SearchAttempt,
)
from evidencegap_backend.agent.stages import AgentStageExecutor
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.agent.workspace import (
    choose_attempt,
    increment_action,
    initialize_workspace,
    register_attempt,
    successful_attempts,
    terminal_count,
    utc_now,
)
from evidencegap_backend.common import EvidenceGapError


class GraphState(TypedDict, total=False):
    workspace: dict[str, Any]
    decomposition_result: dict[str, Any]
    analysis_result: dict[str, Any]
    bundle_result: dict[str, Any]
    gap_result: dict[str, Any]
    output_result: dict[str, Any]
    final_result: dict[str, Any]
    node_history: list[str]


def build_agent_graph(
    *,
    controller: ControllerCallable,
    gap_controller: GapControllerCallable,
    search_tool: StructuredTool,
    stages: AgentStageExecutor,
    run_name: str = "agent_run",
    max_steps: int = 20,
    total_search_budget: int = 8,
    per_claim_search_budget: int = 3,
    max_gap_rounds: int = 2,
    gap_remediation_budget: int = 2,
    controller_retry_count: int = 2,
    gap_controller_retry_count: int = 2,
    trace_writer: AgentTraceWriter | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
    checkpointer: Any = None,
) -> Any:
    """Compile the complete EvidenceGap reasoning harness.

    The executor is deliberately outside state; checkpointed values contain only
    JSON-compatible workspace snapshots and artifact/result dictionaries.
    """

    def save(ws: EvidenceWorkspace, history: list[str], **values: Any) -> GraphState:
        return {
            "workspace": ws.model_dump(mode="json"),
            "node_history": history,
            **values,
        }

    def history(state: GraphState, name: str) -> list[str]:
        return [*state.get("node_history", []), name]

    def trace(ws: EvidenceWorkspace, event: dict[str, Any]) -> None:
        if trace_writer:
            trace_writer.append(
                {
                    "created_at": utc_now(),
                    "evidence_cycle": ws.evidence_cycle,
                    "gap_round": ws.gap_round,
                    **event,
                }
            )

    def progress(
        ws: EvidenceWorkspace,
        message: str,
        *,
        remediation: bool = False,
    ) -> None:
        if progress_callback:
            progress_callback(
                {
                    "stage": (
                        "inference_gap_analysis" if remediation else "claim_analysis"
                    ),
                    "stage_index": 4 if remediation else 2,
                    "total_stages": 5,
                    "message": message,
                    "completed_units": terminal_count(ws),
                    "total_units": len(ws.claims),
                }
            )

    def initialize_run(state: GraphState) -> GraphState:
        result = stages.initialize_run()
        return {**state, **result, "node_history": history(state, "initialize_run")}

    def statement_decomposition(state: GraphState) -> GraphState:
        result = stages.statement_decomposition()
        return {
            **state,
            "decomposition_result": result,
            "node_history": history(state, "statement_decomposition"),
        }

    def initialize(state: GraphState) -> GraphState:
        decomposition_result = state["decomposition_result"]
        ws = initialize_workspace(
            decomposition=decomposition_result["decomposition"],
            run_name=run_name,
            statement=str(decomposition_result["decomposition"]["original_statement"]),
            language=str(decomposition_result["decomposition"]["source_language"]),
            max_steps=max_steps,
            total_search_budget=total_search_budget,
            per_claim_search_budget=per_claim_search_budget,
            max_gap_rounds=max_gap_rounds,
            gap_remediation_budget=gap_remediation_budget,
        )
        ws.phase = "evidence"
        progress(ws, "Retrieving and evaluating evidence for extracted claims")
        return save(
            ws,
            history(state, "initialize_workspace"),
            decomposition_result=decomposition_result,
        )

    def evidence_controller(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
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
        if not (
            source == "deterministic_fallback"
            and decision.action is AgentAction.ABSTAIN
        ):
            validate_decision(ws, decision)
        ws.decision = decision
        ws.decision_source = source
        ws.evidence_controller_decision_count += 1
        if source == "deterministic_fallback":
            ws.deterministic_fallback_decisions += 1
        ws.active_claim_id = decision.claim_id
        if decision.action is not AgentAction.FINISH:
            progress(
                ws,
                f"Agent step {ws.step_count + 1}: {decision.action.value.lower()} "
                f"{decision.claim_id or 'claims'}",
                remediation=ws.gap_round > 0,
            )
        return {**state, **save(ws, history(state, "evidence_controller"))}

    def execute_search(
        state: GraphState,
        ws: EvidenceWorkspace,
        decision: AgentDecision,
        *,
        node_name: str,
        source: str,
    ) -> GraphState:
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
        trace(
            ws,
            {
                "phase": "evidence",
                "controller_type": (
                    "gap_controller" if source == "gap_controller" else "evidence_controller"
                ),
                "event": "action_result",
                "decision_source": source,
                "action": "SEARCH",
                "claim_id": claim.claim_id,
                "query": decision.query,
                "reason": decision.reason,
                "remaining_search_budget": ws.remaining_search_budget,
                "attempt_id": attempt_id,
                "new_articles": len(attempt.new_article_ids),
                "verdict_after_action": attempt.verdict,
                "status": attempt.status,
            },
        )
        return {**state, **save(ws, history(state, node_name))}

    def search(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        decision = ws.decision
        assert decision is not None and decision.action is AgentAction.SEARCH
        return execute_search(
            state, ws, decision, node_name="search_evidence", source=ws.decision_source
        )

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
        node_name = "resolve_claim" if action is AgentAction.RESOLVE else "abstain_claim"
        trace(
            ws,
            {
                "phase": "evidence",
                "controller_type": "evidence_controller",
                "event": "action_result",
                "decision_source": ws.decision_source,
                "action": action.value,
                "claim_id": claim.claim_id,
                "selected_attempt_id": claim.selected_attempt_id,
                "reason": decision.reason,
                "remaining_search_budget": ws.remaining_search_budget,
                "completed_units": terminal_count(ws),
            },
        )
        return {**state, **save(ws, history(state, node_name))}

    def resolve(state: GraphState) -> GraphState:
        return terminal(state, AgentAction.RESOLVE)

    def abstain(state: GraphState) -> GraphState:
        return terminal(state, AgentAction.ABSTAIN)

    def evidence_finish(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.finish_reason = ws.decision.reason if ws.decision else "all claims terminal"
        ws.phase = "materializing"
        increment_action(ws, AgentAction.FINISH)
        trace(
            ws,
            {
                "phase": "evidence",
                "controller_type": "evidence_controller",
                "event": "evidence_cycle_finished",
                "decision_source": ws.decision_source,
                "action": "FINISH",
                "reason": ws.finish_reason,
                "remaining_search_budget": ws.remaining_search_budget,
            },
        )
        return {**state, **save(ws, history(state, "evidence_finish"))}

    def materialize(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.phase = "materializing"
        result = stages.materialize_statement_analysis(ws)
        return {
            **state,
            **save(ws, history(state, "materialize_statement_analysis")),
            "analysis_result": result,
        }

    def bundle(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.phase = "bundle"
        result = stages.build_statement_bundle(ws, state["analysis_result"])
        return {
            **state,
            **save(ws, history(state, "build_statement_bundle")),
            "bundle_result": result,
        }

    def gap_analysis(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.phase = "gap_analysis"
        result = stages.run_inference_gap_analysis(ws, state["bundle_result"])
        ws.gap_round += 1
        ws.latest_gap_summary = compact_gap_summary(
            ws, result["inference_gap_bundle"]
        )
        return {
            **state,
            **save(ws, history(state, "run_inference_gap_analysis")),
            "gap_result": result,
        }

    def gap_decide(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.phase = "gap_decision"
        decision: GapDecision | None = None
        note: str | None = None
        source = "controller"
        gap_bundle = state["gap_result"]["inference_gap_bundle"]
        for _ in range(gap_controller_retry_count + 1):
            try:
                candidate = gap_controller(ws, gap_bundle, note)
                decision = validate_gap_decision(
                    ws, gap_bundle, GapDecision.model_validate(candidate)
                )
                break
            except Exception as exc:
                ws.rejected_gap_decisions += 1
                note = str(exc)[:500]
        if decision is None:
            source = "deterministic_fallback"
            decision = deterministic_gap_fallback(
                ws, reason=f"gap controller returned invalid decisions: {note}"
            )
        validate_gap_decision(ws, gap_bundle, decision)
        ws.gap_decision = decision
        ws.gap_decision_source = source
        ws.gap_controller_decision_count += 1
        if source == "deterministic_fallback":
            ws.deterministic_fallback_decisions += 1
        event = {
            "phase": "gap",
            "controller_type": "gap_controller",
            "event": "decision",
            "decision_source": source,
            "action": decision.action.value,
            "target_gap_id": decision.target_gap_id,
            "claim_id": decision.claim_id,
            "query": decision.query,
            "reason": decision.reason,
            "remaining_gap_remediation_budget": ws.remaining_gap_remediation_budget,
            "remaining_search_budget": ws.remaining_search_budget,
        }
        trace(ws, event)
        round_record = stages.record_gap_round(ws, decision)
        ws.gap_history.append(
            {
                "evidence_cycle": ws.evidence_cycle,
                "gap_round": ws.gap_round,
                **event,
                "round_artifact": round_record,
            }
        )
        return {**state, **save(ws, history(state, "gap_controller"))}

    def reopen(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        decision = ws.gap_decision
        assert decision is not None
        claim = ws.claims[str(decision.claim_id)]
        claim.status = "pending"
        claim.remaining_problem = decision.remaining_problem
        claim.terminal_reason = None
        claim.reopen_count += 1
        claim.reopened_for_gap_id = decision.target_gap_id
        ws.active_claim_id = claim.claim_id
        ws.evidence_cycle += 1
        ws.gap_remediation_count += 1
        ws.remaining_gap_remediation_budget -= 1
        ws.phase = "evidence"
        progress(
            ws,
            f"Gap review requested additional evidence for {claim.claim_id}",
            remediation=True,
        )
        trace(
            ws,
            {
                "phase": "gap",
                "controller_type": "gap_controller",
                "event": "claim_reopened",
                "decision_source": ws.gap_decision_source,
                "action": "REOPEN_CLAIM",
                "target_gap_id": decision.target_gap_id,
                "claim_id": claim.claim_id,
                "remaining_problem": decision.remaining_problem,
            },
        )
        return {**state, **save(ws, history(state, "reopen_claim"))}

    def gap_search(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        gap_decision = ws.gap_decision
        assert gap_decision is not None
        decision = AgentDecision(
            action=AgentAction.SEARCH,
            claim_id=gap_decision.claim_id,
            query=gap_decision.query,
            remaining_problem=gap_decision.remaining_problem,
            reason=gap_decision.reason,
        )
        validate_decision(ws, decision)
        progress(
            ws,
            f"Searching targeted evidence for gap {gap_decision.target_gap_id}",
            remediation=True,
        )
        ws.decision = decision
        ws.decision_source = "gap_controller"
        return execute_search(
            state,
            ws,
            decision,
            node_name="execute_gap_requested_search",
            source="gap_controller",
        )

    def output(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.phase = "output"
        result = stages.generate_output(
            ws, state["bundle_result"], state["gap_result"]
        )
        output_meta = result.get("output")
        presentation_meta = (
            output_meta.get("presentation_bundle")
            if isinstance(output_meta, Mapping)
            else None
        )
        ws.final_output_path = result.get("presentation_bundle_path") or (
            presentation_meta.get("path")
            if isinstance(presentation_meta, Mapping)
            else None
        )
        if not ws.final_output_path:
            raise EvidenceGapError(
                "Output node did not receive a presentation bundle path"
            )
        return {
            **state,
            **save(ws, history(state, "generate_output")),
            "output_result": result,
        }

    def finalize(state: GraphState) -> GraphState:
        ws = EvidenceWorkspace.model_validate(state["workspace"])
        ws.phase = "finished"
        ws.status = "finished"
        ws.finished_at = datetime.now(timezone.utc).isoformat()
        staged_state = {**state, "workspace": ws.model_dump(mode="json")}
        result = stages.finalize_run(ws, staged_state)
        return {
            **staged_state,
            "final_result": result,
            "node_history": history(state, "finalize_run"),
        }

    def route_evidence(state: GraphState) -> str:
        return str(state["workspace"]["decision"]["action"]).lower()

    def route_gap(state: GraphState) -> str:
        return str(state["workspace"]["gap_decision"]["action"]).lower()

    graph = StateGraph(GraphState)
    graph.add_node("initialize_run", initialize_run)
    graph.add_node("statement_decomposition", statement_decomposition)
    graph.add_node("initialize_workspace", initialize)
    graph.add_node("evidence_controller", evidence_controller)
    graph.add_node("search_evidence", search)
    graph.add_node("resolve_claim", resolve)
    graph.add_node("abstain_claim", abstain)
    graph.add_node("evidence_finish", evidence_finish)
    graph.add_node("materialize_statement_analysis", materialize)
    graph.add_node("build_statement_bundle", bundle)
    graph.add_node("run_inference_gap_analysis", gap_analysis)
    graph.add_node("gap_controller", gap_decide)
    graph.add_node("reopen_claim", reopen)
    graph.add_node("execute_gap_requested_search", gap_search)
    graph.add_node("generate_output", output)
    graph.add_node("finalize_run", finalize)

    graph.add_edge(START, "initialize_run")
    graph.add_edge("initialize_run", "statement_decomposition")
    graph.add_edge("statement_decomposition", "initialize_workspace")
    graph.add_edge("initialize_workspace", "evidence_controller")
    graph.add_conditional_edges(
        "evidence_controller",
        route_evidence,
        {
            "search": "search_evidence",
            "resolve": "resolve_claim",
            "abstain": "abstain_claim",
            "finish": "evidence_finish",
        },
    )
    graph.add_edge("search_evidence", "evidence_controller")
    graph.add_edge("resolve_claim", "evidence_controller")
    graph.add_edge("abstain_claim", "evidence_controller")
    graph.add_edge("evidence_finish", "materialize_statement_analysis")
    graph.add_edge("materialize_statement_analysis", "build_statement_bundle")
    graph.add_edge("build_statement_bundle", "run_inference_gap_analysis")
    graph.add_edge("run_inference_gap_analysis", "gap_controller")
    graph.add_conditional_edges(
        "gap_controller",
        route_gap,
        {
            "request_more_evidence": "reopen_claim",
            "accept_gaps": "generate_output",
            "abstain": "generate_output",
        },
    )
    graph.add_edge("reopen_claim", "execute_gap_requested_search")
    graph.add_edge("execute_gap_requested_search", "evidence_controller")
    graph.add_edge("generate_output", "finalize_run")
    graph.add_edge("finalize_run", END)
    return graph.compile(checkpointer=checkpointer)
