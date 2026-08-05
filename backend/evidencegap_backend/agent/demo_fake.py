from __future__ import annotations

import argparse
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

from langgraph.checkpoint.sqlite import SqliteSaver

from evidencegap_backend.agent.gap_controller import compact_gap_summary
from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.schemas import AgentDecision, GapDecision, SearchAttempt
from evidencegap_backend.agent.tools import create_search_evidence_tool
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.agent.workspace import normalize_query, utc_now
from evidencegap_backend.common import atomic_write_json, sha256_file, sha256_text
from evidencegap_backend.pipeline.retrieval_adapters import runtime_claim_id


class FakeStageExecutor:
    """Resource-free executors that still persist every graph stage artifact."""

    def __init__(self, run_dir: Path, trace: AgentTraceWriter) -> None:
        self.run_dir = run_dir
        self.trace = trace
        self.calls = {name: 0 for name in ("analysis", "bundle", "gap", "output")}
        claim = "Drug A reduces biomarker B."
        self.decomposition = {
            "schema_version": "fake-1",
            "contract_id": "fake.decomposition",
            "statement_id": "statement_fake",
            "original_statement": claim,
            "source_language": "English",
            "claims": [
                {
                    "claim_id": runtime_claim_id(claim),
                    "source_text": claim,
                    "source_spans": [{"start": 0, "end": len(claim)}],
                    "canonical_claim_en": claim,
                }
            ],
            "inference_steps": [],
        }

    def initialize_run(self) -> dict[str, Any]:
        atomic_write_json(self.run_dir / "request.json", {"statement": self.decomposition["original_statement"]})
        return {"initialized": True}

    def statement_decomposition(self) -> dict[str, Any]:
        directory = self.run_dir / "decomposition"
        atomic_write_json(directory / "decomposition.json", self.decomposition)
        atomic_write_json(directory / "request.json", {"statement": self.decomposition["original_statement"]})
        atomic_write_json(directory / "run_manifest.json", {"stage": "decomposition", "seconds": 0.0})
        return {
            "run_name": "decomposition",
            "statement_id": self.decomposition["statement_id"],
            "decomposition": self.decomposition,
            "artifact_dir": str(directory),
        }

    def materialize_statement_analysis(self, workspace: Any) -> dict[str, Any]:
        self.calls["analysis"] += 1
        directory = self.run_dir / "analysis"
        rows = []
        graphs = {}
        for claim_id in workspace.claim_order:
            claim = workspace.claims[claim_id]
            attempt = next(a for a in claim.attempts if a.attempt_id == claim.selected_attempt_id)
            graph = json.loads(Path(attempt.graph_bundle_path).read_text(encoding="utf-8"))
            graphs[claim_id] = graph
            rows.append({
                "claim_id": claim_id,
                "canonical_claim_en": claim.canonical_claim_en,
                "selected_attempt_id": attempt.attempt_id,
                "graph_bundle_path": attempt.graph_bundle_path,
                "verdict": attempt.verdict,
                "status": "completed",
            })
        value = {
            "analysis_status": "completed",
            "claim_results": rows,
            "evidence_cycle": workspace.evidence_cycle,
        }
        atomic_write_json(directory / "statement_result.json", value)
        atomic_write_json(directory / "request.json", {"evidence_cycle": workspace.evidence_cycle})
        atomic_write_json(directory / "run_manifest.json", {"stage": "analysis", "seconds": 0.0})
        return {
            "analysis_status": "completed",
            "statement_result": value,
            "decomposition": self.decomposition,
            "claim_graph_bundles": graphs,
            "artifact_dir": str(directory),
        }

    def build_statement_bundle(self, workspace: Any, analysis_result: Mapping[str, Any]) -> dict[str, Any]:
        self.calls["bundle"] += 1
        directory = self.run_dir / "bundle"
        selected = {
            cid: workspace.claims[cid].selected_attempt_id for cid in workspace.claim_order
        }
        value = {"statement_id": "statement_fake", "selected_attempts": selected, "round": self.calls["bundle"]}
        atomic_write_json(directory / "statement_bundle.json", value)
        atomic_write_json(directory / "request.json", {"analysis_cycle": workspace.evidence_cycle})
        atomic_write_json(directory / "run_manifest.json", {"stage": "bundle", "seconds": 0.0})
        return {
            "statement_bundle": value,
            "total_claims": len(selected),
            "completed_claims": len(selected),
            "failed_claims": 0,
            "articles": len(selected),
            "evidence": len(selected),
            "artifact_dir": str(directory),
        }

    def run_inference_gap_analysis(self, workspace: Any, bundle_result: Mapping[str, Any]) -> dict[str, Any]:
        self.calls["gap"] += 1
        directory = self.run_dir / "gaps"
        detected = self.calls["gap"] == 1
        value = {
            "statement_id": "statement_fake",
            "inference_gap_analyses": [{
                "inference_step_id": "inference_1",
                "scope_gap": {
                    "detected": detected,
                    "subtype": "outcome_scope",
                    "affected_dimensions": ["outcome"],
                    "supported_basis": "Biomarker outcome",
                    "unsupported_extension": "Clinical outcomes",
                    "reason": "Direct clinical outcomes were not retrieved",
                    "closure_requirement": "Targeted clinical outcome evidence",
                },
                "causal_gap": {"detected": False},
            }],
            "summary": {"scope_gaps": int(detected), "causal_gaps": 0},
            "bundle_round": bundle_result["statement_bundle"]["round"],
        }
        atomic_write_json(directory / "inference_gap_analysis.json", value)
        atomic_write_json(directory / "request.json", {"bundle_round": self.calls["gap"]})
        atomic_write_json(directory / "run_manifest.json", {"stage": "gaps", "seconds": 0.0})
        return {
            "inference_gap_bundle": value,
            "total_inference_steps": 1,
            "scope_gaps": int(detected),
            "causal_gaps": 0,
            "api_requests": 0,
            "artifact_dir": str(directory),
        }

    def generate_output(self, workspace: Any, bundle_result: Mapping[str, Any], gap_result: Mapping[str, Any]) -> dict[str, Any]:
        self.calls["output"] += 1
        directory = self.run_dir / "output"
        value = {
            "contract_id": "phase077.presentation-bundle.v1",
            "statement_id": "statement_fake",
            "bundle_round": bundle_result["statement_bundle"]["round"],
            "formal_gaps": gap_result["inference_gap_bundle"],
        }
        path = directory / "presentation_bundle.json"
        atomic_write_json(path, value)
        atomic_write_json(directory / "request.json", {"gap_round": workspace.gap_round})
        atomic_write_json(directory / "run_manifest.json", {"stage": "output", "seconds": 0.0})
        return {
            "presentation_bundle": value,
            "output": {
                "presentation_bundle": {
                    "path": str(path),
                    "sha256": sha256_file(path),
                }
            },
            "output_language": "English",
            "localized": False,
            "api_requests": 0,
            "artifact_dir": str(directory),
        }

    def record_gap_round(self, workspace: Any, decision: GapDecision) -> dict[str, Any]:
        rounds_dir = self.run_dir / "agent/gap_rounds"
        gap_summary = dict(workspace.latest_gap_summary or {})
        gap_summary_path = (
            rounds_dir / f"round_{workspace.gap_round:03d}_gap_summary.json"
        )
        atomic_write_json(gap_summary_path, gap_summary)
        value = {
            "gap_round": workspace.gap_round,
            "evidence_cycle": workspace.evidence_cycle,
            "selected_attempts": {cid: workspace.claims[cid].selected_attempt_id for cid in workspace.claim_order},
            "decision": decision.model_dump(mode="json"),
            "analysis_sha256": sha256_text((self.run_dir / "analysis/statement_result.json").read_text()),
            "bundle_sha256": sha256_text((self.run_dir / "bundle/statement_bundle.json").read_text()),
            "gap_sha256": sha256_text((self.run_dir / "gaps/inference_gap_analysis.json").read_text()),
            "gap_summary": gap_summary,
            "gap_summary_artifact": {
                "path": str(gap_summary_path),
                "sha256": sha256_file(gap_summary_path),
            },
        }
        path = rounds_dir / f"round_{workspace.gap_round:03d}.json"
        atomic_write_json(path, value)
        return {"path": str(path), **value}

    def finalize_run(self, workspace: Any, state: Mapping[str, Any]) -> dict[str, Any]:
        self.trace.write_workspace(workspace.model_dump(mode="json"))
        manifest = {
            "execution_mode": "langgraph_end_to_end_fake",
            "node_history": state["node_history"],
            "stage_calls": self.calls,
            "gap_rounds": workspace.gap_round,
            "gap_remediation_count": workspace.gap_remediation_count,
            "final_gap_decision": workspace.gap_decision.model_dump(mode="json"),
        }
        atomic_write_json(self.run_dir / "agent/agent_manifest.json", manifest)
        atomic_write_json(self.run_dir / "run_manifest.json", {"run_type": "fake_agent_smoke", "seconds": 0.0})
        output = state["output_result"]
        return {
            "status": "COMPLETED",
            "artifact_status": "PASS",
            "run_name": "fake_agent_demo",
            "artifact_dir": str(self.run_dir),
            "presentation_bundle_path": workspace.final_output_path,
            "presentation_bundle": output["presentation_bundle"],
            "node_history": state["node_history"] + ["finalize_run"],
            "stage_calls": dict(self.calls),
        }


def run_demo(
    output_dir: Path,
    *,
    request_remediation: bool = True,
    gap_controller_override: Callable[..., GapDecision] | None = None,
) -> tuple[Path, dict[str, Any]]:
    run_dir = output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    agent_dir = run_dir / "agent"
    trace = AgentTraceWriter(agent_dir)
    stages = FakeStageExecutor(run_dir, trace)
    claim_id = stages.decomposition["claims"][0]["claim_id"]

    def evidence_controller(workspace: Any, _note: str | None = None) -> AgentDecision:
        claim = workspace.claims[claim_id]
        if not claim.attempts:
            return AgentDecision(action="SEARCH", claim_id=claim_id, query="Drug A biomarker B randomized trial", reason="Establish direct evidence")
        if claim.status == "pending":
            return AgentDecision(action="RESOLVE", claim_id=claim_id, selected_attempt_id=claim.attempts[-1].attempt_id, reason="Use the latest successful attempt")
        return AgentDecision(action="FINISH", reason="All claims are terminal for this evidence cycle")

    def gap_controller(workspace: Any, gap_bundle: Mapping[str, Any], _note: str | None = None) -> GapDecision:
        if request_remediation and workspace.gap_round == 1:
            gap_id = compact_gap_summary(workspace, gap_bundle)["gaps"][0]["gap_id"]
            return GapDecision(action="REQUEST_MORE_EVIDENCE", target_gap_id=gap_id, claim_id=claim_id, query="Drug A direct clinical outcomes randomized trial", remaining_problem="Current evidence covers only a biomarker", reason="Direct outcomes may improve evidence coverage")
        return GapDecision(action="ACCEPT_GAPS", reason="The remaining formal gap should be preserved")

    def fake_search(request: Any) -> SearchAttempt:
        now = utc_now()
        attempt_dir = run_dir / "analysis/agent_attempts" / request.claim_id / request.attempt_id
        graph_path = attempt_dir / "final_graph.json"
        atomic_write_json(graph_path, {"claim_id": request.claim_id, "query": request.query, "nodes": []})
        return SearchAttempt(
            attempt_id=request.attempt_id,
            claim_id=request.claim_id,
            query=request.query,
            normalized_query=normalize_query(request.query),
            artifact_dir=str(attempt_dir),
            graph_bundle_path=str(graph_path),
            verdict="supported",
            article_counts={"total": 1, "support": 1, "refute": 0, "insufficient": 0},
            article_ids=[f"article:{request.attempt_id}"],
            new_article_ids=[f"article:{request.attempt_id}"],
            direct_evidence_articles=1,
            utility_score=121,
            status="successful",
            started_at=now,
            finished_at=now,
        )

    connection = sqlite3.connect(agent_dir / "checkpoints.sqlite", check_same_thread=False)
    try:
        graph = build_agent_graph(
            controller=evidence_controller,
            gap_controller=gap_controller_override or gap_controller,
            search_tool=create_search_evidence_tool(fake_search),
            stages=stages,
            run_name="fake_agent_demo",
            max_steps=12,
            total_search_budget=4,
            per_claim_search_budget=3,
            max_gap_rounds=2,
            gap_remediation_budget=1,
            trace_writer=trace,
            checkpointer=SqliteSaver(connection),
        )
        (agent_dir / "execution_graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        state = graph.invoke(
            {"node_history": []},
            config={"configurable": {"thread_id": "fake-agent-demo"}, "recursion_limit": 100},
        )
    finally:
        connection.close()
    return run_dir, dict(state["final_result"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the resource-free full EvidenceGap Agent graph")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    target = args.output_dir or Path(tempfile.mkdtemp(prefix="evidencegap-agent-demo-"))
    artifact_dir, result = run_demo(target)
    print("START")
    for node in result["node_history"]:
        print(f"→ {node}")
    print("→ END")
    print(f"Artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
