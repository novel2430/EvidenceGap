from __future__ import annotations

import argparse
import sqlite3
import tempfile
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.schemas import AgentDecision, SearchAttempt
from evidencegap_backend.agent.tools import create_search_evidence_tool
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.agent.workspace import (
    initialize_workspace,
    normalize_query,
    utc_now,
)
from evidencegap_backend.common import atomic_write_json
from evidencegap_backend.pipeline.retrieval_adapters import runtime_claim_id


def run_demo(output_dir: Path) -> tuple[Path, dict]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = AgentTraceWriter(output_dir)
    claims = ["Drug A reduces biomarker B.", "Drug A prevents every complication."]
    decomposition = {
        "original_statement": " ".join(claims),
        "source_language": "English",
        "claims": [
            {
                "claim_id": runtime_claim_id(text),
                "source_text": text,
                "source_spans": [
                    {
                        "start": sum(len(x) + 1 for x in claims[:i]),
                        "end": sum(len(x) + 1 for x in claims[:i]) + len(text),
                    }
                ],
                "canonical_claim_en": text,
            }
            for i, text in enumerate(claims)
        ],
    }
    workspace = initialize_workspace(
        run_name="fake_agent_demo",
        statement=decomposition["original_statement"],
        language="English",
        decomposition=decomposition,
        max_steps=12,
        total_search_budget=4,
        per_claim_search_budget=2,
    )

    def controller(ws, _note=None):
        first, second = (ws.claims[c] for c in ws.claim_order)
        if first.status == "pending" and not first.attempts:
            return AgentDecision(
                action="SEARCH",
                claim_id=first.claim_id,
                query="Drug A biomarker B randomized trial",
                reason="Establish direct evidence for the first claim",
            )
        if first.status == "pending":
            return AgentDecision(
                action="RESOLVE",
                claim_id=first.claim_id,
                reason="The successful attempt provides direct evidence",
            )
        if second.status == "pending" and not second.attempts:
            return AgentDecision(
                action="SEARCH",
                claim_id=second.claim_id,
                query="Drug A all complications clinical outcomes",
                reason="Test the broad outcome claim directly",
            )
        if second.status == "pending":
            return AgentDecision(
                action="ABSTAIN",
                claim_id=second.claim_id,
                reason="The broad claim remains insufficient and further search has low value",
            )
        return AgentDecision(action="FINISH", reason="All claims are terminal")

    def fake_search(request):
        now = utc_now()
        resolved = request.claim_id == workspace.claim_order[0]
        return SearchAttempt(
            attempt_id=request.attempt_id,
            claim_id=request.claim_id,
            query=request.query,
            normalized_query=normalize_query(request.query),
            verdict="supported" if resolved else "insufficient",
            article_counts={
                "total": 2,
                "support": 1 if resolved else 0,
                "refute": 0,
                "insufficient": 1 if resolved else 2,
            },
            article_ids=[
                f"pmid:{request.attempt_id}:1",
                f"pmid:{request.attempt_id}:2",
            ],
            new_article_ids=[
                f"pmid:{request.attempt_id}:1",
                f"pmid:{request.attempt_id}:2",
            ],
            direct_evidence_articles=1 if resolved else 0,
            utility_score=122 if resolved else 2,
            status="successful",
            started_at=now,
            finished_at=now,
        )

    connection = sqlite3.connect(
        output_dir / "checkpoints.sqlite", check_same_thread=False
    )
    try:
        graph = build_agent_graph(
            controller=controller,
            search_tool=create_search_evidence_tool(fake_search),
            trace_writer=trace,
            checkpointer=SqliteSaver(connection),
        )
        (output_dir / "execution_graph.mmd").write_text(
            graph.get_graph().draw_mermaid(), encoding="utf-8"
        )
        result = graph.invoke(
            {"workspace": workspace.model_dump(mode="json"), "node_history": []},
            config={"configurable": {"thread_id": "fake_agent_demo"}},
        )
    finally:
        connection.close()
    trace.write_workspace(result["workspace"])
    atomic_write_json(
        output_dir / "agent_manifest.json",
        {
            "schema_version": "1.0.0",
            "contract_id": "evidencegap.agent-harness.v1",
            "execution_mode": "langgraph_agent",
            "checkpoint_backend": "sqlite",
            "action_counts": result["workspace"]["action_counts"],
        },
    )
    return output_dir, result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the real LangGraph harness with fake controller and search"
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    target = args.output_dir or Path(tempfile.mkdtemp(prefix="evidencegap-agent-demo-"))
    artifact_dir, result = run_demo(target)
    print("START → " + " → ".join(result["node_history"]) + " → END")
    print(f"Agent artifacts: {artifact_dir}")


if __name__ == "__main__":
    main()
