from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidencegap_backend.agent.demo_fake import run_demo
from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.policies import (
    InvalidDecision,
    deterministic_fallback,
    validate_decision,
)
from evidencegap_backend.agent.schemas import (
    AgentDecision,
    EvidenceWorkspace,
    SearchAttempt,
)
from evidencegap_backend.agent.tools import create_search_evidence_tool
from evidencegap_backend.agent.workspace import (
    initialize_workspace,
    normalize_query,
    register_attempt,
)
from evidencegap_backend.pipeline.analysis import run_analysis
from evidencegap_backend.pipeline.retrieval_adapters import (
    retrieve_runtime_articles,
    runtime_claim_id,
)


def decomposition() -> dict:
    statement = "A reduces B."
    return {
        "statement_id": "statement_fixture",
        "original_statement": statement,
        "source_language": "English",
        "claims": [
            {
                "claim_id": runtime_claim_id(statement),
                "source_text": statement,
                "source_spans": [{"start": 0, "end": len(statement)}],
                "canonical_claim_en": statement,
            }
        ],
        "inference_steps": [],
    }


def workspace(**overrides) -> EvidenceWorkspace:
    values = {
        "run_name": "test",
        "statement": "A reduces B.",
        "language": "English",
        "decomposition": decomposition(),
        "max_steps": 10,
        "total_search_budget": 2,
        "per_claim_search_budget": 2,
    }
    values.update(overrides)
    return initialize_workspace(**values)


def attempt(
    claim_id: str, *, query: str = "A B trial", verdict: str = "supported"
) -> SearchAttempt:
    return SearchAttempt(
        attempt_id="attempt_001",
        claim_id=claim_id,
        query=query,
        normalized_query=normalize_query(query),
        verdict=verdict,
        article_counts={"total": 1, "support": 1},
        article_ids=["pmid:1"],
        new_article_ids=["pmid:1"],
        direct_evidence_articles=1,
        utility_score=121,
        status="successful",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
    )


def test_decision_schema_and_policy_guards() -> None:
    ws = workspace()
    cid = ws.claim_order[0]
    with pytest.raises(ValidationError):
        AgentDecision(action="SEARCH", claim_id=cid, reason="missing query")
    with pytest.raises(InvalidDecision, match="successful"):
        validate_decision(
            ws, AgentDecision(action="RESOLVE", claim_id=cid, reason="too early")
        )
    with pytest.raises(InvalidDecision, match="terminal"):
        validate_decision(ws, AgentDecision(action="FINISH", reason="too early"))
    register_attempt(ws, attempt(cid))
    with pytest.raises(InvalidDecision, match="selected_attempt_id"):
        validate_decision(
            ws,
            AgentDecision(
                action="RESOLVE",
                claim_id=cid,
                selected_attempt_id="missing",
                reason="bad selection",
            ),
        )
    with pytest.raises(InvalidDecision, match="duplicates"):
        validate_decision(
            ws,
            AgentDecision(
                action="SEARCH", claim_id=cid, query=" A   B TRIAL ", reason="duplicate"
            ),
        )
    ws.remaining_search_budget = 0
    with pytest.raises(InvalidDecision, match="total search budget"):
        validate_decision(
            ws,
            AgentDecision(
                action="SEARCH", claim_id=cid, query="different", reason="no budget"
            ),
        )
    assert (
        deterministic_fallback(ws, reason="budget exhausted").action.value == "ABSTAIN"
    )


def test_workspace_json_round_trip_and_terminal_fields() -> None:
    ws = workspace()
    cid = ws.claim_order[0]
    register_attempt(ws, attempt(cid))
    restored = EvidenceWorkspace.model_validate_json(ws.model_dump_json())
    assert restored.claims[cid].seen_article_ids == ["pmid:1"]
    assert restored.claims[cid].selected_attempt_id == "attempt_001"
    assert restored.claims[cid].used_queries == ["A B trial"]


def test_query_decoupling_is_explicit_and_backward_compatible() -> None:
    assert inspect.signature(run_analysis).parameters["retrieval_query"].default is None
    assert (
        inspect.signature(retrieve_runtime_articles).parameters["query_text"].default
        is None
    )
    source = inspect.getsource(retrieve_runtime_articles)
    assert "bm25_backend.search(query_text" in source
    assert "claim_text=query_text" in source
    analysis_source = inspect.getsource(run_analysis)
    assert (
        '"prompt_request": {\n                "claim_id": claim_id,\n                "claim_text": claim_text'
        in analysis_source
    )


def test_real_langgraph_loop_checkpoint_trace_and_artifacts(tmp_path: Path) -> None:
    artifact_dir, result = run_demo(tmp_path)
    assert result["node_history"] == [
        "initialize_workspace",
        "controller",
        "search_evidence",
        "controller",
        "resolve_claim",
        "controller",
        "search_evidence",
        "controller",
        "abstain_claim",
        "controller",
        "finalize_statement_analysis",
        "build_statement_bundle",
        "inference_gap_analysis",
        "generate_output",
    ]
    assert result["workspace"]["action_counts"] == {
        "SEARCH": 2,
        "RESOLVE": 1,
        "ABSTAIN": 1,
        "FINISH": 1,
    }
    assert {p.name for p in artifact_dir.iterdir()} >= {
        "workspace.json",
        "action_trace.jsonl",
        "agent_manifest.json",
        "execution_graph.mmd",
        "checkpoints.sqlite",
    }
    lines = [
        json.loads(line)
        for line in (artifact_dir / "action_trace.jsonl").read_text().splitlines()
    ]
    assert [row["action"] for row in lines] == [
        "SEARCH",
        "RESOLVE",
        "SEARCH",
        "ABSTAIN",
        "FINISH",
    ]
    with sqlite3.connect(artifact_dir / "checkpoints.sqlite") as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        assert "checkpoints" in tables
        assert connection.execute("select count(*) from checkpoints").fetchone()[0] > 0
    assert "RuntimeResources" not in (artifact_dir / "workspace.json").read_text()


def test_controller_failures_use_traced_deterministic_fallback(tmp_path: Path) -> None:
    ws = workspace(total_search_budget=1)

    def broken_controller(_workspace, _note=None):
        raise ValueError("invalid JSON")

    def search(request):
        return attempt(request.claim_id, query=request.query)

    from evidencegap_backend.agent.tracing import AgentTraceWriter

    trace = AgentTraceWriter(tmp_path)
    graph = build_agent_graph(
        controller=broken_controller,
        search_tool=create_search_evidence_tool(search),
        controller_retry_count=1,
        trace_writer=trace,
    )
    result = graph.invoke({"workspace": ws.model_dump(mode="json"), "node_history": []})
    assert result["workspace"]["status"] == "finished"
    assert result["workspace"]["rejected_decisions"] == 6
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    assert [row["action"] for row in events] == ["SEARCH", "ABSTAIN", "FINISH"]
    assert {row["decision_source"] for row in events} == {"deterministic_fallback"}
