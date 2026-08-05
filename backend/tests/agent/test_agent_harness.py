from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import ValidationError

from evidencegap_backend.agent.demo_fake import FakeStageExecutor, run_demo
from evidencegap_backend.agent.gap_controller import compact_gap_summary
from evidencegap_backend.agent.gap_policies import InvalidGapDecision, validate_gap_decision
from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.policies import (
    InvalidDecision,
    deterministic_fallback,
    validate_decision,
)
from evidencegap_backend.agent.schemas import (
    AgentDecision,
    EvidenceWorkspace,
    GapDecision,
    SearchAttempt,
)
from evidencegap_backend.agent.tools import create_search_evidence_tool
from evidencegap_backend.agent.stages import AgentRuntimeContext, ProductionStageExecutor
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.agent.workspace import (
    initialize_workspace,
    normalize_query,
    register_attempt,
)
from evidencegap_backend.pipeline.analysis import run_analysis
from evidencegap_backend.config import AgentConfig, LLMStageConfig, PipelineConfig
from evidencegap_backend.common import sha256_file
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
        deterministic_fallback(ws, reason="budget exhausted").action.value
        == "RESOLVE"
    )


@pytest.mark.parametrize("verdict", ["supported", "insufficient"])
def test_fallback_resolves_any_successful_formal_verdict(verdict: str) -> None:
    ws = workspace(total_search_budget=0)
    cid = ws.claim_order[0]
    register_attempt(ws, attempt(cid, verdict=verdict))

    decision = deterministic_fallback(ws, reason="controller failed")

    assert decision.action.value == "RESOLVE"
    assert decision.claim_id == cid
    assert decision.selected_attempt_id == "attempt_001"
    assert decision.reason == "Select the best successful attempt deterministically."


def test_fallback_abstains_only_without_success_and_search_path() -> None:
    ws = workspace(total_search_budget=0)
    decision = deterministic_fallback(ws, reason="controller failed")

    assert decision.action.value == "ABSTAIN"
    assert decision.selected_attempt_id is None
    assert decision.remaining_problem == "No successful attempt is available."
    assert validate_decision(ws, decision) == decision


def test_finish_cannot_skip_pending_successful_claim() -> None:
    ws = workspace(total_search_budget=0)
    cid = ws.claim_order[0]
    register_attempt(ws, attempt(cid, verdict="supported"))

    with pytest.raises(InvalidDecision, match="terminal"):
        validate_decision(ws, AgentDecision(action="FINISH", reason="too early"))

    resolve = deterministic_fallback(ws, reason="invalid early finish")
    assert resolve.action.value == "RESOLVE"
    ws.claims[cid].status = "resolved"
    finish = deterministic_fallback(ws, reason="ignored")
    assert finish == AgentDecision(
        action="FINISH", reason="All claims are terminal."
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


def test_demo_uses_runtime_claim_id_from_defining_module() -> None:
    from evidencegap_backend.agent import demo_fake

    assert demo_fake.runtime_claim_id.__module__ == (
        "evidencegap_backend.pipeline.retrieval_adapters"
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
        "initialize_run",
        "statement_decomposition",
        "initialize_workspace",
        "evidence_controller",
        "search_evidence",
        "evidence_controller",
        "resolve_claim",
        "evidence_controller",
        "evidence_finish",
        "materialize_statement_analysis",
        "build_statement_bundle",
        "run_inference_gap_analysis",
        "gap_controller",
        "reopen_claim",
        "execute_gap_requested_search",
        "evidence_controller",
        "resolve_claim",
        "evidence_controller",
        "evidence_finish",
        "materialize_statement_analysis",
        "build_statement_bundle",
        "run_inference_gap_analysis",
        "gap_controller",
        "generate_output",
        "finalize_run",
    ]
    assert result["stage_calls"] == {
        "analysis": 2,
        "bundle": 2,
        "gap": 2,
        "output": 1,
    }
    agent_dir = artifact_dir / "agent"
    assert {p.name for p in agent_dir.iterdir()} >= {
        "workspace.json",
        "action_trace.jsonl",
        "agent_manifest.json",
        "execution_graph.mmd",
        "checkpoints.sqlite",
    }
    lines = [json.loads(line) for line in (agent_dir / "action_trace.jsonl").read_text().splitlines()]
    assert any(row["action"] == "REQUEST_MORE_EVIDENCE" for row in lines)
    assert any(row["action"] == "ACCEPT_GAPS" for row in lines)
    assert any(row.get("query") == "Drug A direct clinical outcomes randomized trial" for row in lines)
    with sqlite3.connect(agent_dir / "checkpoints.sqlite") as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table'"
            )
        }
        assert "checkpoints" in tables
        assert connection.execute("select count(*) from checkpoints").fetchone()[0] > 0
        checkpoints = list(
            SqliteSaver(connection).list(
                {"configurable": {"thread_id": "fake-agent-demo"}}
            )
        )
        checkpoint_histories = [
            row.checkpoint.get("channel_values", {}).get("node_history", [])
            for row in checkpoints
        ]
        for node in (
            "statement_decomposition",
            "materialize_statement_analysis",
            "build_statement_bundle",
            "run_inference_gap_analysis",
            "gap_controller",
            "generate_output",
        ):
            assert any(node in nodes for nodes in checkpoint_histories)
    workspace_text = (agent_dir / "workspace.json").read_text()
    assert "RuntimeResources" not in workspace_text
    final_workspace = json.loads(workspace_text)
    assert final_workspace["phase"] == "finished"
    assert final_workspace["final_output_path"].endswith(
        "output/presentation_bundle.json"
    )
    reopened = next(iter(final_workspace["claims"].values()))
    assert reopened["status"] == "resolved"
    assert reopened["verdict"] == "supported"
    assert reopened["reopen_count"] == 1
    assert reopened["canonical_claim_en"] == "Drug A reduces biomarker B."
    assert len(reopened["attempts"]) == 2
    assert reopened["used_queries"] == [
        "Drug A biomarker B randomized trial",
        "Drug A direct clinical outcomes randomized trial",
    ]
    assert sorted((agent_dir / "gap_rounds").glob("round_[0-9][0-9][0-9].json")) == [
        agent_dir / "gap_rounds/round_001.json",
        agent_dir / "gap_rounds/round_002.json",
    ]
    assert sorted((agent_dir / "gap_rounds").glob("round_*_gap_summary.json")) == [
        agent_dir / "gap_rounds/round_001_gap_summary.json",
        agent_dir / "gap_rounds/round_002_gap_summary.json",
    ]
    round_one = json.loads((agent_dir / "gap_rounds/round_001.json").read_text())
    round_two = json.loads((agent_dir / "gap_rounds/round_002.json").read_text())
    assert round_one["analysis_sha256"] != round_two["analysis_sha256"]
    assert set(round_one["selected_attempts"].values()) == {"attempt_001"}
    assert set(round_two["selected_attempts"].values()) == {"attempt_002"}
    assert round_one["gap_summary"]["gaps"][0]["gap_type"] == "scope_gap"
    assert round_one["gap_summary"]["gaps"][0]["reason"] == (
        "Direct clinical outcomes were not retrieved"
    )
    assert round_two["gap_summary"]["gaps"] == []
    assert sha256_file(
        Path(round_one["gap_summary_artifact"]["path"])
    ) == round_one["gap_summary_artifact"]["sha256"]


def test_gap_accept_path_runs_each_downstream_stage_once(tmp_path: Path) -> None:
    _artifact_dir, result = run_demo(tmp_path, request_remediation=False)
    assert result["stage_calls"] == {
        "analysis": 1,
        "bundle": 1,
        "gap": 1,
        "output": 1,
    }
    assert "reopen_claim" not in result["node_history"]


def test_gap_controller_failure_uses_deterministic_abstain(tmp_path: Path) -> None:
    def broken_gap_controller(*_args):
        raise ValueError("invalid JSON")

    artifact_dir, result = run_demo(
        tmp_path,
        request_remediation=False,
        gap_controller_override=broken_gap_controller,
    )
    final_workspace = json.loads(
        (artifact_dir / "agent/workspace.json").read_text(encoding="utf-8")
    )
    assert result["stage_calls"]["output"] == 1
    assert final_workspace["gap_decision"]["action"] == "ABSTAIN"
    assert final_workspace["gap_decision_source"] == "deterministic_fallback"
    assert final_workspace["rejected_gap_decisions"] == 3
    assert final_workspace["deterministic_fallback_decisions"] == 1


def test_production_stage_executor_calls_real_downstream_functions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"bundle": 0, "gap": 0, "output": 0}

    def fake_bundle(*_args, **_kwargs):
        calls["bundle"] += 1
        return {"statement_bundle": {"round": 1}}

    def fake_gap(*_args, **_kwargs):
        calls["gap"] += 1
        return {"inference_gap_bundle": {"inference_gap_analyses": []}}

    def fake_output(*_args, **_kwargs):
        calls["output"] += 1
        return {
            "presentation_bundle": {},
            "output": {
                "presentation_bundle": {
                    "path": "output/presentation_bundle.json",
                    "sha256": "fake-sha",
                }
            },
        }

    monkeypatch.setattr("evidencegap_backend.agent.stages.run_statement_bundle", fake_bundle)
    monkeypatch.setattr("evidencegap_backend.agent.stages.run_inference_gap_analysis", fake_gap)
    monkeypatch.setattr("evidencegap_backend.agent.stages.run_output_module", fake_output)
    stage = LLMStageConfig(provider="deepseek", model="fake", api_key_env="FAKE")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executor = ProductionStageExecutor(
        AgentRuntimeContext(
            root=tmp_path,
            run_dir=run_dir,
            run_name="run",
            statement="A reduces B.",
            language="English",
            stage_configs={name: stage for name in (
                "statement_decomposition", "article_evidence", "inference_gap",
                "localization", "agent_controller", "agent_gap_controller"
            )},
            pipeline_config=PipelineConfig(),
            agent_config=AgentConfig(checkpoint_enabled=False),
            trace_writer=AgentTraceWriter(run_dir / "agent"),
        )
    )
    ws = workspace()
    bundle = executor.build_statement_bundle(ws, {
        "decomposition": ws.decomposition,
        "statement_result": {},
        "claim_graph_bundles": {},
    })
    gap = executor.run_inference_gap_analysis(ws, bundle)
    output = executor.generate_output(ws, bundle, gap)
    assert calls == {"bundle": 1, "gap": 1, "output": 1}
    assert output["presentation_bundle_path"] == "output/presentation_bundle.json"


def test_controller_failures_use_traced_deterministic_fallback(tmp_path: Path) -> None:
    def broken_controller(_workspace, _note=None):
        raise ValueError("invalid JSON")

    def search(request):
        value = attempt(request.claim_id, query=request.query)
        attempt_dir = tmp_path / "analysis/agent_attempts" / request.claim_id / request.attempt_id
        graph_path = attempt_dir / "final_graph.json"
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text('{"nodes": []}', encoding="utf-8")
        return value.model_copy(update={
            "artifact_dir": str(attempt_dir),
            "graph_bundle_path": str(graph_path),
        })

    from evidencegap_backend.agent.tracing import AgentTraceWriter

    trace = AgentTraceWriter(tmp_path / "agent")
    stages = FakeStageExecutor(tmp_path, trace)
    graph = build_agent_graph(
        controller=broken_controller,
        gap_controller=lambda *_args: GapDecision(action="ACCEPT_GAPS", reason="accept formal gaps"),
        search_tool=create_search_evidence_tool(search),
        stages=stages,
        run_name="fallback",
        total_search_budget=1,
        controller_retry_count=1,
        trace_writer=trace,
    )
    result = graph.invoke({"node_history": []}, config={"recursion_limit": 80})
    assert result["workspace"]["status"] == "finished"
    assert result["workspace"]["rejected_decisions"] == 6
    events = [json.loads(line) for line in trace.path.read_text().splitlines()]
    evidence_actions = [
        row
        for row in events
        if row.get("controller_type") == "evidence_controller"
        and row.get("event") in {"action_result", "evidence_cycle_finished"}
    ]
    assert [row["action"] for row in evidence_actions] == ["SEARCH", "RESOLVE", "FINISH"]
    assert {row["decision_source"] for row in evidence_actions} == {"deterministic_fallback"}
    final_claim = next(iter(result["workspace"]["claims"].values()))
    assert final_claim["status"] == "resolved"
    assert final_claim["verdict"] == "supported"


def test_gap_policy_rejects_unknown_duplicate_and_exhausted_requests() -> None:
    ws = workspace(max_gap_rounds=2, gap_remediation_budget=1)
    cid = ws.claim_order[0]
    register_attempt(ws, attempt(cid))
    ws.claims[cid].status = "resolved"
    gap_bundle = {
        "inference_gap_analyses": [{
            "inference_step_id": "step_1",
            "scope_gap": {"detected": True, "reason": "missing outcome"},
            "causal_gap": {"detected": False},
        }]
    }
    gap_id = compact_gap_summary(ws, gap_bundle)["gaps"][0]["gap_id"]
    with pytest.raises(InvalidGapDecision, match="target_gap_id"):
        validate_gap_decision(ws, gap_bundle, GapDecision(action="REQUEST_MORE_EVIDENCE", target_gap_id="missing", claim_id=cid, query="new query", reason="test"))
    with pytest.raises(InvalidGapDecision, match="claim_id"):
        validate_gap_decision(ws, gap_bundle, GapDecision(action="REQUEST_MORE_EVIDENCE", target_gap_id=gap_id, claim_id="unknown", query="new query", reason="test"))
    with pytest.raises(InvalidGapDecision, match="duplicates"):
        validate_gap_decision(ws, gap_bundle, GapDecision(action="REQUEST_MORE_EVIDENCE", target_gap_id=gap_id, claim_id=cid, query="A B trial", reason="test"))
    ws.remaining_gap_remediation_budget = 0
    with pytest.raises(InvalidGapDecision, match="remediation budget"):
        validate_gap_decision(ws, gap_bundle, GapDecision(action="REQUEST_MORE_EVIDENCE", target_gap_id=gap_id, claim_id=cid, query="new query", reason="test"))
    ws.remaining_gap_remediation_budget = 1
    ws.gap_round = ws.max_gap_rounds
    with pytest.raises(InvalidGapDecision, match="maximum gap rounds"):
        validate_gap_decision(ws, gap_bundle, GapDecision(action="REQUEST_MORE_EVIDENCE", target_gap_id=gap_id, claim_id=cid, query="new query", reason="test"))
