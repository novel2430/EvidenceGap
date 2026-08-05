from __future__ import annotations

import re
import sqlite3
import time
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping

from langgraph.checkpoint.sqlite import SqliteSaver

from evidencegap_backend.agent.controller import EvidenceController
from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.schemas import EvidenceWorkspace
from evidencegap_backend.agent.tools import (
    create_analysis_executor,
    create_search_evidence_tool,
)
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.agent.workspace import initialize_workspace
from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_write_json,
    load_json,
    relative_path,
    require_empty_or_force,
    sha256_file,
)
from evidencegap_backend.config import AgentConfig, LLMStageConfig, PipelineConfig
from evidencegap_backend.pipeline.statement_analysis import (
    STATEMENT_ANALYSIS_CONTRACT_ID,
    STATEMENT_ANALYSIS_SCHEMA_VERSION,
    validate_statement_analysis_artifact,
    validate_statement_analysis_bundle,
)
from evidencegap_backend.pipeline.statement_decomposition import (
    validate_decomposition_bundle,
    validate_statement_decomposition_artifact,
)
from evidencegap_backend.pipeline.statement_run import run_statement_pipeline


def _safe(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty")
    return cleaned


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run_agent_statement_analysis(
    root: Path,
    *,
    decomposition_artifact_dir: Path,
    run_name: str,
    provider: str,
    model: str | None = None,
    artifact_root: Path | None = None,
    decomposition_bundle: Mapping[str, Any] | None = None,
    progress_callback: Callable[..., None] | None = None,
    force: bool = False,
    controller_config: LLMStageConfig,
    agent_config: AgentConfig,
    **analysis_kwargs: Any,
) -> dict[str, Any]:
    root = root.resolve()
    started = time.perf_counter()
    decomposition_dir = _resolve(root, decomposition_artifact_dir)
    validate_statement_decomposition_artifact(decomposition_dir)
    decomposition = (
        dict(decomposition_bundle)
        if decomposition_bundle is not None
        else load_json(decomposition_dir / "decomposition.json")
    )
    decomposition_validation = validate_decomposition_bundle(decomposition)
    name = _safe(run_name)
    base = artifact_root.resolve() if artifact_root else root
    target = base / name
    require_empty_or_force(target, force=force)
    target.mkdir(parents=True, exist_ok=False)
    attempts_root = target / "agent_attempts"
    agent_dir = base / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    trace = AgentTraceWriter(agent_dir)
    workspace = initialize_workspace(
        run_name=name,
        statement=str(decomposition["original_statement"]),
        language=str(decomposition["source_language"]),
        decomposition=decomposition,
        max_steps=agent_config.max_steps,
        total_search_budget=agent_config.total_search_budget,
        per_claim_search_budget=agent_config.per_claim_search_budget,
    )
    pipeline_settings = analysis_kwargs.get("pipeline_config") or PipelineConfig()
    executor_kwargs = dict(analysis_kwargs)
    executor_kwargs.pop("decomposition_bundle", None)
    executor_kwargs.pop("progress_callback", None)
    executor_kwargs.pop("force", None)
    executor_kwargs.update({"provider": provider, "model": model})
    tool = create_search_evidence_tool(
        create_analysis_executor(
            root=root, analysis_kwargs=executor_kwargs, attempts_root=attempts_root
        )
    )
    controller = EvidenceController(controller_config)

    def progress_finalize(ws: EvidenceWorkspace) -> None:
        if progress_callback:
            progress_callback(
                sum(c.status != "pending" for c in ws.claims.values()), len(ws.claims)
            )

    def action_progress(ws: EvidenceWorkspace, decision: Any) -> None:
        if not progress_callback:
            return
        completed = sum(c.status != "pending" for c in ws.claims.values())
        claim_label = decision.claim_id or "all claims"
        detail = f" with query: {decision.query}" if decision.query else ""
        progress_callback(
            completed,
            len(ws.claims),
            f"Agent step {ws.step_count + 1}: {decision.action.value.lower()} {claim_label}{detail}",
        )

    checkpoint_path = agent_dir / "checkpoints.sqlite"
    connection: sqlite3.Connection | None = None
    try:
        checkpointer = None
        if agent_config.checkpoint_enabled:
            connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            checkpointer = SqliteSaver(connection)
        graph = build_agent_graph(
            controller=controller,
            search_tool=tool,
            controller_retry_count=agent_config.controller_retry_count,
            trace_writer=trace,
            checkpointer=checkpointer,
            finalize=progress_finalize,
            action_callback=action_progress,
        )
        (agent_dir / "execution_graph.mmd").write_text(
            graph.get_graph().draw_mermaid(), encoding="utf-8"
        )
        final_state = graph.invoke(
            {"workspace": workspace.model_dump(mode="json"), "node_history": []},
            config={
                "configurable": {"thread_id": name},
                "recursion_limit": max(50, agent_config.max_steps * 4 + 20),
            },
        )
    finally:
        if connection is not None:
            connection.close()
    final_workspace = EvidenceWorkspace.model_validate(final_state["workspace"])
    trace.write_workspace(final_workspace.model_dump(mode="json"))

    claim_results: list[dict[str, Any]] = []
    graph_bundles: dict[str, dict[str, Any]] = {}
    for claim_id in final_workspace.claim_order:
        claim = final_workspace.claims[claim_id]
        selected = next(
            (
                a
                for a in claim.attempts
                if a.attempt_id == claim.selected_attempt_id
                and a.status == "successful"
            ),
            None,
        )
        if selected is None:
            claim_results.append(
                {
                    "claim_id": claim_id,
                    "source_text": claim.source_text,
                    "source_spans": claim.source_spans,
                    "canonical_claim_en": claim.canonical_claim_en,
                    "status": "failed",
                    "phase07_artifact_dir": None,
                    "graph_bundle_path": None,
                    "verdict": None,
                    "error": claim.last_error
                    or claim.terminal_reason
                    or "No successful evidence attempt",
                    "agent_status": claim.status,
                }
            )
        else:
            graph_path = _resolve(root, str(selected.graph_bundle_path))
            graph_bundles[claim_id] = load_json(graph_path)
            claim_results.append(
                {
                    "claim_id": claim_id,
                    "source_text": claim.source_text,
                    "source_spans": claim.source_spans,
                    "canonical_claim_en": claim.canonical_claim_en,
                    "status": "completed",
                    "phase07_artifact_dir": selected.artifact_dir,
                    "graph_bundle_path": selected.graph_bundle_path,
                    "verdict": selected.verdict,
                    "error": None,
                    "agent_status": claim.status,
                    "selected_attempt_id": selected.attempt_id,
                }
            )
    completed = sum(x["status"] == "completed" for x in claim_results)
    failed = len(claim_results) - completed
    analysis_status = (
        "completed"
        if not failed
        else ("failed" if not completed else "partial_failure")
    )
    bundle = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "statement_id": decomposition_validation["statement_id"],
        "original_statement": decomposition["original_statement"],
        "source_language": decomposition["source_language"],
        "analysis_status": analysis_status,
        "analysis_context": pipeline_settings.analysis_context(),
        "claim_results": claim_results,
        "summary": {
            "total_claims": len(claim_results),
            "completed_claims": completed,
            "failed_claims": failed,
        },
    }
    validate_statement_analysis_bundle(bundle)
    result_path = target / "statement_result.json"
    atomic_write_json(result_path, bundle)
    decomposition_path = decomposition_dir / "decomposition.json"
    request = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "run_name": name,
        "decomposition_artifact_dir": relative_path(root, decomposition_dir),
        "decomposition_path": relative_path(root, decomposition_path),
        "decomposition_sha256": sha256_file(decomposition_path),
    }
    atomic_write_json(target / "request.json", request)
    manifest = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "run_type": "langgraph_agent_multi_claim_analysis",
        "run_name": name,
        "statement_id": decomposition_validation["statement_id"],
        "analysis_status": analysis_status,
        "execution": {
            "provider": provider,
            "model": model,
            "orchestration": "langgraph",
            "decomposition_handoff": "in_memory_handoff"
            if decomposition_bundle
            else "artifact_reload",
        },
        "counts": dict(bundle["summary"]),
        "source": {
            "decomposition_artifact_dir": relative_path(root, decomposition_dir),
            "decomposition": {
                "path": relative_path(root, decomposition_path),
                "sha256": sha256_file(decomposition_path),
            },
        },
        "outputs": {
            "statement_result": {
                "path": relative_path(root, result_path),
                "sha256": sha256_file(result_path),
            }
        },
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_write_json(target / "run_manifest.json", manifest)
    validate_statement_analysis_artifact(target)
    agent_manifest = {
        "schema_version": "1.0.0",
        "contract_id": "evidencegap.agent-harness.v1",
        "run_name": name,
        "execution_mode": "langgraph_agent",
        "langgraph_enabled": True,
        "checkpoint_backend": "sqlite"
        if agent_config.checkpoint_enabled
        else "disabled",
        "checkpoint_path": str(checkpoint_path)
        if agent_config.checkpoint_enabled
        else None,
        "controller": {
            "provider": controller_config.provider,
            "model": controller_config.model,
        },
        "max_steps": agent_config.max_steps,
        "initial_search_budget": final_workspace.initial_search_budget,
        "remaining_search_budget": final_workspace.remaining_search_budget,
        "action_counts": final_workspace.action_counts,
        "search_attempt_count": sum(
            len(c.attempts) for c in final_workspace.claims.values()
        ),
        "resolved_claims": sum(
            c.status == "resolved" for c in final_workspace.claims.values()
        ),
        "abstained_claims": sum(
            c.status == "abstained" for c in final_workspace.claims.values()
        ),
        "failed_claims": sum(
            c.status == "failed" for c in final_workspace.claims.values()
        ),
        "artifacts": {
            "workspace": str(agent_dir / "workspace.json"),
            "action_trace": str(trace.path),
            "execution_graph": str(agent_dir / "execution_graph.mmd"),
        },
        "total_seconds": round(time.perf_counter() - started, 6),
    }
    atomic_write_json(agent_dir / "agent_manifest.json", agent_manifest)
    return {
        **validate_statement_analysis_bundle(bundle),
        "status": analysis_status.upper(),
        "artifact_status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "statement_result_path": relative_path(root, result_path),
        "statement_result": bundle,
        "decomposition": decomposition,
        "claim_graph_bundles": graph_bundles,
    }


def run_agent_statement_pipeline(
    root: Path,
    *,
    agent_config: AgentConfig,
    controller_config: LLMStageConfig,
    **kwargs: Any,
) -> dict[str, Any]:
    runner = partial(
        run_agent_statement_analysis,
        controller_config=controller_config,
        agent_config=agent_config,
    )
    return run_statement_pipeline(root, statement_analysis_runner=runner, **kwargs)
