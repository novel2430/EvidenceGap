from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Protocol

from evidencegap_backend.agent.artifacts import (
    gap_round_record,
    materialize_statement_analysis,
)
from evidencegap_backend.agent.schemas import EvidenceWorkspace, GapDecision
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_write_json,
    relative_path,
    sha256_file,
    sha256_text,
)
from evidencegap_backend.config import AgentConfig, LLMStageConfig, PipelineConfig
from evidencegap_backend.output.presentation import run_output_module
from evidencegap_backend.pipeline.inference_gap_analysis import (
    run_inference_gap_analysis,
)
from evidencegap_backend.pipeline.statement_bundle import run_statement_bundle
from evidencegap_backend.pipeline.statement_decomposition import (
    run_statement_decomposition,
)
from evidencegap_backend.pipeline.statement_run import (
    STATEMENT_RUN_CONTRACT_ID,
    STATEMENT_RUN_SCHEMA_VERSION,
    _STAGE_NAMES,
    _stage_execution_config,
    _stage_meta,
    build_execution_summary,
    validate_statement_pipeline_artifact,
)

if TYPE_CHECKING:
    from evidencegap_backend.resources import RuntimeResources

ProgressCallback = Callable[[Mapping[str, Any]], None]


class AgentStageExecutor(Protocol):
    def initialize_run(self) -> dict[str, Any]: ...

    def statement_decomposition(self) -> dict[str, Any]: ...

    def materialize_statement_analysis(
        self, workspace: EvidenceWorkspace
    ) -> dict[str, Any]: ...

    def build_statement_bundle(
        self, workspace: EvidenceWorkspace, analysis_result: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def run_inference_gap_analysis(
        self, workspace: EvidenceWorkspace, bundle_result: Mapping[str, Any]
    ) -> dict[str, Any]: ...

    def generate_output(
        self,
        workspace: EvidenceWorkspace,
        bundle_result: Mapping[str, Any],
        gap_result: Mapping[str, Any],
    ) -> dict[str, Any]: ...

    def record_gap_round(
        self, workspace: EvidenceWorkspace, decision: GapDecision
    ) -> dict[str, Any]: ...

    def finalize_run(
        self,
        workspace: EvidenceWorkspace,
        state: Mapping[str, Any],
    ) -> dict[str, Any]: ...


@dataclass
class AgentRuntimeContext:
    root: Path
    run_dir: Path
    run_name: str
    statement: str
    language: str
    stage_configs: Mapping[str, LLMStageConfig]
    pipeline_config: PipelineConfig
    agent_config: AgentConfig
    trace_writer: AgentTraceWriter
    runtime_resources: "RuntimeResources | None" = None
    progress_callback: ProgressCallback | None = None
    resolved_config_snapshot: Mapping[str, Any] | None = None
    device: str = "cuda:0"
    amp: str = "fp16"
    corpus_dir: Path | None = None
    article_input_dir: Path | None = None
    bm25_index_dir: Path | None = None
    medcpt_index_dir: Path | None = None
    bmretriever_index_dir: Path | None = None
    cross_encoder_model_dir: Path | None = None
    stanza_model_dir: Path | None = None
    stanza_package: str = "genia"
    stanza_batch_size: int = 32
    cross_encoder_batch_size: int = 16
    section_mode: str = "auto"
    allow_cpu_fallback: bool = False
    cache_dir: Path | None = None
    started: float = field(default_factory=time.perf_counter)

    @property
    def agent_dir(self) -> Path:
        return self.run_dir / "agent"

    @property
    def attempts_root(self) -> Path:
        return self.run_dir / _STAGE_NAMES["analysis"] / "agent_attempts"

    def emit(
        self,
        *,
        stage: str,
        stage_index: int,
        message: str,
        completed_units: int | None = None,
        total_units: int | None = None,
    ) -> None:
        if self.progress_callback:
            self.progress_callback(
                {
                    "stage": stage,
                    "stage_index": stage_index,
                    "total_stages": 5,
                    "message": message,
                    "completed_units": completed_units,
                    "total_units": total_units,
                }
            )


class ProductionStageExecutor:
    """Real deterministic/LLM stage calls used by production graph nodes."""

    def __init__(self, context: AgentRuntimeContext) -> None:
        self.context = context

    def initialize_run(self) -> dict[str, Any]:
        ctx = self.context
        request_path = ctx.run_dir / "request.json"
        atomic_write_json(
            request_path,
            {
                "schema_version": STATEMENT_RUN_SCHEMA_VERSION,
                "contract_id": STATEMENT_RUN_CONTRACT_ID,
                "run_name": ctx.run_name,
                "statement": ctx.statement,
                "statement_sha256": sha256_text(ctx.statement),
                "language": ctx.language,
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        resolved_config_path: Path | None = None
        if ctx.resolved_config_snapshot is not None:
            resolved_config_path = ctx.run_dir / "resolved_config.json"
            atomic_write_json(resolved_config_path, dict(ctx.resolved_config_snapshot))
        return {
            "request_path": relative_path(ctx.root, request_path),
            "resolved_config_path": (
                None
                if resolved_config_path is None
                else relative_path(ctx.root, resolved_config_path)
            ),
        }

    def statement_decomposition(self) -> dict[str, Any]:
        ctx = self.context
        stage = ctx.stage_configs["statement_decomposition"]
        ctx.emit(
            stage="statement_decomposition",
            stage_index=1,
            message="Decomposing statement into verifiable biomedical claims",
        )
        return run_statement_decomposition(
            ctx.root,
            statement=ctx.statement,
            provider=stage.provider,
            run_name=_STAGE_NAMES["decomposition"],
            model=stage.model,
            api_key_env=stage.api_key_env,
            base_url=stage.base_url,
            max_tokens=stage.max_tokens,
            timeout_seconds=stage.timeout_seconds,
            max_retries=stage.max_retries,
            thinking=bool(stage.thinking),
            prompt_override=stage.prompt,
            artifact_root=ctx.run_dir,
            force=False,
        )

    def materialize_statement_analysis(
        self, workspace: EvidenceWorkspace
    ) -> dict[str, Any]:
        ctx = self.context
        stage = ctx.stage_configs["article_evidence"]
        if workspace.gap_round:
            ctx.emit(
                stage="inference_gap_analysis",
                stage_index=4,
                message="Rebuilding statement analysis after supplemental search",
            )
        return materialize_statement_analysis(
            ctx.root,
            workspace=workspace,
            decomposition_artifact_dir=(
                ctx.run_dir / _STAGE_NAMES["decomposition"]
            ),
            artifact_dir=ctx.run_dir / _STAGE_NAMES["analysis"],
            pipeline_config=ctx.pipeline_config,
            provider=stage.provider,
            model=stage.model,
            device=ctx.device,
            amp=ctx.amp,
            request_batch_size=stage.request_batch_size or 1,
            max_tokens=stage.max_tokens,
            thinking=bool(stage.thinking),
            runtime_resources_present=ctx.runtime_resources is not None,
        )

    def build_statement_bundle(
        self, workspace: EvidenceWorkspace, analysis_result: Mapping[str, Any]
    ) -> dict[str, Any]:
        ctx = self.context
        if workspace.gap_round:
            stage, index = "inference_gap_analysis", 4
            message = "Rebuilding evidence bundle after supplemental search"
        else:
            stage, index = "statement_bundle", 3
            message = "Building the statement evidence bundle"
        ctx.emit(stage=stage, stage_index=index, message=message)
        return run_statement_bundle(
            ctx.root,
            statement_analysis_artifact_dir=(
                ctx.run_dir / _STAGE_NAMES["analysis"]
            ),
            run_name=_STAGE_NAMES["bundle"],
            artifact_root=ctx.run_dir,
            decomposition=analysis_result["decomposition"],
            statement_result=analysis_result["statement_result"],
            graphs_by_claim=analysis_result["claim_graph_bundles"],
            force=(ctx.run_dir / _STAGE_NAMES["bundle"]).exists(),
        )

    def run_inference_gap_analysis(
        self, workspace: EvidenceWorkspace, bundle_result: Mapping[str, Any]
    ) -> dict[str, Any]:
        ctx = self.context
        stage = ctx.stage_configs["inference_gap"]
        ctx.emit(
            stage="inference_gap_analysis",
            stage_index=4,
            message=(
                f"Reanalyzing inference gaps, round {workspace.gap_round + 1} "
                f"of {workspace.max_gap_rounds}"
            ),
        )
        return run_inference_gap_analysis(
            ctx.root,
            statement_bundle_artifact_dir=ctx.run_dir / _STAGE_NAMES["bundle"],
            provider=stage.provider,
            run_name=_STAGE_NAMES["gaps"],
            model=stage.model,
            api_key_env=stage.api_key_env,
            base_url=stage.base_url,
            max_tokens=stage.max_tokens,
            timeout_seconds=stage.timeout_seconds,
            max_retries=stage.max_retries,
            thinking=stage.thinking,
            prompt_override=stage.prompt,
            artifact_root=ctx.run_dir,
            statement_bundle=bundle_result["statement_bundle"],
            force=(ctx.run_dir / _STAGE_NAMES["gaps"]).exists(),
        )

    def generate_output(
        self,
        workspace: EvidenceWorkspace,
        bundle_result: Mapping[str, Any],
        gap_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        ctx = self.context
        stage = ctx.stage_configs["localization"]
        ctx.emit(
            stage="output_generation",
            stage_index=5,
            message="Preparing the presentation bundle",
        )
        result = run_output_module(
            ctx.root,
            statement_bundle_artifact_dir=ctx.run_dir / _STAGE_NAMES["bundle"],
            inference_gap_artifact_dir=ctx.run_dir / _STAGE_NAMES["gaps"],
            run_name=_STAGE_NAMES["output"],
            language=ctx.language,
            provider=stage.provider,
            model=stage.model,
            api_key_env=stage.api_key_env,
            base_url=stage.base_url,
            max_tokens=stage.max_tokens,
            request_batch_size=stage.request_batch_size or 32,
            timeout_seconds=stage.timeout_seconds,
            max_retries=stage.max_retries,
            prompt_override=stage.prompt,
            artifact_root=ctx.run_dir,
            statement_bundle=bundle_result["statement_bundle"],
            gap_bundle=gap_result["inference_gap_bundle"],
            force=False,
        )
        output = result.get("output")
        presentation = (
            output.get("presentation_bundle")
            if isinstance(output, Mapping)
            else None
        )
        if not isinstance(presentation, Mapping) or not presentation.get("path"):
            raise EvidenceGapError(
                "Output stage did not return a presentation bundle path"
            )
        return {
            **result,
            "presentation_bundle_path": str(presentation["path"]),
        }

    def record_gap_round(
        self, workspace: EvidenceWorkspace, decision: GapDecision
    ) -> dict[str, Any]:
        ctx = self.context
        return gap_round_record(
            ctx.root,
            agent_dir=ctx.agent_dir,
            workspace=workspace,
            statement_analysis_dir=ctx.run_dir / _STAGE_NAMES["analysis"],
            statement_bundle_dir=ctx.run_dir / _STAGE_NAMES["bundle"],
            gap_dir=ctx.run_dir / _STAGE_NAMES["gaps"],
            gap_decision=decision.model_dump(mode="json"),
        )

    def finalize_run(
        self, workspace: EvidenceWorkspace, state: Mapping[str, Any]
    ) -> dict[str, Any]:
        ctx = self.context
        decomposition_result = state["decomposition_result"]
        analysis_result = state["analysis_result"]
        bundle_result = state["bundle_result"]
        gap_result = state["gap_result"]
        output_result = state["output_result"]

        ctx.trace_writer.write_workspace(workspace.model_dump(mode="json"))
        agent_manifest = {
            "schema_version": "2.0.0",
            "contract_id": "evidencegap.agent-harness.v2",
            "run_name": ctx.run_name,
            "execution_mode": "langgraph_end_to_end_agent",
            "langgraph_enabled": True,
            "checkpoint_backend": (
                "sqlite" if ctx.agent_config.checkpoint_enabled else "disabled"
            ),
            "controller": {
                "evidence": ctx.stage_configs["agent_controller"].safe_dict(),
                "gap": ctx.stage_configs["agent_gap_controller"].safe_dict(),
            },
            "max_steps": ctx.agent_config.max_steps,
            "initial_search_budget": workspace.initial_search_budget,
            "remaining_search_budget": workspace.remaining_search_budget,
            "action_counts": workspace.action_counts,
            "evidence_controller_decisions": workspace.evidence_controller_decision_count,
            "gap_controller_decisions": workspace.gap_controller_decision_count,
            "deterministic_fallback_decisions": workspace.deterministic_fallback_decisions,
            "rejected_evidence_decisions": workspace.rejected_decisions,
            "rejected_gap_decisions": workspace.rejected_gap_decisions,
            "gap_rounds": workspace.gap_round,
            "gap_remediation_count": workspace.gap_remediation_count,
            "gap_requested_searches": workspace.gap_remediation_count,
            "accepted_gap_count": sum(
                row.get("action") == "ACCEPT_GAPS" for row in workspace.gap_history
            ),
            "final_gap_decision": (
                None
                if workspace.gap_decision is None
                else workspace.gap_decision.model_dump(mode="json")
            ),
            "search_attempt_count": sum(
                len(claim.attempts) for claim in workspace.claims.values()
            ),
            "resolved_claims": sum(
                claim.status == "resolved" for claim in workspace.claims.values()
            ),
            "abstained_claims": sum(
                claim.status == "abstained" for claim in workspace.claims.values()
            ),
            "failed_claims": sum(
                claim.status == "failed" for claim in workspace.claims.values()
            ),
            "artifacts": {
                "workspace": relative_path(ctx.root, ctx.agent_dir / "workspace.json"),
                "action_trace": relative_path(ctx.root, ctx.trace_writer.path),
                "execution_graph": relative_path(
                    ctx.root, ctx.agent_dir / "execution_graph.mmd"
                ),
                "checkpoints": (
                    relative_path(ctx.root, ctx.agent_dir / "checkpoints.sqlite")
                    if ctx.agent_config.checkpoint_enabled
                    else None
                ),
                "gap_rounds": relative_path(ctx.root, ctx.agent_dir / "gap_rounds"),
            },
            "total_seconds": round(time.perf_counter() - ctx.started, 6),
        }
        agent_manifest_path = ctx.agent_dir / "agent_manifest.json"
        atomic_write_json(agent_manifest_path, agent_manifest)

        stage_dirs = {
            key: ctx.run_dir / directory for key, directory in _STAGE_NAMES.items()
        }
        total_seconds = round(time.perf_counter() - ctx.started, 6)
        execution_summary = build_execution_summary(
            stage_dirs, total_seconds=total_seconds
        )
        counts = {
            key: int(bundle_result[key])
            for key in (
                "total_claims",
                "completed_claims",
                "failed_claims",
                "articles",
                "evidence",
            )
        }
        counts.update(
            {
                "total_inference_steps": int(gap_result["total_inference_steps"]),
                "scope_gaps": int(gap_result["scope_gaps"]),
                "causal_gaps": int(gap_result["causal_gaps"]),
                "gap_api_requests": int(gap_result["api_requests"]),
                "translation_api_requests": int(output_result["api_requests"]),
            }
        )
        bundle_path = stage_dirs["bundle"] / "statement_bundle.json"
        gap_path = stage_dirs["gaps"] / "inference_gap_analysis.json"
        presentation_path = stage_dirs["output"] / "presentation_bundle.json"
        resolved_config_path = ctx.run_dir / "resolved_config.json"
        manifest = {
            "schema_version": STATEMENT_RUN_SCHEMA_VERSION,
            "contract_id": STATEMENT_RUN_CONTRACT_ID,
            "run_type": "agentic_evidencegap_end_to_end",
            "run_name": ctx.run_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "statement_id": decomposition_result["statement_id"],
            "analysis_status": analysis_result["analysis_status"],
            "output_language": output_result["output_language"],
            "localized": output_result["localized"],
            "execution": {
                "provider": ctx.stage_configs["article_evidence"].provider,
                "model": ctx.stage_configs["article_evidence"].model,
                "device": ctx.device,
                "amp": ctx.amp,
                "orchestration": "langgraph",
                "agent_harness": {
                    "enabled": True,
                    "evidence_actions": [
                        "SEARCH",
                        "RESOLVE",
                        "ABSTAIN",
                        "FINISH",
                    ],
                    "gap_actions": [
                        "REQUEST_MORE_EVIDENCE",
                        "ACCEPT_GAPS",
                        "ABSTAIN",
                    ],
                    "checkpoint": (
                        "sqlite"
                        if ctx.agent_config.checkpoint_enabled
                        else "disabled"
                    ),
                },
                "llm_stages": {
                    name: _stage_execution_config(stage)
                    for name, stage in ctx.stage_configs.items()
                },
                "pipeline": ctx.pipeline_config.safe_dict(),
                "resource_lifecycle": (
                    "engine_resident"
                    if ctx.runtime_resources is not None
                    else "per_call"
                ),
                "stage_handoff": "langgraph_state_with_artifact_persistence",
            },
            "stages": {
                key: _stage_meta(ctx.root, directory)
                for key, directory in stage_dirs.items()
            },
            "execution_summary": execution_summary,
            "counts": counts,
            "outputs": {
                **(
                    {
                        "resolved_config": {
                            "path": relative_path(ctx.root, resolved_config_path),
                            "sha256": sha256_file(resolved_config_path),
                        }
                    }
                    if resolved_config_path.is_file()
                    else {}
                ),
                "statement_bundle": {
                    "path": relative_path(ctx.root, bundle_path),
                    "sha256": sha256_file(bundle_path),
                },
                "inference_gap_analysis": {
                    "path": relative_path(ctx.root, gap_path),
                    "sha256": sha256_file(gap_path),
                },
                "presentation_bundle": {
                    "path": relative_path(ctx.root, presentation_path),
                    "sha256": sha256_file(presentation_path),
                },
            },
            "agent": {
                "artifact_dir": relative_path(ctx.root, ctx.agent_dir),
                "manifest": {
                    "path": relative_path(ctx.root, agent_manifest_path),
                    "sha256": sha256_file(agent_manifest_path),
                },
                "action_trace": {
                    "path": relative_path(ctx.root, ctx.trace_writer.path),
                    "sha256": sha256_file(ctx.trace_writer.path),
                },
                "workspace": {
                    "path": relative_path(ctx.root, ctx.agent_dir / "workspace.json"),
                    "sha256": sha256_file(ctx.agent_dir / "workspace.json"),
                },
            },
            "seconds": total_seconds,
        }
        atomic_write_json(ctx.run_dir / "run_manifest.json", manifest)
        validate_statement_pipeline_artifact(ctx.run_dir)
        return {
            "status": str(analysis_result["analysis_status"]).upper(),
            "artifact_status": "PASS",
            "run_name": ctx.run_name,
            "artifact_dir": relative_path(ctx.root, ctx.run_dir),
            "statement_id": decomposition_result["statement_id"],
            "analysis_status": analysis_result["analysis_status"],
            "output_language": output_result["output_language"],
            "localized": output_result["localized"],
            "execution_summary": execution_summary,
            **counts,
            "empty_claims": counts["total_claims"] == 0,
            "statement_bundle_path": relative_path(ctx.root, bundle_path),
            "inference_gap_analysis_path": relative_path(ctx.root, gap_path),
            "presentation_bundle_path": relative_path(ctx.root, presentation_path),
            "presentation_bundle": output_result["presentation_bundle"],
        }
