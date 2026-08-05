from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from langgraph.checkpoint.sqlite import SqliteSaver

from evidencegap_backend.agent.controller import EvidenceController
from evidencegap_backend.agent.gap_controller import GapController
from evidencegap_backend.agent.graph import build_agent_graph
from evidencegap_backend.agent.stages import AgentRuntimeContext, ProductionStageExecutor
from evidencegap_backend.agent.tools import create_analysis_executor, create_search_evidence_tool
from evidencegap_backend.agent.tracing import AgentTraceWriter
from evidencegap_backend.common import EvidenceGapError, require_empty_or_force
from evidencegap_backend.config import AgentConfig, LLMStageConfig, PipelineConfig
from evidencegap_backend.pipeline.statement_run import DEFAULT_ARTIFACT_ROOT, _safe_name

if TYPE_CHECKING:
    from evidencegap_backend.resources import RuntimeResources


def run_agent_statement_pipeline(
    root: Path,
    *,
    statement: str,
    run_name: str,
    provider: str,
    agent_config: AgentConfig,
    controller_config: LLMStageConfig,
    gap_controller_config: LLMStageConfig,
    model: str | None = None,
    device: str = "cuda:0",
    amp: str = "fp16",
    artifact_root: Path | None = None,
    corpus_dir: Path | None = None,
    article_input_dir: Path | None = None,
    bm25_index_dir: Path | None = None,
    medcpt_index_dir: Path | None = None,
    bmretriever_index_dir: Path | None = None,
    cross_encoder_model_dir: Path | None = None,
    stanza_model_dir: Path | None = None,
    stanza_package: str = "genia",
    stanza_batch_size: int = 32,
    cross_encoder_batch_size: int = 16,
    section_mode: str = "auto",
    allow_cpu_fallback: bool = False,
    api_key_env: str | None = None,
    base_url: str | None = None,
    decomposition_max_tokens: int = 2048,
    request_batch_size: int = 2,
    max_tokens: int = 4096,
    gap_max_tokens: int = 4096,
    language: str = "English",
    translation_max_tokens: int = 8192,
    translation_request_batch_size: int = 32,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    decomposition_thinking: bool = False,
    analysis_thinking: bool | None = None,
    gap_thinking: bool | None = None,
    cache_dir: Path | None = None,
    runtime_resources: "RuntimeResources | None" = None,
    stage_configs: Mapping[str, LLMStageConfig] | None = None,
    pipeline_config: PipelineConfig | None = None,
    resolved_config_snapshot: Mapping[str, Any] | None = None,
    progress_callback: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run decomposition through presentation output inside one StateGraph."""

    root = root.resolve()
    statement = statement.strip()
    language = language.strip()
    if not statement:
        raise EvidenceGapError("Statement cannot be blank")
    if not language:
        raise EvidenceGapError("language cannot be blank")
    name = _safe_name(run_name)
    base = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    run_dir = base / name
    require_empty_or_force(run_dir, force=force)
    run_dir.mkdir(parents=True, exist_ok=False)
    agent_dir = run_dir / "agent"
    agent_dir.mkdir(parents=True, exist_ok=True)
    trace = AgentTraceWriter(agent_dir)

    defaults = {
        "statement_decomposition": LLMStageConfig(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            max_tokens=decomposition_max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            thinking=decomposition_thinking,
        ),
        "article_evidence": LLMStageConfig(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            thinking=analysis_thinking,
            request_batch_size=request_batch_size,
        ),
        "inference_gap": LLMStageConfig(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            max_tokens=gap_max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            thinking=gap_thinking,
        ),
        "localization": LLMStageConfig(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            max_tokens=translation_max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            thinking=False,
            request_batch_size=translation_request_batch_size,
        ),
        "agent_controller": controller_config,
        "agent_gap_controller": gap_controller_config,
    }
    if stage_configs:
        unknown = set(stage_configs) - set(defaults)
        if unknown:
            raise EvidenceGapError(f"Unknown LLM stage configuration: {sorted(unknown)}")
        defaults.update(stage_configs)

    context = AgentRuntimeContext(
        root=root,
        run_dir=run_dir,
        run_name=name,
        statement=statement,
        language=language,
        stage_configs=defaults,
        pipeline_config=pipeline_config or PipelineConfig(),
        agent_config=agent_config,
        trace_writer=trace,
        runtime_resources=runtime_resources,
        progress_callback=progress_callback,
        resolved_config_snapshot=resolved_config_snapshot,
        device=device,
        amp=amp,
        corpus_dir=corpus_dir,
        article_input_dir=article_input_dir,
        bm25_index_dir=bm25_index_dir,
        medcpt_index_dir=medcpt_index_dir,
        bmretriever_index_dir=bmretriever_index_dir,
        cross_encoder_model_dir=cross_encoder_model_dir,
        stanza_model_dir=stanza_model_dir,
        stanza_package=stanza_package,
        stanza_batch_size=stanza_batch_size,
        cross_encoder_batch_size=cross_encoder_batch_size,
        section_mode=section_mode,
        allow_cpu_fallback=allow_cpu_fallback,
        cache_dir=cache_dir,
    )
    article = defaults["article_evidence"]
    analysis_kwargs = {
        "provider": article.provider,
        "model": article.model,
        "device": device,
        "amp": amp,
        "corpus_dir": corpus_dir,
        "article_input_dir": article_input_dir,
        "bm25_index_dir": bm25_index_dir,
        "medcpt_index_dir": medcpt_index_dir,
        "bmretriever_index_dir": bmretriever_index_dir,
        "cross_encoder_model_dir": cross_encoder_model_dir,
        "stanza_model_dir": stanza_model_dir,
        "stanza_package": stanza_package,
        "stanza_batch_size": stanza_batch_size,
        "cross_encoder_batch_size": cross_encoder_batch_size,
        "section_mode": section_mode,
        "allow_cpu_fallback": allow_cpu_fallback,
        "api_key_env": article.api_key_env,
        "base_url": article.base_url,
        "request_batch_size": article.request_batch_size or 1,
        "max_tokens": article.max_tokens,
        "timeout_seconds": article.timeout_seconds,
        "max_retries": article.max_retries,
        "thinking": article.thinking,
        "prompt_override": article.prompt,
        "pipeline_config": context.pipeline_config,
        "cache_dir": cache_dir,
        "runtime_resources": runtime_resources,
        "force": False,
    }
    tool = create_search_evidence_tool(
        create_analysis_executor(
            root=root,
            analysis_kwargs=analysis_kwargs,
            attempts_root=context.attempts_root,
        )
    )
    connection: sqlite3.Connection | None = None
    try:
        checkpointer = None
        if agent_config.checkpoint_enabled:
            checkpoint_path = agent_dir / "checkpoints.sqlite"
            connection = sqlite3.connect(checkpoint_path, check_same_thread=False)
            checkpointer = SqliteSaver(connection)
        graph = build_agent_graph(
            controller=EvidenceController(controller_config),
            gap_controller=GapController(gap_controller_config),
            search_tool=tool,
            stages=ProductionStageExecutor(context),
            run_name=name,
            max_steps=agent_config.max_steps,
            total_search_budget=agent_config.total_search_budget,
            per_claim_search_budget=agent_config.per_claim_search_budget,
            max_gap_rounds=agent_config.max_gap_rounds,
            gap_remediation_budget=agent_config.gap_remediation_budget,
            controller_retry_count=agent_config.controller_retry_count,
            gap_controller_retry_count=agent_config.gap_controller_retry_count,
            trace_writer=trace,
            progress_callback=progress_callback,
            checkpointer=checkpointer,
        )
        (agent_dir / "execution_graph.mmd").write_text(
            graph.get_graph().draw_mermaid(), encoding="utf-8"
        )
        state = graph.invoke(
            {"node_history": []},
            config={
                "configurable": {"thread_id": name},
                "recursion_limit": max(80, agent_config.max_steps * 8 + 40),
            },
        )
    finally:
        if connection is not None:
            connection.close()
    return dict(state["final_result"])
