from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidencegap_backend.common import (
    EvidenceGapError,
    load_json,
    workspace_root_context,
)
from evidencegap_backend.config import BackendConfig
from evidencegap_backend.pipeline.statement_run import (
    run_statement_pipeline,
    validate_statement_pipeline_artifact,
)
from evidencegap_backend.resources import RuntimeResources


@dataclass(frozen=True)
class StatementAnalysisResult:
    run: dict[str, Any]
    artifact_dir: Path
    presentation_bundle_path: Path
    presentation_bundle: dict[str, Any]


class EvidenceGapEngine:
    """Long-lived Python runtime for the complete EvidenceGap 07.7 pipeline."""

    def __init__(
        self,
        config: BackendConfig,
        *,
        resources: RuntimeResources | None = None,
    ) -> None:
        self.config = config
        self._resources = resources or RuntimeResources(config)
        self._run_lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return bool(self._resources.loaded)

    @property
    def runtime_status(self) -> dict[str, Any]:
        return self._resources.status()

    def load(self, *, validate_resources: bool = True) -> None:
        with self._run_lock:
            if not self.config.workspace_root.is_dir():
                raise EvidenceGapError(
                    f"workspace_root does not exist: {self.config.workspace_root}"
                )
            self.config.artifact_root.mkdir(parents=True, exist_ok=True)
            if self.config.cache_dir is not None:
                self.config.cache_dir.mkdir(parents=True, exist_ok=True)
            self._resources.load(validate_paths=validate_resources)

    def close(self) -> None:
        with self._run_lock:
            self._resources.close()

    def analyze_statement(
        self,
        *,
        statement: str,
        run_name: str,
        language: str | None = None,
        force: bool = False,
        validate: bool = True,
    ) -> StatementAnalysisResult:
        with self._run_lock:
            if not self.loaded:
                raise EvidenceGapError("EvidenceGapEngine.load() must be called first")
            cfg = self.config
            with workspace_root_context(cfg.workspace_root):
                run = run_statement_pipeline(
                    cfg.workspace_root,
                    statement=statement,
                    run_name=run_name,
                    provider=cfg.provider,
                    model=cfg.model,
                    device=cfg.device,
                    amp=cfg.amp,
                    artifact_root=cfg.artifact_root,
                    corpus_dir=cfg.corpus_dir,
                    article_input_dir=cfg.article_input_dir,
                    bm25_index_dir=cfg.bm25_index_dir,
                    medcpt_index_dir=cfg.medcpt_index_dir,
                    bmretriever_index_dir=cfg.bmretriever_index_dir,
                    cross_encoder_model_dir=cfg.cross_encoder_model_dir,
                    stanza_model_dir=cfg.stanza_model_dir,
                    stanza_package=cfg.stanza_package,
                    stanza_batch_size=cfg.stanza_batch_size,
                    cross_encoder_batch_size=cfg.cross_encoder_batch_size,
                    section_mode=cfg.section_mode,
                    allow_cpu_fallback=cfg.allow_cpu_fallback,
                    api_key_env=cfg.api_key_env,
                    base_url=cfg.base_url,
                    decomposition_max_tokens=cfg.decomposition_max_tokens,
                    request_batch_size=cfg.request_batch_size,
                    max_tokens=cfg.max_tokens,
                    gap_max_tokens=cfg.gap_max_tokens,
                    language=language or cfg.default_language,
                    translation_max_tokens=cfg.translation_max_tokens,
                    translation_request_batch_size=(
                        cfg.translation_request_batch_size
                    ),
                    timeout_seconds=cfg.timeout_seconds,
                    max_retries=cfg.max_retries,
                    decomposition_thinking=cfg.decomposition_thinking,
                    analysis_thinking=cfg.analysis_thinking,
                    gap_thinking=cfg.gap_thinking,
                    cache_dir=cfg.cache_dir,
                    runtime_resources=self._resources,
                    stage_configs=cfg.llm_stages,
                    pipeline_config=cfg.pipeline,
                    resolved_config_snapshot=cfg.safe_dict(),
                    force=force,
                )
                artifact_value = Path(str(run["artifact_dir"]))
                artifact_dir = (
                    artifact_value
                    if artifact_value.is_absolute()
                    else cfg.workspace_root / artifact_value
                ).resolve()
                if validate:
                    validate_statement_pipeline_artifact(artifact_dir)
            presentation_value = Path(str(run["presentation_bundle_path"]))
            presentation_path = (
                presentation_value
                if presentation_value.is_absolute()
                else cfg.workspace_root / presentation_value
            ).resolve()
            bundle_value = run.pop("presentation_bundle", None)
            bundle = (
                dict(bundle_value)
                if isinstance(bundle_value, dict)
                else load_json(presentation_path)
            )
            if not isinstance(bundle, dict):
                raise EvidenceGapError(
                    f"Expected presentation bundle object: {presentation_path}"
                )
            self._resources.record_analysis_run()
            return StatementAnalysisResult(
                run=run,
                artifact_dir=artifact_dir,
                presentation_bundle_path=presentation_path,
                presentation_bundle=bundle,
            )
