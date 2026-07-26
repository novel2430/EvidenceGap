from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from evidencegap_backend.common import (
    EvidenceGapError,
    load_json,
    sha256_text,
    workspace_root_context,
)
from evidencegap_backend.config import BackendConfig
from evidencegap_backend.output.presentation import (
    run_output_module,
    validate_output_artifact,
)
from evidencegap_backend.pipeline.contracts import RuntimeArticle
from evidencegap_backend.pipeline.sentence_materialization import (
    canonicalize_article_text,
)
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


@dataclass(frozen=True)
class LocalizationResult:
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
        progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
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
                    progress_callback=progress_callback,
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

    def get_article_context(
        self,
        *,
        presentation_bundle: Mapping[str, Any],
        article_node_id: str,
    ) -> dict[str, Any]:
        """Rebuild the exact canonical article text and verify evidence offsets."""

        with self._run_lock:
            if not self.loaded:
                raise EvidenceGapError("EvidenceGapEngine.load() must be called first")
            articles = presentation_bundle.get("articles")
            evidence = presentation_bundle.get("evidence")
            if not isinstance(articles, list) or not isinstance(evidence, list):
                raise EvidenceGapError("Presentation bundle article data is invalid")
            matches = [
                row
                for row in articles
                if isinstance(row, Mapping)
                and row.get("article_node_id") == article_node_id
            ]
            if len(matches) != 1:
                raise EvidenceGapError(
                    f"Article node does not belong to this run: {article_node_id}"
                )
            article = matches[0]
            article_id = str(article.get("article_id") or "").strip()
            claim_id = str(article.get("claim_id") or "").strip()
            if not article_id or not claim_id:
                raise EvidenceGapError("Presentation article identity is incomplete")
            source = self._resources.fetch_article_texts([article_id])[article_id]
            runtime_article = RuntimeArticle.from_mapping(
                {
                    **source,
                    "article_rank": article.get("rank"),
                }
            )
            canonical = canonicalize_article_text(
                runtime_article,
                section_mode=self.config.section_mode,
            )
            fingerprint = sha256_text(canonical.source_text)
            article_evidence = [
                row
                for row in evidence
                if isinstance(row, Mapping)
                and row.get("article_node_id") == article_node_id
            ]
            stored_fingerprints = {
                str(row.get("source_text_fingerprint") or "")
                for row in article_evidence
                if str(row.get("source_text_fingerprint") or "")
            }
            if stored_fingerprints and stored_fingerprints != {fingerprint}:
                raise EvidenceGapError(
                    "Article source fingerprint changed; precise evidence offsets "
                    "cannot be trusted"
                )

            evidence_spans: list[dict[str, Any]] = []
            for row in article_evidence:
                start = int(row.get("character_start", -1))
                end = int(row.get("character_end", -1))
                text = str(row.get("text") or "")
                if not 0 <= start < end <= len(canonical.source_text):
                    raise EvidenceGapError("Stored evidence offsets are out of range")
                if canonical.source_text[start:end] != text:
                    raise EvidenceGapError(
                        "Stored evidence text does not match canonical article offsets"
                    )
                evidence_spans.append(
                    {
                        "evidence_id": str(row.get("evidence_id") or ""),
                        "claim_id": str(row.get("claim_id") or ""),
                        "section": row.get("section"),
                        "section_index": int(row.get("section_index", -1)),
                        "sentence_index": int(row.get("sentence_index", -1)),
                        "character_start": start,
                        "character_end": end,
                        "text": text,
                    }
                )
            evidence_spans.sort(
                key=lambda row: (row["character_start"], row["evidence_id"])
            )
            sections = [
                {
                    "sentence_type": segment.sentence_type,
                    "section": segment.section,
                    "section_index": segment.section_index,
                    "character_start": segment.document_start,
                    "character_end": segment.document_start + len(segment.text),
                }
                for segment in canonical.segments
            ]
            return {
                "article_node_id": article_node_id,
                "article_id": article_id,
                "claim_id": claim_id,
                "pmid": runtime_article.pmid,
                "title": runtime_article.title,
                "canonical_text": canonical.source_text,
                "source_text_fingerprint": fingerprint,
                "fingerprint_verified": bool(stored_fingerprints),
                "sections": sections,
                "evidence_spans": evidence_spans,
            }

    def localize_statement_run(
        self,
        *,
        artifact_dir: Path,
        localization_name: str,
        language: str,
        artifact_root: Path,
        force: bool = False,
        validate: bool = True,
    ) -> LocalizationResult:
        """Create a presentation variant without rerunning evidence analysis."""

        with self._run_lock:
            if not self.loaded:
                raise EvidenceGapError("EvidenceGapEngine.load() must be called first")
            cfg = self.config
            artifact_dir = artifact_dir.resolve()
            with workspace_root_context(cfg.workspace_root):
                validate_statement_pipeline_artifact(artifact_dir)
                manifest = load_json(artifact_dir / "run_manifest.json")
                if not isinstance(manifest, Mapping):
                    raise EvidenceGapError("Invalid source statement run manifest")
                stages = manifest.get("stages")
                if not isinstance(stages, Mapping):
                    raise EvidenceGapError("Source statement run stages are missing")

                def stage_dir(name: str) -> Path:
                    meta = stages.get(name)
                    if not isinstance(meta, Mapping):
                        raise EvidenceGapError(
                            f"Source statement run stage is missing: {name}"
                        )
                    value = Path(str(meta.get("artifact_dir") or ""))
                    return (
                        value
                        if value.is_absolute()
                        else cfg.workspace_root / value
                    ).resolve()

                localization_stage = cfg.llm_stages["localization"]
                result = run_output_module(
                    cfg.workspace_root,
                    statement_bundle_artifact_dir=stage_dir("bundle"),
                    inference_gap_artifact_dir=stage_dir("gaps"),
                    run_name=localization_name,
                    language=language,
                    provider=localization_stage.provider,
                    model=localization_stage.model,
                    api_key_env=localization_stage.api_key_env,
                    base_url=localization_stage.base_url,
                    max_tokens=localization_stage.max_tokens,
                    request_batch_size=(
                        localization_stage.request_batch_size or 32
                    ),
                    timeout_seconds=localization_stage.timeout_seconds,
                    max_retries=localization_stage.max_retries,
                    prompt_override=localization_stage.prompt,
                    artifact_root=artifact_root,
                    force=force,
                )
                result_artifact_value = Path(str(result["artifact_dir"]))
                result_artifact_dir = (
                    result_artifact_value
                    if result_artifact_value.is_absolute()
                    else cfg.workspace_root / result_artifact_value
                ).resolve()
                if validate:
                    validate_output_artifact(result_artifact_dir)
            presentation_path = (
                result_artifact_dir / "presentation_bundle.json"
            ).resolve()
            bundle_value = result.pop("presentation_bundle", None)
            bundle = (
                dict(bundle_value)
                if isinstance(bundle_value, Mapping)
                else load_json(presentation_path)
            )
            if not isinstance(bundle, dict):
                raise EvidenceGapError(
                    f"Expected localized presentation bundle: {presentation_path}"
                )
            return LocalizationResult(
                run=result,
                artifact_dir=result_artifact_dir,
                presentation_bundle_path=presentation_path,
                presentation_bundle=bundle,
            )
