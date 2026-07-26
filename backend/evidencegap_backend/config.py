from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from evidencegap_backend.prompting import PromptOverride

DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/statement_run")
DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")
DEFAULT_ARTICLE_INPUT_DIR = Path("artifacts/v1/dense/article_inputs")
DEFAULT_BM25_INDEX_DIR = Path("artifacts/v1/bm25_index")
DEFAULT_MEDCPT_INDEX_DIR = Path("artifacts/v1/dense/medcpt/faiss_index")
DEFAULT_BMRETRIEVER_INDEX_DIR = Path(
    "artifacts/v1/dense/bmretriever/faiss_index"
)
DEFAULT_CROSS_ENCODER_MODEL_DIR = Path("models/v1/medcpt-cross")
DEFAULT_STANZA_MODEL_DIR = Path("models/v1/stanza")
ANALYSIS_CONTEXT_SCHEMA_VERSION = "1.0.0"


def validate_analysis_context(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the deterministic methodology boundary exposed to clients."""

    expected_fields = {
        "schema_version",
        "scope",
        "is_systematic_review",
        "is_clinical_recommendation",
        "is_final_medical_truth",
        "aggregation_method",
        "uses_confidence_weighting",
        "retrieval_methods",
        "fusion_method",
        "reranker",
        "source_depth",
        "dense_nprobe",
        "rrf_k",
        "rerank_depth",
        "article_top_k",
        "max_evidence_sentences_per_article",
    }
    if set(value) != expected_fields:
        raise ValueError("Invalid analysis context fields")
    expected_literals = {
        "schema_version": ANALYSIS_CONTEXT_SCHEMA_VERSION,
        "scope": "retrieved_top_articles",
        "is_systematic_review": False,
        "is_clinical_recommendation": False,
        "is_final_medical_truth": False,
        "aggregation_method": "deterministic_article_count",
        "uses_confidence_weighting": False,
        "fusion_method": "reciprocal_rank_fusion",
        "reranker": "MedCPT Cross-Encoder",
    }
    if any(value.get(key) != expected for key, expected in expected_literals.items()):
        raise ValueError("Invalid analysis context methodology boundary")

    methods = value.get("retrieval_methods")
    if methods != ["BM25", "MedCPT", "BMRetriever"]:
        raise ValueError("Invalid analysis context retrieval methods")

    for key in (
        "source_depth",
        "dense_nprobe",
        "rerank_depth",
        "article_top_k",
        "max_evidence_sentences_per_article",
    ):
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            raise ValueError(f"Invalid analysis context value: {key}")
    rrf_k = value.get("rrf_k")
    if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or rrf_k < 0:
        raise ValueError("Invalid analysis context value: rrf_k")
    if int(value["article_top_k"]) > int(value["rerank_depth"]):
        raise ValueError("analysis context article_top_k exceeds rerank_depth")
    return dict(value)


@dataclass(frozen=True)
class LLMStageConfig:
    """Resolved LLM settings for one logical pipeline stage."""

    provider: str
    model: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    max_tokens: int = 4096
    timeout_seconds: float = 180.0
    max_retries: int = 4
    thinking: bool | None = None
    request_batch_size: int | None = None
    prompt: PromptOverride = field(default_factory=PromptOverride)

    def __post_init__(self) -> None:
        provider = self.provider.strip().lower()
        if not provider:
            raise ValueError("LLM stage provider cannot be blank")
        object.__setattr__(self, "provider", provider)
        for name in ("model", "api_key_env", "base_url"):
            value = getattr(self, name)
            if value is not None:
                value = str(value).strip()
                object.__setattr__(self, name, value or None)
        if self.max_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("LLM max_tokens and timeout_seconds must be positive")
        if self.max_retries < 0:
            raise ValueError("LLM max_retries cannot be negative")
        if self.request_batch_size is not None and self.request_batch_size <= 0:
            raise ValueError("LLM request_batch_size must be positive")

    def safe_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "thinking": self.thinking,
            "request_batch_size": self.request_batch_size,
            "prompt": {
                "system_prompt": self.prompt.system_prompt,
                "additional_instructions": self.prompt.additional_instructions,
                "version": self.prompt.version,
                "source": self.prompt.source,
            },
        }


@dataclass(frozen=True)
class PipelineConfig:
    source_depth: int = 100
    dense_nprobe: int = 1024
    rrf_k: int = 60
    rerank_depth: int = 100
    final_article_top_k: int = 10
    max_evidence_sentences: int = 5

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.source_depth,
                self.dense_nprobe,
                self.rerank_depth,
                self.final_article_top_k,
                self.max_evidence_sentences,
            )
        ):
            raise ValueError("Pipeline depth and evidence limits must be positive")
        if self.rrf_k < 0:
            raise ValueError("rrf_k cannot be negative")
        if self.rerank_depth > self.source_depth * 3:
            raise ValueError("rerank_depth exceeds the maximum fused candidate pool")
        if self.final_article_top_k > self.rerank_depth:
            raise ValueError("final_article_top_k cannot exceed rerank_depth")

    def safe_dict(self) -> dict[str, Any]:
        return {
            "retrieval": {
                "source_depth": self.source_depth,
                "dense_nprobe": self.dense_nprobe,
                "rrf_k": self.rrf_k,
                "rerank_depth": self.rerank_depth,
                "final_article_top_k": self.final_article_top_k,
            },
            "article_evidence": {
                "max_evidence_sentences": self.max_evidence_sentences,
            },
        }

    def analysis_context(self) -> dict[str, Any]:
        """Return the stable, user-facing methodology description for this run."""

        context = {
            "schema_version": ANALYSIS_CONTEXT_SCHEMA_VERSION,
            "scope": "retrieved_top_articles",
            "is_systematic_review": False,
            "is_clinical_recommendation": False,
            "is_final_medical_truth": False,
            "aggregation_method": "deterministic_article_count",
            "uses_confidence_weighting": False,
            "retrieval_methods": ["BM25", "MedCPT", "BMRetriever"],
            "fusion_method": "reciprocal_rank_fusion",
            "reranker": "MedCPT Cross-Encoder",
            "source_depth": self.source_depth,
            "dense_nprobe": self.dense_nprobe,
            "rrf_k": self.rrf_k,
            "rerank_depth": self.rerank_depth,
            "article_top_k": self.final_article_top_k,
            "max_evidence_sentences_per_article": self.max_evidence_sentences,
        }
        return validate_analysis_context(context)


def _resolve(root: Path, value: Path | None, default: Path) -> Path:
    path = value if value is not None else default
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class BackendConfig:
    """Explicit configuration for the independent 07.7 backend runtime."""

    workspace_root: Path
    provider: str
    model: str | None = None
    device: str = "cuda:0"
    amp: str = "fp16"
    artifact_root: Path | None = None
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
    article_cache_size: int = 5000
    section_mode: str = "auto"
    allow_cpu_fallback: bool = False
    api_key_env: str | None = None
    base_url: str | None = None
    decomposition_max_tokens: int = 2048
    request_batch_size: int = 1
    max_tokens: int = 8192
    gap_max_tokens: int = 4096
    translation_max_tokens: int = 8192
    translation_request_batch_size: int = 32
    timeout_seconds: float = 180.0
    max_retries: int = 4
    decomposition_thinking: bool = False
    analysis_thinking: bool | None = None
    gap_thinking: bool | None = None
    cache_dir: Path | None = None
    default_language: str = "English"
    config_path: Path | None = None
    decomposition_llm: LLMStageConfig | None = None
    article_evidence_llm: LLMStageConfig | None = None
    inference_gap_llm: LLMStageConfig | None = None
    localization_llm: LLMStageConfig | None = None
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def __post_init__(self) -> None:
        root = self.workspace_root.resolve()
        object.__setattr__(self, "workspace_root", root)
        provider = self.provider.strip().lower()
        if not provider:
            raise ValueError("provider cannot be blank")
        object.__setattr__(self, "provider", provider)
        object.__setattr__(
            self,
            "artifact_root",
            _resolve(root, self.artifact_root, DEFAULT_ARTIFACT_ROOT),
        )
        object.__setattr__(
            self, "corpus_dir", _resolve(root, self.corpus_dir, DEFAULT_CORPUS_DIR)
        )
        object.__setattr__(
            self,
            "article_input_dir",
            _resolve(root, self.article_input_dir, DEFAULT_ARTICLE_INPUT_DIR),
        )
        object.__setattr__(
            self,
            "bm25_index_dir",
            _resolve(root, self.bm25_index_dir, DEFAULT_BM25_INDEX_DIR),
        )
        object.__setattr__(
            self,
            "medcpt_index_dir",
            _resolve(root, self.medcpt_index_dir, DEFAULT_MEDCPT_INDEX_DIR),
        )
        object.__setattr__(
            self,
            "bmretriever_index_dir",
            _resolve(root, self.bmretriever_index_dir, DEFAULT_BMRETRIEVER_INDEX_DIR),
        )
        object.__setattr__(
            self,
            "cross_encoder_model_dir",
            _resolve(
                root, self.cross_encoder_model_dir, DEFAULT_CROSS_ENCODER_MODEL_DIR
            ),
        )
        object.__setattr__(
            self,
            "stanza_model_dir",
            _resolve(root, self.stanza_model_dir, DEFAULT_STANZA_MODEL_DIR),
        )
        if self.article_cache_size <= 0:
            raise ValueError("article_cache_size must be positive")
        if self.stanza_batch_size <= 0 or self.cross_encoder_batch_size <= 0:
            raise ValueError("Runtime batch sizes must be positive")
        if self.cache_dir is not None:
            object.__setattr__(
                self, "cache_dir", _resolve(root, self.cache_dir, self.cache_dir)
            )
        if self.config_path is not None:
            object.__setattr__(self, "config_path", self.config_path.resolve())
        language = self.default_language.strip()
        if not language:
            raise ValueError("default_language cannot be blank")
        object.__setattr__(self, "default_language", language)

        default_kwargs = {
            "provider": provider,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
        }
        if self.decomposition_llm is None:
            object.__setattr__(
                self,
                "decomposition_llm",
                LLMStageConfig(
                    **default_kwargs,
                    max_tokens=self.decomposition_max_tokens,
                    thinking=self.decomposition_thinking,
                ),
            )
        if self.article_evidence_llm is None:
            object.__setattr__(
                self,
                "article_evidence_llm",
                LLMStageConfig(
                    **default_kwargs,
                    max_tokens=self.max_tokens,
                    thinking=self.analysis_thinking,
                    request_batch_size=self.request_batch_size,
                ),
            )
        if self.inference_gap_llm is None:
            object.__setattr__(
                self,
                "inference_gap_llm",
                LLMStageConfig(
                    **default_kwargs,
                    max_tokens=self.gap_max_tokens,
                    thinking=self.gap_thinking,
                ),
            )
        if self.localization_llm is None:
            object.__setattr__(
                self,
                "localization_llm",
                LLMStageConfig(
                    **default_kwargs,
                    max_tokens=self.translation_max_tokens,
                    thinking=False,
                    request_batch_size=self.translation_request_batch_size,
                ),
            )

    @property
    def llm_stages(self) -> Mapping[str, LLMStageConfig]:
        assert self.decomposition_llm is not None
        assert self.article_evidence_llm is not None
        assert self.inference_gap_llm is not None
        assert self.localization_llm is not None
        return {
            "statement_decomposition": self.decomposition_llm,
            "article_evidence": self.article_evidence_llm,
            "inference_gap": self.inference_gap_llm,
            "localization": self.localization_llm,
        }

    @property
    def required_resource_paths(self) -> tuple[Path, ...]:
        return (
            self.corpus_dir,
            self.article_input_dir,
            self.bm25_index_dir,
            self.medcpt_index_dir,
            self.bmretriever_index_dir,
            self.cross_encoder_model_dir,
            self.stanza_model_dir,
        )

    def safe_dict(self) -> dict[str, Any]:
        """Resolved configuration snapshot with secret values excluded."""

        return {
            "config_version": 1,
            "config_path": None if self.config_path is None else str(self.config_path),
            "workspace_root": str(self.workspace_root),
            "runtime": {
                "device": self.device,
                "amp": self.amp,
                "allow_cpu_fallback": self.allow_cpu_fallback,
                "article_cache_size": self.article_cache_size,
                "stanza_batch_size": self.stanza_batch_size,
                "cross_encoder_batch_size": self.cross_encoder_batch_size,
                "section_mode": self.section_mode,
            },
            "resources": {
                "artifact_root": str(self.artifact_root),
                "corpus_dir": str(self.corpus_dir),
                "article_input_dir": str(self.article_input_dir),
                "bm25_index_dir": str(self.bm25_index_dir),
                "medcpt_index_dir": str(self.medcpt_index_dir),
                "bmretriever_index_dir": str(self.bmretriever_index_dir),
                "cross_encoder_model_dir": str(self.cross_encoder_model_dir),
                "stanza_model_dir": str(self.stanza_model_dir),
                "stanza_package": self.stanza_package,
                "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
            },
            "llm": {
                "stages": {
                    name: stage.safe_dict() for name, stage in self.llm_stages.items()
                }
            },
            "pipeline": self.pipeline.safe_dict(),
            "output": {"default_language": self.default_language},
        }
