from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from evidencegap_backend.common import EvidenceGapError
from evidencegap_backend.config import (
    BackendConfig,
    LLMStageConfig,
    PipelineConfig,
)
from evidencegap_backend.prompting import PromptOverride
from evidencegap_backend.stance.llm_judge import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
)

_STAGE_ENV_PREFIXES = {
    "statement_decomposition": "DECOMPOSITION",
    "article_evidence": "ARTICLE_EVIDENCE",
    "inference_gap": "INFERENCE_GAP",
    "localization": "LOCALIZATION",
}


@dataclass(frozen=True)
class ConfigDocument:
    path: Path | None
    base_dir: Path
    data: Mapping[str, Any]


def _value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _path(environ: Mapping[str, str], name: str) -> Path | None:
    value = _value(environ, name)
    return Path(value) if value is not None else None


def _bool_value(value: Any, *, label: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise EvidenceGapError(f"{label} must be a boolean value")


def _bool(
    environ: Mapping[str, str],
    name: str,
    default: bool,
) -> bool:
    value = _value(environ, name)
    return default if value is None else _bool_value(value, label=name)


def _optional_bool(
    environ: Mapping[str, str],
    name: str,
) -> bool | None:
    value = _value(environ, name)
    return None if value is None else _bool_value(value, label=name)


def _int(
    environ: Mapping[str, str],
    name: str,
    default: int,
) -> int:
    value = _value(environ, name)
    return default if value is None else int(value)


def _float(
    environ: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    value = _value(environ, name)
    return default if value is None else float(value)


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise EvidenceGapError(f"{label} must be a JSON object")
    return value


def _nested(data: Mapping[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _text(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value).strip()
    return rendered or None


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _provider_default(provider: str, values: Mapping[str, str], *, label: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(
            f"{label} must be one of {SUPPORTED_PROVIDERS}, got {provider!r}"
        )
    return values[normalized]


def _resolve_from(base: Path, value: Any) -> Path | None:
    text = _text(value)
    if text is None:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config_document(
    environ: Mapping[str, str] | None = None,
) -> ConfigDocument:
    env = os.environ if environ is None else environ
    explicit = _value(env, "EVIDENCEGAP_CONFIG")
    workspace_hint = Path(
        _value(env, "EVIDENCEGAP_WORKSPACE_ROOT") or Path.cwd()
    ).resolve()
    path = Path(explicit).expanduser() if explicit else workspace_hint / "config.json"
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        if explicit:
            raise EvidenceGapError(f"Configured EVIDENCEGAP_CONFIG does not exist: {path}")
        return ConfigDocument(path=None, base_dir=workspace_hint, data={})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON configuration {path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise EvidenceGapError(f"Configuration root must be a JSON object: {path}")
    version = raw.get("config_version", 1)
    if int(version) != 1:
        raise EvidenceGapError(f"Unsupported config_version {version!r}; expected 1")
    return ConfigDocument(path=path.resolve(), base_dir=path.parent.resolve(), data=raw)


def _prompt_override(
    *,
    document: ConfigDocument,
    stage_name: str,
    environ: Mapping[str, str],
) -> PromptOverride:
    stage = _mapping(
        _nested(document.data, "llm", "stages", stage_name),
        label=f"llm.stages.{stage_name}",
    )
    prompt = _mapping(stage.get("prompt"), label=f"llm.stages.{stage_name}.prompt")
    prefix = _STAGE_ENV_PREFIXES[stage_name]

    system_file_value = (
        _value(environ, f"EVIDENCEGAP_{prefix}_PROMPT_FILE")
        or _text(prompt.get("system_file"))
    )
    additional_file_value = (
        _value(environ, f"EVIDENCEGAP_{prefix}_PROMPT_EXTRA_FILE")
        or _text(prompt.get("additional_instructions_file"))
    )
    version = (
        _value(environ, f"EVIDENCEGAP_{prefix}_PROMPT_VERSION")
        or _text(prompt.get("version"))
    )

    system_prompt: str | None = None
    additional: str | None = None
    sources: list[str] = []
    if system_file_value:
        system_path = _resolve_from(document.base_dir, system_file_value)
        assert system_path is not None
        if not system_path.is_file():
            raise EvidenceGapError(f"Prompt file does not exist: {system_path}")
        system_prompt = system_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise EvidenceGapError(f"Prompt file is blank: {system_path}")
        sources.append(str(system_path))
    elif _text(prompt.get("system")):
        system_prompt = _text(prompt.get("system"))
        sources.append(
            f"{document.path or document.base_dir / 'config.json'}#llm.stages.{stage_name}.prompt.system"
        )
    if additional_file_value:
        extra_path = _resolve_from(document.base_dir, additional_file_value)
        assert extra_path is not None
        if not extra_path.is_file():
            raise EvidenceGapError(
                f"Prompt additional-instructions file does not exist: {extra_path}"
            )
        additional = extra_path.read_text(encoding="utf-8").strip()
        if not additional:
            raise EvidenceGapError(
                f"Prompt additional-instructions file is blank: {extra_path}"
            )
        sources.append(str(extra_path))
    elif _text(prompt.get("additional_instructions")):
        additional = _text(prompt.get("additional_instructions"))
        sources.append(
            f"{document.path or document.base_dir / 'config.json'}#llm.stages.{stage_name}.prompt.additional_instructions"
        )
    return PromptOverride(
        system_prompt=system_prompt,
        additional_instructions=additional,
        version=version,
        source=" + ".join(sources) if sources else None,
    )


def _stage_llm_config(
    *,
    document: ConfigDocument,
    stage_name: str,
    environ: Mapping[str, str],
    default_provider: str,
    default_model: str | None,
    default_api_key_env: str | None,
    default_base_url: str | None,
    default_max_tokens: int,
    default_timeout_seconds: float,
    default_max_retries: int,
    default_thinking: bool | None,
    default_request_batch_size: int | None = None,
) -> LLMStageConfig:
    defaults = _mapping(_nested(document.data, "llm", "defaults"), label="llm.defaults")
    stage = _mapping(
        _nested(document.data, "llm", "stages", stage_name),
        label=f"llm.stages.{stage_name}",
    )
    prefix = _STAGE_ENV_PREFIXES[stage_name]

    global_provider_env = _value(environ, "EVIDENCEGAP_PROVIDER")
    global_model_env = _value(environ, "EVIDENCEGAP_MODEL")
    global_api_key_env = _value(environ, "EVIDENCEGAP_API_KEY_ENV")
    global_base_url_env = _value(environ, "EVIDENCEGAP_BASE_URL")
    global_timeout_env = _value(environ, "EVIDENCEGAP_TIMEOUT_SECONDS")
    global_retries_env = _value(environ, "EVIDENCEGAP_MAX_RETRIES")

    provider = (
        _value(environ, f"EVIDENCEGAP_{prefix}_PROVIDER")
        or global_provider_env
        or _text(stage.get("provider"))
        or _text(defaults.get("provider"))
        or default_provider
    )
    provider = provider.strip().lower()
    _provider_default(provider, DEFAULT_MODELS, label=f"{stage_name}.provider")
    same_as_default_provider = provider == default_provider.strip().lower()
    model = (
        _value(environ, f"EVIDENCEGAP_{prefix}_MODEL")
        or global_model_env
        or _text(stage.get("model"))
        or (default_model if same_as_default_provider else None)
        or _provider_default(provider, DEFAULT_MODELS, label=f"{stage_name}.provider")
    )
    api_key_env = (
        _value(environ, f"EVIDENCEGAP_{prefix}_API_KEY_ENV")
        or global_api_key_env
        or _text(stage.get("api_key_env"))
        or (default_api_key_env if same_as_default_provider else None)
        or _provider_default(
            provider, DEFAULT_API_KEY_ENVS, label=f"{stage_name}.provider"
        )
    )
    base_url = (
        _value(environ, f"EVIDENCEGAP_{prefix}_BASE_URL")
        or global_base_url_env
        or _text(stage.get("base_url"))
        or (default_base_url if same_as_default_provider else None)
        or _provider_default(provider, DEFAULT_BASE_URLS, label=f"{stage_name}.provider")
    )

    stage_tokens_env = _value(environ, f"EVIDENCEGAP_{prefix}_MAX_TOKENS")
    if stage_name == "article_evidence":
        stage_tokens_env = stage_tokens_env or _value(environ, "EVIDENCEGAP_MAX_TOKENS")
    elif stage_name == "inference_gap":
        stage_tokens_env = stage_tokens_env or _value(environ, "EVIDENCEGAP_GAP_MAX_TOKENS")
    elif stage_name == "localization":
        stage_tokens_env = stage_tokens_env or _value(
            environ, "EVIDENCEGAP_TRANSLATION_MAX_TOKENS"
        )
    elif stage_name == "statement_decomposition":
        stage_tokens_env = stage_tokens_env or _value(
            environ, "EVIDENCEGAP_DECOMPOSITION_MAX_TOKENS"
        )
    max_tokens = int(
        _first_not_none(
            stage_tokens_env,
            stage.get("max_tokens"),
            defaults.get("max_tokens"),
            default_max_tokens,
        )
    )

    timeout_seconds = float(
        _first_not_none(
            _value(environ, f"EVIDENCEGAP_{prefix}_TIMEOUT_SECONDS"),
            global_timeout_env,
            stage.get("timeout_seconds"),
            defaults.get("timeout_seconds"),
            default_timeout_seconds,
        )
    )
    max_retries = int(
        _first_not_none(
            _value(environ, f"EVIDENCEGAP_{prefix}_MAX_RETRIES"),
            global_retries_env,
            stage.get("max_retries"),
            defaults.get("max_retries"),
            default_max_retries,
        )
    )

    thinking_env = _value(environ, f"EVIDENCEGAP_{prefix}_THINKING")
    if thinking_env is None:
        legacy_name = {
            "statement_decomposition": "EVIDENCEGAP_DECOMPOSITION_THINKING",
            "article_evidence": "EVIDENCEGAP_ANALYSIS_THINKING",
            "inference_gap": "EVIDENCEGAP_GAP_THINKING",
        }.get(stage_name)
        thinking_env = _value(environ, legacy_name) if legacy_name else None
    thinking_value = stage.get("thinking", defaults.get("thinking", default_thinking))
    thinking = (
        _bool_value(thinking_env, label=f"EVIDENCEGAP_{prefix}_THINKING")
        if thinking_env is not None
        else (
            None
            if thinking_value is None
            else _bool_value(thinking_value, label=f"{stage_name}.thinking")
        )
    )

    batch_size: int | None = default_request_batch_size
    if default_request_batch_size is not None:
        batch_env = _value(environ, f"EVIDENCEGAP_{prefix}_REQUEST_BATCH_SIZE")
        if stage_name == "article_evidence":
            batch_env = batch_env or _value(environ, "EVIDENCEGAP_REQUEST_BATCH_SIZE")
        elif stage_name == "localization":
            batch_env = batch_env or _value(
                environ, "EVIDENCEGAP_TRANSLATION_REQUEST_BATCH_SIZE"
            )
        batch_size = int(
            _first_not_none(
                batch_env,
                stage.get("request_batch_size"),
                defaults.get("request_batch_size"),
                default_request_batch_size,
            )
        )

    return LLMStageConfig(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        thinking=thinking,
        request_batch_size=batch_size,
        prompt=_prompt_override(
            document=document,
            stage_name=stage_name,
            environ=environ,
        ),
    )


def backend_config_from_env(
    environ: Mapping[str, str] | None = None,
    *,
    document: ConfigDocument | None = None,
) -> BackendConfig:
    """Resolve defaults < config.json < environment variables."""

    env = os.environ if environ is None else environ
    doc = document or load_config_document(env)
    data = doc.data
    runtime = _mapping(data.get("runtime"), label="runtime")
    resources = _mapping(data.get("resources"), label="resources")
    output = _mapping(data.get("output"), label="output")
    pipeline_root = _mapping(data.get("pipeline"), label="pipeline")
    retrieval = _mapping(pipeline_root.get("retrieval"), label="pipeline.retrieval")
    article_evidence_pipeline = _mapping(
        pipeline_root.get("article_evidence"),
        label="pipeline.article_evidence",
    )
    llm_defaults = _mapping(_nested(data, "llm", "defaults"), label="llm.defaults")

    configured_workspace = _resolve_from(doc.base_dir, data.get("workspace_root"))
    workspace_root = Path(
        _value(env, "EVIDENCEGAP_WORKSPACE_ROOT")
        or configured_workspace
        or Path.cwd()
    ).resolve()

    configured_default_provider = (
        _text(llm_defaults.get("provider")) or "deepseek"
    ).lower()
    provider_env = _value(env, "EVIDENCEGAP_PROVIDER")
    default_provider = (provider_env or configured_default_provider).lower()
    _provider_default(default_provider, DEFAULT_MODELS, label="llm.defaults.provider")
    provider_changed_by_env = (
        provider_env is not None and default_provider != configured_default_provider
    )
    default_model = (
        _value(env, "EVIDENCEGAP_MODEL")
        or (None if provider_changed_by_env else _text(llm_defaults.get("model")))
        or _provider_default(default_provider, DEFAULT_MODELS, label="llm.defaults.provider")
    )
    default_api_key_env = (
        _value(env, "EVIDENCEGAP_API_KEY_ENV")
        or (
            None
            if provider_changed_by_env
            else _text(llm_defaults.get("api_key_env"))
        )
        or _provider_default(
            default_provider, DEFAULT_API_KEY_ENVS, label="llm.defaults.provider"
        )
    )
    default_base_url = (
        _value(env, "EVIDENCEGAP_BASE_URL")
        or (None if provider_changed_by_env else _text(llm_defaults.get("base_url")))
        or _provider_default(
            default_provider, DEFAULT_BASE_URLS, label="llm.defaults.provider"
        )
    )
    default_timeout = float(
        _value(env, "EVIDENCEGAP_TIMEOUT_SECONDS")
        or llm_defaults.get("timeout_seconds")
        or 180.0
    )
    default_retries = int(
        _first_not_none(
            _value(env, "EVIDENCEGAP_MAX_RETRIES"),
            llm_defaults.get("max_retries"),
            4,
        )
    )

    decomposition = _stage_llm_config(
        document=doc,
        stage_name="statement_decomposition",
        environ=env,
        default_provider=default_provider,
        default_model=default_model,
        default_api_key_env=default_api_key_env,
        default_base_url=default_base_url,
        default_max_tokens=2048,
        default_timeout_seconds=default_timeout,
        default_max_retries=default_retries,
        default_thinking=False,
    )
    article_evidence = _stage_llm_config(
        document=doc,
        stage_name="article_evidence",
        environ=env,
        default_provider=default_provider,
        default_model=default_model,
        default_api_key_env=default_api_key_env,
        default_base_url=default_base_url,
        default_max_tokens=8192,
        default_timeout_seconds=default_timeout,
        default_max_retries=default_retries,
        default_thinking=None,
        default_request_batch_size=1,
    )
    inference_gap = _stage_llm_config(
        document=doc,
        stage_name="inference_gap",
        environ=env,
        default_provider=default_provider,
        default_model=default_model,
        default_api_key_env=default_api_key_env,
        default_base_url=default_base_url,
        default_max_tokens=4096,
        default_timeout_seconds=default_timeout,
        default_max_retries=default_retries,
        default_thinking=None,
    )
    localization = _stage_llm_config(
        document=doc,
        stage_name="localization",
        environ=env,
        default_provider=default_provider,
        default_model=default_model,
        default_api_key_env=default_api_key_env,
        default_base_url=default_base_url,
        default_max_tokens=8192,
        default_timeout_seconds=default_timeout,
        default_max_retries=default_retries,
        default_thinking=False,
        default_request_batch_size=32,
    )

    pipeline = PipelineConfig(
        source_depth=_int(
            env,
            "EVIDENCEGAP_RETRIEVAL_SOURCE_DEPTH",
            int(retrieval.get("source_depth", 100)),
        ),
        dense_nprobe=_int(
            env,
            "EVIDENCEGAP_RETRIEVAL_DENSE_NPROBE",
            int(retrieval.get("dense_nprobe", 1024)),
        ),
        rrf_k=_int(
            env,
            "EVIDENCEGAP_RETRIEVAL_RRF_K",
            int(retrieval.get("rrf_k", 60)),
        ),
        rerank_depth=_int(
            env,
            "EVIDENCEGAP_RETRIEVAL_RERANK_DEPTH",
            int(retrieval.get("rerank_depth", 100)),
        ),
        final_article_top_k=_int(
            env,
            "EVIDENCEGAP_RETRIEVAL_FINAL_ARTICLE_TOP_K",
            int(retrieval.get("final_article_top_k", 10)),
        ),
        max_evidence_sentences=_int(
            env,
            "EVIDENCEGAP_MAX_EVIDENCE_SENTENCES",
            int(article_evidence_pipeline.get("max_evidence_sentences", 5)),
        ),
    )

    def resource_path(env_name: str, key: str) -> Path | None:
        env_path = _path(env, env_name)
        if env_path is not None:
            return env_path
        value = resources.get(key)
        return Path(str(value)) if value is not None else None

    return BackendConfig(
        workspace_root=workspace_root,
        provider=default_provider,
        model=default_model,
        device=_value(env, "EVIDENCEGAP_DEVICE") or _text(runtime.get("device")) or "cuda:0",
        amp=_value(env, "EVIDENCEGAP_AMP") or _text(runtime.get("amp")) or "fp16",
        artifact_root=resource_path("EVIDENCEGAP_ARTIFACT_ROOT", "artifact_root"),
        corpus_dir=resource_path("EVIDENCEGAP_CORPUS_DIR", "corpus_dir"),
        article_input_dir=resource_path(
            "EVIDENCEGAP_ARTICLE_INPUT_DIR", "article_input_dir"
        ),
        bm25_index_dir=resource_path("EVIDENCEGAP_BM25_INDEX_DIR", "bm25_index_dir"),
        medcpt_index_dir=resource_path(
            "EVIDENCEGAP_MEDCPT_INDEX_DIR", "medcpt_index_dir"
        ),
        bmretriever_index_dir=resource_path(
            "EVIDENCEGAP_BMRETRIEVER_INDEX_DIR", "bmretriever_index_dir"
        ),
        cross_encoder_model_dir=resource_path(
            "EVIDENCEGAP_CROSS_ENCODER_MODEL_DIR", "cross_encoder_model_dir"
        ),
        stanza_model_dir=resource_path(
            "EVIDENCEGAP_STANZA_MODEL_DIR", "stanza_model_dir"
        ),
        stanza_package=(
            _value(env, "EVIDENCEGAP_STANZA_PACKAGE")
            or _text(resources.get("stanza_package"))
            or "genia"
        ),
        stanza_batch_size=_int(
            env,
            "EVIDENCEGAP_STANZA_BATCH_SIZE",
            int(runtime.get("stanza_batch_size", 32)),
        ),
        cross_encoder_batch_size=_int(
            env,
            "EVIDENCEGAP_CROSS_ENCODER_BATCH_SIZE",
            int(runtime.get("cross_encoder_batch_size", 16)),
        ),
        article_cache_size=_int(
            env,
            "EVIDENCEGAP_ARTICLE_CACHE_SIZE",
            int(runtime.get("article_cache_size", 5000)),
        ),
        section_mode=(
            _value(env, "EVIDENCEGAP_SECTION_MODE")
            or _text(runtime.get("section_mode"))
            or "auto"
        ),
        allow_cpu_fallback=(
            _bool(
                env,
                "EVIDENCEGAP_ALLOW_CPU_FALLBACK",
                _bool_value(
                    runtime.get("allow_cpu_fallback", False),
                    label="runtime.allow_cpu_fallback",
                ),
            )
        ),
        cache_dir=resource_path("EVIDENCEGAP_CACHE_DIR", "cache_dir"),
        default_language=(
            _value(env, "EVIDENCEGAP_DEFAULT_LANGUAGE")
            or _text(output.get("default_language"))
            or "English"
        ),
        config_path=doc.path,
        decomposition_llm=decomposition,
        article_evidence_llm=article_evidence,
        inference_gap_llm=inference_gap,
        localization_llm=localization,
        pipeline=pipeline,
    )


@dataclass(frozen=True)
class ApiConfig:
    """Small HTTP/task-layer configuration kept separate from the algorithm."""

    run_store_root: Path
    max_queue_size: int = 16
    max_statement_chars: int = 20_000
    validate_resources: bool = True
    cors_origins: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_store_root", self.run_store_root.resolve())
        if self.max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive")
        if self.max_statement_chars <= 0:
            raise ValueError("max_statement_chars must be positive")

    @classmethod
    def from_env(
        cls,
        backend_config: BackendConfig,
        environ: Mapping[str, str] | None = None,
        *,
        document: ConfigDocument | None = None,
    ) -> "ApiConfig":
        env = os.environ if environ is None else environ
        doc = document or load_config_document(env)
        api = _mapping(doc.data.get("api"), label="api")
        run_store_value = _path(env, "EVIDENCEGAP_API_RUN_STORE_ROOT")
        if run_store_value is None and api.get("run_store_root") is not None:
            run_store_value = Path(str(api["run_store_root"]))
        run_store_root = (
            run_store_value
            if run_store_value is not None
            else backend_config.workspace_root / "artifacts/v1/api_runs"
        )
        if not run_store_root.is_absolute():
            run_store_root = backend_config.workspace_root / run_store_root
        raw_origins = _value(env, "EVIDENCEGAP_CORS_ORIGINS")
        if raw_origins is not None:
            origins = tuple(
                item.strip() for item in raw_origins.split(",") if item.strip()
            )
        else:
            configured_origins = api.get("cors_origins", [])
            if not isinstance(configured_origins, list) or any(
                not isinstance(item, str) for item in configured_origins
            ):
                raise EvidenceGapError("api.cors_origins must be a string array")
            origins = tuple(item.strip() for item in configured_origins if item.strip())
        return cls(
            run_store_root=run_store_root,
            max_queue_size=_int(
                env,
                "EVIDENCEGAP_API_MAX_QUEUE_SIZE",
                int(api.get("max_queue_size", 16)),
            ),
            max_statement_chars=_int(
                env,
                "EVIDENCEGAP_API_MAX_STATEMENT_CHARS",
                int(api.get("max_statement_chars", 20_000)),
            ),
            validate_resources=_bool(
                env,
                "EVIDENCEGAP_API_VALIDATE_RESOURCES",
                _bool_value(
                    api.get("validate_resources", True),
                    label="api.validate_resources",
                ),
            ),
            cors_origins=origins,
        )
