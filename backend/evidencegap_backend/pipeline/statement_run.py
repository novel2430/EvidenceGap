from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_write_json,
    relative_path,
    require_empty_or_force,
    sha256_file,
    sha256_text,
    find_workspace_root,
)
from evidencegap_backend.config import LLMStageConfig, PipelineConfig
from evidencegap_backend.output.presentation import run_output_module, validate_output_artifact
from evidencegap_backend.pipeline.inference_gap_analysis import (
    run_inference_gap_analysis,
    validate_inference_gap_analysis_artifact,
)
from evidencegap_backend.pipeline.statement_analysis import (
    run_statement_analysis,
    validate_statement_analysis_artifact,
)
from evidencegap_backend.pipeline.statement_bundle import (
    run_statement_bundle,
    validate_statement_bundle_artifact,
)
from evidencegap_backend.pipeline.statement_decomposition import (
    run_statement_decomposition,
    validate_statement_decomposition_artifact,
)

STATEMENT_RUN_SCHEMA_VERSION = "2.0.0"
STATEMENT_RUN_CONTRACT_ID = "phase077.complete-run.v1"
if TYPE_CHECKING:
    from evidencegap_backend.resources import RuntimeResources

DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/statement_run")

_STAGE_NAMES = {
    "decomposition": "decomposition",
    "analysis": "analysis",
    "bundle": "bundle",
    "gaps": "gaps",
    "output": "output",
}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty")
    return cleaned


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing required JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceGapError(f"Expected JSON object in {path}")
    return value


def _find_repo_root(start: Path) -> Path:
    return find_workspace_root(start)


def _stage_meta(root: Path, artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise EvidenceGapError(f"Missing nested stage manifest: {manifest_path}")
    return {
        "artifact_dir": relative_path(root, artifact_dir),
        "run_manifest": {
            "path": relative_path(root, manifest_path),
            "sha256": sha256_file(manifest_path),
        },
    }


def _resolve_deepseek_thinking(
    provider: str,
    value: bool | None,
    *,
    label: str,
) -> bool:
    if provider == "deepseek":
        return True if value is None else value
    if value is True:
        raise EvidenceGapError(f"{label} thinking is only supported for DeepSeek")
    return False


def _stage_execution_config(stage: LLMStageConfig) -> dict[str, Any]:
    value = stage.safe_dict()
    prompt = dict(value.get("prompt") or {})
    prompt.pop("system_prompt", None)
    prompt.pop("additional_instructions", None)
    value["prompt"] = prompt
    return value


def run_statement_pipeline(
    root: Path,
    *,
    statement: str,
    run_name: str,
    provider: str,
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
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    statement = statement.strip()
    language = language.strip()
    if not statement:
        raise EvidenceGapError("Statement cannot be blank")
    if not language:
        raise EvidenceGapError("language cannot be blank")
    if any(
        value <= 0
        for value in (
            decomposition_max_tokens,
            request_batch_size,
            max_tokens,
            gap_max_tokens,
            translation_max_tokens,
            translation_request_batch_size,
            timeout_seconds,
        )
    ):
        raise EvidenceGapError("Run token, batch, and timeout parameters must be positive")
    if max_retries < 0:
        raise EvidenceGapError("max_retries cannot be negative")

    legacy_stage_configs = {
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
    }
    llm_stages = dict(legacy_stage_configs)
    if stage_configs is not None:
        unknown = set(stage_configs) - set(llm_stages)
        if unknown:
            raise EvidenceGapError(f"Unknown LLM stage configuration: {sorted(unknown)}")
        llm_stages.update(stage_configs)
    decomposition_stage = llm_stages["statement_decomposition"]
    article_stage = llm_stages["article_evidence"]
    gap_stage = llm_stages["inference_gap"]
    localization_stage = llm_stages["localization"]
    pipeline_settings = pipeline_config or PipelineConfig()

    resolved_analysis_thinking = _resolve_deepseek_thinking(
        article_stage.provider,
        article_stage.thinking,
        label="Analysis",
    )
    resolved_gap_thinking = _resolve_deepseek_thinking(
        gap_stage.provider,
        gap_stage.thinking,
        label="Gap analysis",
    )
    resolved_decomposition_thinking = bool(decomposition_stage.thinking)
    if decomposition_stage.provider != "deepseek" and resolved_decomposition_thinking:
        raise EvidenceGapError("Decomposition thinking is only supported for DeepSeek")
    if localization_stage.thinking:
        raise EvidenceGapError("Localization thinking is not supported")

    name = _safe_name(run_name)
    base = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    target = base / name
    require_empty_or_force(target, force=force)
    target.mkdir(parents=True, exist_ok=False)

    started = time.perf_counter()
    request_path = target / "request.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": STATEMENT_RUN_SCHEMA_VERSION,
            "contract_id": STATEMENT_RUN_CONTRACT_ID,
            "run_name": name,
            "statement": statement,
            "statement_sha256": sha256_text(statement),
            "language": language,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    resolved_config_path: Path | None = None
    if resolved_config_snapshot is not None:
        resolved_config_path = target / "resolved_config.json"
        atomic_write_json(resolved_config_path, dict(resolved_config_snapshot))

    decomposition_result = run_statement_decomposition(
        root,
        statement=statement,
        provider=decomposition_stage.provider,
        run_name=_STAGE_NAMES["decomposition"],
        model=decomposition_stage.model,
        api_key_env=decomposition_stage.api_key_env,
        base_url=decomposition_stage.base_url,
        max_tokens=decomposition_stage.max_tokens,
        timeout_seconds=decomposition_stage.timeout_seconds,
        max_retries=decomposition_stage.max_retries,
        thinking=resolved_decomposition_thinking,
        prompt_override=decomposition_stage.prompt,
        artifact_root=target,
        force=False,
    )
    decomposition_dir = target / _STAGE_NAMES["decomposition"]

    decomposition_value = decomposition_result.get("decomposition")
    analysis_result = run_statement_analysis(
        root,
        decomposition_artifact_dir=decomposition_dir,
        run_name=_STAGE_NAMES["analysis"],
        provider=article_stage.provider,
        model=article_stage.model,
        device=device,
        amp=amp,
        artifact_root=target,
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
        api_key_env=article_stage.api_key_env,
        base_url=article_stage.base_url,
        request_batch_size=article_stage.request_batch_size or 1,
        max_tokens=article_stage.max_tokens,
        timeout_seconds=article_stage.timeout_seconds,
        max_retries=article_stage.max_retries,
        thinking=resolved_analysis_thinking,
        prompt_override=article_stage.prompt,
        pipeline_config=pipeline_settings,
        cache_dir=cache_dir,
        runtime_resources=runtime_resources,
        decomposition_bundle=(
            decomposition_value
            if isinstance(decomposition_value, Mapping)
            else None
        ),
        force=False,
    )
    analysis_dir = target / _STAGE_NAMES["analysis"]

    bundle_kwargs: dict[str, Any] = {}
    if (
        isinstance(analysis_result.get("decomposition"), Mapping)
        and isinstance(analysis_result.get("statement_result"), Mapping)
        and isinstance(analysis_result.get("claim_graph_bundles"), Mapping)
    ):
        bundle_kwargs = {
            "decomposition": analysis_result["decomposition"],
            "statement_result": analysis_result["statement_result"],
            "graphs_by_claim": analysis_result["claim_graph_bundles"],
        }
    bundle_result = run_statement_bundle(
        root,
        statement_analysis_artifact_dir=analysis_dir,
        run_name=_STAGE_NAMES["bundle"],
        artifact_root=target,
        force=False,
        **bundle_kwargs,
    )
    bundle_dir = target / _STAGE_NAMES["bundle"]
    bundle_path = bundle_dir / "statement_bundle.json"

    statement_bundle_value = bundle_result.get("statement_bundle")
    gap_kwargs = (
        {"statement_bundle": statement_bundle_value}
        if isinstance(statement_bundle_value, Mapping)
        else {}
    )
    gap_result = run_inference_gap_analysis(
        root,
        statement_bundle_artifact_dir=bundle_dir,
        provider=gap_stage.provider,
        run_name=_STAGE_NAMES["gaps"],
        model=gap_stage.model,
        api_key_env=gap_stage.api_key_env,
        base_url=gap_stage.base_url,
        max_tokens=gap_stage.max_tokens,
        timeout_seconds=gap_stage.timeout_seconds,
        max_retries=gap_stage.max_retries,
        thinking=resolved_gap_thinking,
        prompt_override=gap_stage.prompt,
        artifact_root=target,
        force=False,
        **gap_kwargs,
    )
    gap_dir = target / _STAGE_NAMES["gaps"]
    gap_path = gap_dir / "inference_gap_analysis.json"

    gap_bundle_value = gap_result.get("inference_gap_bundle")
    output_kwargs: dict[str, Any] = {}
    if isinstance(statement_bundle_value, Mapping) and isinstance(
        gap_bundle_value, Mapping
    ):
        output_kwargs = {
            "statement_bundle": statement_bundle_value,
            "gap_bundle": gap_bundle_value,
        }
    output_result = run_output_module(
        root,
        statement_bundle_artifact_dir=bundle_dir,
        inference_gap_artifact_dir=gap_dir,
        run_name=_STAGE_NAMES["output"],
        language=language,
        provider=localization_stage.provider,
        model=localization_stage.model,
        api_key_env=localization_stage.api_key_env,
        base_url=localization_stage.base_url,
        max_tokens=localization_stage.max_tokens,
        request_batch_size=localization_stage.request_batch_size or 32,
        timeout_seconds=localization_stage.timeout_seconds,
        max_retries=localization_stage.max_retries,
        prompt_override=localization_stage.prompt,
        artifact_root=target,
        force=False,
        **output_kwargs,
    )
    output_dir = target / _STAGE_NAMES["output"]
    presentation_path = output_dir / "presentation_bundle.json"

    stage_artifacts = {
        key: _stage_meta(root, target / directory)
        for key, directory in _STAGE_NAMES.items()
    }
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
    manifest = {
        "schema_version": STATEMENT_RUN_SCHEMA_VERSION,
        "contract_id": STATEMENT_RUN_CONTRACT_ID,
        "run_type": "phase077_end_to_end_presentation",
        "run_name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "statement_id": decomposition_result["statement_id"],
        "analysis_status": analysis_result["analysis_status"],
        "output_language": output_result["output_language"],
        "localized": output_result["localized"],
        "execution": {
            "provider": article_stage.provider,
            "model": article_stage.model,
            "device": device,
            "amp": amp,
            "llm_stages": {
                name: _stage_execution_config(stage)
                for name, stage in llm_stages.items()
            },
            "pipeline": pipeline_settings.safe_dict(),
            "decomposition_max_tokens": decomposition_stage.max_tokens,
            "analysis_request_batch_size": article_stage.request_batch_size,
            "analysis_max_tokens": article_stage.max_tokens,
            "gap_max_tokens": gap_stage.max_tokens,
            "translation_max_tokens": localization_stage.max_tokens,
            "translation_request_batch_size": localization_stage.request_batch_size,
            "decomposition_thinking": (
                resolved_decomposition_thinking
                if decomposition_stage.provider == "deepseek"
                else None
            ),
            "analysis_thinking": (
                resolved_analysis_thinking
                if article_stage.provider == "deepseek"
                else None
            ),
            "gap_thinking": (
                resolved_gap_thinking if gap_stage.provider == "deepseek" else None
            ),
            "resource_lifecycle": (
                "engine_resident"
                if runtime_resources is not None
                else "per_call"
            ),
            "stage_handoff": "in_memory_with_artifact_persistence",
        },
        "stages": stage_artifacts,
        "counts": counts,
        "outputs": {
            **(
                {
                    "resolved_config": {
                        "path": relative_path(root, resolved_config_path),
                        "sha256": sha256_file(resolved_config_path),
                    }
                }
                if resolved_config_path is not None
                else {}
            ),
            "statement_bundle": {
                "path": relative_path(root, bundle_path),
                "sha256": sha256_file(bundle_path),
            },
            "inference_gap_analysis": {
                "path": relative_path(root, gap_path),
                "sha256": sha256_file(gap_path),
            },
            "presentation_bundle": {
                "path": relative_path(root, presentation_path),
                "sha256": sha256_file(presentation_path),
            },
        },
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_write_json(target / "run_manifest.json", manifest)

    return {
        "status": str(analysis_result["analysis_status"]).upper(),
        "artifact_status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "statement_id": decomposition_result["statement_id"],
        "analysis_status": analysis_result["analysis_status"],
        "output_language": output_result["output_language"],
        "localized": output_result["localized"],
        **counts,
        "empty_claims": counts["total_claims"] == 0,
        "statement_bundle_path": relative_path(root, bundle_path),
        "inference_gap_analysis_path": relative_path(root, gap_path),
        "presentation_bundle_path": relative_path(root, presentation_path),
        "presentation_bundle": (
            output_result["presentation_bundle"]
            if isinstance(output_result.get("presentation_bundle"), Mapping)
            else _read_json_object(presentation_path)
        ),
    }


def validate_statement_pipeline_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    request = _read_json_object(artifact_dir / "request.json")
    manifest = _read_json_object(artifact_dir / "run_manifest.json")
    for document, label in ((request, "request"), (manifest, "manifest")):
        if document.get("schema_version") != STATEMENT_RUN_SCHEMA_VERSION:
            raise EvidenceGapError(f"Unexpected statement run {label} schema_version")
        if document.get("contract_id") != STATEMENT_RUN_CONTRACT_ID:
            raise EvidenceGapError(f"Unexpected statement run {label} contract_id")

    statement = str(request.get("statement") or "")
    if not statement or sha256_text(statement) != str(
        request.get("statement_sha256") or ""
    ):
        raise EvidenceGapError("Statement run request identity mismatch")
    if request.get("run_name") != manifest.get("run_name"):
        raise EvidenceGapError("Statement run name mismatch")
    if request.get("language") != manifest.get("output_language"):
        raise EvidenceGapError("Statement run output language mismatch")

    root = _find_repo_root(artifact_dir)
    stages = manifest.get("stages")
    if not isinstance(stages, Mapping) or set(stages) != set(_STAGE_NAMES):
        raise EvidenceGapError("Statement run stages are incomplete")

    stage_dirs: dict[str, Path] = {}
    for key in _STAGE_NAMES:
        meta = stages.get(key)
        if not isinstance(meta, Mapping):
            raise EvidenceGapError(f"Invalid statement run stage metadata: {key}")
        stage_dir = _resolve(root, str(meta.get("artifact_dir") or ""))
        manifest_meta = meta.get("run_manifest")
        if not isinstance(manifest_meta, Mapping):
            raise EvidenceGapError(f"Missing nested stage manifest metadata: {key}")
        nested_manifest = _resolve(root, str(manifest_meta.get("path") or ""))
        if (
            stage_dir != artifact_dir / _STAGE_NAMES[key]
            or nested_manifest != stage_dir / "run_manifest.json"
            or not nested_manifest.is_file()
            or sha256_file(nested_manifest) != str(manifest_meta.get("sha256") or "")
        ):
            raise EvidenceGapError(f"Statement run stage checksum mismatch: {key}")
        stage_dirs[key] = stage_dir

    decomposition_validation = validate_statement_decomposition_artifact(
        stage_dirs["decomposition"]
    )
    analysis_validation = validate_statement_analysis_artifact(stage_dirs["analysis"])
    bundle_validation = validate_statement_bundle_artifact(stage_dirs["bundle"])
    gap_validation = validate_inference_gap_analysis_artifact(stage_dirs["gaps"])
    output_validation = validate_output_artifact(stage_dirs["output"])

    analysis_request = _read_json_object(stage_dirs["analysis"] / "request.json")
    if _resolve(
        root, str(analysis_request.get("decomposition_artifact_dir") or "")
    ) != stage_dirs["decomposition"]:
        raise EvidenceGapError("Statement run analysis source mismatch")
    bundle_manifest = _read_json_object(stage_dirs["bundle"] / "run_manifest.json")
    bundle_source = bundle_manifest.get("source")
    if not isinstance(bundle_source, Mapping) or _resolve(
        root, str(bundle_source.get("statement_analysis_artifact_dir") or "")
    ) != stage_dirs["analysis"]:
        raise EvidenceGapError("Statement run bundle source mismatch")

    gap_request = _read_json_object(stage_dirs["gaps"] / "request.json")
    if _resolve(
        root, str(gap_request.get("statement_bundle_artifact_dir") or "")
    ) != stage_dirs["bundle"]:
        raise EvidenceGapError("Statement run gap source mismatch")
    output_request = _read_json_object(stage_dirs["output"] / "request.json")
    if _resolve(
        root, str(output_request.get("statement_bundle_artifact_dir") or "")
    ) != stage_dirs["bundle"] or _resolve(
        root, str(output_request.get("inference_gap_artifact_dir") or "")
    ) != stage_dirs["gaps"]:
        raise EvidenceGapError("Statement run output source mismatch")

    statement_ids = {
        decomposition_validation.get("statement_id"),
        analysis_validation.get("statement_id"),
        bundle_validation.get("statement_id"),
        gap_validation.get("statement_id"),
        output_validation.get("statement_id"),
        manifest.get("statement_id"),
    }
    if len(statement_ids) != 1:
        raise EvidenceGapError("Statement run statement identity mismatch")
    if analysis_validation.get("analysis_status") != bundle_validation.get(
        "analysis_status"
    ) or manifest.get("analysis_status") != bundle_validation.get("analysis_status"):
        raise EvidenceGapError("Statement run analysis status mismatch")
    if output_validation.get("output_language") != manifest.get("output_language"):
        raise EvidenceGapError("Statement run presentation language mismatch")
    if bool(output_validation.get("localized")) != bool(manifest.get("localized")):
        raise EvidenceGapError("Statement run localization status mismatch")

    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise EvidenceGapError("Statement run output metadata is missing")
    expected_output_paths = {
        "statement_bundle": stage_dirs["bundle"] / "statement_bundle.json",
        "inference_gap_analysis": stage_dirs["gaps"] / "inference_gap_analysis.json",
        "presentation_bundle": stage_dirs["output"] / "presentation_bundle.json",
    }
    for key, expected_path in expected_output_paths.items():
        output_meta = outputs.get(key)
        if not isinstance(output_meta, Mapping):
            raise EvidenceGapError(f"Statement run {key} metadata is missing")
        actual_path = _resolve(root, str(output_meta.get("path") or ""))
        if (
            actual_path != expected_path
            or not actual_path.is_file()
            or sha256_file(actual_path) != str(output_meta.get("sha256") or "")
        ):
            raise EvidenceGapError(f"Statement run {key} checksum mismatch")
    resolved_config_meta = outputs.get("resolved_config")
    if resolved_config_meta is not None:
        if not isinstance(resolved_config_meta, Mapping):
            raise EvidenceGapError("Statement run resolved_config metadata is invalid")
        resolved_config_path = _resolve(
            root, str(resolved_config_meta.get("path") or "")
        )
        if (
            resolved_config_path != artifact_dir / "resolved_config.json"
            or not resolved_config_path.is_file()
            or sha256_file(resolved_config_path)
            != str(resolved_config_meta.get("sha256") or "")
        ):
            raise EvidenceGapError("Statement run resolved_config checksum mismatch")

    expected_counts = {
        key: int(bundle_validation[key])
        for key in (
            "total_claims",
            "completed_claims",
            "failed_claims",
            "articles",
            "evidence",
        )
    }
    expected_counts.update(
        {
            "total_inference_steps": int(gap_validation["total_inference_steps"]),
            "scope_gaps": int(gap_validation["scope_gaps"]),
            "causal_gaps": int(gap_validation["causal_gaps"]),
        }
    )
    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(key, -1)) != value for key, value in expected_counts.items()
    ):
        raise EvidenceGapError("Statement run count mismatch")

    return {
        "status": "PASS",
        "run_name": manifest.get("run_name"),
        "statement_id": manifest.get("statement_id"),
        "analysis_status": manifest.get("analysis_status"),
        "output_language": manifest.get("output_language"),
        "localized": bool(manifest.get("localized")),
        **expected_counts,
        "empty_claims": expected_counts["total_claims"] == 0,
        "statement_bundle_path": relative_path(
            root, expected_output_paths["statement_bundle"]
        ),
        "inference_gap_analysis_path": relative_path(
            root, expected_output_paths["inference_gap_analysis"]
        ),
        "presentation_bundle_path": relative_path(
            root, expected_output_paths["presentation_bundle"]
        ),
        "checksums": "PASS",
    }
