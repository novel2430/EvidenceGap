from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_write_json,
    relative_path,
    require_empty_or_force,
    sha256_file,
    sha256_text,
    find_workspace_root,
)
from evidencegap_backend.config import PipelineConfig
from evidencegap_backend.prompting import PromptOverride
from evidencegap_backend.pipeline.article_evidence import (
    build_article_prompt_inputs,
    load_article_prompt_inputs,
    run_article_evidence_extractor,
    validate_article_evidence_artifact,
)
from evidencegap_backend.pipeline.claim_aggregation import (
    run_claim_aggregation,
    validate_claim_aggregation_artifact,
)
from evidencegap_backend.pipeline.final_graph import (
    run_final_graph,
    validate_final_graph_artifact,
)
from evidencegap_backend.pipeline.retrieval_adapters import (
    RUNTIME_RETRIEVAL_CONTRACT_ID,
    RUNTIME_RETRIEVAL_SCHEMA_VERSION,
    retrieve_runtime_articles,
    runtime_claim_id,
)
from evidencegap_backend.pipeline.sentence_materialization import (
    DEFAULT_STANZA_MODEL_DIR,
    materialize_runtime_sentences,
    validate_runtime_sentence_artifact,
)


if TYPE_CHECKING:
    from evidencegap_backend.resources import RuntimeResources

ANALYSIS_SCHEMA_VERSION = "1.0.0"
ANALYSIS_CONTRACT_ID = "phase07.analysis.v1"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/analyze")

_STAGE_DIRS = {
    "article_retrieval": "article_retrieval",
    "sentence_materialization": "sentence_materialization",
    "article_evidence": "article_evidence",
    "claim_aggregation": "claim_aggregation",
    "final_graph": "final_graph",
}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty")
    return cleaned


def _clean_claim(value: str) -> str:
    claim = " ".join(str(value).strip().split())
    if not claim:
        raise EvidenceGapError("claim cannot be empty")
    return claim


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing analysis artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceGapError(f"Expected JSON object in {path}")
    return value


def _find_repo_root(start: Path) -> Path:
    return find_workspace_root(start)


def _manifest_output(root: Path, path: Path) -> dict[str, str]:
    return {
        "path": relative_path(root, path),
        "sha256": sha256_file(path),
    }


def run_analysis(
    root: Path,
    *,
    claim: str,
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
    request_batch_size: int = 2,
    max_tokens: int = 4096,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    thinking: bool = False,
    cache_dir: Path | None = None,
    runtime_resources: "RuntimeResources | None" = None,
    prompt_override: PromptOverride | None = None,
    pipeline_config: PipelineConfig | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    claim_text = _clean_claim(claim)
    claim_id = runtime_claim_id(claim_text)
    name = _safe_name(run_name)
    base = artifact_root.resolve() if artifact_root else (root / DEFAULT_ARTIFACT_ROOT)
    target = base / name
    require_empty_or_force(target, force=force)
    target.mkdir(parents=True, exist_ok=False)
    pipeline_settings = pipeline_config or PipelineConfig()

    started = time.perf_counter()
    request_path = target / "request.json"
    atomic_write_json(
        request_path,
        {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "contract_id": ANALYSIS_CONTRACT_ID,
            "run_name": name,
            "claim_id": claim_id,
            "claim_text": claim_text,
            "claim_text_sha256": sha256_text(claim_text),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    article_result = retrieve_runtime_articles(
        root,
        claim_id=claim_id,
        claim_text=claim_text,
        artifact_dir=target / _STAGE_DIRS["article_retrieval"],
        device=device,
        amp=amp,
        corpus_dir=corpus_dir,
        article_input_dir=article_input_dir,
        bm25_index_dir=bm25_index_dir,
        medcpt_index_dir=medcpt_index_dir,
        bmretriever_index_dir=bmretriever_index_dir,
        cross_encoder_model_dir=cross_encoder_model_dir,
        cross_encoder_batch_size=cross_encoder_batch_size,
        source_depth=pipeline_settings.source_depth,
        dense_nprobe=pipeline_settings.dense_nprobe,
        rrf_k=pipeline_settings.rrf_k,
        rerank_depth=pipeline_settings.rerank_depth,
        final_article_top_k=pipeline_settings.final_article_top_k,
        runtime_resources=runtime_resources,
        force=False,
    )
    runtime_articles_input = _resolve(
        root, article_result["outputs"]["runtime_articles_input"]["path"]
    )

    sentence_result = materialize_runtime_sentences(
        root,
        input_path=runtime_articles_input,
        run_name=_STAGE_DIRS["sentence_materialization"],
        model_dir=(
            stanza_model_dir.resolve()
            if stanza_model_dir is not None
            else (root / DEFAULT_STANZA_MODEL_DIR).resolve()
        ),
        device=device,
        package=stanza_package,
        batch_size=stanza_batch_size,
        section_mode=section_mode,
        allow_cpu_fallback=allow_cpu_fallback,
        artifact_root=target,
        force=False,
        splitter=(
            None
            if runtime_resources is None
            else runtime_resources.sentence_splitter
        ),
        runtime_articles=(
            article_result.get("top_article_rows")
            if isinstance(article_result.get("top_article_rows"), list)
            else None
        ),
    )

    article_rows_value = article_result.get("top_article_rows")
    sentence_rows_value = sentence_result.get("runtime_sentence_rows")
    prompt_kwargs: dict[str, Any] = {}
    if isinstance(article_rows_value, list) and isinstance(
        sentence_rows_value, list
    ):
        prompt_kwargs = {
            "prompt_request": {
                "claim_id": claim_id,
                "claim_text": claim_text,
            },
            "prompt_items": build_article_prompt_inputs(
                claim_id=claim_id,
                claim_text=claim_text,
                articles=article_rows_value,
                runtime_sentences=sentence_rows_value,
            ),
        }
    article_evidence_result = run_article_evidence_extractor(
        root,
        retrieval_artifact_dir=target,
        provider=provider,
        model=model,
        run_name=_STAGE_DIRS["article_evidence"],
        api_key_env=api_key_env,
        base_url=base_url,
        request_batch_size=request_batch_size,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        thinking=thinking,
        cache_dir=cache_dir,
        prompt_override=prompt_override,
        max_evidence_sentences=pipeline_settings.max_evidence_sentences,
        artifact_root=target,
        force=False,
        **prompt_kwargs,
    )

    article_evidence_rows = article_evidence_result.get("article_evidence_rows")
    aggregation_kwargs = (
        {"article_rows": article_evidence_rows}
        if isinstance(article_evidence_rows, list)
        else {}
    )
    aggregation_result = run_claim_aggregation(
        root,
        article_evidence_artifact_dir=target / _STAGE_DIRS["article_evidence"],
        run_name=_STAGE_DIRS["claim_aggregation"],
        artifact_root=target,
        force=False,
        **aggregation_kwargs,
    )
    claim_result_value = aggregation_result.get("claim_result")
    graph_kwargs: dict[str, Any] = {}
    if isinstance(claim_result_value, dict) and isinstance(
        article_evidence_rows, list
    ):
        graph_kwargs = {
            "claim_result": claim_result_value,
            "article_rows": article_evidence_rows,
        }
    graph_result = run_final_graph(
        root,
        claim_aggregation_artifact_dir=target / _STAGE_DIRS["claim_aggregation"],
        run_name=_STAGE_DIRS["final_graph"],
        artifact_root=target,
        force=False,
        **graph_kwargs,
    )

    stage_manifests = {
        key: target / directory / "run_manifest.json"
        for key, directory in _STAGE_DIRS.items()
    }
    article_evidence_manifest = _read_json(stage_manifests["article_evidence"])
    graph_bundle_path = target / _STAGE_DIRS["final_graph"] / "graph_bundle.json"
    manifest = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "contract_id": ANALYSIS_CONTRACT_ID,
        "run_type": "phase07_end_to_end_analysis",
        "run_name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_id": claim_id,
        "claim_text_sha256": sha256_text(claim_text),
        "pipeline": [
            "phase04_runtime_article_retrieval",
            "phase07_runtime_sentence_materialization",
            "phase07_article_llm_evidence_extractor",
            "phase07_claim_aggregation",
            "phase07_final_graph",
        ],
        "excluded_pipeline_stages": [
            "phase05_runtime_sentence_retrieval",
            "phase06_sentence_level_stance_judgment",
        ],
        "execution": {
            "device": device,
            "amp": amp,
            "provider": article_evidence_manifest.get("provider", provider),
            "model": article_evidence_manifest.get("model", model),
            "prompt_version": article_evidence_manifest.get("prompt_version"),
            "request_batch_size": request_batch_size,
            "max_tokens": max_tokens,
            "thinking": thinking if provider == "deepseek" else None,
            "stanza_package": stanza_package,
            "stanza_batch_size": stanza_batch_size,
            "cross_encoder_batch_size": cross_encoder_batch_size,
            "section_mode": section_mode,
            "allow_cpu_fallback": allow_cpu_fallback,
            "resource_lifecycle": (
                "engine_resident"
                if runtime_resources is not None
                else "per_call"
            ),
            "stage_handoff": "in_memory_with_artifact_persistence",
        },
        "counts": {
            "top_articles": int(article_result["top_articles"]),
            "runtime_sentences": int(sentence_result["sentences"]),
            "article_evidence_rows": int(article_evidence_result["articles"]),
            "evidence_selections": int(article_evidence_result["evidence_selections"]),
            "support_articles": int(aggregation_result["support_articles"]),
            "refute_articles": int(aggregation_result["refute_articles"]),
            "insufficient_articles": int(aggregation_result["insufficient_articles"]),
        },
        "verdict": aggregation_result["verdict"],
        "request": _manifest_output(root, request_path),
        "stages": {
            key: {
                "artifact_dir": relative_path(root, target / directory),
                "manifest": _manifest_output(root, stage_manifests[key]),
            }
            for key, directory in _STAGE_DIRS.items()
        },
        "output": {
            "graph_bundle": _manifest_output(root, graph_bundle_path),
        },
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_write_json(target / "run_manifest.json", manifest)
    validation = validate_analysis_artifact(target)
    return {
        "status": "PASS",
        "run_name": name,
        "claim_id": claim_id,
        "artifact_dir": relative_path(root, target),
        "verdict": aggregation_result["verdict"],
        "graph_bundle_path": relative_path(root, graph_bundle_path),
        "validation": validation,
        "graph_bundle": (
            graph_result["graph_bundle"]
            if isinstance(graph_result.get("graph_bundle"), dict)
            else _read_json(graph_bundle_path)
        ),
    }


def validate_analysis_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest = _read_json(artifact_dir / "run_manifest.json")
    request = _read_json(artifact_dir / "request.json")
    if manifest.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected analysis manifest schema_version")
    if manifest.get("contract_id") != ANALYSIS_CONTRACT_ID:
        raise EvidenceGapError("Unexpected analysis manifest contract_id")
    if request.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected analysis request schema_version")
    if request.get("contract_id") != ANALYSIS_CONTRACT_ID:
        raise EvidenceGapError("Unexpected analysis request contract_id")

    root = _find_repo_root(artifact_dir)
    request_meta = manifest.get("request", {})
    request_path = _resolve(root, str(request_meta.get("path", "")))
    if request_path != artifact_dir / "request.json":
        raise EvidenceGapError("Analysis request path mismatch")
    if sha256_file(request_path) != str(request_meta.get("sha256", "")):
        raise EvidenceGapError("Analysis request checksum mismatch")

    expected_pipeline = [
        "phase04_runtime_article_retrieval",
        "phase07_runtime_sentence_materialization",
        "phase07_article_llm_evidence_extractor",
        "phase07_claim_aggregation",
        "phase07_final_graph",
    ]
    if manifest.get("pipeline") != expected_pipeline:
        raise EvidenceGapError("Unexpected analysis pipeline")
    if (artifact_dir / "evidence_retrieval").exists():
        raise EvidenceGapError("Analysis artifact must not contain Phase 05 sentence retrieval")

    stages = manifest.get("stages")
    if not isinstance(stages, dict) or set(stages) != set(_STAGE_DIRS):
        raise EvidenceGapError("Analysis stage manifest coverage mismatch")
    for key, directory in _STAGE_DIRS.items():
        expected_dir = artifact_dir / directory
        stage = stages[key]
        stage_dir = _resolve(root, str(stage.get("artifact_dir", "")))
        if stage_dir != expected_dir:
            raise EvidenceGapError(f"Analysis stage path mismatch: {key}")
        stage_manifest = stage.get("manifest", {})
        stage_manifest_path = _resolve(root, str(stage_manifest.get("path", "")))
        if stage_manifest_path != expected_dir / "run_manifest.json":
            raise EvidenceGapError(f"Analysis stage manifest path mismatch: {key}")
        if sha256_file(stage_manifest_path) != str(stage_manifest.get("sha256", "")):
            raise EvidenceGapError(f"Analysis stage manifest checksum mismatch: {key}")

    article_manifest = _read_json(
        artifact_dir / _STAGE_DIRS["article_retrieval"] / "run_manifest.json"
    )
    if article_manifest.get("schema_version") != RUNTIME_RETRIEVAL_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected article retrieval schema_version")
    if article_manifest.get("contract_id") != RUNTIME_RETRIEVAL_CONTRACT_ID:
        raise EvidenceGapError("Unexpected article retrieval contract_id")
    if article_manifest.get("claim_id") != request.get("claim_id"):
        raise EvidenceGapError("Article retrieval claim_id mismatch")
    top_article_rows = int(
        article_manifest.get("outputs", {}).get("top_articles", {}).get("rows", -1)
    )
    expected_top_articles = int(
        article_manifest.get("parameters", {}).get("final_article_top_k", -1)
    )
    if top_article_rows != expected_top_articles or expected_top_articles <= 0:
        raise EvidenceGapError("Analysis article retrieval depth mismatch")

    validate_runtime_sentence_artifact(
        artifact_dir / _STAGE_DIRS["sentence_materialization"]
    )
    _, prompt_inputs, _ = load_article_prompt_inputs(artifact_dir)
    if len(prompt_inputs) != expected_top_articles:
        raise EvidenceGapError("Analysis prompt input article depth mismatch")
    validate_article_evidence_artifact(artifact_dir / _STAGE_DIRS["article_evidence"])
    aggregation = validate_claim_aggregation_artifact(
        artifact_dir / _STAGE_DIRS["claim_aggregation"]
    )
    graph = validate_final_graph_artifact(artifact_dir / _STAGE_DIRS["final_graph"])

    graph_meta = manifest.get("output", {}).get("graph_bundle", {})
    graph_path = _resolve(root, str(graph_meta.get("path", "")))
    if graph_path != artifact_dir / _STAGE_DIRS["final_graph"] / "graph_bundle.json":
        raise EvidenceGapError("Analysis graph bundle path mismatch")
    if sha256_file(graph_path) != str(graph_meta.get("sha256", "")):
        raise EvidenceGapError("Analysis graph bundle checksum mismatch")
    bundle = _read_json(graph_path)
    if bundle.get("claim_id") != request.get("claim_id"):
        raise EvidenceGapError("Final graph claim_id mismatch")
    if bundle.get("claim_text") != request.get("claim_text"):
        raise EvidenceGapError("Final graph claim_text mismatch")
    if bundle.get("verdict") != aggregation.get("verdict"):
        raise EvidenceGapError("Final graph verdict mismatch")
    if manifest.get("verdict") != aggregation.get("verdict"):
        raise EvidenceGapError("Analysis manifest verdict mismatch")

    return {
        "status": "PASS",
        "run_name": manifest.get("run_name"),
        "claim_id": request.get("claim_id"),
        "top_articles": top_article_rows,
        "verdict": aggregation.get("verdict"),
        "node_counts": graph.get("node_counts"),
        "relation_counts": graph.get("relation_counts"),
        "phase05_sentence_retrieval_used": False,
        "checksums": "PASS",
    }
