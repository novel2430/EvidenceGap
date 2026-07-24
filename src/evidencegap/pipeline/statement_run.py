from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evidencegap.common import (
    EvidenceGapError,
    atomic_write_json,
    relative_path,
    require_empty_or_force,
    sha256_file,
    sha256_text,
)
from evidencegap.pipeline.statement_analysis import (
    run_statement_analysis,
    validate_statement_analysis_artifact,
)
from evidencegap.pipeline.statement_bundle import (
    run_statement_bundle,
    validate_statement_bundle_artifact,
)
from evidencegap.pipeline.statement_decomposition import (
    run_statement_decomposition,
    validate_statement_decomposition_artifact,
)

STATEMENT_RUN_SCHEMA_VERSION = "1.0.0"
STATEMENT_RUN_CONTRACT_ID = "phase075.statement-run.v1"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/statement_run")

_STAGE_NAMES = {
    "decomposition": "decomposition",
    "analysis": "analysis",
    "bundle": "bundle",
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
    current = start.resolve()
    while current.parent != current:
        if (current / "src/evidencegap").exists():
            return current
        current = current.parent
    return start.resolve()


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
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    decomposition_thinking: bool = False,
    analysis_thinking: bool | None = None,
    cache_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    statement = statement.strip()
    if not statement:
        raise EvidenceGapError("Statement cannot be blank")
    if decomposition_max_tokens <= 0:
        raise EvidenceGapError("decomposition_max_tokens must be positive")
    if provider == "deepseek":
        resolved_analysis_thinking = (
            True if analysis_thinking is None else analysis_thinking
        )
    else:
        if decomposition_thinking or analysis_thinking is True:
            raise EvidenceGapError("Thinking mode is only supported for DeepSeek")
        resolved_analysis_thinking = False

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
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    decomposition_result = run_statement_decomposition(
        root,
        statement=statement,
        provider=provider,
        run_name=_STAGE_NAMES["decomposition"],
        model=model,
        api_key_env=api_key_env,
        base_url=base_url,
        max_tokens=decomposition_max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        thinking=decomposition_thinking,
        artifact_root=target,
        force=False,
    )
    decomposition_dir = target / _STAGE_NAMES["decomposition"]

    analysis_result = run_statement_analysis(
        root,
        decomposition_artifact_dir=decomposition_dir,
        run_name=_STAGE_NAMES["analysis"],
        provider=provider,
        model=model,
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
        api_key_env=api_key_env,
        base_url=base_url,
        request_batch_size=request_batch_size,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        thinking=resolved_analysis_thinking,
        cache_dir=cache_dir,
        force=False,
    )
    analysis_dir = target / _STAGE_NAMES["analysis"]

    bundle_result = run_statement_bundle(
        root,
        statement_analysis_artifact_dir=analysis_dir,
        run_name=_STAGE_NAMES["bundle"],
        artifact_root=target,
        force=False,
    )
    bundle_dir = target / _STAGE_NAMES["bundle"]
    bundle_path = bundle_dir / "statement_bundle.json"

    stages = {
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
    manifest = {
        "schema_version": STATEMENT_RUN_SCHEMA_VERSION,
        "contract_id": STATEMENT_RUN_CONTRACT_ID,
        "run_type": "phase075_end_to_end_statement_analysis",
        "run_name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "statement_id": decomposition_result["statement_id"],
        "analysis_status": analysis_result["analysis_status"],
        "execution": {
            "provider": provider,
            "model": model,
            "device": device,
            "amp": amp,
            "decomposition_max_tokens": decomposition_max_tokens,
            "request_batch_size": request_batch_size,
            "max_tokens": max_tokens,
            "decomposition_thinking": (
                decomposition_thinking if provider == "deepseek" else None
            ),
            "analysis_thinking": (
                resolved_analysis_thinking if provider == "deepseek" else None
            ),
        },
        "stages": stages,
        "counts": counts,
        "output": {
            "statement_bundle": {
                "path": relative_path(root, bundle_path),
                "sha256": sha256_file(bundle_path),
            }
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
        **counts,
        "empty_claims": counts["total_claims"] == 0,
        "statement_bundle_path": relative_path(root, bundle_path),
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

    statement_ids = {
        decomposition_validation.get("statement_id"),
        analysis_validation.get("statement_id"),
        bundle_validation.get("statement_id"),
        manifest.get("statement_id"),
    }
    if len(statement_ids) != 1:
        raise EvidenceGapError("Statement run statement identity mismatch")
    if analysis_validation.get("analysis_status") != bundle_validation.get(
        "analysis_status"
    ) or manifest.get("analysis_status") != bundle_validation.get("analysis_status"):
        raise EvidenceGapError("Statement run analysis status mismatch")

    output = manifest.get("output")
    output_meta = output.get("statement_bundle") if isinstance(output, Mapping) else None
    if not isinstance(output_meta, Mapping):
        raise EvidenceGapError("Statement run output metadata is missing")
    bundle_path = _resolve(root, str(output_meta.get("path") or ""))
    if (
        bundle_path != stage_dirs["bundle"] / "statement_bundle.json"
        or not bundle_path.is_file()
        or sha256_file(bundle_path) != str(output_meta.get("sha256") or "")
    ):
        raise EvidenceGapError("Statement run output checksum mismatch")

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
        **expected_counts,
        "empty_claims": expected_counts["total_claims"] == 0,
        "statement_bundle_path": relative_path(root, bundle_path),
        "checksums": "PASS",
    }
