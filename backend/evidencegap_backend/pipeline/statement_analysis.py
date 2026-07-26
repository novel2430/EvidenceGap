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
    find_workspace_root,
)
from evidencegap_backend.config import PipelineConfig
from evidencegap_backend.prompting import PromptOverride
from evidencegap_backend.pipeline.analysis import run_analysis, validate_analysis_artifact
from evidencegap_backend.pipeline.retrieval_adapters import runtime_claim_id
from evidencegap_backend.pipeline.statement_decomposition import (
    validate_decomposition_bundle,
    validate_statement_decomposition_artifact,
)

STATEMENT_ANALYSIS_SCHEMA_VERSION = "1.0.0"
STATEMENT_ANALYSIS_CONTRACT_ID = "phase075.statement-analysis.v1"
if TYPE_CHECKING:
    from evidencegap_backend.resources import RuntimeResources

DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/statement_analysis")


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


def _analysis_status(*, total: int, completed: int, failed: int) -> str:
    if total == 0 or failed == 0:
        return "completed"
    if completed == 0:
        return "failed"
    return "partial_failure"


def validate_statement_analysis_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    if bundle.get("schema_version") != STATEMENT_ANALYSIS_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected statement analysis schema_version")
    if bundle.get("contract_id") != STATEMENT_ANALYSIS_CONTRACT_ID:
        raise EvidenceGapError("Unexpected statement analysis contract_id")

    statement_id = str(bundle.get("statement_id") or "").strip()
    original_statement = str(bundle.get("original_statement") or "").strip()
    source_language = str(bundle.get("source_language") or "").strip()
    if not statement_id or not original_statement or not source_language:
        raise EvidenceGapError("Statement analysis source fields cannot be blank")

    claims = bundle.get("claim_results")
    if not isinstance(claims, list):
        raise EvidenceGapError("Statement analysis claim_results must be an array")

    completed = 0
    failed = 0
    claim_ids: set[str] = set()
    for row in claims:
        if not isinstance(row, Mapping):
            raise EvidenceGapError("Statement analysis claim result must be an object")
        claim_id = str(row.get("claim_id") or "").strip()
        source_text = str(row.get("source_text") or "").strip()
        canonical = str(row.get("canonical_claim_en") or "").strip()
        status = str(row.get("status") or "").strip()
        if (
            not claim_id
            or claim_id != runtime_claim_id(canonical)
            or claim_id in claim_ids
            or not source_text
            or not canonical
            or source_text not in original_statement
        ):
            raise EvidenceGapError("Invalid statement analysis claim identity")
        claim_ids.add(claim_id)

        verdict = row.get("verdict")
        graph_path = row.get("graph_bundle_path")
        error = row.get("error")
        if status == "completed":
            completed += 1
            if verdict not in {"supported", "refuted", "mixed", "insufficient"}:
                raise EvidenceGapError("Completed claim has invalid verdict")
            if (
                not str(row.get("phase07_artifact_dir") or "").strip()
                or not str(graph_path or "").strip()
                or error is not None
            ):
                raise EvidenceGapError("Completed claim result is incomplete")
        elif status == "failed":
            failed += 1
            if verdict is not None or graph_path is not None or not str(error or "").strip():
                raise EvidenceGapError("Failed claim result is inconsistent")
        else:
            raise EvidenceGapError(f"Unexpected claim analysis status: {status!r}")

    summary = bundle.get("summary")
    if not isinstance(summary, Mapping):
        raise EvidenceGapError("Statement analysis summary must be an object")
    total = len(claims)
    if (
        int(summary.get("total_claims", -1)) != total
        or int(summary.get("completed_claims", -1)) != completed
        or int(summary.get("failed_claims", -1)) != failed
    ):
        raise EvidenceGapError("Statement analysis summary count mismatch")

    expected_status = _analysis_status(
        total=total,
        completed=completed,
        failed=failed,
    )
    if bundle.get("analysis_status") != expected_status:
        raise EvidenceGapError("Statement analysis status mismatch")

    return {
        "status": "PASS",
        "statement_id": statement_id,
        "analysis_status": expected_status,
        "total_claims": total,
        "completed_claims": completed,
        "failed_claims": failed,
        "empty_claims": total == 0,
    }


def run_statement_analysis(
    root: Path,
    *,
    decomposition_artifact_dir: Path,
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
    decomposition_bundle: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    decomposition_dir = _resolve(root, decomposition_artifact_dir)
    decomposition_path = decomposition_dir / "decomposition.json"
    if decomposition_bundle is None:
        validate_statement_decomposition_artifact(decomposition_dir)
        decomposition = _read_json_object(decomposition_path)
        source_handoff = "artifact_reload"
    else:
        decomposition = dict(decomposition_bundle)
        source_handoff = "in_memory_handoff"
    decomposition_validation = validate_decomposition_bundle(decomposition)

    name = _safe_name(run_name)
    base = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    target = base / name
    require_empty_or_force(target, force=force)
    target.mkdir(parents=True, exist_ok=False)
    claims_root = target / "claims"
    claims_root.mkdir()

    started = time.perf_counter()
    request = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "run_name": name,
        "statement_id": decomposition_validation["statement_id"],
        "decomposition_artifact_dir": relative_path(root, decomposition_dir),
        "decomposition_path": relative_path(root, decomposition_path),
        "decomposition_sha256": sha256_file(decomposition_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(target / "request.json", request)

    claim_results: list[dict[str, Any]] = []
    claim_graph_bundles: dict[str, dict[str, Any]] = {}
    for claim in decomposition["claims"]:
        claim_id = str(claim["claim_id"])
        source_text = str(claim["source_text"])
        canonical_claim_en = str(claim["canonical_claim_en"])
        claim_artifact_dir = claims_root / claim_id
        try:
            result = run_analysis(
                root,
                claim=canonical_claim_en,
                run_name=claim_id,
                provider=provider,
                model=model,
                device=device,
                amp=amp,
                artifact_root=claims_root,
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
                thinking=thinking,
                cache_dir=cache_dir,
                runtime_resources=runtime_resources,
                prompt_override=prompt_override,
                pipeline_config=pipeline_config,
                force=False,
            )
            if result.get("claim_id") != claim_id:
                raise EvidenceGapError(
                    f"Phase 07 claim identity mismatch for {claim_id}"
                )
            graph_bundle = result.get("graph_bundle")
            if not isinstance(graph_bundle, dict):
                graph_path = _resolve(
                    root, str(result.get("graph_bundle_path") or "")
                )
                graph_bundle = _read_json_object(graph_path)
            claim_graph_bundles[claim_id] = graph_bundle
            claim_results.append(
                {
                    "claim_id": claim_id,
                    "source_text": source_text,
                    "canonical_claim_en": canonical_claim_en,
                    "status": "completed",
                    "phase07_artifact_dir": relative_path(root, claim_artifact_dir),
                    "graph_bundle_path": str(result["graph_bundle_path"]),
                    "verdict": str(result["verdict"]),
                    "error": None,
                }
            )
        except EvidenceGapError as exc:
            claim_results.append(
                {
                    "claim_id": claim_id,
                    "source_text": source_text,
                    "canonical_claim_en": canonical_claim_en,
                    "status": "failed",
                    "phase07_artifact_dir": (
                        relative_path(root, claim_artifact_dir)
                        if claim_artifact_dir.exists()
                        else None
                    ),
                    "graph_bundle_path": None,
                    "verdict": None,
                    "error": str(exc),
                }
            )

    completed = sum(row["status"] == "completed" for row in claim_results)
    failed = sum(row["status"] == "failed" for row in claim_results)
    analysis_status = _analysis_status(
        total=len(claim_results),
        completed=completed,
        failed=failed,
    )
    bundle = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "statement_id": decomposition_validation["statement_id"],
        "original_statement": decomposition["original_statement"],
        "source_language": decomposition["source_language"],
        "analysis_status": analysis_status,
        "claim_results": claim_results,
        "summary": {
            "total_claims": len(claim_results),
            "completed_claims": completed,
            "failed_claims": failed,
        },
    }
    validation = validate_statement_analysis_bundle(bundle)
    result_path = target / "statement_result.json"
    atomic_write_json(result_path, bundle)

    manifest = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "run_type": "phase075_multi_claim_phase07_analysis",
        "run_name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "statement_id": decomposition_validation["statement_id"],
        "analysis_status": analysis_status,
        "execution": {
            "provider": provider,
            "model": model,
            "device": device,
            "amp": amp,
            "request_batch_size": request_batch_size,
            "max_tokens": max_tokens,
            "thinking": thinking if provider == "deepseek" else None,
            "resource_lifecycle": (
                "engine_resident"
                if runtime_resources is not None
                else "per_call"
            ),
            "decomposition_handoff": source_handoff,
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

    return {
        **validation,
        "status": analysis_status.upper(),
        "artifact_status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "statement_result_path": relative_path(root, result_path),
        "statement_result": bundle,
        "decomposition": decomposition,
        "claim_graph_bundles": claim_graph_bundles,
    }


def validate_statement_analysis_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest = _read_json_object(artifact_dir / "run_manifest.json")
    request = _read_json_object(artifact_dir / "request.json")
    if manifest.get("schema_version") != STATEMENT_ANALYSIS_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected statement analysis manifest schema_version")
    if manifest.get("contract_id") != STATEMENT_ANALYSIS_CONTRACT_ID:
        raise EvidenceGapError("Unexpected statement analysis manifest contract_id")
    if request.get("schema_version") != STATEMENT_ANALYSIS_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected statement analysis request schema_version")
    if request.get("contract_id") != STATEMENT_ANALYSIS_CONTRACT_ID:
        raise EvidenceGapError("Unexpected statement analysis request contract_id")

    root = _find_repo_root(artifact_dir)
    output_meta = manifest.get("outputs", {}).get("statement_result", {})
    output_path = _resolve(root, str(output_meta.get("path") or ""))
    if not output_path.is_file() or sha256_file(output_path) != str(
        output_meta.get("sha256") or ""
    ):
        raise EvidenceGapError("Statement analysis output checksum mismatch")
    bundle = _read_json_object(output_path)
    validation = validate_statement_analysis_bundle(bundle)

    decomposition_dir = _resolve(
        root, str(request.get("decomposition_artifact_dir") or "")
    )
    validate_statement_decomposition_artifact(decomposition_dir)
    decomposition_path = decomposition_dir / "decomposition.json"
    if sha256_file(decomposition_path) != str(
        request.get("decomposition_sha256") or ""
    ):
        raise EvidenceGapError("Statement analysis decomposition checksum mismatch")
    decomposition = _read_json_object(decomposition_path)
    validate_decomposition_bundle(decomposition)

    if (
        bundle.get("statement_id") != decomposition.get("statement_id")
        or bundle.get("original_statement") != decomposition.get("original_statement")
        or bundle.get("source_language") != decomposition.get("source_language")
    ):
        raise EvidenceGapError("Statement analysis source does not match decomposition")

    expected_claims = decomposition.get("claims", [])
    actual_claims = bundle.get("claim_results", [])
    if len(expected_claims) != len(actual_claims):
        raise EvidenceGapError("Statement analysis claim count differs from decomposition")
    for expected, actual in zip(expected_claims, actual_claims, strict=True):
        for key in ("claim_id", "source_text", "canonical_claim_en"):
            if actual.get(key) != expected.get(key):
                raise EvidenceGapError(
                    f"Statement analysis claim does not match decomposition: {key}"
                )
        if actual.get("status") != "completed":
            continue
        phase07_dir = _resolve(root, str(actual.get("phase07_artifact_dir") or ""))
        phase07_validation = validate_analysis_artifact(phase07_dir)
        if phase07_validation.get("claim_id") != actual.get("claim_id"):
            raise EvidenceGapError("Nested Phase 07 claim identity mismatch")
        if phase07_validation.get("verdict") != actual.get("verdict"):
            raise EvidenceGapError("Nested Phase 07 verdict mismatch")
        graph_path = _resolve(root, str(actual.get("graph_bundle_path") or ""))
        if not graph_path.is_file():
            raise EvidenceGapError("Nested Phase 07 graph bundle is missing")

    counts = manifest.get("counts", {})
    if any(
        int(counts.get(key, -1)) != int(bundle["summary"][key])
        for key in ("total_claims", "completed_claims", "failed_claims")
    ):
        raise EvidenceGapError("Statement analysis manifest count mismatch")
    if manifest.get("analysis_status") != validation["analysis_status"]:
        raise EvidenceGapError("Statement analysis manifest status mismatch")

    return {
        **validation,
        "run_name": manifest.get("run_name"),
        "checksums": "PASS",
    }
