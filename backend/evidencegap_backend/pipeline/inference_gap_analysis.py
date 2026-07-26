from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
    find_workspace_root,
)
from evidencegap_backend.pipeline.statement_bundle import (
    validate_statement_bundle,
    validate_statement_bundle_artifact,
)
from evidencegap_backend.prompting import (
    PromptOverride,
    ResolvedPrompt,
    resolve_builtin_prompt,
)
from evidencegap_backend.stance.llm_judge import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    ProviderResponse,
    _ProviderError,
    call_structured_llm,
)

INFERENCE_GAP_ANALYSIS_SCHEMA_VERSION = "1.0.0"
INFERENCE_GAP_ANALYSIS_CONTRACT_ID = "phase077.inference-gap-analysis.v1"
INFERENCE_GAP_ANALYSIS_PROMPT_VERSION = "phase077_inference_gap_analysis_v2"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/output/inference_gap_analysis")

_VERDICT_TO_EVIDENCE_STATE = {
    "supported": "SUPPORTED",
    "refuted": "REFUTED",
    "mixed": "CONFLICTED",
    "insufficient": "INSUFFICIENT",
}


def response_json_schema() -> dict[str, Any]:
    gap_schema = {
        "type": "object",
        "properties": {
            "detected": {"type": "boolean"},
            "reason": {"type": ["string", "null"]},
        },
        "required": ["detected", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "analyses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "inference_step_id": {"type": "string"},
                        "scope_gap": gap_schema,
                        "causal_gap": gap_schema,
                    },
                    "required": [
                        "inference_step_id",
                        "scope_gap",
                        "causal_gap",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["analyses"],
        "additionalProperties": False,
    }


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid name: {value!r}")
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


def build_gap_analysis_input(statement_bundle: Mapping[str, Any]) -> dict[str, Any]:
    validate_statement_bundle(statement_bundle)
    inference_steps = statement_bundle.get("inference_steps", [])
    connected_claim_ids = {
        claim_id
        for step in inference_steps
        for claim_id in [
            *step["premise_claim_ids"],
            step["conclusion_claim_id"],
        ]
    }

    evidence_by_id = {
        str(item["evidence_id"]): item
        for item in statement_bundle.get("evidence", [])
    }
    articles_by_claim: dict[str, list[dict[str, Any]]] = {}
    for article in statement_bundle.get("articles", []):
        claim_id = str(article["claim_id"])
        if claim_id not in connected_claim_ids:
            continue
        evidence_texts = [
            {
                "evidence_id": evidence_id,
                "section": evidence_by_id[evidence_id].get("section"),
                "text": str(evidence_by_id[evidence_id].get("text") or ""),
            }
            for evidence_id in article.get("evidence_ids", [])
            if evidence_id in evidence_by_id
        ]
        articles_by_claim.setdefault(claim_id, []).append(
            {
                "article_node_id": str(article["article_node_id"]),
                "pmid": article.get("pmid"),
                "title": str(article.get("title") or ""),
                "stance": str(article.get("stance") or ""),
                "confidence": float(article.get("confidence") or 0.0),
                "rationale": str(article.get("rationale") or ""),
                "evidence": evidence_texts,
            }
        )

    claims = []
    for claim in statement_bundle.get("claims", []):
        claim_id = str(claim["claim_id"])
        if claim_id not in connected_claim_ids:
            continue
        status = str(claim.get("analysis_status") or "")
        verdict = claim.get("verdict")
        evidence_state = (
            _VERDICT_TO_EVIDENCE_STATE.get(str(verdict))
            if status == "completed"
            else "ERROR"
        )
        claims.append(
            {
                "claim_id": claim_id,
                "source_text": str(claim.get("source_text") or ""),
                "canonical_claim_en": str(
                    claim.get("canonical_claim_en") or ""
                ),
                "analysis_status": status,
                "evidence_state": evidence_state,
                "claim_rationale": claim.get("rationale"),
                "articles": articles_by_claim.get(claim_id, []),
            }
        )

    statement = statement_bundle["statement"]
    return {
        "statement_id": str(statement["statement_id"]),
        "original_statement": str(statement["original_text"]),
        "claims": claims,
        "inference_steps": [
            {
                "inference_step_id": str(step["inference_step_id"]),
                "premise_claim_ids": list(step["premise_claim_ids"]),
                "conclusion_claim_id": str(step["conclusion_claim_id"]),
            }
            for step in inference_steps
        ],
    }


def build_user_prompt(
    statement_bundle: Mapping[str, Any], *, retry_note: str | None = None
) -> str:
    payload = build_gap_analysis_input(statement_bundle)
    parts = [
        (
            "Analyze every inference step below for SCOPE_GAP and CAUSAL_GAP. "
            "Keep all claim evidence states and inference references unchanged."
        ),
        "INPUT ARGUMENT AND EVIDENCE JSON:",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ]
    if retry_note:
        parts.extend(
            [
                "The previous response was invalid. Correct this issue and return the complete JSON object:",
                retry_note,
            ]
        )
    return "\n\n".join(parts)


def _validate_gap_result(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _ProviderError(f"{label} must be an object", retryable=True)
    if set(value) != {"detected", "reason"}:
        raise _ProviderError(
            f"{label} must contain exactly detected and reason", retryable=True
        )
    detected = value.get("detected")
    if not isinstance(detected, bool):
        raise _ProviderError(f"{label}.detected must be boolean", retryable=True)
    raw_reason = value.get("reason")
    if raw_reason is None:
        reason = None
    elif isinstance(raw_reason, str):
        reason = raw_reason.strip() or None
    else:
        raise _ProviderError(f"{label}.reason must be string or null", retryable=True)
    if detected and reason is None:
        raise _ProviderError(
            f"{label}.reason must be non-empty when detected is true",
            retryable=True,
        )
    if not detected and reason is not None:
        raise _ProviderError(
            f"{label}.reason must be null when detected is false",
            retryable=True,
        )
    return {"detected": detected, "reason": reason}


def validate_response_payload(
    payload: Mapping[str, Any], *, statement_bundle: Mapping[str, Any]
) -> list[dict[str, Any]]:
    validate_statement_bundle(statement_bundle)
    if set(payload) != {"analyses"}:
        raise _ProviderError(
            "Response must contain exactly the analyses field", retryable=True
        )
    raw_analyses = payload.get("analyses")
    if not isinstance(raw_analyses, list):
        raise _ProviderError("analyses must be an array", retryable=True)

    expected_ids = [
        str(step["inference_step_id"])
        for step in statement_bundle.get("inference_steps", [])
    ]
    expected_id_set = set(expected_ids)
    analyses_by_id: dict[str, dict[str, Any]] = {}
    for raw in raw_analyses:
        if not isinstance(raw, Mapping):
            raise _ProviderError("Each analysis must be an object", retryable=True)
        if set(raw) != {"inference_step_id", "scope_gap", "causal_gap"}:
            raise _ProviderError(
                "Each analysis must contain exactly inference_step_id, scope_gap, and causal_gap",
                retryable=True,
            )
        inference_step_id = str(raw.get("inference_step_id") or "").strip()
        if inference_step_id not in expected_id_set:
            raise _ProviderError(
                f"Unknown inference_step_id: {inference_step_id!r}",
                retryable=True,
            )
        if inference_step_id in analyses_by_id:
            raise _ProviderError(
                f"Duplicate inference_step_id: {inference_step_id}",
                retryable=True,
            )
        analyses_by_id[inference_step_id] = {
            "inference_step_id": inference_step_id,
            "scope_gap": _validate_gap_result(
                raw.get("scope_gap"), label="scope_gap"
            ),
            "causal_gap": _validate_gap_result(
                raw.get("causal_gap"), label="causal_gap"
            ),
        }

    if set(analyses_by_id) != expected_id_set:
        missing = sorted(expected_id_set - set(analyses_by_id))
        raise _ProviderError(
            f"Missing analyses for inference_step_ids: {missing}", retryable=True
        )
    return [analyses_by_id[inference_step_id] for inference_step_id in expected_ids]


def build_inference_gap_analysis_bundle(
    statement_bundle: Mapping[str, Any],
    analyses: list[dict[str, Any]],
    *,
    source_statement_bundle_sha256: str,
) -> dict[str, Any]:
    validate_statement_bundle(statement_bundle)
    try:
        normalized_analyses = validate_response_payload(
            {"analyses": analyses}, statement_bundle=statement_bundle
        )
    except _ProviderError as exc:
        raise EvidenceGapError(str(exc)) from exc
    bundle = {
        "schema_version": INFERENCE_GAP_ANALYSIS_SCHEMA_VERSION,
        "contract_id": INFERENCE_GAP_ANALYSIS_CONTRACT_ID,
        "statement_id": str(statement_bundle["statement"]["statement_id"]),
        "source_statement_bundle_sha256": source_statement_bundle_sha256,
        "inference_gap_analyses": normalized_analyses,
        "summary": {
            "total_inference_steps": len(normalized_analyses),
            "scope_gaps": sum(
                bool(row["scope_gap"]["detected"])
                for row in normalized_analyses
            ),
            "causal_gaps": sum(
                bool(row["causal_gap"]["detected"])
                for row in normalized_analyses
            ),
        },
    }
    validate_inference_gap_analysis_bundle(bundle, statement_bundle=statement_bundle)
    return bundle


def validate_inference_gap_analysis_bundle(
    bundle: Mapping[str, Any], *, statement_bundle: Mapping[str, Any]
) -> dict[str, Any]:
    validate_statement_bundle(statement_bundle)
    if (
        bundle.get("schema_version") != INFERENCE_GAP_ANALYSIS_SCHEMA_VERSION
        or bundle.get("contract_id") != INFERENCE_GAP_ANALYSIS_CONTRACT_ID
    ):
        raise EvidenceGapError("Unexpected inference gap analysis contract")
    if bundle.get("statement_id") != statement_bundle["statement"]["statement_id"]:
        raise EvidenceGapError("Inference gap analysis statement identity mismatch")
    source_sha = str(bundle.get("source_statement_bundle_sha256") or "").strip()
    if not source_sha:
        raise EvidenceGapError("Inference gap analysis source checksum is missing")

    raw_analyses = bundle.get("inference_gap_analyses")
    if not isinstance(raw_analyses, list):
        raise EvidenceGapError("inference_gap_analyses must be an array")
    try:
        analyses = validate_response_payload(
            {"analyses": raw_analyses}, statement_bundle=statement_bundle
        )
    except _ProviderError as exc:
        raise EvidenceGapError(str(exc)) from exc

    expected_summary = {
        "total_inference_steps": len(analyses),
        "scope_gaps": sum(
            bool(row["scope_gap"]["detected"]) for row in analyses
        ),
        "causal_gaps": sum(
            bool(row["causal_gap"]["detected"]) for row in analyses
        ),
    }
    summary = bundle.get("summary")
    if not isinstance(summary, Mapping) or any(
        int(summary.get(key, -1)) != value
        for key, value in expected_summary.items()
    ):
        raise EvidenceGapError("Inference gap analysis summary mismatch")
    return {
        "status": "PASS",
        "statement_id": str(bundle["statement_id"]),
        **expected_summary,
    }


def _call_with_retries(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    statement_bundle: Mapping[str, Any],
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    thinking: bool,
    prompt: ResolvedPrompt,
) -> tuple[list[dict[str, Any]], ProviderResponse, int]:
    retry_note: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = call_structured_llm(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=prompt.system_prompt,
                user_prompt=build_user_prompt(
                    statement_bundle, retry_note=retry_note
                ),
                response_schema=response_json_schema(),
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                thinking=thinking,
            )
            analyses = validate_response_payload(
                response.payload, statement_bundle=statement_bundle
            )
            return analyses, response, attempt
        except _ProviderError as exc:
            if attempt >= max_retries or not exc.retryable:
                raise
            retry_note = str(exc)
            time.sleep(min(2**attempt, 16) + random.random())
    raise AssertionError("retry loop exhausted")


def run_inference_gap_analysis(
    root: Path,
    *,
    statement_bundle_artifact_dir: Path,
    provider: str,
    run_name: str,
    model: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 4096,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    thinking: bool | None = None,
    prompt_override: PromptOverride | None = None,
    artifact_root: Path | None = None,
    statement_bundle: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    source_dir = _resolve(root, statement_bundle_artifact_dir)
    statement_bundle_path = source_dir / "statement_bundle.json"
    if statement_bundle is None:
        validate_statement_bundle_artifact(source_dir)
        statement_bundle_value = _read_json_object(statement_bundle_path)
        source_handoff = "artifact_reload"
    else:
        statement_bundle_value = dict(statement_bundle)
        source_handoff = "in_memory_handoff"
    validate_statement_bundle(statement_bundle_value)

    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(f"provider must be one of {SUPPORTED_PROVIDERS}")
    model = (model or DEFAULT_MODELS[provider]).strip()
    api_key_env = (api_key_env or DEFAULT_API_KEY_ENVS[provider]).strip()
    base_url = (base_url or DEFAULT_BASE_URLS[provider]).strip()
    if max_tokens <= 0 or timeout_seconds <= 0:
        raise EvidenceGapError("max_tokens and timeout_seconds must be positive")
    if max_retries < 0:
        raise EvidenceGapError("max_retries cannot be negative")
    resolved_thinking = provider == "deepseek" if thinking is None else thinking
    if provider != "deepseek" and resolved_thinking:
        raise EvidenceGapError("--thinking is only supported for DeepSeek")
    prompt = resolve_builtin_prompt(
        prompt_name="inference_gap.txt",
        default_version=INFERENCE_GAP_ANALYSIS_PROMPT_VERSION,
        override=prompt_override,
    )

    name = _safe_name(run_name)
    target = (
        artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    ) / name
    source_sha = sha256_file(statement_bundle_path)
    started = time.perf_counter()

    if statement_bundle_value.get("inference_steps"):
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise EvidenceGapError(
                f"Missing API key environment variable {api_key_env}"
            )
        analyses, response, retries = _call_with_retries(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            statement_bundle=statement_bundle_value,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            thinking=resolved_thinking,
            prompt=prompt,
        )
        usage = dict(response.usage)
        provider_response: dict[str, Any] | None = {
            "request_id": response.request_id,
            "raw_response_sha256": response.raw_response_sha256,
            "finish_reason": response.finish_reason,
        }
        api_requests = 1
    else:
        analyses = []
        retries = 0
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        provider_response = None
        api_requests = 0

    output_bundle = build_inference_gap_analysis_bundle(
        statement_bundle_value,
        analyses,
        source_statement_bundle_sha256=source_sha,
    )
    validation = validate_inference_gap_analysis_bundle(
        output_bundle, statement_bundle=statement_bundle_value
    )

    with atomic_directory(target, force=force) as staging:
        request_path = staging / "request.json"
        atomic_write_json(
            request_path,
            {
                "schema_version": INFERENCE_GAP_ANALYSIS_SCHEMA_VERSION,
                "contract_id": INFERENCE_GAP_ANALYSIS_CONTRACT_ID,
                "run_name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "statement_bundle_artifact_dir": relative_path(root, source_dir),
                "statement_bundle": {
                    "path": relative_path(root, statement_bundle_path),
                    "sha256": source_sha,
                },
            },
        )
        output_path = staging / "inference_gap_analysis.json"
        atomic_write_json(output_path, output_bundle)
        output_meta = {
            "path": relative_path(root, target / output_path.name),
            "sha256": sha256_file(output_path),
        }
        atomic_write_json(
            staging / "run_manifest.json",
            {
                "schema_version": INFERENCE_GAP_ANALYSIS_SCHEMA_VERSION,
                "contract_id": INFERENCE_GAP_ANALYSIS_CONTRACT_ID,
                "run_type": "phase077_inference_gap_analysis",
                "run_name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_handoff": source_handoff,
                "provider": provider,
                "model": model,
                **prompt.manifest_fields(),
                "parameters": {
                    "max_tokens": max_tokens,
                    "max_retries": max_retries,
                    "thinking": (
                        resolved_thinking if provider == "deepseek" else None
                    ),
                },
                "source": {
                    "statement_bundle_artifact_dir": relative_path(root, source_dir),
                    "statement_bundle": {
                        "path": relative_path(root, statement_bundle_path),
                        "sha256": source_sha,
                    },
                },
                "counts": {
                    **dict(output_bundle["summary"]),
                    "api_requests": api_requests,
                    "retries": retries,
                },
                "usage": usage,
                "provider_response": provider_response,
                "outputs": {"inference_gap_analysis": output_meta},
                "seconds": round(time.perf_counter() - started, 6),
            },
        )

    return {
        **validation,
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "output": {"inference_gap_analysis": output_meta},
        "api_requests": api_requests,
        "inference_gap_bundle": output_bundle,
    }


def validate_inference_gap_analysis_artifact(
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    request = _read_json_object(artifact_dir / "request.json")
    manifest = _read_json_object(artifact_dir / "run_manifest.json")
    for document, label in ((request, "request"), (manifest, "manifest")):
        if (
            document.get("schema_version")
            != INFERENCE_GAP_ANALYSIS_SCHEMA_VERSION
            or document.get("contract_id")
            != INFERENCE_GAP_ANALYSIS_CONTRACT_ID
        ):
            raise EvidenceGapError(
                f"Unexpected inference gap analysis {label} contract"
            )
    if request.get("run_name") != manifest.get("run_name"):
        raise EvidenceGapError("Inference gap analysis run name mismatch")

    root = _find_repo_root(artifact_dir)
    source_dir = _resolve(
        root, str(request.get("statement_bundle_artifact_dir") or "")
    )
    validate_statement_bundle_artifact(source_dir)
    manifest_source_root = manifest.get("source")
    if not isinstance(manifest_source_root, Mapping):
        raise EvidenceGapError("Inference gap analysis source metadata is missing")
    if source_dir != _resolve(
        root,
        str(manifest_source_root.get("statement_bundle_artifact_dir") or ""),
    ):
        raise EvidenceGapError("Inference gap analysis source directory mismatch")

    source_meta = request.get("statement_bundle")
    manifest_source = manifest_source_root.get("statement_bundle")
    if not isinstance(source_meta, Mapping) or source_meta != manifest_source:
        raise EvidenceGapError("Inference gap analysis source metadata mismatch")
    statement_bundle_path = _resolve(root, str(source_meta.get("path") or ""))
    if (
        statement_bundle_path != source_dir / "statement_bundle.json"
        or not statement_bundle_path.is_file()
        or sha256_file(statement_bundle_path) != str(source_meta.get("sha256") or "")
    ):
        raise EvidenceGapError("Inference gap analysis source checksum mismatch")
    statement_bundle = _read_json_object(statement_bundle_path)

    outputs = manifest.get("outputs")
    output_meta = (
        outputs.get("inference_gap_analysis")
        if isinstance(outputs, Mapping)
        else None
    )
    if not isinstance(output_meta, Mapping):
        raise EvidenceGapError("Inference gap analysis output metadata is missing")
    output_path = _resolve(root, str(output_meta.get("path") or ""))
    if (
        output_path != artifact_dir / "inference_gap_analysis.json"
        or not output_path.is_file()
        or sha256_file(output_path) != str(output_meta.get("sha256") or "")
    ):
        raise EvidenceGapError("Inference gap analysis output checksum mismatch")
    bundle = _read_json_object(output_path)
    if bundle.get("source_statement_bundle_sha256") != source_meta.get("sha256"):
        raise EvidenceGapError("Inference gap analysis source hash mismatch")
    validation = validate_inference_gap_analysis_bundle(
        bundle, statement_bundle=statement_bundle
    )

    counts = manifest.get("counts")
    if not isinstance(counts, Mapping) or any(
        int(counts.get(key, -1)) != value
        for key, value in {
            "total_inference_steps": validation["total_inference_steps"],
            "scope_gaps": validation["scope_gaps"],
            "causal_gaps": validation["causal_gaps"],
        }.items()
    ):
        raise EvidenceGapError("Inference gap analysis manifest count mismatch")
    expected_api_requests = 1 if validation["total_inference_steps"] else 0
    if int(counts.get("api_requests", -1)) != expected_api_requests:
        raise EvidenceGapError("Inference gap analysis api request count mismatch")
    return {
        **validation,
        "run_name": manifest.get("run_name"),
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "checksums": "PASS",
    }
