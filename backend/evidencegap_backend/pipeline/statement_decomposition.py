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
    sha256_text,
)
from evidencegap_backend.pipeline.retrieval_adapters import runtime_claim_id
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

STATEMENT_DECOMPOSITION_SCHEMA_VERSION = "1.2.0"
STATEMENT_DECOMPOSITION_CONTRACT_ID = "phase075.statement-decomposition.v1"
STATEMENT_DECOMPOSITION_PROMPT_VERSION = "phase075_statement_decomposition_v3"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/statement_decomposition")


def response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "source_language": {
                "type": "string",
                "description": (
                    "Concise BCP 47 language tag for the input statement, such as "
                    "en, zh-TW, ja, or ko."
                ),
            },
            "claims": {
                "type": "array",
                "description": (
                    "The smallest complete argument-preserving set of independently "
                    "verifiable biomedical claims extracted from the statement."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "claim_ref": {"type": "string"},
                        "source_text": {"type": "string"},
                        "canonical_claim_en": {"type": "string"},
                    },
                    "required": [
                        "claim_ref",
                        "source_text",
                        "canonical_claim_en",
                    ],
                    "additionalProperties": False,
                },
            },
            "inference_steps": {
                "type": "array",
                "description": (
                    "Direct premise-to-conclusion relationships explicitly asserted "
                    "by the input. These links describe the source argument and do "
                    "not validate its reasoning."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "premise_claim_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "conclusion_claim_ref": {"type": "string"},
                    },
                    "required": [
                        "premise_claim_refs",
                        "conclusion_claim_ref",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["source_language", "claims", "inference_steps"],
        "additionalProperties": False,
    }


def build_user_prompt(statement: str, *, retry_note: str | None = None) -> str:
    parts = [
        (
            "Decompose this biomedical statement into independently verifiable "
            "claims and preserve every explicit premise-to-conclusion relationship "
            "among those claims. Do not analyze inference validity or gaps."
        ),
        "INPUT STATEMENT:",
        statement,
    ]
    if retry_note:
        parts.extend(
            [
                "The previous response was invalid. Correct this issue and return the complete JSON object:",
                retry_note,
            ]
        )
    return "\n\n".join(parts)


def runtime_inference_step_id(
    premise_claim_ids: list[str], conclusion_claim_id: str
) -> str:
    normalized_premises = sorted(str(value).strip() for value in premise_claim_ids)
    conclusion = str(conclusion_claim_id).strip()
    identity_payload = json.dumps(
        {
            "premise_claim_ids": normalized_premises,
            "conclusion_claim_id": conclusion,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "inference_" + sha256_text(identity_payload)[:24]


def _statement_id(statement: str) -> str:
    return "statement_" + sha256_text(statement.strip())[:24]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid name: {value!r}")
    return cleaned


def _duplicate_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).rstrip(".?!")


def exact_source_spans(statement: str, source_text: str) -> list[dict[str, int]]:
    """Return every exact occurrence of ``source_text`` in ``statement``."""

    if not source_text:
        return []
    spans: list[dict[str, int]] = []
    start = 0
    while True:
        index = statement.find(source_text, start)
        if index < 0:
            break
        spans.append(
            {
                "character_start": index,
                "character_end": index + len(source_text),
            }
        )
        start = index + 1
    return spans


def validate_response_payload(
    payload: Mapping[str, Any], *, original_statement: str
) -> dict[str, Any]:
    # Language metadata is useful for display but is not part of biomedical
    # claim extraction. Do not reject an otherwise valid decomposition when a
    # provider leaves it blank, especially for a legitimate empty-claim result.
    source_language = str(payload.get("source_language") or "").strip() or "und"

    raw_claims = payload.get("claims")
    if not isinstance(raw_claims, list):
        raise _ProviderError("claims must be an array", retryable=True)

    claims: list[dict[str, str]] = []
    ref_to_id: dict[str, str] = {}
    duplicate_keys: set[str] = set()
    for index, raw_claim in enumerate(raw_claims, start=1):
        if not isinstance(raw_claim, Mapping):
            raise _ProviderError(
                f"claims[{index - 1}] must be an object", retryable=True
            )
        expected_ref = f"C{index}"
        claim_ref = str(raw_claim.get("claim_ref") or "").strip()
        if claim_ref != expected_ref:
            raise _ProviderError(
                f"claims[{index - 1}].claim_ref must be {expected_ref!r}",
                retryable=True,
            )
        source_text = str(raw_claim.get("source_text") or "").strip()
        source_spans = exact_source_spans(original_statement, source_text)
        if not source_text or not source_spans:
            raise _ProviderError(
                f"{claim_ref}.source_text must be an exact contiguous quote from the input",
                retryable=True,
            )
        canonical_claim_en = str(
            raw_claim.get("canonical_claim_en") or ""
        ).strip()
        if not canonical_claim_en or "\n" in canonical_claim_en:
            raise _ProviderError(
                f"{claim_ref}.canonical_claim_en must be one English sentence",
                retryable=True,
            )
        if not re.search(r"[A-Za-z]", canonical_claim_en):
            raise _ProviderError(
                f"{claim_ref}.canonical_claim_en must be written in English",
                retryable=True,
            )
        duplicate_key = _duplicate_key(canonical_claim_en)
        if duplicate_key in duplicate_keys:
            raise _ProviderError(
                "claims must not contain duplicate canonical English claims",
                retryable=True,
            )
        duplicate_keys.add(duplicate_key)
        claim_id = runtime_claim_id(canonical_claim_en)
        ref_to_id[claim_ref] = claim_id
        claims.append(
            {
                "claim_id": claim_id,
                "source_text": source_text,
                "source_spans": source_spans,
                "canonical_claim_en": canonical_claim_en,
            }
        )

    raw_steps = payload.get("inference_steps")
    if not isinstance(raw_steps, list):
        raise _ProviderError("inference_steps must be an array", retryable=True)
    if not claims and raw_steps:
        raise _ProviderError(
            "inference_steps must be empty when claims is empty", retryable=True
        )

    inference_steps: list[dict[str, Any]] = []
    inference_step_keys: set[tuple[tuple[str, ...], str]] = set()
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, Mapping):
            raise _ProviderError(
                f"inference_steps[{index}] must be an object", retryable=True
            )
        premise_refs = raw_step.get("premise_claim_refs")
        if not isinstance(premise_refs, list) or not premise_refs or any(
            not isinstance(value, str) for value in premise_refs
        ):
            raise _ProviderError(
                "premise_claim_refs must be a non-empty string array",
                retryable=True,
            )
        premise_refs = [value.strip() for value in premise_refs]
        conclusion_ref = str(
            raw_step.get("conclusion_claim_ref") or ""
        ).strip()
        referenced = [*premise_refs, conclusion_ref]
        unknown = [value for value in referenced if value not in ref_to_id]
        if unknown:
            raise _ProviderError(
                f"inference_steps references unknown claims: {unknown}",
                retryable=True,
            )
        if (
            any(not value for value in premise_refs)
            or len(premise_refs) != len(set(premise_refs))
            or conclusion_ref in premise_refs
        ):
            raise _ProviderError("Invalid inference step references", retryable=True)
        step_key = (tuple(sorted(premise_refs)), conclusion_ref)
        if step_key in inference_step_keys:
            raise _ProviderError(
                "inference_steps must not contain duplicate relationships",
                retryable=True,
            )
        inference_step_keys.add(step_key)
        premise_claim_ids = [ref_to_id[value] for value in premise_refs]
        conclusion_claim_id = ref_to_id[conclusion_ref]
        inference_steps.append(
            {
                "inference_step_id": runtime_inference_step_id(
                    premise_claim_ids, conclusion_claim_id
                ),
                "premise_claim_ids": premise_claim_ids,
                "conclusion_claim_id": conclusion_claim_id,
            }
        )

    return {
        "schema_version": STATEMENT_DECOMPOSITION_SCHEMA_VERSION,
        "contract_id": STATEMENT_DECOMPOSITION_CONTRACT_ID,
        "statement_id": _statement_id(original_statement),
        "original_statement": original_statement,
        "source_language": source_language,
        "claims": claims,
        "inference_steps": inference_steps,
    }


def validate_decomposition_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    if bundle.get("schema_version") != STATEMENT_DECOMPOSITION_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected statement decomposition schema_version")
    if bundle.get("contract_id") != STATEMENT_DECOMPOSITION_CONTRACT_ID:
        raise EvidenceGapError("Unexpected statement decomposition contract_id")

    original_statement = str(bundle.get("original_statement") or "").strip()
    source_language = str(bundle.get("source_language") or "").strip()
    if not original_statement or not source_language:
        raise EvidenceGapError("Statement decomposition source fields cannot be blank")
    if bundle.get("statement_id") != _statement_id(original_statement):
        raise EvidenceGapError("Statement decomposition statement_id mismatch")

    claims = bundle.get("claims")
    if not isinstance(claims, list):
        raise EvidenceGapError("Statement decomposition claims must be an array")
    claim_ids: set[str] = set()
    duplicate_keys: set[str] = set()
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise EvidenceGapError("Statement decomposition claim must be an object")
        source_text = str(claim.get("source_text") or "").strip()
        source_spans = claim.get("source_spans")
        canonical = str(claim.get("canonical_claim_en") or "").strip()
        claim_id = str(claim.get("claim_id") or "").strip()
        expected_spans = exact_source_spans(original_statement, source_text)
        if not source_text or not expected_spans:
            raise EvidenceGapError("Statement decomposition source_text is not grounded")
        if source_spans != expected_spans:
            raise EvidenceGapError("Statement decomposition source_spans mismatch")
        if not canonical or claim_id != runtime_claim_id(canonical):
            raise EvidenceGapError("Statement decomposition claim identity mismatch")
        duplicate_key = _duplicate_key(canonical)
        if claim_id in claim_ids or duplicate_key in duplicate_keys:
            raise EvidenceGapError("Statement decomposition contains duplicate claims")
        claim_ids.add(claim_id)
        duplicate_keys.add(duplicate_key)

    inference_steps = bundle.get("inference_steps")
    if not isinstance(inference_steps, list):
        raise EvidenceGapError(
            "Statement decomposition inference_steps must be an array"
        )
    if not claims and inference_steps:
        raise EvidenceGapError("Empty decomposition cannot contain inference steps")
    inference_step_keys: set[tuple[tuple[str, ...], str]] = set()
    inference_step_ids: set[str] = set()
    for step in inference_steps:
        if not isinstance(step, Mapping):
            raise EvidenceGapError("Statement decomposition step must be an object")
        premises = step.get("premise_claim_ids")
        conclusion = str(step.get("conclusion_claim_id") or "").strip()
        if (
            not isinstance(premises, list)
            or not premises
            or any(
                not isinstance(value, str) or not value.strip()
                for value in premises
            )
        ):
            raise EvidenceGapError("Inference premises must be a non-empty array")
        premise_ids = [value.strip() for value in premises]
        if (
            len(premise_ids) != len(set(premise_ids))
            or not set(premise_ids).issubset(claim_ids)
            or conclusion not in claim_ids
            or conclusion in premise_ids
        ):
            raise EvidenceGapError("Invalid statement decomposition inference step")
        step_key = (tuple(sorted(premise_ids)), conclusion)
        inference_step_id = str(step.get("inference_step_id") or "").strip()
        expected_step_id = runtime_inference_step_id(premise_ids, conclusion)
        if inference_step_id != expected_step_id:
            raise EvidenceGapError(
                "Statement decomposition inference_step_id mismatch"
            )
        if (
            step_key in inference_step_keys
            or inference_step_id in inference_step_ids
        ):
            raise EvidenceGapError(
                "Statement decomposition contains duplicate inference steps"
            )
        inference_step_keys.add(step_key)
        inference_step_ids.add(inference_step_id)

    return {
        "status": "PASS",
        "statement_id": str(bundle["statement_id"]),
        "claims": len(claims),
        "inference_steps": len(inference_steps),
        "empty_claims": len(claims) == 0,
    }


def _call_with_retries(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    statement: str,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    thinking: bool,
    prompt: ResolvedPrompt,
) -> tuple[dict[str, Any], ProviderResponse, int]:
    retry_note: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = call_structured_llm(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=prompt.system_prompt,
                user_prompt=build_user_prompt(statement, retry_note=retry_note),
                response_schema=response_json_schema(),
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                thinking=thinking,
            )
            bundle = validate_response_payload(
                response.payload,
                original_statement=statement,
            )
            return bundle, response, attempt
        except _ProviderError as exc:
            if attempt >= max_retries or not exc.retryable:
                raise
            retry_note = str(exc)
            time.sleep(min(2**attempt, 16) + random.random())
    raise AssertionError("retry loop exhausted")


def run_statement_decomposition(
    root: Path,
    *,
    statement: str,
    provider: str,
    run_name: str,
    model: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 2048,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    thinking: bool = False,
    prompt_override: PromptOverride | None = None,
    artifact_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    statement = statement.strip()
    if not statement:
        raise EvidenceGapError("Statement cannot be blank")
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
    if provider != "deepseek" and thinking:
        raise EvidenceGapError("--thinking is only supported for DeepSeek")
    prompt = resolve_builtin_prompt(
        prompt_name="statement_decomposition.txt",
        default_version=STATEMENT_DECOMPOSITION_PROMPT_VERSION,
        override=prompt_override,
    )

    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise EvidenceGapError(f"Missing API key environment variable {api_key_env}")

    name = _safe_name(run_name)
    target = (
        artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    ) / name
    started = time.perf_counter()
    bundle, response, retries = _call_with_retries(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        statement=statement,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        thinking=thinking,
        prompt=prompt,
    )
    validation = validate_decomposition_bundle(bundle)

    with atomic_directory(target, force=force) as staging:
        output_path = staging / "decomposition.json"
        atomic_write_json(output_path, bundle)
        output_meta = {
            "path": relative_path(root, target / output_path.name),
            "sha256": sha256_file(output_path),
        }
        manifest = {
            "schema_version": STATEMENT_DECOMPOSITION_SCHEMA_VERSION,
            "contract_id": STATEMENT_DECOMPOSITION_CONTRACT_ID,
            "run_type": "phase075_multilingual_claim_decomposition",
            "run_name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            **prompt.manifest_fields(),
            "parameters": {
                "max_tokens": max_tokens,
                "max_retries": max_retries,
                "thinking": thinking if provider == "deepseek" else None,
            },
            "counts": {
                "claims": validation["claims"],
                "inference_steps": validation["inference_steps"],
                "api_requests": 1,
                "retries": retries,
            },
            "usage": dict(response.usage),
            "provider_response": {
                "request_id": response.request_id,
                "raw_response_sha256": response.raw_response_sha256,
                "finish_reason": response.finish_reason,
            },
            "outputs": {"decomposition": output_meta},
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_write_json(staging / "run_manifest.json", manifest)

    return {
        "status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "statement_id": validation["statement_id"],
        "claims": validation["claims"],
        "inference_steps": validation["inference_steps"],
        "empty_claims": validation["empty_claims"],
        "outputs": {"decomposition": output_meta},
        "decomposition": bundle,
    }


def validate_statement_decomposition_artifact(
    artifact_dir: Path,
) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    try:
        manifest = json.loads(
            (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise EvidenceGapError(
            f"Missing statement decomposition manifest: {artifact_dir}"
        ) from exc
    if manifest.get("schema_version") != STATEMENT_DECOMPOSITION_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected statement decomposition schema_version")
    if manifest.get("contract_id") != STATEMENT_DECOMPOSITION_CONTRACT_ID:
        raise EvidenceGapError("Unexpected statement decomposition contract_id")

    output_meta = manifest["outputs"]["decomposition"]
    output_path = Path(str(output_meta["path"]))
    if not output_path.is_absolute():
        root = find_workspace_root(artifact_dir)
        output_path = root / output_path
    if sha256_file(output_path) != str(output_meta["sha256"]):
        raise EvidenceGapError("Statement decomposition output checksum mismatch")

    bundle = json.loads(output_path.read_text(encoding="utf-8"))
    validation = validate_decomposition_bundle(bundle)
    if int(manifest["counts"]["claims"]) != validation["claims"] or int(
        manifest["counts"]["inference_steps"]
    ) != validation["inference_steps"]:
        raise EvidenceGapError("Statement decomposition manifest count mismatch")
    return {
        "status": "PASS",
        "run_name": manifest.get("run_name"),
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        **validation,
        "checksums": "PASS",
    }
