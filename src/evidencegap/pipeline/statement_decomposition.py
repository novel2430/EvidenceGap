from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
    sha256_text,
)
from evidencegap.pipeline.retrieval_adapters import runtime_claim_id
from evidencegap.stance.llm_judge import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    ProviderResponse,
    _ProviderError,
    call_structured_llm,
)

STATEMENT_DECOMPOSITION_SCHEMA_VERSION = "1.0.0"
STATEMENT_DECOMPOSITION_CONTRACT_ID = "phase075.statement-decomposition.v1"
STATEMENT_DECOMPOSITION_PROMPT_VERSION = "phase075_statement_decomposition_v2"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/statement_decomposition")

SYSTEM_PROMPT = """You extract biomedical claims from a user statement.

Your only task is to identify claims that can be directly supported or refuted by biomedical research literature and rewrite each one as a canonical English claim.

Rules:
- Use only information stated in the input. Do not use outside knowledge.
- Extract only directly verifiable biomedical claims.
- Exclude policy proposals, value judgments, personal advice, calls to action, rhetorical language, and other statements that biomedical literature cannot directly verify.
- Do not invent a biomedical claim merely to produce output. If none exists, return an empty claims list and an empty inference_steps list.
- Each claim must contain exactly one independently testable biomedical outcome assertion.
- When one source clause says that the same exposure or intervention affects multiple separable outcomes, create one claim per outcome and repeat the shared population, exposure or intervention, comparator, dose, timing, uncertainty, and causal strength in each claim.
- Do not merge separable outcomes merely because they share the same population, exposure, intervention, or sentence. For example, "sleep insufficiency increases obesity and type 2 diabetes risk" must become two claims: one about obesity risk and one about type 2 diabetes risk.
- Also keep prevention separate from treatment, disease incidence separate from severity or control in existing patients, and benefits separate from harms when they are independently testable.
- Do not split a single established composite endpoint or a phrase whose words jointly name one outcome.
- The same exact source_text quote may ground more than one claim when that quote contains multiple independently testable outcomes.
- Return the smallest complete set of non-duplicate claims.
- Preserve every material qualifier from the source, including population, intervention or exposure, comparator, dose, outcome, timing, uncertainty, causal strength, and prevention-versus-treatment scope.
- Do not strengthen possibility into certainty, association into causation, or a narrow population into a general population.
- source_language must identify the input language with a concise BCP 47 language tag such as en, zh-TW, ja, or ko.
- source_text must be an exact contiguous quote copied from the input statement.
- canonical_claim_en must be a concise English declarative sentence.
- Use claim_ref values C1, C2, C3, ... in claims array order.
- inference_steps may only describe explicit premise-to-conclusion relationships among the extracted biomedical claims. Do not invent missing reasoning. If no such relationship is explicit, return an empty inference_steps list.
- Return JSON only."""


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
        "Extract directly verifiable biomedical claims from this statement.",
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


def _statement_id(statement: str) -> str:
    return "statement_" + sha256_text(statement.strip())[:24]


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid name: {value!r}")
    return cleaned


def _duplicate_key(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).rstrip(".?!")


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
        if not source_text or source_text not in original_statement:
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
        inference_steps.append(
            {
                "premise_claim_ids": [ref_to_id[value] for value in premise_refs],
                "conclusion_claim_id": ref_to_id[conclusion_ref],
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
        canonical = str(claim.get("canonical_claim_en") or "").strip()
        claim_id = str(claim.get("claim_id") or "").strip()
        if not source_text or source_text not in original_statement:
            raise EvidenceGapError("Statement decomposition source_text is not grounded")
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
    for step in inference_steps:
        if not isinstance(step, Mapping):
            raise EvidenceGapError("Statement decomposition step must be an object")
        premises = step.get("premise_claim_ids")
        conclusion = str(step.get("conclusion_claim_id") or "")
        if not isinstance(premises, list) or not premises:
            raise EvidenceGapError("Inference premises must be a non-empty array")
        premise_ids = [str(value) for value in premises]
        if (
            len(premise_ids) != len(set(premise_ids))
            or not set(premise_ids).issubset(claim_ids)
            or conclusion not in claim_ids
            or conclusion in premise_ids
        ):
            raise EvidenceGapError("Invalid statement decomposition inference step")

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
) -> tuple[dict[str, Any], ProviderResponse, int]:
    retry_note: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = call_structured_llm(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=SYSTEM_PROMPT,
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
            "prompt_version": STATEMENT_DECOMPOSITION_PROMPT_VERSION,
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
        root = artifact_dir
        while root.parent != root and not (root / "src/evidencegap").exists():
            root = root.parent
        output_path = root / output_path if (root / "src/evidencegap").exists() else artifact_dir / output_path.name
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
