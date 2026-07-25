from __future__ import annotations

import copy
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
)
from evidencegap.pipeline.inference_gap_analysis import (
    validate_inference_gap_analysis_artifact,
    validate_inference_gap_analysis_bundle,
)
from evidencegap.pipeline.statement_bundle import (
    validate_statement_bundle,
    validate_statement_bundle_artifact,
)
from evidencegap.stance.llm_judge import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    ProviderResponse,
    _ProviderError,
    call_structured_llm,
)

PRESENTATION_BUNDLE_SCHEMA_VERSION = "1.0.0"
PRESENTATION_BUNDLE_CONTRACT_ID = "phase077.presentation-bundle.v1"
LOCALIZATION_PROMPT_VERSION = "phase077_output_localization_v2"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/output/presentation")

_VERDICT_TO_EVIDENCE_STATE = {
    "supported": "SUPPORTED",
    "refuted": "REFUTED",
    "mixed": "CONFLICTED",
    "insufficient": "INSUFFICIENT",
}
_EVIDENCE_STATES = (*_VERDICT_TO_EVIDENCE_STATE.values(), "ERROR")
_ARGUMENT_ROLES = ("PREMISE", "INTERMEDIATE", "CONCLUSION", "STANDALONE")
_GAP_TYPES = ("SCOPE_GAP", "CAUSAL_GAP")
_ENGLISH_ALIASES = {"en", "en-us", "en-gb", "english", "english (us)", "english (uk)"}

SYSTEM_PROMPT = """You localize an already validated biomedical evidence presentation.

Translate only the supplied text values into the requested target language. Preserve biomedical meaning, uncertainty, causal strength, populations, interventions or exposures, comparators, outcomes, doses, timing, numbers, abbreviations, and negation.

Rules:
- Return exactly one translation for every supplied text_id and no extra items.
- Copy every text_id exactly. Never translate, rename, omit, or invent an ID.
- Translate only the text field. Do not add explanations, citations, markdown, labels, or facts.
- Do not change verdicts, evidence states, gap types, confidence values, PMIDs, claim relationships, or any other structured value.
- Keep article titles and evidence sentences faithful rather than simplifying them.
- Return JSON only."""


def response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "translations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text_id": {"type": "string"},
                        "text": {"type": "string"},
                    },
                    "required": ["text_id", "text"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["translations"],
        "additionalProperties": False,
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing required JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceGapError(f"Expected JSON object in {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid run name: {value!r}")
    return cleaned


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current.parent != current:
        if (current / "src/evidencegap").exists():
            return current
        current = current.parent
    return start.resolve()


def _language(value: str | None) -> str:
    normalized = (value or "English").strip()
    if not normalized:
        raise EvidenceGapError("language cannot be blank")
    return normalized


def _needs_translation(language: str) -> bool:
    return language.casefold() not in _ENGLISH_ALIASES


def _claim_graph_metadata(
    claims: list[Mapping[str, Any]], steps: list[Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, list[str]]]:
    premise_of = {str(claim["claim_id"]): [] for claim in claims}
    conclusion_of = {str(claim["claim_id"]): [] for claim in claims}
    for step in steps:
        step_id = str(step["inference_step_id"])
        for claim_id in step["premise_claim_ids"]:
            premise_of[str(claim_id)].append(step_id)
        conclusion_of[str(step["conclusion_claim_id"])].append(step_id)

    roles: dict[str, str] = {}
    for claim_id in premise_of:
        if premise_of[claim_id] and conclusion_of[claim_id]:
            roles[claim_id] = "INTERMEDIATE"
        elif premise_of[claim_id]:
            roles[claim_id] = "PREMISE"
        elif conclusion_of[claim_id]:
            roles[claim_id] = "CONCLUSION"
        else:
            roles[claim_id] = "STANDALONE"
    return roles, premise_of, conclusion_of


def build_presentation_bundle(
    statement_bundle: Mapping[str, Any],
    gap_bundle: Mapping[str, Any],
    *,
    output_language: str,
    statement_bundle_sha256: str,
    gap_bundle_sha256: str,
) -> dict[str, Any]:
    validate_statement_bundle(statement_bundle)
    validate_inference_gap_analysis_bundle(gap_bundle, statement_bundle=statement_bundle)
    if gap_bundle.get("source_statement_bundle_sha256") != statement_bundle_sha256:
        raise EvidenceGapError("Inference gap analysis does not match statement bundle")

    language = _language(output_language)
    claims = copy.deepcopy(list(statement_bundle["claims"]))
    steps = copy.deepcopy(list(statement_bundle["inference_steps"]))
    roles, premise_of, conclusion_of = _claim_graph_metadata(claims, steps)
    gap_by_step = {
        str(row["inference_step_id"]): row
        for row in gap_bundle["inference_gap_analyses"]
    }

    for claim in claims:
        claim_id = str(claim["claim_id"])
        claim["evidence_state"] = (
            _VERDICT_TO_EVIDENCE_STATE[str(claim["verdict"])]
            if claim["analysis_status"] == "completed"
            else "ERROR"
        )
        claim["argument_role"] = roles[claim_id]
        claim["premise_inference_step_ids"] = premise_of[claim_id]
        claim["conclusion_inference_step_ids"] = conclusion_of[claim_id]
        claim["display_text"] = claim["canonical_claim_en"]
        claim["display_rationale"] = claim.get("rationale")

    for step in steps:
        analysis = gap_by_step[str(step["inference_step_id"])]
        step["gaps"] = []
        for field, gap_type in (("scope_gap", "SCOPE_GAP"), ("causal_gap", "CAUSAL_GAP")):
            if analysis[field]["detected"]:
                step["gaps"].append(
                    {
                        "gap_type": gap_type,
                        "detection_method": "llm",
                        "reason_en": analysis[field]["reason"],
                        "display_reason": analysis[field]["reason"],
                    }
                )

    articles = copy.deepcopy(list(statement_bundle["articles"]))
    for article in articles:
        article["display_title"] = article["title"]
        article["display_rationale"] = article["rationale"]
    evidence = copy.deepcopy(list(statement_bundle["evidence"]))
    for item in evidence:
        item["display_text"] = item["text"]

    statement = copy.deepcopy(dict(statement_bundle["statement"]))
    statement["display_text"] = statement["original_text"]
    bundle = {
        "schema_version": PRESENTATION_BUNDLE_SCHEMA_VERSION,
        "contract_id": PRESENTATION_BUNDLE_CONTRACT_ID,
        "output_language": language,
        "localized": _needs_translation(language),
        "source_statement_bundle_sha256": statement_bundle_sha256,
        "source_inference_gap_analysis_sha256": gap_bundle_sha256,
        "statement": statement,
        "claims": claims,
        "inference_steps": steps,
        "articles": articles,
        "evidence": evidence,
    }
    bundle["summary"] = _summary(bundle)
    validate_presentation_bundle(bundle, statement_bundle=statement_bundle, gap_bundle=gap_bundle)
    return bundle


def _summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    claims = bundle["claims"]
    steps = bundle["inference_steps"]
    return {
        "total_claims": len(claims),
        "evidence_states": {
            state: sum(claim["evidence_state"] == state for claim in claims)
            for state in _EVIDENCE_STATES
        },
        "argument_roles": {
            role: sum(claim["argument_role"] == role for claim in claims)
            for role in _ARGUMENT_ROLES
        },
        "total_inference_steps": len(steps),
        "gaps": {
            gap_type: sum(
                gap["gap_type"] == gap_type
                for step in steps
                for gap in step["gaps"]
            )
            for gap_type in _GAP_TYPES
        },
        "articles": len(bundle["articles"]),
        "evidence": len(bundle["evidence"]),
    }


def _translation_units(bundle: Mapping[str, Any]) -> list[dict[str, str]]:
    units: list[dict[str, str]] = []

    def add(text_id: str, text: Any) -> None:
        if isinstance(text, str) and text.strip():
            units.append({"text_id": text_id, "text": text})

    add("statement", bundle["statement"]["display_text"])
    for claim in bundle["claims"]:
        claim_id = str(claim["claim_id"])
        add(f"claim:{claim_id}:text", claim["display_text"])
        add(f"claim:{claim_id}:rationale", claim.get("display_rationale"))
    for article in bundle["articles"]:
        article_id = str(article["article_node_id"])
        add(f"article:{article_id}:title", article["display_title"])
        add(f"article:{article_id}:rationale", article["display_rationale"])
    for item in bundle["evidence"]:
        add(f"evidence:{item['evidence_id']}", item["display_text"])
    for step in bundle["inference_steps"]:
        for gap in step["gaps"]:
            add(
                f"inference:{step['inference_step_id']}:{gap['gap_type']}",
                gap["display_reason"],
            )
    return units


def build_localization_prompt(
    units: list[Mapping[str, str]],
    target_language: str,
    retry_note: str | None = None,
) -> str:
    parts = [
        f"TARGET LANGUAGE: {target_language}",
        "Translate every item and return the same text_id values exactly.",
        json.dumps({"texts": units}, ensure_ascii=False, indent=2),
    ]
    if retry_note:
        parts.extend(["The previous response was invalid. Fix it:", retry_note])
    return "\n\n".join(parts)


def validate_localization_response(
    payload: Mapping[str, Any], expected_units: list[Mapping[str, str]]
) -> dict[str, str]:
    if set(payload) != {"translations"} or not isinstance(payload["translations"], list):
        raise _ProviderError("Response must contain exactly a translations array", retryable=True)
    expected_ids = [str(row["text_id"]) for row in expected_units]
    values: dict[str, str] = {}
    for row in payload["translations"]:
        if not isinstance(row, Mapping) or set(row) != {"text_id", "text"}:
            raise _ProviderError("Each translation needs exactly text_id and text", retryable=True)
        text_id = str(row["text_id"]).strip()
        text = str(row["text"]).strip()
        if text_id not in expected_ids:
            raise _ProviderError(f"Unknown text_id: {text_id}", retryable=True)
        if text_id in values:
            raise _ProviderError(f"Duplicate text_id: {text_id}", retryable=True)
        if not text:
            raise _ProviderError(f"Blank translation: {text_id}", retryable=True)
        values[text_id] = text
    if set(values) != set(expected_ids):
        raise _ProviderError("Missing translations for one or more text_ids", retryable=True)
    return {text_id: values[text_id] for text_id in expected_ids}


def apply_localization(bundle: Mapping[str, Any], translations: Mapping[str, str]) -> dict[str, Any]:
    localized = copy.deepcopy(dict(bundle))
    localized["statement"]["display_text"] = translations["statement"]
    for claim in localized["claims"]:
        claim_id = str(claim["claim_id"])
        claim["display_text"] = translations[f"claim:{claim_id}:text"]
        if claim.get("display_rationale"):
            claim["display_rationale"] = translations[f"claim:{claim_id}:rationale"]
    for article in localized["articles"]:
        article_id = str(article["article_node_id"])
        if article.get("display_title"):
            article["display_title"] = translations[f"article:{article_id}:title"]
        if article.get("display_rationale"):
            article["display_rationale"] = translations[f"article:{article_id}:rationale"]
    for item in localized["evidence"]:
        if item.get("display_text"):
            item["display_text"] = translations[f"evidence:{item['evidence_id']}"]
    for step in localized["inference_steps"]:
        for gap in step["gaps"]:
            gap["display_reason"] = translations[
                f"inference:{step['inference_step_id']}:{gap['gap_type']}"
            ]
    return localized


def _localize_batch(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    units: list[Mapping[str, str]],
    language: str,
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[dict[str, str], ProviderResponse, int]:
    retry_note: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = call_structured_llm(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_localization_prompt(units, language, retry_note),
                response_schema=response_json_schema(),
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                thinking=False,
            )
            return validate_localization_response(response.payload, units), response, attempt
        except _ProviderError as exc:
            if attempt >= max_retries or not exc.retryable:
                raise
            retry_note = str(exc)
            time.sleep(min(2**attempt, 16) + random.random())
    raise AssertionError("retry loop exhausted")


def _localize(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    bundle: Mapping[str, Any],
    language: str,
    max_tokens: int,
    request_batch_size: int,
    timeout_seconds: float,
    max_retries: int,
) -> tuple[dict[str, str], list[ProviderResponse], int]:
    expected = _translation_units(bundle)
    translations: dict[str, str] = {}
    responses: list[ProviderResponse] = []
    retries = 0
    for start in range(0, len(expected), request_batch_size):
        batch = expected[start : start + request_batch_size]
        batch_translations, response, batch_retries = _localize_batch(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            units=batch,
            language=language,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        translations.update(batch_translations)
        responses.append(response)
        retries += batch_retries
    if set(translations) != {str(row["text_id"]) for row in expected}:
        raise EvidenceGapError("Localization batches did not cover every translation unit")
    return translations, responses, retries


def _without_added_fields(row: Mapping[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    value = copy.deepcopy(dict(row))
    for field in fields:
        value.pop(field, None)
    return value


def validate_presentation_bundle(
    bundle: Mapping[str, Any],
    *,
    statement_bundle: Mapping[str, Any],
    gap_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    validate_statement_bundle(statement_bundle)
    validate_inference_gap_analysis_bundle(gap_bundle, statement_bundle=statement_bundle)
    if (
        bundle.get("schema_version") != PRESENTATION_BUNDLE_SCHEMA_VERSION
        or bundle.get("contract_id") != PRESENTATION_BUNDLE_CONTRACT_ID
    ):
        raise EvidenceGapError("Unexpected presentation bundle contract")
    language = str(bundle.get("output_language") or "").strip()
    if not language or bundle.get("localized") != _needs_translation(language):
        raise EvidenceGapError("Invalid presentation language metadata")
    if _without_added_fields(bundle["statement"], ("display_text",)) != statement_bundle["statement"]:
        raise EvidenceGapError("Presentation statement changed source data")
    if not str(bundle["statement"].get("display_text") or "").strip():
        raise EvidenceGapError("Presentation statement display text is blank")

    claim_added = (
        "evidence_state",
        "argument_role",
        "premise_inference_step_ids",
        "conclusion_inference_step_ids",
        "display_text",
        "display_rationale",
    )
    if len(bundle["claims"]) != len(statement_bundle["claims"]):
        raise EvidenceGapError("Presentation Claim count mismatch")
    for row, source in zip(bundle["claims"], statement_bundle["claims"], strict=True):
        if _without_added_fields(row, claim_added) != source:
            raise EvidenceGapError("Presentation Claim changed source data")
        expected_state = (
            _VERDICT_TO_EVIDENCE_STATE[str(source["verdict"])]
            if source["analysis_status"] == "completed"
            else "ERROR"
        )
        if row.get("evidence_state") != expected_state or row.get("argument_role") not in _ARGUMENT_ROLES:
            raise EvidenceGapError("Invalid presentation Claim enrichment")
        if not str(row.get("display_text") or "").strip():
            raise EvidenceGapError("Presentation Claim display text is blank")

    gap_source = {str(row["inference_step_id"]): row for row in gap_bundle["inference_gap_analyses"]}
    if len(bundle["inference_steps"]) != len(statement_bundle["inference_steps"]):
        raise EvidenceGapError("Presentation inference count mismatch")
    for row, source in zip(bundle["inference_steps"], statement_bundle["inference_steps"], strict=True):
        if _without_added_fields(row, ("gaps",)) != source:
            raise EvidenceGapError("Presentation inference changed source data")
        expected = []
        analysis = gap_source[str(source["inference_step_id"])]
        if analysis["scope_gap"]["detected"]:
            expected.append("SCOPE_GAP")
        if analysis["causal_gap"]["detected"]:
            expected.append("CAUSAL_GAP")
        if [gap.get("gap_type") for gap in row["gaps"]] != expected:
            raise EvidenceGapError("Presentation Gap mismatch")
        if any(
            gap.get("detection_method") != "llm"
            or not str(gap.get("reason_en") or "").strip()
            or not str(gap.get("display_reason") or "").strip()
            for gap in row["gaps"]
        ):
            raise EvidenceGapError("Invalid presentation Gap")

    for name, added, display_field in (
        ("articles", ("display_title", "display_rationale"), "display_title"),
        ("evidence", ("display_text",), "display_text"),
    ):
        if len(bundle[name]) != len(statement_bundle[name]):
            raise EvidenceGapError(f"Presentation {name} count mismatch")
        for row, source in zip(bundle[name], statement_bundle[name], strict=True):
            if _without_added_fields(row, added) != source:
                raise EvidenceGapError(f"Presentation {name} changed source data")
            if not str(row.get(display_field) or "").strip():
                raise EvidenceGapError(f"Presentation {name} display text is blank")

    expected_summary = _summary(bundle)
    if bundle.get("summary") != expected_summary:
        raise EvidenceGapError("Presentation summary mismatch")
    return {
        "status": "PASS",
        "statement_id": bundle["statement"]["statement_id"],
        "output_language": language,
        "localized": bool(bundle["localized"]),
        **expected_summary,
    }


def run_output_module(
    root: Path,
    *,
    statement_bundle_artifact_dir: Path,
    inference_gap_artifact_dir: Path,
    run_name: str,
    language: str = "English",
    provider: str = "deepseek",
    model: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 8192,
    request_batch_size: int = 32,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    artifact_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    statement_dir = _resolve(root, statement_bundle_artifact_dir)
    gap_dir = _resolve(root, inference_gap_artifact_dir)
    validate_statement_bundle_artifact(statement_dir)
    validate_inference_gap_analysis_artifact(gap_dir)
    statement_path = statement_dir / "statement_bundle.json"
    gap_path = gap_dir / "inference_gap_analysis.json"
    statement_bundle = _read_json(statement_path)
    gap_bundle = _read_json(gap_path)
    statement_sha = sha256_file(statement_path)
    gap_sha = sha256_file(gap_path)

    language = _language(language)
    localized = _needs_translation(language)
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(f"provider must be one of {SUPPORTED_PROVIDERS}")
    model = (model or DEFAULT_MODELS[provider]).strip()
    api_key_env = (api_key_env or DEFAULT_API_KEY_ENVS[provider]).strip()
    base_url = (base_url or DEFAULT_BASE_URLS[provider]).strip()
    if (
        max_tokens <= 0
        or request_batch_size <= 0
        or timeout_seconds <= 0
        or max_retries < 0
    ):
        raise EvidenceGapError("Invalid localization runtime parameters")

    output = build_presentation_bundle(
        statement_bundle,
        gap_bundle,
        output_language=language,
        statement_bundle_sha256=statement_sha,
        gap_bundle_sha256=gap_sha,
    )
    started = time.perf_counter()
    if localized:
        api_key = os.environ.get(api_key_env, "").strip()
        if not api_key:
            raise EvidenceGapError(f"Missing API key environment variable {api_key_env}")
        translations, responses, retries = _localize(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            bundle=output,
            language=language,
            max_tokens=max_tokens,
            request_batch_size=request_batch_size,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        output = apply_localization(output, translations)
        usage = {
            key: sum(int(response.usage.get(key, 0)) for response in responses)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        provider_responses = [
            {
                "request_id": response.request_id,
                "raw_response_sha256": response.raw_response_sha256,
                "finish_reason": response.finish_reason,
            }
            for response in responses
        ]
        translation_count = len(translations)
        api_requests = len(responses)
    else:
        retries = 0
        usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        provider_responses: list[dict[str, Any]] = []
        translation_count = 0
        api_requests = 0

    validation = validate_presentation_bundle(
        output, statement_bundle=statement_bundle, gap_bundle=gap_bundle
    )
    name = _safe_name(run_name)
    target = (artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT) / name
    source_meta = {
        "statement_bundle": {"path": relative_path(root, statement_path), "sha256": statement_sha},
        "inference_gap_analysis": {"path": relative_path(root, gap_path), "sha256": gap_sha},
    }
    with atomic_directory(target, force=force) as staging:
        atomic_write_json(
            staging / "request.json",
            {
                "schema_version": PRESENTATION_BUNDLE_SCHEMA_VERSION,
                "contract_id": PRESENTATION_BUNDLE_CONTRACT_ID,
                "run_name": name,
                "language": language,
                "request_batch_size": request_batch_size,
                "statement_bundle_artifact_dir": relative_path(root, statement_dir),
                "inference_gap_artifact_dir": relative_path(root, gap_dir),
                "sources": source_meta,
            },
        )
        output_path = staging / "presentation_bundle.json"
        atomic_write_json(output_path, output)
        output_meta = {
            "path": relative_path(root, target / output_path.name),
            "sha256": sha256_file(output_path),
        }
        atomic_write_json(
            staging / "run_manifest.json",
            {
                "schema_version": PRESENTATION_BUNDLE_SCHEMA_VERSION,
                "contract_id": PRESENTATION_BUNDLE_CONTRACT_ID,
                "run_type": "phase077_output_module",
                "run_name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "output_language": language,
                "localized": localized,
                "provider": provider if localized else None,
                "model": model if localized else None,
                "prompt_version": LOCALIZATION_PROMPT_VERSION if localized else None,
                "parameters": {
                    "max_tokens": max_tokens,
                    "request_batch_size": request_batch_size,
                    "timeout_seconds": timeout_seconds,
                    "max_retries": max_retries,
                },
                "source": {
                    "statement_bundle_artifact_dir": relative_path(root, statement_dir),
                    "inference_gap_artifact_dir": relative_path(root, gap_dir),
                    **source_meta,
                },
                "counts": {
                    "claims": validation["total_claims"],
                    "inference_steps": validation["total_inference_steps"],
                    "articles": validation["articles"],
                    "evidence": validation["evidence"],
                    "scope_gaps": validation["gaps"]["SCOPE_GAP"],
                    "causal_gaps": validation["gaps"]["CAUSAL_GAP"],
                    "translation_units": translation_count,
                    "api_requests": api_requests,
                    "retries": retries,
                },
                "usage": usage,
                "provider_responses": provider_responses,
                "outputs": {"presentation_bundle": output_meta},
                "seconds": round(time.perf_counter() - started, 6),
            },
        )
    return {
        **validation,
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "output": {"presentation_bundle": output_meta},
        "api_requests": api_requests,
    }


def validate_output_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    request = _read_json(artifact_dir / "request.json")
    manifest = _read_json(artifact_dir / "run_manifest.json")
    if any(
        document.get("schema_version") != PRESENTATION_BUNDLE_SCHEMA_VERSION
        or document.get("contract_id") != PRESENTATION_BUNDLE_CONTRACT_ID
        for document in (request, manifest)
    ):
        raise EvidenceGapError("Unexpected presentation artifact contract")
    if request.get("run_name") != manifest.get("run_name"):
        raise EvidenceGapError("Presentation run name mismatch")

    root = _find_repo_root(artifact_dir)
    statement_dir = _resolve(root, request["statement_bundle_artifact_dir"])
    gap_dir = _resolve(root, request["inference_gap_artifact_dir"])
    validate_statement_bundle_artifact(statement_dir)
    validate_inference_gap_analysis_artifact(gap_dir)
    source = manifest.get("source")
    if not isinstance(source, Mapping):
        raise EvidenceGapError("Presentation source metadata is missing")
    for key, path in (
        ("statement_bundle", statement_dir / "statement_bundle.json"),
        ("inference_gap_analysis", gap_dir / "inference_gap_analysis.json"),
    ):
        request_meta = request.get("sources", {}).get(key)
        manifest_meta = source.get(key)
        if request_meta != manifest_meta or not isinstance(request_meta, Mapping):
            raise EvidenceGapError(f"Presentation {key} source metadata mismatch")
        if _resolve(root, request_meta["path"]) != path or sha256_file(path) != request_meta["sha256"]:
            raise EvidenceGapError(f"Presentation {key} source checksum mismatch")

    output_meta = manifest.get("outputs", {}).get("presentation_bundle")
    output_path = artifact_dir / "presentation_bundle.json"
    if not isinstance(output_meta, Mapping) or _resolve(root, output_meta["path"]) != output_path:
        raise EvidenceGapError("Presentation output metadata is invalid")
    if sha256_file(output_path) != output_meta["sha256"]:
        raise EvidenceGapError("Presentation output checksum mismatch")

    statement_bundle = _read_json(statement_dir / "statement_bundle.json")
    gap_bundle = _read_json(gap_dir / "inference_gap_analysis.json")
    bundle = _read_json(output_path)
    if (
        bundle.get("source_statement_bundle_sha256") != sha256_file(statement_dir / "statement_bundle.json")
        or bundle.get("source_inference_gap_analysis_sha256") != sha256_file(gap_dir / "inference_gap_analysis.json")
    ):
        raise EvidenceGapError("Presentation source hash mismatch")
    validation = validate_presentation_bundle(
        bundle, statement_bundle=statement_bundle, gap_bundle=gap_bundle
    )
    counts = manifest.get("counts")
    parameters = manifest.get("parameters")
    request_batch_size = request.get("request_batch_size")
    if (
        not isinstance(parameters, Mapping)
        or not isinstance(request_batch_size, int)
        or request_batch_size <= 0
        or parameters.get("request_batch_size") != request_batch_size
    ):
        raise EvidenceGapError("Presentation localization batch metadata is invalid")
    translation_units = len(_translation_units(bundle)) if validation["localized"] else 0
    api_requests = (
        (translation_units + request_batch_size - 1) // request_batch_size
        if validation["localized"]
        else 0
    )
    expected = {
        "claims": validation["total_claims"],
        "inference_steps": validation["total_inference_steps"],
        "articles": validation["articles"],
        "evidence": validation["evidence"],
        "scope_gaps": validation["gaps"]["SCOPE_GAP"],
        "causal_gaps": validation["gaps"]["CAUSAL_GAP"],
        "translation_units": translation_units,
        "api_requests": api_requests,
    }
    if not isinstance(counts, Mapping) or any(int(counts.get(key, -1)) != value for key, value in expected.items()):
        raise EvidenceGapError("Presentation manifest count mismatch")
    provider_responses = manifest.get("provider_responses")
    if not isinstance(provider_responses, list) or len(provider_responses) != api_requests:
        raise EvidenceGapError("Presentation provider response count mismatch")
    return {
        **validation,
        "run_name": manifest["run_name"],
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        "checksums": "PASS",
    }
