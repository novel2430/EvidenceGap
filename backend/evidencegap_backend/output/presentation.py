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

from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
    find_workspace_root,
)
from evidencegap_backend.config import validate_analysis_context
from evidencegap_backend.pipeline.inference_gap_analysis import (
    validate_inference_gap_analysis_artifact,
    validate_inference_gap_analysis_bundle,
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

PRESENTATION_BUNDLE_SCHEMA_VERSION = "1.4.0"
PRESENTATION_BUNDLE_CONTRACT_ID = "phase09c.presentation-bundle.v3"
LOCALIZATION_PROMPT_VERSION = "phase09b_output_localization_v1"
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
_CLAIM_INFERENCE_INTEGRITY_STATES = (
    "INTACT",
    "GAPPED",
    "NOT_APPLICABLE",
    "ERROR",
)
_STEP_INFERENCE_INTEGRITY_STATES = ("INTACT", "GAPPED")
_ENGLISH_ALIASES = {"en", "en-us", "en-gb", "english", "english (us)", "english (uk)"}


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
    return find_workspace_root(start)


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


def _build_claim_audit(
    *,
    evidence_status: str,
    incoming_step_ids: list[str],
    step_integrity_by_id: Mapping[str, str],
) -> dict[str, Any]:
    if evidence_status == "ERROR":
        inference_integrity = "ERROR"
        affecting_step_ids: list[str] = []
    elif not incoming_step_ids:
        inference_integrity = "NOT_APPLICABLE"
        affecting_step_ids = []
    else:
        affecting_step_ids = [
            step_id
            for step_id in incoming_step_ids
            if step_integrity_by_id[step_id] == "GAPPED"
        ]
        inference_integrity = "GAPPED" if affecting_step_ids else "INTACT"
    return {
        "evidence_status": evidence_status,
        "inference_integrity": inference_integrity,
        "affecting_inference_step_ids": affecting_step_ids,
    }


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
                gap = analysis[field]
                step["gaps"].append(
                    {
                        "gap_type": gap_type,
                        "subtype": gap["subtype"],
                        "affected_dimensions": list(gap["affected_dimensions"]),
                        "supported_basis": gap["supported_basis"],
                        "unsupported_extension": gap["unsupported_extension"],
                        "detection_method": "llm",
                        "reason_en": gap["reason"],
                        "closure_requirement_en": gap["closure_requirement"],
                        "display_reason": gap["reason"],
                        "display_closure_requirement": gap["closure_requirement"],
                    }
                )
        step["inference_integrity"] = (
            "GAPPED" if step["gaps"] else "INTACT"
        )

    step_integrity_by_id = {
        str(step["inference_step_id"]): str(step["inference_integrity"])
        for step in steps
    }
    for claim in claims:
        claim["audit"] = _build_claim_audit(
            evidence_status=str(claim["evidence_state"]),
            incoming_step_ids=list(claim["conclusion_inference_step_ids"]),
            step_integrity_by_id=step_integrity_by_id,
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
        "analysis_context": copy.deepcopy(dict(statement_bundle["analysis_context"])),
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
        "claim_inference_integrity": {
            state: sum(
                claim["audit"]["inference_integrity"] == state
                for claim in claims
            )
            for state in _CLAIM_INFERENCE_INTEGRITY_STATES
        },
        "total_inference_steps": len(steps),
        "inference_step_integrity": {
            state: sum(step["inference_integrity"] == state for step in steps)
            for state in _STEP_INFERENCE_INTEGRITY_STATES
        },
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
        "terminal_conclusions": [
            {
                "claim_id": claim["claim_id"],
                "evidence_status": claim["audit"]["evidence_status"],
                "inference_integrity": claim["audit"]["inference_integrity"],
                "affecting_inference_step_ids": list(
                    claim["audit"]["affecting_inference_step_ids"]
                ),
            }
            for claim in claims
            if claim["argument_role"] == "CONCLUSION"
        ],
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
            prefix = f"inference:{step['inference_step_id']}:{gap['gap_type']}"
            add(f"{prefix}:reason", gap["display_reason"])
            add(
                f"{prefix}:closure_requirement",
                gap["display_closure_requirement"],
            )
    return units


def build_localization_prompt(
    units: list[Mapping[str, str]],
    target_language: str,
    retry_note: str | None = None,
) -> str:
    parts = [
        f"TARGET LANGUAGE: {target_language}",
        "Translate every item and copy every text_id exactly.",
        "Return exactly this JSON shape and no other top-level keys:",
        json.dumps(
            {
                "translations": [
                    {
                        "text_id": "COPY_THE_INPUT_TEXT_ID_EXACTLY",
                        "text": "TRANSLATED_TEXT",
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        "Do not return a top-level texts, result, output, or data key.",
        "INPUT TRANSLATION UNITS:",
        json.dumps({"translation_units": units}, ensure_ascii=False, indent=2),
    ]
    if retry_note:
        parts.extend(
            [
                "The previous response was invalid. Correct only its JSON structure and retry:",
                retry_note,
            ]
        )
    return "\n\n".join(parts)


def _translation_rows(payload: Mapping[str, Any]) -> list[Any]:
    """Extract a safe, semantically equivalent translation-array shape.

    DeepSeek's ``json_object`` mode guarantees a JSON object but does not enforce
    the supplied JSON Schema. It may mirror the input key (``texts``) or wrap the
    requested object once. We normalize only these narrow structural variants;
    the strict text_id coverage and row validation below remain unchanged.
    """

    current: Mapping[str, Any] = payload
    for _ in range(2):
        keys = set(current)
        if keys == {"translations"}:
            rows = current["translations"]
            if not isinstance(rows, list):
                raise _ProviderError(
                    "translations must be an array", retryable=True
                )
            return rows
        if keys == {"texts"}:
            rows = current["texts"]
            if not isinstance(rows, list):
                raise _ProviderError("texts must be an array", retryable=True)
            return rows
        if len(keys) == 1:
            wrapper = next(iter(keys))
            nested = current[wrapper]
            if wrapper in {"result", "output", "data"} and isinstance(
                nested, Mapping
            ):
                current = nested
                continue
        break

    received = sorted(str(key) for key in current)
    raise _ProviderError(
        "Response must contain exactly a translations array; "
        f"received top-level keys: {received}",
        retryable=True,
    )


def validate_localization_response(
    payload: Mapping[str, Any], expected_units: list[Mapping[str, str]]
) -> dict[str, str]:
    rows = _translation_rows(payload)
    expected_ids = [str(row["text_id"]) for row in expected_units]
    values: dict[str, str] = {}
    for row in rows:
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
            prefix = f"inference:{step['inference_step_id']}:{gap['gap_type']}"
            gap["display_reason"] = translations[f"{prefix}:reason"]
            gap["display_closure_requirement"] = translations[
                f"{prefix}:closure_requirement"
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
    prompt: ResolvedPrompt,
) -> tuple[dict[str, str], ProviderResponse, int]:
    retry_note: str | None = None
    for attempt in range(max_retries + 1):
        try:
            response = call_structured_llm(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=prompt.system_prompt,
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
    prompt: ResolvedPrompt,
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
            prompt=prompt,
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
    if bundle.get("analysis_context") != statement_bundle.get("analysis_context"):
        raise EvidenceGapError("Presentation analysis context changed source data")
    try:
        validate_analysis_context(bundle.get("analysis_context") or {})
    except ValueError as exc:
        raise EvidenceGapError("Invalid presentation analysis context") from exc
    if _without_added_fields(bundle["statement"], ("display_text",)) != statement_bundle["statement"]:
        raise EvidenceGapError("Presentation statement changed source data")
    if not str(bundle["statement"].get("display_text") or "").strip():
        raise EvidenceGapError("Presentation statement display text is blank")

    gap_source = {
        str(row["inference_step_id"]): row
        for row in gap_bundle["inference_gap_analyses"]
    }
    step_integrity_by_id = {
        step_id: (
            "GAPPED"
            if analysis["scope_gap"]["detected"]
            or analysis["causal_gap"]["detected"]
            else "INTACT"
        )
        for step_id, analysis in gap_source.items()
    }

    claim_added = (
        "evidence_state",
        "argument_role",
        "premise_inference_step_ids",
        "conclusion_inference_step_ids",
        "audit",
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
        expected_audit = _build_claim_audit(
            evidence_status=expected_state,
            incoming_step_ids=list(row.get("conclusion_inference_step_ids") or []),
            step_integrity_by_id=step_integrity_by_id,
        )
        if row.get("audit") != expected_audit:
            raise EvidenceGapError("Invalid presentation Claim audit")
        if not str(row.get("display_text") or "").strip():
            raise EvidenceGapError("Presentation Claim display text is blank")

    if len(bundle["inference_steps"]) != len(statement_bundle["inference_steps"]):
        raise EvidenceGapError("Presentation inference count mismatch")
    for row, source in zip(bundle["inference_steps"], statement_bundle["inference_steps"], strict=True):
        if _without_added_fields(row, ("gaps", "inference_integrity")) != source:
            raise EvidenceGapError("Presentation inference changed source data")
        expected = []
        analysis = gap_source[str(source["inference_step_id"])]
        if analysis["scope_gap"]["detected"]:
            expected.append("SCOPE_GAP")
        if analysis["causal_gap"]["detected"]:
            expected.append("CAUSAL_GAP")
        if [gap.get("gap_type") for gap in row["gaps"]] != expected:
            raise EvidenceGapError("Presentation Gap mismatch")
        expected_integrity = "GAPPED" if expected else "INTACT"
        if row.get("inference_integrity") != expected_integrity:
            raise EvidenceGapError("Invalid presentation inference integrity")
        source_by_type = {
            "SCOPE_GAP": analysis["scope_gap"],
            "CAUSAL_GAP": analysis["causal_gap"],
        }
        for gap in row["gaps"]:
            source_gap = source_by_type[str(gap.get("gap_type"))]
            if (
                gap.get("detection_method") != "llm"
                or gap.get("subtype") != source_gap["subtype"]
                or gap.get("affected_dimensions")
                != source_gap["affected_dimensions"]
                or gap.get("supported_basis") != source_gap["supported_basis"]
                or gap.get("unsupported_extension")
                != source_gap["unsupported_extension"]
                or gap.get("reason_en") != source_gap["reason"]
                or gap.get("closure_requirement_en")
                != source_gap["closure_requirement"]
                or not str(gap.get("display_reason") or "").strip()
                or not str(gap.get("display_closure_requirement") or "").strip()
            ):
                raise EvidenceGapError("Invalid presentation Gap")
            if not bundle["localized"] and (
                gap["display_reason"] != gap["reason_en"]
                or gap["display_closure_requirement"]
                != gap["closure_requirement_en"]
            ):
                raise EvidenceGapError(
                    "Unlocalized presentation Gap display text changed source data"
                )

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
    prompt_override: PromptOverride | None = None,
    artifact_root: Path | None = None,
    statement_bundle: Mapping[str, Any] | None = None,
    gap_bundle: Mapping[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    statement_dir = _resolve(root, statement_bundle_artifact_dir)
    gap_dir = _resolve(root, inference_gap_artifact_dir)
    statement_path = statement_dir / "statement_bundle.json"
    gap_path = gap_dir / "inference_gap_analysis.json"
    if (statement_bundle is None) != (gap_bundle is None):
        raise EvidenceGapError(
            "statement_bundle and gap_bundle must be provided together"
        )
    if statement_bundle is None or gap_bundle is None:
        validate_statement_bundle_artifact(statement_dir)
        validate_inference_gap_analysis_artifact(gap_dir)
        statement_bundle_value = _read_json(statement_path)
        gap_bundle_value = _read_json(gap_path)
        source_handoff = "artifact_reload"
    else:
        statement_bundle_value = dict(statement_bundle)
        gap_bundle_value = dict(gap_bundle)
        source_handoff = "in_memory_handoff"
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
    prompt = resolve_builtin_prompt(
        prompt_name="localization.txt",
        default_version=LOCALIZATION_PROMPT_VERSION,
        override=prompt_override,
    )

    output = build_presentation_bundle(
        statement_bundle_value,
        gap_bundle_value,
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
            prompt=prompt,
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
        output,
        statement_bundle=statement_bundle_value,
        gap_bundle=gap_bundle_value,
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
                "source_handoff": source_handoff,
                "output_language": language,
                "localized": localized,
                "provider": provider if localized else None,
                "model": model if localized else None,
                **(
                    prompt.manifest_fields()
                    if localized
                    else {
                        "prompt_version": None,
                        "prompt_sha256": None,
                        "prompt_source": None,
                    }
                ),
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
        "presentation_bundle": output,
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
