from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    require_empty_or_force,
    sha256_text,
)
from evidencegap.stance.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    RUN_SCHEMA_VERSION,
    iter_inputs,
    validate_input_artifact,
    validate_prediction_artifact,
    write_predictions_atomic,
)
from evidencegap.stance.contracts import (
    EVIDENCE_TYPES,
    SCHEMA_VERSION,
    STANCE_LABELS,
    TASK_ID,
    StanceInput,
    StancePrediction,
    canonical_evidence_type,
    canonical_stance_label,
)
from evidencegap.stance.evaluation import (
    evaluate_prediction_rows,
    render_evaluation_markdown,
)

PROMPT_VERSION = "phase06_llm_stance_v2"
SUPPORTED_PROVIDERS = ("deepseek", "anthropic")
DEFAULT_MODELS = {
    "deepseek": "deepseek-v4-pro",
    "anthropic": "claude-sonnet-4-6",
}
DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com",
    "anthropic": "https://api.anthropic.com",
}
DEFAULT_API_KEY_ENVS = {
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_THINKING_DISABLED_MODELS = frozenset({"claude-sonnet-5"})

EVIDENCE_TYPE_DESCRIPTIONS = {
    "direct_result": (
        "A substantive efficacy, effectiveness, association, comparison, or other "
        "reported result that directly bears on the claim."
    ),
    "background": (
        "Prior knowledge, motivation, general context, or literature discussion "
        "without a result that directly establishes the claim."
    ),
    "method": (
        "Study design, recruitment, intervention procedure, measurement, analysis "
        "method, or other methodological information."
    ),
    "population_or_scope": (
        "A population, setting, intervention, outcome, time frame, or applicability "
        "constraint that limits how broadly the evidence applies."
    ),
    "safety": (
        "Adverse events, harms, tolerability, side effects, mortality risk, or other "
        "safety-related evidence."
    ),
    "statistical_uncertainty": (
        "Statistical non-significance, wide confidence intervals, low power, "
        "imprecision, inconsistent estimates, or explicit uncertainty."
    ),
    "mixed_or_other": (
        "Multiple equally important evidence types, a study limitation that does "
        "not fit a more specific category, or content not covered above."
    ),
}

EVIDENCE_TYPE_PRIORITY = (
    "safety",
    "statistical_uncertainty",
    "population_or_scope",
    "direct_result",
    "method",
    "background",
    "mixed_or_other",
)


def _evidence_type_prompt_block() -> str:
    lines = [
        "Evidence type:",
        "Choose exactly one of the following seven values. Do not invent, rename, combine, or pluralize categories.",
    ]
    for evidence_type in EVIDENCE_TYPES:
        lines.append(f"- {evidence_type}: {EVIDENCE_TYPE_DESCRIPTIONS[evidence_type]}")
    lines.extend(
        [
            "When more than one category applies, choose the first applicable category in this priority order:",
            " > ".join(EVIDENCE_TYPE_PRIORITY),
            "Use mixed_or_other only when no more specific category applies.",
            "The evidence_type value must exactly match one of the seven values above.",
        ]
    )
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are an evidence-grounded medical stance classifier.

Judge only the relationship between the supplied EVIDENCE and CLAIM. Do not use outside medical knowledge, web knowledge, or assumptions that are not stated in the evidence.

Labels:
- support: the evidence directly increases confidence that the claim is true.
- refute: the evidence directly increases confidence that the claim is false or materially contradicts it.
- insufficient: the evidence is merely related, provides background or methods only, lacks the needed result, depends on missing context, or does not establish either direction.

Important rules:
- Relevance is not support.
- A study being mentioned is not evidence of its conclusion.
- Match population, intervention/exposure, comparator, outcome, direction, and scope.
- Do not convert association into causation.
- A non-significant result refutes an affirmative effect claim only when the evidence directly tests the same proposition; otherwise choose insufficient.
- For a sentence unit, classify that sentence with the supplied adjacent context used only to resolve references.
- For a bundle unit, classify the bundle as a whole.
- Probabilities are self-assessed stance probabilities and must sum to 1.
- Rationales must be one concise English sentence grounded in the supplied text.

{_evidence_type_prompt_block()}

Return JSON only, with exactly one result for every input_id and no extra items."""


def response_json_schema() -> dict[str, Any]:
    result_schema = {
        "type": "object",
        "properties": {
            "input_id": {"type": "string"},
            "label": {"type": "string", "enum": list(STANCE_LABELS)},
            "probabilities": {
                "type": "object",
                "properties": {
                    "support": {"type": "number"},
                    "refute": {"type": "number"},
                    "insufficient": {"type": "number"},
                },
                "required": ["support", "refute", "insufficient"],
                "additionalProperties": False,
            },
            "rationale": {"type": "string"},
            "evidence_type": {
                "type": "string",
                "enum": list(EVIDENCE_TYPES),
                "description": (
                    "Choose exactly one evidence category using this priority order: "
                    + " > ".join(EVIDENCE_TYPE_PRIORITY)
                    + ". Do not invent new category names."
                ),
            },
            "requires_context": {"type": "boolean"},
        },
        "required": [
            "input_id",
            "label",
            "probabilities",
            "rationale",
            "evidence_type",
            "requires_context",
        ],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": result_schema,
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


_EXAMPLE_JSON = {
    "results": [
        {
            "input_id": "example-id",
            "label": "support",
            "probabilities": {
                "support": 0.9,
                "refute": 0.02,
                "insufficient": 0.08,
            },
            "rationale": "The reported outcome directly matches the direction asserted by the claim.",
            "evidence_type": "direct_result",
            "requires_context": False,
        }
    ]
}


def _model_identity(
    *,
    provider: str,
    model: str,
    base_url: str,
) -> tuple[str, str, str]:
    """Return stable hashes for the active prompt, schema, and model setup."""

    schema_hash = sha256_text(
        json.dumps(response_json_schema(), ensure_ascii=False, sort_keys=True)
    )
    prompt_hash = sha256_text(SYSTEM_PROMPT)
    model_fingerprint = sha256_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url.rstrip("/"),
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "response_schema_sha256": schema_hash,
                "anthropic_thinking": (
                    _anthropic_thinking_config(model)
                    if provider == "anthropic"
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    return schema_hash, prompt_hash, model_fingerprint


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid name: {value!r}")
    return cleaned


def _input_payload(item: StanceInput) -> dict[str, Any]:
    return {
        "input_id": item.input_id,
        "evidence_unit": item.evidence_unit,
        "claim": item.claim_text,
        "evidence": item.evidence_text,
        "context_before": item.context_before,
        "context_after": item.context_after,
    }


def _user_prompt(inputs: Sequence[StanceInput], *, retry_note: str | None = None) -> str:
    payload = [_input_payload(item) for item in inputs]
    parts = [
        "Classify every item below. Return results in the same order as the input items.",
        "The response must be a JSON object matching this example shape:",
        json.dumps(_EXAMPLE_JSON, ensure_ascii=False, indent=2),
        (
            "For evidence_type, choose exactly one of: "
            + ", ".join(EVIDENCE_TYPES)
            + ". Do not invent or rename categories."
        ),
        "INPUT ITEMS JSON:",
        json.dumps(payload, ensure_ascii=False, indent=2),
    ]
    if retry_note:
        parts.append(
            "The previous response was invalid. Correct the following issue and return a complete JSON object: "
            + retry_note
        )
    return "\n\n".join(parts)


@dataclass(frozen=True)
class ProviderResponse:
    payload: Mapping[str, Any]
    request_id: str | None
    usage: Mapping[str, int]
    raw_response_sha256: str
    finish_reason: str | None


class _ProviderError(EvidenceGapError):
    def __init__(self, message: str, *, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


_RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


def _post_json(
    url: str,
    *,
    headers: Mapping[str, str],
    body: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[Mapping[str, Any], str]:
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raw_error = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(raw_error)
            detail = json.dumps(error_payload, ensure_ascii=False)
        except json.JSONDecodeError:
            detail = raw_error[:2000]
        raise _ProviderError(
            f"HTTP {exc.code} from {url}: {detail}",
            retryable=exc.code in _RETRYABLE_HTTP_STATUS,
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise _ProviderError(
            f"Network error calling {url}: {exc}", retryable=True
        ) from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _ProviderError(
            f"Provider returned non-JSON HTTP response: {raw[:1000]!r}",
            retryable=True,
        ) from exc
    if not isinstance(payload, Mapping):
        raise _ProviderError("Provider response must be a JSON object", retryable=True)
    return payload, raw


def _parse_json_content(content: str) -> Mapping[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        raise _ProviderError("Provider returned empty content", retryable=True)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise _ProviderError(
            f"Provider content is not valid JSON: {exc}", retryable=True
        ) from exc
    if not isinstance(payload, Mapping):
        raise _ProviderError("Provider JSON content must be an object", retryable=True)
    return payload



def call_structured_llm(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_schema: Mapping[str, Any],
    max_tokens: int,
    timeout_seconds: float,
    thinking: bool = False,
) -> ProviderResponse:
    """Call a supported provider with a task-specific structured JSON prompt.

    Phase 06 and Phase 07 use the same provider transport, response parsing,
    truncation checks, and usage accounting.  Task-specific validation and
    caching remain in the caller because their contracts differ.
    """

    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(
            f"provider must be one of {SUPPORTED_PROVIDERS}, got {provider!r}"
        )
    if provider != "deepseek" and thinking:
        raise EvidenceGapError("thinking is only supported for DeepSeek")
    if provider == "deepseek":
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
            "stream": False,
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        response, raw = _post_json(
            base_url.rstrip("/") + "/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            body=body,
            timeout_seconds=timeout_seconds,
        )
        try:
            choice = response["choices"][0]
            content = choice["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise _ProviderError(
                "Unexpected DeepSeek response shape: "
                + json.dumps(response, ensure_ascii=False)[:2000],
                retryable=True,
            ) from exc
        finish_reason = (
            None if choice.get("finish_reason") is None else str(choice["finish_reason"])
        )
        if finish_reason == "length":
            raise _ProviderError(
                "DeepSeek output was truncated (finish_reason=length); increase --max-tokens or reduce --request-batch-size",
                retryable=False,
            )
        usage_raw = response.get("usage") or {}
        return ProviderResponse(
            payload=_parse_json_content(str(content or "")),
            request_id=None if response.get("id") is None else str(response["id"]),
            usage={
                "input_tokens": int(usage_raw.get("prompt_tokens") or 0),
                "output_tokens": int(usage_raw.get("completion_tokens") or 0),
                "total_tokens": int(usage_raw.get("total_tokens") or 0),
            },
            raw_response_sha256=sha256_text(raw),
            finish_reason=finish_reason,
        )

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": dict(response_schema),
            }
        },
    }
    thinking_config = _anthropic_thinking_config(model)
    if thinking_config is not None:
        body["thinking"] = thinking_config
    response, raw = _post_json(
        base_url.rstrip("/") + "/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        body=body,
        timeout_seconds=timeout_seconds,
    )
    stop_reason = (
        None if response.get("stop_reason") is None else str(response["stop_reason"])
    )
    if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
        raise _ProviderError(
            f"Claude output was truncated (stop_reason={stop_reason}); increase --max-tokens or reduce --request-batch-size",
            retryable=False,
        )
    if stop_reason == "refusal":
        raise _ProviderError("Claude refused the structured request", retryable=False)
    content_blocks = response.get("content")
    if not isinstance(content_blocks, list):
        raise _ProviderError("Unexpected Claude content shape", retryable=True)
    text_blocks = [
        str(block.get("text"))
        for block in content_blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    if not text_blocks:
        raise _ProviderError("Claude returned no text content block", retryable=True)
    usage_raw = response.get("usage") or {}
    input_tokens = int(usage_raw.get("input_tokens") or 0)
    output_tokens = int(usage_raw.get("output_tokens") or 0)
    return ProviderResponse(
        payload=_parse_json_content("\n".join(text_blocks)),
        request_id=None if response.get("id") is None else str(response["id"]),
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        raw_response_sha256=sha256_text(raw),
        finish_reason=stop_reason,
    )

def _call_deepseek(
    *,
    api_key: str,
    base_url: str,
    model: str,
    inputs: Sequence[StanceInput],
    max_tokens: int,
    timeout_seconds: float,
    retry_note: str | None,
    thinking: bool,
) -> ProviderResponse:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _user_prompt(inputs, retry_note=retry_note)},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "stream": False,
        "thinking": {"type": "enabled" if thinking else "disabled"},
    }
    response, raw = _post_json(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        body=body,
        timeout_seconds=timeout_seconds,
    )
    try:
        choice = response["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise _ProviderError(
            f"Unexpected DeepSeek response shape: {json.dumps(response, ensure_ascii=False)[:2000]}",
            retryable=True,
        ) from exc
    finish_reason = None if choice.get("finish_reason") is None else str(choice["finish_reason"])
    if finish_reason == "length":
        raise _ProviderError(
            "DeepSeek output was truncated (finish_reason=length); increase --max-tokens or reduce --request-batch-size",
            retryable=False,
        )
    usage_raw = response.get("usage") or {}
    usage = {
        "input_tokens": int(usage_raw.get("prompt_tokens") or 0),
        "output_tokens": int(usage_raw.get("completion_tokens") or 0),
        "total_tokens": int(usage_raw.get("total_tokens") or 0),
    }
    return ProviderResponse(
        payload=_parse_json_content(str(content or "")),
        request_id=None if response.get("id") is None else str(response["id"]),
        usage=usage,
        raw_response_sha256=sha256_text(raw),
        finish_reason=finish_reason,
    )


def _anthropic_thinking_config(model: str) -> dict[str, str] | None:
    """Return model-specific Anthropic thinking configuration.

    Claude Sonnet 5 enables adaptive thinking when the field is omitted. This
    stance-classification workload intentionally disables it to avoid paying
    for reasoning tokens that are not part of the structured result. Older
    Anthropic models retain their existing request shape.
    """

    normalized = model.strip().lower()
    if normalized in ANTHROPIC_THINKING_DISABLED_MODELS:
        return {"type": "disabled"}
    return None


def _call_anthropic(
    *,
    api_key: str,
    base_url: str,
    model: str,
    inputs: Sequence[StanceInput],
    max_tokens: int,
    timeout_seconds: float,
    retry_note: str | None,
) -> ProviderResponse:
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": _user_prompt(inputs, retry_note=retry_note)}
        ],
        "output_config": {
            "format": {
                "type": "json_schema",
                "schema": response_json_schema(),
            }
        },
    }
    thinking_config = _anthropic_thinking_config(model)
    if thinking_config is not None:
        body["thinking"] = thinking_config
    response, raw = _post_json(
        base_url.rstrip("/") + "/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        },
        body=body,
        timeout_seconds=timeout_seconds,
    )
    stop_reason = None if response.get("stop_reason") is None else str(response["stop_reason"])
    if stop_reason in {"max_tokens", "model_context_window_exceeded"}:
        raise _ProviderError(
            f"Claude output was truncated (stop_reason={stop_reason}); increase --max-tokens or reduce --request-batch-size",
            retryable=False,
        )
    if stop_reason == "refusal":
        raise _ProviderError("Claude refused the stance request", retryable=False)
    content_blocks = response.get("content")
    if not isinstance(content_blocks, list):
        raise _ProviderError("Unexpected Claude content shape", retryable=True)
    text_blocks = [
        str(block.get("text"))
        for block in content_blocks
        if isinstance(block, Mapping) and block.get("type") == "text"
    ]
    if not text_blocks:
        raise _ProviderError("Claude returned no text content block", retryable=True)
    usage_raw = response.get("usage") or {}
    input_tokens = int(usage_raw.get("input_tokens") or 0)
    output_tokens = int(usage_raw.get("output_tokens") or 0)
    return ProviderResponse(
        payload=_parse_json_content("\n".join(text_blocks)),
        request_id=None if response.get("id") is None else str(response["id"]),
        usage={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        raw_response_sha256=sha256_text(raw),
        finish_reason=stop_reason,
    )


def _validate_result_payload(
    payload: Mapping[str, Any],
    inputs: Sequence[StanceInput],
) -> list[dict[str, Any]]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list):
        raise _ProviderError("Response JSON must contain a results array", retryable=True)
    expected_ids = [item.input_id for item in inputs]
    if len(raw_results) != len(expected_ids):
        raise _ProviderError(
            f"Expected {len(expected_ids)} results, got {len(raw_results)}",
            retryable=True,
        )
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_results):
        if not isinstance(raw, Mapping):
            raise _ProviderError(f"results[{index}] must be an object", retryable=True)
        input_id = str(raw.get("input_id", "")).strip()
        if input_id != expected_ids[index]:
            raise _ProviderError(
                f"results[{index}].input_id must be {expected_ids[index]!r}, got {input_id!r}",
                retryable=True,
            )
        if input_id in seen:
            raise _ProviderError(f"Duplicate result input_id {input_id!r}", retryable=True)
        seen.add(input_id)
        raw_label_value = str(raw.get("label", "")).strip()
        probabilities_raw = raw.get("probabilities")
        if not isinstance(probabilities_raw, Mapping):
            raise _ProviderError(
                f"{input_id}: probabilities must be an object", retryable=True
            )
        try:
            probabilities = {
                label_name: float(probabilities_raw[label_name])
                for label_name in STANCE_LABELS
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise _ProviderError(
                f"{input_id}: probabilities require numeric support/refute/insufficient",
                retryable=True,
            ) from exc
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in probabilities.values()
        ):
            raise _ProviderError(
                f"{input_id}: probabilities must be finite values in [0, 1]",
                retryable=True,
            )
        probability_sum = sum(probabilities.values())
        if abs(probability_sum - 1.0) > 0.01:
            raise _ProviderError(
                f"{input_id}: probabilities sum to {probability_sum:.6f}, not 1",
                retryable=True,
            )
        if probability_sum <= 0:
            raise _ProviderError(f"{input_id}: probability sum is zero", retryable=True)
        probabilities = {
            key: value / probability_sum for key, value in probabilities.items()
        }
        expected_label = max(STANCE_LABELS, key=lambda key: probabilities[key])
        label_recovered = bool(raw.get("label_recovered", False))
        label_original_value = raw.get("label_original_value")
        try:
            label = canonical_stance_label(raw_label_value)
        except EvidenceGapError:
            # A generated label is a core field, but when the model also returned a
            # complete numeric stance distribution we can recover deterministically
            # from its argmax instead of rebilling the same batch.
            label = expected_label
            label_recovered = True
            label_original_value = raw_label_value or "blank"
        probability_reconciled = bool(raw.get("probability_reconciled", False))
        probability_original_argmax = raw.get("probability_original_argmax")
        if probability_original_argmax is not None:
            try:
                probability_original_argmax = canonical_stance_label(
                    str(probability_original_argmax)
                )
            except EvidenceGapError:
                probability_original_argmax = None
        if label != expected_label:
            # The explicit stance label is the primary model decision. Self-assessed
            # probabilities are auxiliary metadata and may occasionally disagree by
            # a small amount. Reconcile locally instead of retrying a valid response
            # and spending API tokens again.
            probability_original_argmax = expected_label
            probabilities[label], probabilities[expected_label] = (
                probabilities[expected_label],
                probabilities[label],
            )
            max_other = max(
                probabilities[key] for key in STANCE_LABELS if key != label
            )
            if probabilities[label] <= max_other:
                probabilities[label] += 1e-6
                normalized_sum = sum(probabilities.values())
                probabilities = {
                    key: value / normalized_sum
                    for key, value in probabilities.items()
                }
            probability_reconciled = True
        rationale = str(raw.get("rationale", "")).strip()
        if not rationale:
            raise _ProviderError(f"{input_id}: rationale is blank", retryable=True)
        evidence_type = canonical_evidence_type(
            None if raw.get("evidence_type") is None else str(raw.get("evidence_type"))
        )
        requires_context = raw.get("requires_context")
        if not isinstance(requires_context, bool):
            raise _ProviderError(
                f"{input_id}: requires_context must be boolean", retryable=True
            )
        validated.append(
            {
                "input_id": input_id,
                "label": label,
                "probabilities": probabilities,
                "rationale": rationale,
                "evidence_type": evidence_type,
                "requires_context": requires_context,
                "label_recovered": label_recovered,
                "label_original_value": label_original_value,
                "probability_reconciled": probability_reconciled,
                "probability_original_argmax": probability_original_argmax,
            }
        )
    return validated


def _request_fingerprint(
    *,
    provider: str,
    model: str,
    base_url: str,
    inputs: Sequence[StanceInput],
    max_tokens: int,
    thinking: bool,
) -> str:
    value = {
        "provider": provider,
        "model": model,
        "base_url": base_url.rstrip("/"),
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "response_schema": response_json_schema(),
        "inputs": [_input_payload(item) for item in inputs],
        "max_tokens": max_tokens,
        "thinking": thinking if provider == "deepseek" else None,
        "anthropic_thinking": (
            _anthropic_thinking_config(model) if provider == "anthropic" else None
        ),
    }
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _read_cache(path: Path, *, request_hash: str) -> tuple[list[dict[str, Any]], ProviderResponse] | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceGapError(f"Invalid LLM cache file {path}: {exc}") from exc
    if cached.get("request_hash") != request_hash:
        raise EvidenceGapError(f"LLM cache hash mismatch: {path}")
    payload = cached.get("payload")
    validated = cached.get("validated_results")
    if not isinstance(payload, Mapping) or not isinstance(validated, list):
        raise EvidenceGapError(f"Incomplete LLM cache file: {path}")
    response = ProviderResponse(
        payload=payload,
        request_id=cached.get("request_id"),
        usage={
            key: int((cached.get("usage") or {}).get(key) or 0)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
        raw_response_sha256=str(cached["raw_response_sha256"]),
        finish_reason=cached.get("finish_reason"),
    )
    return [dict(item) for item in validated], response


def _write_cache(
    path: Path,
    *,
    request_hash: str,
    provider: str,
    model: str,
    input_ids: Sequence[str],
    response: ProviderResponse,
    validated_results: Sequence[Mapping[str, Any]],
) -> None:
    atomic_write_json(
        path,
        {
            "cache_schema_version": "1.0.0",
            "request_hash": request_hash,
            "provider": provider,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "input_ids": list(input_ids),
            "request_id": response.request_id,
            "usage": dict(response.usage),
            "finish_reason": response.finish_reason,
            "raw_response_sha256": response.raw_response_sha256,
            "payload": response.payload,
            "validated_results": list(validated_results),
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    )


def _call_with_retries(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    inputs: Sequence[StanceInput],
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    thinking: bool,
) -> tuple[list[dict[str, Any]], ProviderResponse, int]:
    retry_note: str | None = None
    for attempt in range(max_retries + 1):
        try:
            if provider == "deepseek":
                response = _call_deepseek(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    inputs=inputs,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    retry_note=retry_note,
                    thinking=thinking,
                )
            elif provider == "anthropic":
                response = _call_anthropic(
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    inputs=inputs,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    retry_note=retry_note,
                )
            else:
                raise AssertionError(provider)
            validated = _validate_result_payload(response.payload, inputs)
            return validated, response, attempt
        except _ProviderError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise EvidenceGapError(
                    f"{provider} request failed after {attempt + 1} attempt(s): {exc}"
                ) from exc
            retry_note = str(exc)[:1000]
            delay = min(30.0, 1.5 * (2**attempt)) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    raise AssertionError("unreachable")


def _prediction_from_result(
    stance_input: StanceInput,
    result: Mapping[str, Any],
    *,
    run_name: str,
    model_name: str,
    model_fingerprint: str,
    input_sha256: str,
    provider: str,
    response: ProviderResponse,
) -> StancePrediction:
    probabilities = result["probabilities"]
    predicted_label = canonical_stance_label(str(result["label"]))
    confidence = float(probabilities[predicted_label])
    ordered_probabilities = sorted(
        (float(probabilities[label]) for label in STANCE_LABELS), reverse=True
    )
    probability_margin = ordered_probabilities[0] - ordered_probabilities[1]
    return StancePrediction(
        stance_input=stance_input,
        run_name=run_name,
        model_name=model_name,
        model_fingerprint=model_fingerprint,
        stance_input_artifact_sha256=input_sha256,
        predicted_label=predicted_label,
        probability_support=float(probabilities["support"]),
        probability_refute=float(probabilities["refute"]),
        probability_insufficient=float(probabilities["insufficient"]),
        confidence=confidence,
        probability_margin=probability_margin,
        abstained=False,
        rationale=str(result["rationale"]),
        evidence_type=str(result["evidence_type"]),
        requires_context=bool(result["requires_context"]),
        provider=provider,
        provider_request_id=response.request_id,
        raw_response_sha256=response.raw_response_sha256,
        prompt_version=PROMPT_VERSION,
    )


def _batches(items: Sequence[StanceInput], size: int) -> Iterable[Sequence[StanceInput]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]



def _selection_group_id(item: StanceInput) -> str:
    return item.query_id or item.claim_id


def _select_inputs(
    inputs: Sequence[StanceInput],
    *,
    offset: int = 0,
    limit: int | None = None,
    query_offset: int = 0,
    query_limit: int | None = None,
    query_sample_size: int | None = None,
    query_sample_seed: int = 20260722,
) -> tuple[list[StanceInput], dict[str, Any]]:
    """Select rows while preserving complete query groups for Phase 05 smoke runs."""

    if offset < 0 or query_offset < 0:
        raise EvidenceGapError("offsets cannot be negative")
    if limit is not None and limit <= 0:
        raise EvidenceGapError("limit must be positive")
    if query_limit is not None and query_limit <= 0:
        raise EvidenceGapError("query_limit must be positive")
    if query_sample_size is not None and query_sample_size <= 0:
        raise EvidenceGapError("query_sample_size must be positive")
    row_selection = offset != 0 or limit is not None
    query_selection = (
        query_offset != 0
        or query_limit is not None
        or query_sample_size is not None
    )
    if row_selection and query_selection:
        raise EvidenceGapError(
            "Row selection (--offset/--limit) cannot be combined with query-level selection"
        )
    if query_sample_size is not None and (query_offset != 0 or query_limit is not None):
        raise EvidenceGapError(
            "--query-sample-size cannot be combined with --query-offset or --query-limit"
        )

    group_order: list[str] = []
    rows_by_group: dict[str, list[StanceInput]] = {}
    for item in inputs:
        group_id = _selection_group_id(item)
        if group_id not in rows_by_group:
            group_order.append(group_id)
            rows_by_group[group_id] = []
        rows_by_group[group_id].append(item)

    selection_mode = "all"
    selected_group_ids = list(group_order)
    if row_selection:
        selection_mode = "rows"
        selected = list(inputs[offset : None if limit is None else offset + limit])
        selected_group_ids = list(dict.fromkeys(_selection_group_id(item) for item in selected))
    else:
        if query_sample_size is not None:
            selection_mode = "query_sample"
            if query_sample_size > len(group_order):
                raise EvidenceGapError(
                    f"query_sample_size={query_sample_size} exceeds available queries={len(group_order)}"
                )
            ranked = sorted(
                group_order,
                key=lambda group_id: (
                    sha256_text(f"{query_sample_seed}:{group_id}"),
                    group_id,
                ),
            )
            chosen = set(ranked[:query_sample_size])
            selected_group_ids = [group_id for group_id in group_order if group_id in chosen]
        elif query_offset != 0 or query_limit is not None:
            selection_mode = "query_range"
            selected_group_ids = group_order[
                query_offset : None
                if query_limit is None
                else query_offset + query_limit
            ]
        selected = [
            item
            for group_id in selected_group_ids
            for item in rows_by_group[group_id]
        ]

    if not selected:
        raise EvidenceGapError("The selected input range is empty")
    selected_group_set = set(selected_group_ids)
    rows_per_selected_group = Counter(
        _selection_group_id(item) for item in selected
    )
    complete_group_selection = all(
        rows_per_selected_group[group_id] == len(rows_by_group[group_id])
        for group_id in selected_group_set
    )
    metadata = {
        "mode": selection_mode,
        "source_rows": len(inputs),
        "source_queries": len(group_order),
        "selected_rows": len(selected),
        "selected_queries": len(selected_group_ids),
        "complete_query_groups": complete_group_selection,
        "offset": offset,
        "limit": limit,
        "query_offset": query_offset,
        "query_limit": query_limit,
        "query_sample_size": query_sample_size,
        "query_sample_seed": query_sample_seed if query_sample_size is not None else None,
        "selection_sha256": sha256_text(
            json.dumps([item.input_id for item in selected], ensure_ascii=False)
        ),
    }
    return selected, metadata


def _execution_plan(
    inputs: Sequence[StanceInput],
    *,
    selection: Mapping[str, Any],
    request_batch_size: int,
) -> dict[str, Any]:
    evidence_characters = sum(len(item.evidence_text) for item in inputs)
    context_characters = sum(
        len(item.context_before or "") + len(item.context_after or "")
        for item in inputs
    )
    claim_characters = sum(len(item.claim_text) for item in inputs)
    duplicate_sentence_groups = 0
    rows_by_group: dict[str, list[StanceInput]] = {}
    for item in inputs:
        rows_by_group.setdefault(_selection_group_id(item), []).append(item)
    rank_gap_groups = 0
    for group_id, group_rows in rows_by_group.items():
        ranks = sorted(
            item.evidence_rank for item in group_rows if item.evidence_rank is not None
        )
        if ranks:
            if ranks != list(range(1, len(ranks) + 1)):
                rank_gap_groups += 1
        indices = [
            item.sentence_index
            for item in group_rows
            if item.sentence_index is not None
        ]
        if len(indices) != len(set(indices)):
            duplicate_sentence_groups += 1
    rows_per_query = Counter(len(rows) for rows in rows_by_group.values())
    return {
        **dict(selection),
        "request_batch_size": request_batch_size,
        "estimated_api_requests": math.ceil(len(inputs) / request_batch_size),
        "datasets": dict(sorted(Counter(item.dataset for item in inputs).items())),
        "splits": dict(sorted(Counter(item.split for item in inputs).items())),
        "evidence_units": dict(
            sorted(Counter(item.evidence_unit for item in inputs).items())
        ),
        "rows_per_query": {str(key): value for key, value in sorted(rows_per_query.items())},
        "rank_gap_queries": rank_gap_groups,
        "duplicate_sentence_queries": duplicate_sentence_groups,
        "claim_characters": claim_characters,
        "evidence_characters": evidence_characters,
        "context_characters": context_characters,
        "total_prompt_text_characters": (
            claim_characters + evidence_characters + context_characters
        ),
        "max_evidence_characters": max(len(item.evidence_text) for item in inputs),
        "context_rows": sum(
            bool(item.context_before or item.context_after) for item in inputs
        ),
    }

def _collect_exact_cached_batches(
    inputs: Sequence[StanceInput],
    *,
    provider: str,
    model: str,
    base_url: str,
    request_batch_size: int,
    max_tokens: int,
    thinking: bool,
    provider_cache: Path,
) -> tuple[
    dict[str, tuple[dict[str, Any], ProviderResponse]],
    dict[str, Any],
]:
    """Collect only cache files matching the exact current request fingerprint."""

    cached_by_input_id: dict[
        str, tuple[dict[str, Any], ProviderResponse]
    ] = {}
    completed_batches = 0
    missing_batches = 0
    cached_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for batch in _batches(inputs, request_batch_size):
        request_hash = _request_fingerprint(
            provider=provider,
            model=model,
            base_url=base_url,
            inputs=batch,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        cache_path = provider_cache / f"{request_hash}.json"
        cached = _read_cache(cache_path, request_hash=request_hash)
        if cached is None:
            missing_batches += 1
            continue
        results, response = cached
        results = _validate_result_payload({"results": results}, batch)
        completed_batches += 1
        for key in cached_usage:
            cached_usage[key] += int(response.usage.get(key, 0))
        for stance_input, result in zip(batch, results):
            if stance_input.input_id in cached_by_input_id:
                raise EvidenceGapError(
                    f"Duplicate cached result for {stance_input.input_id}"
                )
            cached_by_input_id[stance_input.input_id] = (result, response)
    return cached_by_input_id, {
        "expected_batches": math.ceil(len(inputs) / request_batch_size),
        "completed_batches": completed_batches,
        "missing_batches": missing_batches,
        "cached_rows": len(cached_by_input_id),
        "cached_usage": cached_usage,
    }


def _complete_cached_query_ids(
    inputs: Sequence[StanceInput],
    cached_input_ids: set[str],
) -> tuple[set[str], dict[str, Any]]:
    rows_by_group: dict[str, list[StanceInput]] = {}
    group_order: list[str] = []
    for item in inputs:
        group_id = _selection_group_id(item)
        if group_id not in rows_by_group:
            rows_by_group[group_id] = []
            group_order.append(group_id)
        rows_by_group[group_id].append(item)

    complete: set[str] = set()
    incomplete_with_cache = 0
    uncached = 0
    cached_rows_excluded = 0
    for group_id in group_order:
        group_rows = rows_by_group[group_id]
        cached_count = sum(item.input_id in cached_input_ids for item in group_rows)
        if cached_count == len(group_rows):
            complete.add(group_id)
        elif cached_count:
            incomplete_with_cache += 1
            cached_rows_excluded += cached_count
        else:
            uncached += 1
    return complete, {
        "source_queries": len(group_order),
        "complete_cached_queries": len(complete),
        "incomplete_cached_queries": incomplete_with_cache,
        "uncached_queries": uncached,
        "cached_rows_excluded": cached_rows_excluded,
    }


def _render_cache_export_markdown(report: Mapping[str, Any]) -> str:
    coverage = report["coverage"]
    lines = [
        "# Phase 06 Partial LLM Cache Export",
        "",
        f"- Provider/model: `{report['provider']}:{report['model']}`",
        f"- Prompt: `{report['prompt']['version']}`",
        f"- Source queries: {report['source_queries']:,}",
        f"- Exported complete queries: {report['exported_queries']:,} "
        f"({coverage['query_percent']:.1f}%)",
        f"- Source rows: {report['source_rows']:,}",
        f"- Exported rows: {report['rows']:,} ({coverage['row_percent']:.1f}%)",
        f"- Exact cache batches: {report['cache']['completed_batches']:,} / "
        f"{report['cache']['expected_batches']:,}",
        f"- API requests made during export: 0",
        "",
        "## Stance distribution",
        "",
        "| Label | Count | Percent |",
        "|---|---:|---:|",
    ]
    for label in STANCE_LABELS:
        count = int(report["prediction_counts"].get(label, 0))
        percent = 100.0 * count / report["rows"] if report["rows"] else 0.0
        lines.append(f"| {label} | {count:,} | {percent:.1f}% |")
    lines.extend(
        [
            "",
            "## Evidence types",
            "",
            "| Evidence type | Count |",
            "|---|---:|",
        ]
    )
    for evidence_type, count in sorted(
        report["evidence_type_counts"].items(),
        key=lambda item: (-int(item[1]), item[0]),
    ):
        lines.append(f"| {evidence_type} | {int(count):,} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Only exact cache fingerprints for this input artifact, model, Prompt, batch size, and request settings were exported.",
            "- Only complete query groups were retained; partially cached queries were excluded.",
            "- This artifact is a partial Dev inference result, not a stance accuracy evaluation because Phase 05 has no stance gold labels.",
            "",
        ]
    )
    return "\n".join(lines)


def export_llm_stance_cache(
    root: Path,
    *,
    input_path: Path,
    provider: str,
    model: str | None = None,
    run_name: str | None = None,
    base_url: str | None = None,
    request_batch_size: int = 8,
    max_tokens: int = 4096,
    thinking: bool = False,
    cache_dir: Path | None = None,
    artifact_root: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Export all completed exact-match cache batches without making API calls."""

    root = root.resolve()
    input_path = input_path.resolve()
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(
            f"provider must be one of {SUPPORTED_PROVIDERS}, got {provider!r}"
        )
    model = (model or DEFAULT_MODELS[provider]).strip()
    base_url = (base_url or DEFAULT_BASE_URLS[provider]).strip()
    if request_batch_size <= 0:
        raise EvidenceGapError("request_batch_size must be positive")
    if max_tokens <= 0:
        raise EvidenceGapError("max_tokens must be positive")
    if provider != "deepseek" and thinking:
        raise EvidenceGapError("--thinking is only supported for the DeepSeek provider")

    input_validation = validate_input_artifact(input_path)
    input_sha = str(input_validation["sha256"])
    all_inputs = list(iter_inputs(input_path))
    source_query_ids = {
        _selection_group_id(item) for item in all_inputs
    }
    cache_root = (
        cache_dir.resolve()
        if cache_dir
        else root / DEFAULT_ARTIFACT_ROOT / "llm_cache"
    )
    provider_cache = cache_root / provider / _safe_name(model)
    cached_by_input_id, cache_stats = _collect_exact_cached_batches(
        all_inputs,
        provider=provider,
        model=model,
        base_url=base_url,
        request_batch_size=request_batch_size,
        max_tokens=max_tokens,
        thinking=thinking,
        provider_cache=provider_cache,
    )
    if not cached_by_input_id:
        raise EvidenceGapError(
            "No exact-match cache batches were found. Keep input path, provider, "
            "model, request batch size, max tokens, Prompt version, base URL, and "
            "thinking settings identical to the interrupted run."
        )
    complete_query_ids, query_stats = _complete_cached_query_ids(
        all_inputs,
        set(cached_by_input_id),
    )
    export_inputs = [
        item
        for item in all_inputs
        if _selection_group_id(item) in complete_query_ids
    ]
    if not export_inputs:
        raise EvidenceGapError("Cache contains no complete query groups to export")

    name = _safe_name(
        run_name or f"{provider}_{model}_{input_path.parent.name}_partial_cache"
    )
    model_name = f"{provider}:{model}"
    schema_hash, prompt_hash, model_fingerprint = _model_identity(
        provider=provider,
        model=model,
        base_url=base_url,
    )
    base = (
        artifact_root.resolve()
        if artifact_root
        else root / DEFAULT_ARTIFACT_ROOT / "llm_judge"
    )
    target = base / name
    require_empty_or_force(target, force=force)

    started = time.perf_counter()
    predictions: list[StancePrediction] = []
    prediction_counts: Counter[str] = Counter()
    evidence_type_counts: Counter[str] = Counter()
    context_required = 0
    confidence_sum = 0.0
    margin_sum = 0.0
    probability_reconciled_rows = 0
    probability_reconciliation_counts: Counter[str] = Counter()
    label_recovered_rows = 0
    label_recovery_counts: Counter[str] = Counter()
    for stance_input in export_inputs:
        result, response = cached_by_input_id[stance_input.input_id]
        if bool(result.get("label_recovered")):
            label_recovered_rows += 1
            original_label = str(result.get("label_original_value") or "blank")
            label_recovery_counts[f"{original_label}->{result['label']}"] += 1
        if bool(result.get("probability_reconciled")):
            probability_reconciled_rows += 1
            original_argmax = str(
                result.get("probability_original_argmax") or "unknown"
            )
            probability_reconciliation_counts[
                f"{original_argmax}->{result['label']}"
            ] += 1
        prediction = _prediction_from_result(
            stance_input,
            result,
            run_name=name,
            model_name=model_name,
            model_fingerprint=model_fingerprint,
            input_sha256=input_sha,
            provider=provider,
            response=response,
        )
        predictions.append(prediction)
        prediction_counts[prediction.predicted_label] += 1
        evidence_type_counts[str(prediction.evidence_type)] += 1
        context_required += bool(prediction.requires_context)
        confidence_sum += prediction.confidence
        margin_sum += prediction.probability_margin

    with atomic_directory(target, force=force) as staging:
        output_path = staging / "stance_predictions.parquet"
        row_count = write_predictions_atomic(output_path, predictions)
        validation = validate_prediction_artifact(
            output_path,
            expected_input_sha256=input_sha,
            expected_run_name=name,
        )
        elapsed = time.perf_counter() - started
        exported_queries = len(complete_query_ids)
        report = {
            "schema_version": RUN_SCHEMA_VERSION,
            "stance_schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_name": name,
            "run_type": "llm_structured_stance_partial_cache_export",
            "partial": exported_queries < len(source_query_ids),
            "provider": provider,
            "model": model,
            "model_name": model_name,
            "model_fingerprint": model_fingerprint,
            "prompt": {
                "version": PROMPT_VERSION,
                "language": "English",
                "sha256": prompt_hash,
                "response_format": "JSON",
                "response_schema_sha256": schema_hash,
            },
            "parameters": {
                "request_batch_size": request_batch_size,
                "max_tokens": max_tokens,
                "thinking": thinking if provider == "deepseek" else None,
                "anthropic_thinking": (
                    _anthropic_thinking_config(model)
                    if provider == "anthropic"
                    else None
                ),
            },
            "source_input_path": relative_path(root, input_path),
            "source_input_sha256": input_sha,
            "source_rows": len(all_inputs),
            "source_queries": len(source_query_ids),
            "exported_queries": exported_queries,
            "rows": row_count,
            "coverage": {
                "query_fraction": exported_queries / len(source_query_ids),
                "query_percent": 100.0 * exported_queries / len(source_query_ids),
                "row_fraction": row_count / len(all_inputs),
                "row_percent": 100.0 * row_count / len(all_inputs),
            },
            "cache": {
                **cache_stats,
                **query_stats,
                "path": relative_path(root, provider_cache),
                "api_requests_during_export": 0,
            },
            "prediction_counts": dict(sorted(prediction_counts.items())),
            "evidence_type_counts": dict(sorted(evidence_type_counts.items())),
            "requires_context_rows": context_required,
            "mean_confidence": confidence_sum / row_count,
            "mean_probability_margin": margin_sum / row_count,
            "probability_reconciled_rows": probability_reconciled_rows,
            "probability_reconciliation_counts": dict(
                sorted(probability_reconciliation_counts.items())
            ),
            "label_recovered_rows": label_recovered_rows,
            "label_recovery_counts": dict(sorted(label_recovery_counts.items())),
            "elapsed_seconds": elapsed,
            "rows_per_second": row_count / elapsed if elapsed > 0 else None,
            "output_path": relative_path(
                root, target / "stance_predictions.parquet"
            ),
            "output_sha256": validation["sha256"],
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "run_manifest.json", report)

    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_root.mkdir(parents=True, exist_ok=True)
    json_report_path = report_root / f"stance_llm_{name}.json"
    markdown_path = report_root / f"stance_llm_{name}.md"
    atomic_write_json(json_report_path, report)
    markdown_path.write_text(
        _render_cache_export_markdown(report),
        encoding="utf-8",
    )
    return {
        "run_name": name,
        "provider": provider,
        "model": model,
        "prediction_path": report["output_path"],
        "manifest_path": relative_path(root, target / "run_manifest.json"),
        "report_path": relative_path(root, json_report_path),
        "markdown_report_path": relative_path(root, markdown_path),
        "source_queries": report["source_queries"],
        "exported_queries": report["exported_queries"],
        "source_rows": report["source_rows"],
        "exported_rows": report["rows"],
        "coverage": report["coverage"],
        "cache": report["cache"],
        "prediction_counts": report["prediction_counts"],
        "evidence_type_counts": report["evidence_type_counts"],
        "validation": validation,
    }


def run_llm_stance_judge(
    root: Path,
    *,
    input_path: Path,
    provider: str,
    model: str | None = None,
    run_name: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    request_batch_size: int = 8,
    max_tokens: int = 4096,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    offset: int = 0,
    limit: int | None = None,
    query_offset: int = 0,
    query_limit: int | None = None,
    query_sample_size: int | None = None,
    query_sample_seed: int = 20260722,
    dry_run: bool = False,
    thinking: bool = False,
    cache_dir: Path | None = None,
    artifact_root: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    input_path = input_path.resolve()
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(
            f"provider must be one of {SUPPORTED_PROVIDERS}, got {provider!r}"
        )
    model = (model or DEFAULT_MODELS[provider]).strip()
    api_key_env = (api_key_env or DEFAULT_API_KEY_ENVS[provider]).strip()
    base_url = (base_url or DEFAULT_BASE_URLS[provider]).strip()
    if request_batch_size <= 0:
        raise EvidenceGapError("request_batch_size must be positive")
    if max_tokens <= 0:
        raise EvidenceGapError("max_tokens must be positive")
    if timeout_seconds <= 0:
        raise EvidenceGapError("timeout_seconds must be positive")
    if max_retries < 0:
        raise EvidenceGapError("max_retries cannot be negative")
    if provider != "deepseek" and thinking:
        raise EvidenceGapError("--thinking is only supported for the DeepSeek provider")

    input_validation = validate_input_artifact(input_path)
    input_sha = str(input_validation["sha256"])
    all_inputs = list(iter_inputs(input_path))
    selected_inputs, selection = _select_inputs(
        all_inputs,
        offset=offset,
        limit=limit,
        query_offset=query_offset,
        query_limit=query_limit,
        query_sample_size=query_sample_size,
        query_sample_seed=query_sample_seed,
    )
    plan = _execution_plan(
        selected_inputs,
        selection=selection,
        request_batch_size=request_batch_size,
    )
    if dry_run:
        return {
            "status": "DRY_RUN",
            "provider": provider,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "source_input_path": relative_path(root, input_path),
            "source_input_sha256": input_sha,
            "plan": plan,
        }
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise EvidenceGapError(
            f"Missing API key environment variable {api_key_env}; do not pass API keys on the command line"
        )
    name = _safe_name(
        run_name or f"{provider}_{model}_{input_path.parent.name}"
    )
    model_name = f"{provider}:{model}"
    schema_hash = sha256_text(
        json.dumps(response_json_schema(), ensure_ascii=False, sort_keys=True)
    )
    prompt_hash = sha256_text(SYSTEM_PROMPT)
    model_fingerprint = sha256_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url.rstrip("/"),
                "prompt_version": PROMPT_VERSION,
                "prompt_sha256": prompt_hash,
                "response_schema_sha256": schema_hash,
                "anthropic_thinking": (
                    _anthropic_thinking_config(model)
                    if provider == "anthropic"
                    else None
                ),
            },
            sort_keys=True,
        )
    )
    base = (
        artifact_root.resolve()
        if artifact_root
        else root / DEFAULT_ARTIFACT_ROOT / "llm_judge"
    )
    target = base / name
    require_empty_or_force(target, force=force)
    cache_root = (
        cache_dir.resolve()
        if cache_dir
        else root / DEFAULT_ARTIFACT_ROOT / "llm_cache"
    )
    provider_cache = cache_root / provider / _safe_name(model)

    started = time.perf_counter()
    predictions: list[StancePrediction] = []
    prediction_counts: Counter[str] = Counter()
    evidence_type_counts: Counter[str] = Counter()
    context_required = 0
    confidence_sum = 0.0
    margin_sum = 0.0
    api_requests = 0
    cache_hits = 0
    retry_count = 0
    usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    cached_usage_totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    probability_reconciled_rows = 0
    probability_reconciliation_counts: Counter[str] = Counter()
    label_recovered_rows = 0
    label_recovery_counts: Counter[str] = Counter()

    for batch in _batches(selected_inputs, request_batch_size):
        request_hash = _request_fingerprint(
            provider=provider,
            model=model,
            base_url=base_url,
            inputs=batch,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        cache_path = provider_cache / f"{request_hash}.json"
        cached = _read_cache(cache_path, request_hash=request_hash)
        if cached is None:
            results, response, retries = _call_with_retries(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                inputs=batch,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                thinking=thinking,
            )
            _write_cache(
                cache_path,
                request_hash=request_hash,
                provider=provider,
                model=model,
                input_ids=[item.input_id for item in batch],
                response=response,
                validated_results=results,
            )
            api_requests += 1
            retry_count += retries
            for key in usage_totals:
                usage_totals[key] += int(response.usage.get(key, 0))
        else:
            results, response = cached
            # Revalidate cached parsed output against the current input ordering.
            results = _validate_result_payload({"results": results}, batch)
            cache_hits += 1
            for key in cached_usage_totals:
                cached_usage_totals[key] += int(response.usage.get(key, 0))
        for stance_input, result in zip(batch, results):
            if bool(result.get("label_recovered")):
                label_recovered_rows += 1
                original_label = str(result.get("label_original_value") or "blank")
                label_recovery_counts[f"{original_label}->{result['label']}"] += 1
            if bool(result.get("probability_reconciled")):
                probability_reconciled_rows += 1
                original_argmax = str(
                    result.get("probability_original_argmax") or "unknown"
                )
                probability_reconciliation_counts[
                    f"{original_argmax}->{result['label']}"
                ] += 1
            prediction = _prediction_from_result(
                stance_input,
                result,
                run_name=name,
                model_name=model_name,
                model_fingerprint=model_fingerprint,
                input_sha256=input_sha,
                provider=provider,
                response=response,
            )
            predictions.append(prediction)
            prediction_counts[prediction.predicted_label] += 1
            evidence_type_counts[str(prediction.evidence_type)] += 1
            context_required += bool(prediction.requires_context)
            confidence_sum += prediction.confidence
            margin_sum += prediction.probability_margin

    with atomic_directory(target, force=force) as staging:
        output_path = staging / "stance_predictions.parquet"
        row_count = write_predictions_atomic(output_path, predictions)
        validation = validate_prediction_artifact(
            output_path,
            expected_input_sha256=input_sha,
            expected_run_name=name,
        )
        gold_rows = [
            prediction.to_dict()
            for prediction in predictions
            if prediction.stance_input.gold_label is not None
        ]
        metrics = evaluate_prediction_rows(gold_rows) if gold_rows else None
        elapsed = time.perf_counter() - started
        report = {
            "schema_version": RUN_SCHEMA_VERSION,
            "stance_schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_name": name,
            "run_type": "llm_structured_stance_judge",
            "provider": provider,
            "model": model,
            "model_name": model_name,
            "model_fingerprint": model_fingerprint,
            "api": {
                "base_url": base_url.rstrip("/"),
                "api_key_env": api_key_env,
                "anthropic_version": (
                    ANTHROPIC_VERSION if provider == "anthropic" else None
                ),
            },
            "prompt": {
                "version": PROMPT_VERSION,
                "language": "English",
                "sha256": prompt_hash,
                "response_format": "JSON",
                "response_schema_sha256": schema_hash,
            },
            "parameters": {
                "request_batch_size": request_batch_size,
                "max_tokens": max_tokens,
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
                "offset": offset,
                "limit": limit,
                "query_offset": query_offset,
                "query_limit": query_limit,
                "query_sample_size": query_sample_size,
                "query_sample_seed": (
                    query_sample_seed if query_sample_size is not None else None
                ),
                "thinking": thinking if provider == "deepseek" else None,
                "anthropic_thinking": (
                    _anthropic_thinking_config(model)
                    if provider == "anthropic"
                    else None
                ),
            },
            "source_input_path": relative_path(root, input_path),
            "source_input_sha256": input_sha,
            "source_input_rows": len(all_inputs),
            "selected_rows": len(selected_inputs),
            "selected_queries": selection["selected_queries"],
            "selection": selection,
            "execution_plan": plan,
            "rows": row_count,
            "prediction_counts": dict(sorted(prediction_counts.items())),
            "evidence_type_counts": dict(sorted(evidence_type_counts.items())),
            "requires_context_rows": context_required,
            "mean_confidence": confidence_sum / row_count,
            "mean_probability_margin": margin_sum / row_count,
            "probability_reconciled_rows": probability_reconciled_rows,
            "probability_reconciliation_counts": dict(
                sorted(probability_reconciliation_counts.items())
            ),
            "label_recovered_rows": label_recovered_rows,
            "label_recovery_counts": dict(sorted(label_recovery_counts.items())),
            "api_requests": api_requests,
            "cache_hits": cache_hits,
            "retry_count": retry_count,
            "billed_usage": usage_totals,
            "cached_usage": cached_usage_totals,
            "elapsed_seconds": elapsed,
            "rows_per_second": row_count / elapsed if elapsed > 0 else None,
            "metrics": metrics,
            "output_path": relative_path(root, target / "stance_predictions.parquet"),
            "output_sha256": validation["sha256"],
            "cache_dir": relative_path(root, provider_cache),
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "run_manifest.json", report)

    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_root.mkdir(parents=True, exist_ok=True)
    json_report_path = report_root / f"stance_llm_{name}.json"
    atomic_write_json(json_report_path, report)
    markdown_path: Path | None = None
    if report["metrics"] is not None:
        markdown_path = report_root / f"stance_llm_{name}.md"
        markdown_path.write_text(
            render_evaluation_markdown(
                {"metrics": report["metrics"]},
                title=f"Phase 06 LLM Stance Evaluation — {provider}:{model}",
            ),
            encoding="utf-8",
        )
    return {
        "run_name": name,
        "provider": provider,
        "model": model,
        "prediction_path": report["output_path"],
        "manifest_path": relative_path(root, target / "run_manifest.json"),
        "report_path": relative_path(root, json_report_path),
        "markdown_report_path": (
            None if markdown_path is None else relative_path(root, markdown_path)
        ),
        "api_requests": api_requests,
        "cache_hits": cache_hits,
        "billed_usage": usage_totals,
        "cached_usage": cached_usage_totals,
        "probability_reconciled_rows": probability_reconciled_rows,
        "probability_reconciliation_counts": dict(
            sorted(probability_reconciliation_counts.items())
        ),
        "label_recovered_rows": label_recovered_rows,
        "label_recovery_counts": dict(sorted(label_recovery_counts.items())),
        "execution_plan": plan,
        "validation": validation,
        "metrics": report["metrics"],
    }
