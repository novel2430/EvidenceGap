from __future__ import annotations

import json
import math
import os
import random
import re
import time
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
    sha256_file,
    sha256_text,
)
from evidencegap.pipeline.retrieval_adapters import _read_parquet
from evidencegap.stance.contracts import STANCE_LABELS, canonical_stance_label
from evidencegap.stance.llm_judge import (
    DEFAULT_API_KEY_ENVS,
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    SUPPORTED_PROVIDERS,
    ProviderResponse,
    _ProviderError,
    call_structured_llm,
)

ARTICLE_EVIDENCE_SCHEMA_VERSION = "1.0.0"
ARTICLE_EVIDENCE_CONTRACT_ID = "phase07.article-evidence.v1"
ARTICLE_EVIDENCE_PROMPT_VERSION = "phase07_article_evidence_v3"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/article_evidence")
DEFAULT_CACHE_ROOT = Path("artifacts/v1/stance_verification/article_evidence_cache")
MAX_EVIDENCE_SENTENCES = 5

SYSTEM_PROMPT = f"""You are an evidence-grounded medical article verifier.

Judge each supplied article only against the exact CLAIM using only the supplied article text. Do not use outside knowledge, unstated assumptions, or biomedical plausibility to bridge missing evidence.

Labels:
- support: the article reports direct evidence that increases confidence that the exact claim is true.
- refute: the article reports direct evidence that increases confidence that the exact claim is false or materially contradicts it.
- insufficient: the article is related but does not provide enough directly applicable evidence for either direction, or its findings are mixed or ambiguous for the exact claim.

Check claim applicability before assigning a stance.

Applicability rules:
- Before assigning support or refute, verify that the article matches the claim in population or species, intervention or exposure, comparator, outcome, direction, timeframe, estimand, and scope.
- Semantic relatedness is not claim applicability.
- Do not substitute a related disease, condition, comorbidity, proxy exposure, mechanism, biomarker, or surrogate outcome for the exact concept stated in the claim.
- If any material component of the claim is not directly addressed, label the article insufficient.
- If a claim contains multiple independently testable components or outcomes, support or refute requires the article to directly address every asserted component. Evidence for only part of the claim is insufficient for the full claim.

Population and species:
- Unless the claim explicitly concerns animals, cells, or another preclinical system, animal, cellular, in-vitro, and other preclinical studies are insufficient for supporting or refuting a human clinical or epidemiological claim.
- Evidence from a materially different population is insufficient when that difference changes whether the study can answer the claim.
- A study restricted to participants who already have the claimed outcome does not answer whether the exposure increases the risk of developing that outcome.

Exposure and intervention:
- The intervention or exposure must match the claim directly.
- Do not treat a related condition, consequence, or proxy as equivalent to the stated exposure.
- Exposure direction and range must match. Increased and decreased exposure, long and short duration, high and low dose, presence and absence, and acute and chronic exposure are not interchangeable.
- A comparison between two active doses, treatments, or regimens does not establish effectiveness versus placebo, no treatment, no supplementation, or an untreated comparator.
- For a general claim that an intervention is effective, an active-versus-active comparison is insufficient unless the claim specifically compares those interventions or regimens.

Outcome and timeframe:
- The measured outcome must match the claimed outcome.
- Do not treat short-term changes in body weight, biomarkers, laboratory values, physiological measurements, energy expenditure, symptoms, or other surrogate outcomes as direct evidence about the incidence or risk of developing a clinical condition.
- Disease severity, symptom duration, disease control, hospitalization, respiratory support, mortality, biomarkers, or prognosis among existing patients do not by themselves answer a claim about preventing or developing the disease.
- For a claim about incidence or future disease risk, the article must evaluate new occurrence of the outcome after the exposure or use another design capable of addressing future risk.
- Cross-sectional associations are insufficient for claims about incidence, future risk, or temporal effects unless the article directly establishes the required temporal relationship.

Evidence strength and claim wording:
- Match the evidentiary strength to the wording of the claim.
- Do not convert association into a stronger causal conclusion.
- A well-matched longitudinal observational study may support or refute a claim about increased risk or epidemiological association.
- Randomized intervention evidence is not required for a risk or association claim unless the claim specifically asserts an intervention effect or strong causal effect.
- Mechanistic plausibility alone is insufficient for supporting or refuting a clinical or epidemiological claim.
- A review article may support or refute a claim only when the supplied text explicitly reports a directly applicable conclusion grounded in the reviewed evidence. Background statements or general discussion alone are insufficient.

Prevention and treatment:
- A treatment study in patients who already have the disease does not support or refute a prevention claim unless it directly reports disease incidence or another preventive outcome.
- A prevention study does not automatically answer a treatment claim.
- Treatment response, symptom improvement, reduced severity, or improved prognosis does not establish prevention of disease occurrence.

Evidence selection:
- Select evidence only by the supplied sentence IDs.
- Select the smallest sufficient set, at most {MAX_EVIDENCE_SENTENCES} sentences.
- Prefer reported results and conclusions over objectives, background, methods, speculation, or recommendations.
- Do not select headings, titles, isolated fragments, incomplete sentences, or sentences that require unsupported context.
- Every selected sentence must directly contribute to the assigned stance.
- support and refute must select at least one evidence sentence.
- insufficient must return an empty evidence_sentence_ids list.

Output requirements:
- Return exactly one result for each supplied article_id and no extra results.
- `probabilities` MUST be a JSON object with exactly the numeric keys `support`, `refute`, and `insufficient`.
- `probabilities` must never be null, a string, or an array.
- The three probabilities must sum to 1.
- The assigned label must be the unique highest-probability class.
- `evidence_sentence_ids` MUST be a JSON array containing only supplied sentence-ID strings.
- The rationale must be one concise English sentence explaining why the article directly supports, refutes, or is insufficient for the exact claim.
- Do not include reasoning text, Markdown, code fences, commentary, or extra keys outside the required JSON structure.

After any internal reasoning, return only the final JSON object."""


def response_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "article_id": {"type": "string"},
                        "label": {"type": "string", "enum": list(STANCE_LABELS)},
                        "probabilities": {
                            "type": "object",
                            "properties": {
                                "support": {"type": "number"},
                                "refute": {"type": "number"},
                                "insufficient": {"type": "number"},
                            },
                            "required": list(STANCE_LABELS),
                            "additionalProperties": False,
                        },
                        "evidence_sentence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "article_id",
                        "label",
                        "probabilities",
                        "evidence_sentence_ids",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class PromptSentence:
    alias: str
    source: Mapping[str, Any]

    @property
    def sentence_id(self) -> str:
        return str(self.source["sentence_id"])


@dataclass(frozen=True)
class ArticlePromptInput:
    article: Mapping[str, Any]
    sentences: tuple[PromptSentence, ...]

    @property
    def article_id(self) -> str:
        return str(self.article["article_id"])

    @property
    def alias_map(self) -> dict[str, PromptSentence]:
        return {sentence.alias: sentence for sentence in self.sentences}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid name: {value!r}")
    return cleaned


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    os.replace(temp, path)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing article evidence artifact: {path}") from exc
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceGapError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, Mapping):
            raise EvidenceGapError(f"Expected JSON object at {path}:{line_number}")
        rows.append(dict(row))
    return rows


def load_article_prompt_inputs(
    retrieval_artifact_dir: Path,
) -> tuple[dict[str, str], list[ArticlePromptInput], dict[str, Path]]:
    retrieval_artifact_dir = retrieval_artifact_dir.resolve()
    paths = {
        "request": retrieval_artifact_dir / "request.json",
        "top_articles": retrieval_artifact_dir
        / "article_retrieval/top_articles.parquet",
        "runtime_sentences": retrieval_artifact_dir
        / "sentence_materialization/runtime_sentences.parquet",
    }
    try:
        request = json.loads(paths["request"].read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing retrieval request: {paths['request']}") from exc
    claim_id = str(request.get("claim_id") or "").strip()
    claim_text = str(request.get("claim_text") or "").strip()
    if not claim_id or not claim_text:
        raise EvidenceGapError("Retrieval request is missing claim_id or claim_text")

    articles = sorted(
        _read_parquet(paths["top_articles"]),
        key=lambda row: int(row["final_article_rank"]),
    )
    if not articles:
        raise EvidenceGapError("Top article artifact cannot be empty")
    article_ids = [str(row["article_id"]) for row in articles]
    if len(article_ids) != len(set(article_ids)):
        raise EvidenceGapError("Top article artifact contains duplicate article IDs")

    by_article: dict[str, list[Mapping[str, Any]]] = {value: [] for value in article_ids}
    for row in _read_parquet(paths["runtime_sentences"]):
        article_id = str(row.get("article_id") or "")
        if article_id in by_article and str(row.get("sentence_type")) == "abstract":
            by_article[article_id].append(row)

    inputs: list[ArticlePromptInput] = []
    for article in articles:
        article_id = str(article["article_id"])
        rows = sorted(
            by_article[article_id], key=lambda row: int(row["sentence_index"])
        )
        if not rows:
            raise EvidenceGapError(f"Article {article_id} has no abstract sentences")
        inputs.append(
            ArticlePromptInput(
                article=article,
                sentences=tuple(
                    PromptSentence(alias=f"S{index:02d}", source=row)
                    for index, row in enumerate(rows, start=1)
                ),
            )
        )
    return {"claim_id": claim_id, "claim_text": claim_text}, inputs, paths


def _required_output_template(items: Sequence[ArticlePromptInput]) -> dict[str, Any]:
    return {
        "results": [
            {
                "article_id": item.article_id,
                "label": "insufficient",
                "probabilities": {
                    "support": 0.0,
                    "refute": 0.0,
                    "insufficient": 1.0,
                },
                "evidence_sentence_ids": [],
                "rationale": "One concise English sentence grounded in this article.",
            }
            for item in items
        ]
    }


def build_user_prompt(
    *, claim_text: str, items: Sequence[ArticlePromptInput], retry_note: str | None = None
) -> str:
    articles = []
    for item in items:
        articles.append(
            {
                "article_id": item.article_id,
                "title": str(item.article.get("title") or ""),
                "sentences": [
                    {
                        "sentence_id": sentence.alias,
                        "section": str(sentence.source.get("section") or "abstract"),
                        "text": str(sentence.source["sentence_text"]),
                    }
                    for sentence in item.sentences
                ],
            }
        )
    parts = [
        "Judge every article below against the same claim and return results in the same order.",
        "CLAIM:",
        claim_text,
        "ARTICLES JSON:",
        json.dumps(articles, ensure_ascii=False, indent=2),
        "REQUIRED OUTPUT JSON SHAPE (replace the example values, preserve the exact field types and article IDs):",
        json.dumps(_required_output_template(items), ensure_ascii=False, indent=2),
        "Return only this JSON object. In particular, probabilities must be an object, not an array, string, or null.",
    ]
    if retry_note:
        parts.extend(
            [
                "The previous response was invalid. Correct this issue and return the complete JSON object:",
                retry_note,
            ]
        )
    return "\n\n".join(parts)


def validate_response_payload(
    payload: Mapping[str, Any], items: Sequence[ArticlePromptInput]
) -> list[dict[str, Any]]:
    raw_results = payload.get("results")
    if not isinstance(raw_results, list) or len(raw_results) != len(items):
        actual = len(raw_results) if isinstance(raw_results, list) else "non-array"
        raise _ProviderError(
            f"Expected {len(items)} results, got {actual}", retryable=True
        )

    validated: list[dict[str, Any]] = []
    for index, (raw, item) in enumerate(zip(raw_results, items)):
        if not isinstance(raw, Mapping):
            raise _ProviderError(f"results[{index}] must be an object", retryable=True)
        article_id = str(raw.get("article_id") or "").strip()
        if article_id != item.article_id:
            raise _ProviderError(
                f"results[{index}].article_id must be {item.article_id!r}",
                retryable=True,
            )
        try:
            label = canonical_stance_label(str(raw.get("label") or ""))
        except EvidenceGapError as exc:
            raise _ProviderError(str(exc), retryable=True) from exc

        probabilities_raw = raw.get("probabilities")
        if not isinstance(probabilities_raw, Mapping):
            raise _ProviderError("probabilities must be an object", retryable=True)
        probabilities: dict[str, float] = {}
        for stance in STANCE_LABELS:
            try:
                value = float(probabilities_raw[stance])
            except (KeyError, TypeError, ValueError) as exc:
                raise _ProviderError(
                    f"probabilities.{stance} must be numeric", retryable=True
                ) from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise _ProviderError(f"Invalid probability for {stance}", retryable=True)
            probabilities[stance] = value
        if abs(sum(probabilities.values()) - 1.0) > 1e-4:
            raise _ProviderError("Probabilities must sum to 1", retryable=True)
        if label != max(STANCE_LABELS, key=lambda value: probabilities[value]):
            raise _ProviderError("Label must equal probability argmax", retryable=True)

        raw_aliases = raw.get("evidence_sentence_ids")
        if not isinstance(raw_aliases, list) or any(
            not isinstance(value, str) for value in raw_aliases
        ):
            raise _ProviderError(
                "evidence_sentence_ids must be a string array", retryable=True
            )
        aliases = [value.strip() for value in raw_aliases]
        if any(not value for value in aliases) or len(aliases) != len(set(aliases)):
            raise _ProviderError("Evidence IDs must be non-empty and unique", retryable=True)
        if len(aliases) > MAX_EVIDENCE_SENTENCES:
            raise _ProviderError("Too many evidence sentences selected", retryable=True)
        unknown = [value for value in aliases if value not in item.alias_map]
        if unknown:
            raise _ProviderError(f"Unknown evidence sentence IDs: {unknown}", retryable=True)
        if label == "insufficient" and aliases:
            raise _ProviderError(
                "insufficient must use an empty evidence list", retryable=True
            )
        if label != "insufficient" and not aliases:
            raise _ProviderError(
                f"{label} must select at least one evidence sentence", retryable=True
            )
        rationale = str(raw.get("rationale") or "").strip()
        if not rationale:
            raise _ProviderError("Rationale cannot be blank", retryable=True)
        validated.append(
            {
                "article_id": article_id,
                "label": label,
                "probabilities": probabilities,
                "aliases": aliases,
                "rationale": rationale,
            }
        )
    return validated


def _fingerprint(value: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _request_hash(
    *,
    provider: str,
    model: str,
    base_url: str,
    claim_text: str,
    items: Sequence[ArticlePromptInput],
    max_tokens: int,
    thinking: bool,
) -> str:
    return _fingerprint(
        {
            "provider": provider,
            "model": model,
            "base_url": base_url.rstrip("/"),
            "prompt_version": ARTICLE_EVIDENCE_PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "response_schema": response_json_schema(),
            "user_prompt": build_user_prompt(claim_text=claim_text, items=items),
            "max_tokens": max_tokens,
            "thinking": thinking if provider == "deepseek" else None,
        }
    )


def _read_cache(
    path: Path, *, request_hash: str, items: Sequence[ArticlePromptInput]
) -> tuple[list[dict[str, Any]], ProviderResponse] | None:
    if not path.exists():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, Mapping) or cached.get("request_hash") != request_hash:
        return None
    payload = cached.get("payload")
    response = cached.get("response")
    if not isinstance(payload, Mapping) or not isinstance(response, Mapping):
        return None
    results = validate_response_payload(payload, items)
    raw_hash = str(response.get("raw_response_sha256") or "")
    if not raw_hash:
        return None
    return results, ProviderResponse(
        payload=payload,
        request_id=None if response.get("request_id") is None else str(response["request_id"]),
        usage={
            key: int((response.get("usage") or {}).get(key) or 0)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        },
        raw_response_sha256=raw_hash,
        finish_reason=None
        if response.get("finish_reason") is None
        else str(response["finish_reason"]),
    )


def _write_cache(
    path: Path,
    *,
    request_hash: str,
    response: ProviderResponse,
    results: Sequence[Mapping[str, Any]],
) -> None:
    payload = {
        "results": [
            {
                "article_id": result["article_id"],
                "label": result["label"],
                "probabilities": dict(result["probabilities"]),
                "evidence_sentence_ids": list(result["aliases"]),
                "rationale": result["rationale"],
            }
            for result in results
        ]
    }
    atomic_write_json(
        path,
        {
            "request_hash": request_hash,
            "payload": payload,
            "response": {
                "request_id": response.request_id,
                "usage": dict(response.usage),
                "raw_response_sha256": response.raw_response_sha256,
                "finish_reason": response.finish_reason,
            },
        },
    )


def _write_failed_provider_response(
    directory: Path,
    *,
    request_hash: str,
    attempt: int,
    provider: str,
    model: str,
    thinking: bool,
    items: Sequence[ArticlePromptInput],
    error: _ProviderError,
    response: ProviderResponse,
) -> Path:
    path = directory / f"{request_hash}_attempt_{attempt:02d}.json"
    atomic_write_json(
        path,
        {
            "schema_version": ARTICLE_EVIDENCE_SCHEMA_VERSION,
            "request_hash": request_hash,
            "attempt": attempt,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "thinking": thinking if provider == "deepseek" else None,
            "article_ids": [item.article_id for item in items],
            "validation_error": str(error),
            "response": {
                "payload": dict(response.payload),
                "request_id": response.request_id,
                "usage": dict(response.usage),
                "raw_response_sha256": response.raw_response_sha256,
                "finish_reason": response.finish_reason,
            },
        },
    )
    return path


def _call_with_retries(
    *,
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    claim_text: str,
    items: Sequence[ArticlePromptInput],
    max_tokens: int,
    timeout_seconds: float,
    max_retries: int,
    thinking: bool,
    request_hash: str,
    failed_response_dir: Path,
) -> tuple[list[dict[str, Any]], ProviderResponse, int]:
    retry_note: str | None = None
    last_failed_payload: Path | None = None
    for attempt in range(max_retries + 1):
        response: ProviderResponse | None = None
        try:
            response = call_structured_llm(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=build_user_prompt(
                    claim_text=claim_text, items=items, retry_note=retry_note
                ),
                response_schema=response_json_schema(),
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                thinking=thinking,
            )
            try:
                results = validate_response_payload(response.payload, items)
            except _ProviderError as exc:
                last_failed_payload = _write_failed_provider_response(
                    failed_response_dir,
                    request_hash=request_hash,
                    attempt=attempt + 1,
                    provider=provider,
                    model=model,
                    thinking=thinking,
                    items=items,
                    error=exc,
                    response=response,
                )
                raise
            return results, response, attempt
        except _ProviderError as exc:
            if attempt >= max_retries or not exc.retryable:
                if last_failed_payload is not None:
                    raise _ProviderError(
                        f"{exc}; invalid provider payload saved to {last_failed_payload}",
                        retryable=False,
                    ) from exc
                raise
            retry_note = str(exc)
            time.sleep(min(2**attempt, 16) + random.random())
    raise AssertionError("retry loop exhausted")


def _batches(
    items: Sequence[ArticlePromptInput], batch_size: int
) -> Iterable[Sequence[ArticlePromptInput]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _output_row(
    *,
    request: Mapping[str, str],
    item: ArticlePromptInput,
    result: Mapping[str, Any],
    provider: str,
    model: str,
    model_fingerprint: str,
    response: ProviderResponse,
) -> dict[str, Any]:
    probabilities = dict(result["probabilities"])
    label = str(result["label"])
    aliases = list(result["aliases"])
    selected = []
    for alias in aliases:
        sentence = item.alias_map[alias]
        source = sentence.source
        selected.append(
            {
                "evidence_id": "evidence_"
                + sha256_text(
                    f"{ARTICLE_EVIDENCE_CONTRACT_ID}\0{request['claim_id']}\0{item.article_id}\0{sentence.sentence_id}"
                )[:24],
                "sentence_alias": alias,
                "sentence_id": sentence.sentence_id,
                "sentence_index": int(source["sentence_index"]),
                "sentence_index_within_section": int(
                    source["sentence_index_within_section"]
                ),
                "section": str(source["section"]),
                "section_index": int(source["section_index"]),
                "sentence_text": str(source["sentence_text"]),
                "character_start": int(source["character_start"]),
                "character_end": int(source["character_end"]),
                "source_text_fingerprint": str(source["source_text_fingerprint"]),
                "splitter_fingerprint": str(source["splitter_fingerprint"]),
            }
        )
    ordered = sorted((float(value) for value in probabilities.values()), reverse=True)
    return {
        "schema_version": ARTICLE_EVIDENCE_SCHEMA_VERSION,
        "contract_id": ARTICLE_EVIDENCE_CONTRACT_ID,
        "claim_id": request["claim_id"],
        "claim_text": request["claim_text"],
        "article_id": item.article_id,
        "pmid": None if item.article.get("pmid") is None else str(item.article["pmid"]),
        "final_article_rank": int(item.article["final_article_rank"]),
        "title": str(item.article.get("title") or ""),
        "predicted_label": label,
        "probabilities": probabilities,
        "confidence": float(probabilities[label]),
        "probability_margin": ordered[0] - ordered[1],
        "rationale": str(result["rationale"]),
        "selected_evidence": selected,
        "provider": provider,
        "model": model,
        "model_fingerprint": model_fingerprint,
        "prompt_version": ARTICLE_EVIDENCE_PROMPT_VERSION,
        "provider_request_id": response.request_id,
        "raw_response_sha256": response.raw_response_sha256,
    }


def validate_article_evidence_rows(
    rows: Sequence[Mapping[str, Any]], *, source_inputs: Sequence[ArticlePromptInput]
) -> dict[str, Any]:
    expected = [item.article_id for item in source_inputs]
    actual = [str(row.get("article_id") or "") for row in rows]
    if actual != expected:
        raise EvidenceGapError("Article evidence order/coverage does not match Top Articles")
    source_by_article = {item.article_id: item for item in source_inputs}
    selected_ids: set[str] = set()
    label_counts = {label: 0 for label in STANCE_LABELS}
    for row in rows:
        article_id = str(row["article_id"])
        label = canonical_stance_label(str(row["predicted_label"]))
        label_counts[label] += 1
        probabilities = {key: float(value) for key, value in row["probabilities"].items()}
        if set(probabilities) != set(STANCE_LABELS):
            raise EvidenceGapError(f"Invalid probability labels for {article_id}")
        if abs(sum(probabilities.values()) - 1.0) > 1e-4:
            raise EvidenceGapError(f"Probabilities do not sum to one for {article_id}")
        if max(STANCE_LABELS, key=lambda value: probabilities[value]) != label:
            raise EvidenceGapError(f"Label/probability mismatch for {article_id}")
        if abs(float(row.get("confidence")) - probabilities[label]) > 1e-5:
            raise EvidenceGapError(f"Confidence mismatch for {article_id}")
        evidence = list(row.get("selected_evidence") or [])
        if label == "insufficient" and evidence:
            raise EvidenceGapError(f"Insufficient article selected evidence: {article_id}")
        if label != "insufficient" and not evidence:
            raise EvidenceGapError(f"Decisive article has no evidence: {article_id}")
        if len(evidence) > MAX_EVIDENCE_SENTENCES:
            raise EvidenceGapError(f"Too many evidence sentences for {article_id}")
        source = {
            sentence.sentence_id: sentence for sentence in source_by_article[article_id].sentences
        }
        for selected in evidence:
            sentence_id = str(selected.get("sentence_id") or "")
            if not sentence_id or sentence_id in selected_ids:
                raise EvidenceGapError(f"Duplicate or blank sentence_id: {sentence_id!r}")
            selected_ids.add(sentence_id)
            sentence = source.get(sentence_id)
            if sentence is None:
                raise EvidenceGapError(f"Unknown selected sentence: {sentence_id}")
            if str(selected.get("sentence_text")) != str(
                sentence.source["sentence_text"]
            ):
                raise EvidenceGapError(f"Selected sentence text mismatch: {sentence_id}")
    return {
        "status": "PASS",
        "articles": len(rows),
        "evidence_selections": len(selected_ids),
        "article_label_counts": label_counts,
        "title_evidence_count": 0,
        "source_sentence_identity_preserved": True,
    }


def run_article_evidence_extractor(
    root: Path,
    *,
    retrieval_artifact_dir: Path,
    provider: str,
    model: str | None = None,
    run_name: str | None = None,
    api_key_env: str | None = None,
    base_url: str | None = None,
    request_batch_size: int = 2,
    max_tokens: int = 4096,
    timeout_seconds: float = 180.0,
    max_retries: int = 4,
    thinking: bool = False,
    dry_run: bool = False,
    cache_dir: Path | None = None,
    artifact_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    retrieval_artifact_dir = retrieval_artifact_dir.resolve()
    provider = provider.strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise EvidenceGapError(f"provider must be one of {SUPPORTED_PROVIDERS}")
    model = (model or DEFAULT_MODELS[provider]).strip()
    api_key_env = (api_key_env or DEFAULT_API_KEY_ENVS[provider]).strip()
    base_url = (base_url or DEFAULT_BASE_URLS[provider]).strip()
    if request_batch_size <= 0 or max_tokens <= 0 or timeout_seconds <= 0:
        raise EvidenceGapError("Batch size, max tokens, and timeout must be positive")
    if max_retries < 0:
        raise EvidenceGapError("max_retries cannot be negative")
    if provider != "deepseek" and thinking:
        raise EvidenceGapError("--thinking is only supported for DeepSeek")

    request, items, source_paths = load_article_prompt_inputs(retrieval_artifact_dir)
    batches = list(_batches(items, request_batch_size))
    if dry_run:
        return {
            "status": "DRY_RUN",
            "provider": provider,
            "model": model,
            "prompt_version": ARTICLE_EVIDENCE_PROMPT_VERSION,
            "articles": len(items),
            "requests": len(batches),
            "request_batch_size": request_batch_size,
            "estimated_input_characters": sum(
                len(build_user_prompt(claim_text=request["claim_text"], items=batch))
                for batch in batches
            ),
        }

    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise EvidenceGapError(f"Missing API key environment variable {api_key_env}")
    name = _safe_name(run_name or f"{provider}_{model}_{retrieval_artifact_dir.name}")
    target = (
        artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    ) / name
    require_empty_or_force(target, force=force)
    cache_root = cache_dir.resolve() if cache_dir else root / DEFAULT_CACHE_ROOT
    cache_root = cache_root / provider / _safe_name(model)
    model_fingerprint = _fingerprint(
        {
            "provider": provider,
            "model": model,
            "base_url": base_url.rstrip("/"),
            "prompt_version": ARTICLE_EVIDENCE_PROMPT_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "response_schema": response_json_schema(),
            "thinking": thinking if provider == "deepseek" else None,
        }
    )

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    api_requests = cache_hits = retries = 0
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for batch in batches:
        request_hash = _request_hash(
            provider=provider,
            model=model,
            base_url=base_url,
            claim_text=request["claim_text"],
            items=batch,
            max_tokens=max_tokens,
            thinking=thinking,
        )
        cache_path = cache_root / f"{request_hash}.json"
        cached = _read_cache(cache_path, request_hash=request_hash, items=batch)
        if cached is None:
            results, response, batch_retries = _call_with_retries(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
                claim_text=request["claim_text"],
                items=batch,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                thinking=thinking,
                request_hash=request_hash,
                failed_response_dir=(
                    target.parent / "debug" / "article_evidence_failed_responses"
                ),
            )
            _write_cache(
                cache_path,
                request_hash=request_hash,
                response=response,
                results=results,
            )
            api_requests += 1
            retries += batch_retries
            for key in usage:
                usage[key] += int(response.usage.get(key, 0))
        else:
            results, response = cached
            cache_hits += 1
        rows.extend(
            _output_row(
                request=request,
                item=item,
                result=result,
                provider=provider,
                model=model,
                model_fingerprint=model_fingerprint,
                response=response,
            )
            for item, result in zip(batch, results)
        )

    validation = validate_article_evidence_rows(rows, source_inputs=items)
    with atomic_directory(target, force=force) as staging:
        output_path = staging / "article_evidence.jsonl"
        _write_jsonl(output_path, rows)
        outputs = {
            "article_evidence": {
                "path": relative_path(root, target / output_path.name),
                "sha256": sha256_file(output_path),
                "rows": len(rows),
            }
        }
        manifest = {
            "schema_version": ARTICLE_EVIDENCE_SCHEMA_VERSION,
            "contract_id": ARTICLE_EVIDENCE_CONTRACT_ID,
            "run_type": "phase07_article_llm_evidence_extractor",
            "run_name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": provider,
            "model": model,
            "model_fingerprint": model_fingerprint,
            "prompt_version": ARTICLE_EVIDENCE_PROMPT_VERSION,
            "parameters": {
                "request_batch_size": request_batch_size,
                "max_tokens": max_tokens,
                "max_retries": max_retries,
                "thinking": thinking if provider == "deepseek" else None,
                "selection_source": "all_numbered_abstract_sentences",
                "max_evidence_sentences_per_article": MAX_EVIDENCE_SENTENCES,
            },
            "source": {
                "retrieval_artifact_dir": relative_path(root, retrieval_artifact_dir),
                **{
                    key: {
                        "path": relative_path(root, path),
                        "sha256": sha256_file(path),
                    }
                    for key, path in source_paths.items()
                },
            },
            "counts": {
                "articles": len(rows),
                "evidence_selections": validation["evidence_selections"],
                "api_requests": api_requests,
                "cache_hits": cache_hits,
                "retries": retries,
            },
            "usage": usage,
            "validation": validation,
            "outputs": outputs,
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_write_json(staging / "run_manifest.json", manifest)
    return {
        "status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "articles": len(rows),
        "evidence_selections": validation["evidence_selections"],
        "api_requests": api_requests,
        "cache_hits": cache_hits,
        "validation": validation,
        "outputs": outputs,
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def validate_article_evidence_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    try:
        manifest = json.loads(
            (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing article evidence manifest: {artifact_dir}") from exc
    if manifest.get("schema_version") != ARTICLE_EVIDENCE_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected article evidence schema_version")
    if manifest.get("contract_id") != ARTICLE_EVIDENCE_CONTRACT_ID:
        raise EvidenceGapError("Unexpected article evidence contract_id")

    root = artifact_dir
    while root.parent != root and not (root / "src/evidencegap").exists():
        root = root.parent
    if not (root / "src/evidencegap").exists():
        root = artifact_dir
    output_meta = manifest["outputs"]["article_evidence"]
    output_path = _resolve(root, str(output_meta["path"]))
    if sha256_file(output_path) != str(output_meta["sha256"]):
        raise EvidenceGapError("Article evidence checksum mismatch")
    for label in ("request", "top_articles", "runtime_sentences"):
        source_meta = manifest["source"][label]
        source_path = _resolve(root, str(source_meta["path"]))
        if sha256_file(source_path) != str(source_meta["sha256"]):
            raise EvidenceGapError(f"Source checksum mismatch: {label}")
    retrieval_dir = _resolve(root, str(manifest["source"]["retrieval_artifact_dir"]))
    _request, inputs, _paths = load_article_prompt_inputs(retrieval_dir)
    validation = validate_article_evidence_rows(
        _read_jsonl(output_path), source_inputs=inputs
    )
    return {
        "status": "PASS",
        "run_name": manifest.get("run_name"),
        "provider": manifest.get("provider"),
        "model": manifest.get("model"),
        **validation,
        "checksums": "PASS",
    }
