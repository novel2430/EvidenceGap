from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from evidencegap.common import EvidenceGapError, sha256_text

SCHEMA_VERSION = "1.0.0"
TASK_ID = "STANCE-EVIDENCE-3"
INPUT_RECORD_TYPE = "StanceInputRecord"
PREDICTION_RECORD_TYPE = "StancePredictionRecord"


class StanceLabel(str, Enum):
    SUPPORT = "support"
    REFUTE = "refute"
    INSUFFICIENT = "insufficient"


STANCE_LABELS = tuple(label.value for label in StanceLabel)
EVIDENCE_UNITS = ("sentence", "bundle")
EVIDENCE_TYPES = (
    "direct_result",
    "background",
    "method",
    "population_or_scope",
    "safety",
    "statistical_uncertainty",
    "mixed_or_other",
)


def canonical_evidence_type(value: str | None) -> str:
    """Normalize auxiliary provider wording into the stable taxonomy.

    Evidence type does not determine the stance label. Unknown wording therefore
    degrades to ``mixed_or_other`` instead of invalidating and rebilling an
    otherwise valid response.
    """

    normalized = "" if value is None else str(value).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "result": "direct_result",
        "study_result": "direct_result",
        "finding": "direct_result",
        "direct_evidence": "direct_result",
        "context": "background",
        "background_information": "background",
        "methods": "method",
        "methodology": "method",
        "population": "population_or_scope",
        "scope": "population_or_scope",
        "population_constraint": "population_or_scope",
        "scope_limitation": "population_or_scope",
        "safety_signal": "safety",
        "adverse_event": "safety",
        "adverse_events": "safety",
        "uncertainty": "statistical_uncertainty",
        "statistical_limitation": "statistical_uncertainty",
        "non_significant_result": "statistical_uncertainty",
        "study_limitation": "mixed_or_other",
        "study_limitations": "mixed_or_other",
        "limitation": "mixed_or_other",
        "risk_of_bias": "mixed_or_other",
        "mixed": "mixed_or_other",
        "other": "mixed_or_other",
        "unknown": "mixed_or_other",
        "": "mixed_or_other",
    }
    canonical = aliases.get(normalized, normalized)
    return canonical if canonical in EVIDENCE_TYPES else "mixed_or_other"


def canonical_stance_label(value: str | StanceLabel) -> str:
    label = value.value if isinstance(value, StanceLabel) else str(value).strip().lower()
    aliases = {
        "supported": StanceLabel.SUPPORT.value,
        "entailment": StanceLabel.SUPPORT.value,
        "refuted": StanceLabel.REFUTE.value,
        "contradiction": StanceLabel.REFUTE.value,
        "neutral": StanceLabel.INSUFFICIENT.value,
        "not_enough_information": StanceLabel.INSUFFICIENT.value,
        "nei": StanceLabel.INSUFFICIENT.value,
    }
    label = aliases.get(label, label)
    if label not in STANCE_LABELS:
        raise EvidenceGapError(
            f"Unknown stance label {value!r}; expected one of {STANCE_LABELS}"
        )
    return label


def stance_input_fingerprint(
    *,
    claim_text: str,
    evidence_text: str,
    context_before: str | None,
    context_after: str | None,
) -> str:
    payload = {
        "claim_text": claim_text,
        "evidence_text": evidence_text,
        "context_before": context_before,
        "context_after": context_after,
    }
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True))


@dataclass(frozen=True)
class StanceInput:
    input_id: str
    dataset: str
    split: str
    claim_id: str
    query_id: str | None
    claim_text: str
    paper_id: str | None
    sentence_index: int | None
    sentence_type: str | None
    evidence_rank: int | None
    evidence_text: str
    evidence_unit: str
    context_before: str | None = None
    context_after: str | None = None
    retrieval_model: str | None = None
    retrieval_score: float | None = None
    cross_encoder_score: float | None = None
    source_run_name: str | None = None
    source_artifact_sha256: str | None = None
    gold_label: str | None = None
    source_locator: Mapping[str, Any] | None = None

    def validate(self) -> None:
        if not self.input_id.strip():
            raise EvidenceGapError("stance input_id cannot be empty")
        if not self.dataset.strip():
            raise EvidenceGapError(f"{self.input_id}: dataset cannot be empty")
        if not self.split.strip():
            raise EvidenceGapError(f"{self.input_id}: split cannot be empty")
        if not self.claim_id.strip():
            raise EvidenceGapError(f"{self.input_id}: claim_id cannot be empty")
        if not self.claim_text.strip():
            raise EvidenceGapError(f"{self.input_id}: claim_text cannot be empty")
        if not self.evidence_text.strip():
            raise EvidenceGapError(f"{self.input_id}: evidence_text cannot be empty")
        if self.evidence_unit not in EVIDENCE_UNITS:
            raise EvidenceGapError(
                f"{self.input_id}: evidence_unit must be one of {EVIDENCE_UNITS}"
            )
        if self.sentence_index is not None and self.sentence_index < 0:
            raise EvidenceGapError(f"{self.input_id}: sentence_index cannot be negative")
        if self.evidence_rank is not None and self.evidence_rank <= 0:
            raise EvidenceGapError(f"{self.input_id}: evidence_rank must be positive")
        for field_name, value in (
            ("retrieval_score", self.retrieval_score),
            ("cross_encoder_score", self.cross_encoder_score),
        ):
            if value is not None and not math.isfinite(float(value)):
                raise EvidenceGapError(f"{self.input_id}: {field_name} must be finite")
        if self.gold_label is not None:
            canonical_stance_label(self.gold_label)
        if self.evidence_unit == "sentence" and self.sentence_index is None:
            raise EvidenceGapError(
                f"{self.input_id}: sentence evidence requires sentence_index"
            )

    @property
    def input_fingerprint(self) -> str:
        return stance_input_fingerprint(
            claim_text=self.claim_text,
            evidence_text=self.evidence_text,
            context_before=self.context_before,
            context_after=self.context_after,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "record_type": INPUT_RECORD_TYPE,
            "input_id": self.input_id,
            "input_fingerprint": self.input_fingerprint,
            "dataset": self.dataset,
            "split": self.split,
            "claim_id": self.claim_id,
            "query_id": self.query_id,
            "claim_text": self.claim_text,
            "paper_id": self.paper_id,
            "sentence_index": self.sentence_index,
            "sentence_type": self.sentence_type,
            "evidence_rank": self.evidence_rank,
            "evidence_text": self.evidence_text,
            "evidence_unit": self.evidence_unit,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "retrieval_model": self.retrieval_model,
            "retrieval_score": self.retrieval_score,
            "cross_encoder_score": self.cross_encoder_score,
            "source_run_name": self.source_run_name,
            "source_artifact_sha256": self.source_artifact_sha256,
            "gold_label": (
                None if self.gold_label is None else canonical_stance_label(self.gold_label)
            ),
            "source_locator_json": (
                None
                if self.source_locator is None
                else json.dumps(self.source_locator, ensure_ascii=False, sort_keys=True)
            ),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StanceInput":
        locator_value = value.get("source_locator_json")
        if locator_value is None:
            locator = value.get("source_locator")
        elif isinstance(locator_value, str):
            try:
                locator = json.loads(locator_value)
            except json.JSONDecodeError as exc:
                raise EvidenceGapError(
                    f"Invalid source_locator_json for {value.get('input_id')}"
                ) from exc
        else:
            raise EvidenceGapError(
                f"source_locator_json must be a string for {value.get('input_id')}"
            )
        if locator is not None and not isinstance(locator, Mapping):
            raise EvidenceGapError(
                f"source locator must be an object for {value.get('input_id')}"
            )
        instance = cls(
            input_id=str(value["input_id"]),
            dataset=str(value["dataset"]),
            split=str(value["split"]),
            claim_id=str(value["claim_id"]),
            query_id=None if value.get("query_id") is None else str(value["query_id"]),
            claim_text=str(value["claim_text"]),
            paper_id=None if value.get("paper_id") is None else str(value["paper_id"]),
            sentence_index=(
                None if value.get("sentence_index") is None else int(value["sentence_index"])
            ),
            sentence_type=(
                None if value.get("sentence_type") is None else str(value["sentence_type"])
            ),
            evidence_rank=(
                None if value.get("evidence_rank") is None else int(value["evidence_rank"])
            ),
            evidence_text=str(value["evidence_text"]),
            evidence_unit=str(value["evidence_unit"]),
            context_before=(
                None if value.get("context_before") is None else str(value["context_before"])
            ),
            context_after=(
                None if value.get("context_after") is None else str(value["context_after"])
            ),
            retrieval_model=(
                None if value.get("retrieval_model") is None else str(value["retrieval_model"])
            ),
            retrieval_score=(
                None if value.get("retrieval_score") is None else float(value["retrieval_score"])
            ),
            cross_encoder_score=(
                None
                if value.get("cross_encoder_score") is None
                else float(value["cross_encoder_score"])
            ),
            source_run_name=(
                None if value.get("source_run_name") is None else str(value["source_run_name"])
            ),
            source_artifact_sha256=(
                None
                if value.get("source_artifact_sha256") is None
                else str(value["source_artifact_sha256"])
            ),
            gold_label=(
                None if value.get("gold_label") is None else str(value["gold_label"])
            ),
            source_locator=locator,
        )
        instance.validate()
        expected_fingerprint = value.get("input_fingerprint")
        if (
            expected_fingerprint is not None
            and str(expected_fingerprint) != instance.input_fingerprint
        ):
            raise EvidenceGapError(
                f"Input fingerprint mismatch for {instance.input_id}"
            )
        return instance


@dataclass(frozen=True)
class StancePrediction:
    stance_input: StanceInput
    run_name: str
    model_name: str
    model_fingerprint: str
    stance_input_artifact_sha256: str
    predicted_label: str
    probability_support: float
    probability_refute: float
    probability_insufficient: float
    confidence: float
    probability_margin: float
    abstained: bool = False
    rationale: str | None = None
    evidence_type: str | None = None
    requires_context: bool | None = None
    provider: str | None = None
    provider_request_id: str | None = None
    raw_response_sha256: str | None = None
    prompt_version: str | None = None

    def validate(self) -> None:
        self.stance_input.validate()
        if not self.run_name.strip():
            raise EvidenceGapError("stance prediction run_name cannot be empty")
        if not self.model_name.strip():
            raise EvidenceGapError("stance prediction model_name cannot be empty")
        if not self.model_fingerprint.strip():
            raise EvidenceGapError("stance prediction model_fingerprint cannot be empty")
        if not self.stance_input_artifact_sha256.strip():
            raise EvidenceGapError("stance prediction input artifact checksum cannot be empty")
        predicted = canonical_stance_label(self.predicted_label)
        probabilities = {
            StanceLabel.SUPPORT.value: float(self.probability_support),
            StanceLabel.REFUTE.value: float(self.probability_refute),
            StanceLabel.INSUFFICIENT.value: float(self.probability_insufficient),
        }
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in probabilities.values()
        ):
            raise EvidenceGapError(
                f"{self.stance_input.input_id}: stance probabilities must be finite in [0, 1]"
            )
        if abs(sum(probabilities.values()) - 1.0) > 1e-4:
            raise EvidenceGapError(
                f"{self.stance_input.input_id}: stance probabilities do not sum to one"
            )
        expected = max(probabilities, key=probabilities.get)
        if predicted != expected:
            raise EvidenceGapError(
                f"{self.stance_input.input_id}: predicted_label is not argmax probability"
            )
        if abs(float(self.confidence) - probabilities[predicted]) > 1e-5:
            raise EvidenceGapError(
                f"{self.stance_input.input_id}: confidence does not match predicted probability"
            )
        if (
            not math.isfinite(float(self.probability_margin))
            or self.probability_margin < 0.0
            or self.probability_margin > 1.0
        ):
            raise EvidenceGapError(
                f"{self.stance_input.input_id}: probability_margin must be finite and non-negative"
            )
        if self.rationale is not None and not self.rationale.strip():
            raise EvidenceGapError(
                f"{self.stance_input.input_id}: rationale cannot be blank"
            )
        if self.evidence_type is not None and self.evidence_type not in EVIDENCE_TYPES:
            raise EvidenceGapError(
                f"{self.stance_input.input_id}: unknown evidence_type {self.evidence_type!r}"
            )
        for field_name, value in (
            ("provider", self.provider),
            ("provider_request_id", self.provider_request_id),
            ("raw_response_sha256", self.raw_response_sha256),
            ("prompt_version", self.prompt_version),
        ):
            if value is not None and not str(value).strip():
                raise EvidenceGapError(
                    f"{self.stance_input.input_id}: {field_name} cannot be blank"
                )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        base = self.stance_input.to_dict()
        base.update(
            {
                "record_type": PREDICTION_RECORD_TYPE,
                "run_name": self.run_name,
                "model_name": self.model_name,
                "model_fingerprint": self.model_fingerprint,
                "stance_input_artifact_sha256": self.stance_input_artifact_sha256,
                "predicted_label": canonical_stance_label(self.predicted_label),
                "probability_support": float(self.probability_support),
                "probability_refute": float(self.probability_refute),
                "probability_insufficient": float(self.probability_insufficient),
                "confidence": float(self.confidence),
                "probability_margin": float(self.probability_margin),
                "abstained": bool(self.abstained),
                "rationale": self.rationale,
                "evidence_type": self.evidence_type,
                "requires_context": self.requires_context,
                "provider": self.provider,
                "provider_request_id": self.provider_request_id,
                "raw_response_sha256": self.raw_response_sha256,
                "prompt_version": self.prompt_version,
            }
        )
        return base
