from __future__ import annotations

import copy

import pytest

from evidencegap_backend.pipeline.article_evidence import (
    APPLICABILITY_DIMENSIONS,
    ArticlePromptInput,
    PromptSentence,
    response_json_schema,
    validate_response_payload,
)
from evidencegap_backend.stance.llm_judge import _ProviderError


def _item() -> ArticlePromptInput:
    return ArticlePromptInput(
        article={"article_id": "pmid:1", "title": "Fixture"},
        sentences=(
            PromptSentence(
                alias="S01",
                source={
                    "sentence_id": "sentence:1",
                    "sentence_text": "The study measured a surrogate biomarker.",
                },
            ),
        ),
    )


def _payload() -> dict[str, object]:
    return {
        "results": [
            {
                "article_id": "pmid:1",
                "label": "insufficient",
                "probabilities": {
                    "support": 0.05,
                    "refute": 0.05,
                    "insufficient": 0.9,
                },
                "evidence_sentence_ids": [],
                "rationale": "The article reports a surrogate rather than the claimed clinical outcome.",
                "applicability": {
                    "population_or_species": "MATCH",
                    "intervention_or_exposure": "MATCH",
                    "comparator": "NOT_REPORTED",
                    "outcome": "MISMATCH",
                    "direction": "MATCH",
                    "timeframe": "NOT_REPORTED",
                    "causal_strength": "MATCH",
                    "prevention_treatment_scope": "NOT_APPLICABLE",
                },
                "applicability_issues": [
                    {
                        "dimension": "outcome",
                        "code": "SURROGATE_OUTCOME",
                        "reason": "The article measures a biomarker rather than the claimed clinical outcome.",
                    }
                ],
            }
        ]
    }


def test_phase9a_schema_requires_compact_applicability_output() -> None:
    result_schema = response_json_schema()["properties"]["results"]["items"]
    assert set(result_schema["properties"]["applicability"]["required"]) == set(
        APPLICABILITY_DIMENSIONS
    )
    assert "applicability" in result_schema["required"]
    assert "applicability_issues" in result_schema["required"]


def test_phase9a_validates_and_normalizes_applicability() -> None:
    result = validate_response_payload(_payload(), [_item()])[0]

    assert result["applicability"]["outcome"] == "MISMATCH"
    assert result["applicability_issues"] == [
        {
            "dimension": "outcome",
            "code": "SURROGATE_OUTCOME",
            "reason": "The article measures a biomarker rather than the claimed clinical outcome.",
        }
    ]


def test_phase9a_rejects_missing_or_spurious_mismatch_issues() -> None:
    missing = _payload()
    missing["results"][0]["applicability_issues"] = []  # type: ignore[index]
    with pytest.raises(_ProviderError, match="exactly one issue"):
        validate_response_payload(missing, [_item()])

    spurious = copy.deepcopy(_payload())
    spurious["results"][0]["applicability"]["outcome"] = "MATCH"  # type: ignore[index]
    with pytest.raises(_ProviderError, match="exactly one issue"):
        validate_response_payload(spurious, [_item()])
