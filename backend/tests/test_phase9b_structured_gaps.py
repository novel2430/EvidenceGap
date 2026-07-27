from __future__ import annotations

import copy

import pytest

import evidencegap_backend.pipeline.inference_gap_analysis as gap_module
from evidencegap_backend.pipeline.inference_gap_analysis import (
    CAUSAL_GAP_SUBTYPES,
    GAP_AFFECTED_DIMENSIONS,
    SCOPE_GAP_SUBTYPES,
    build_gap_analysis_input,
    response_json_schema,
    validate_response_payload,
)
from evidencegap_backend.stance.llm_judge import _ProviderError


def _statement_bundle() -> dict[str, object]:
    return {
        "statement": {
            "statement_id": "statement:1",
            "original_text": "Marker X improves, therefore all complications are prevented.",
        },
        "claims": [
            {
                "claim_id": "claim:premise",
                "source_text": "Marker X improves",
                "canonical_claim_en": "The intervention improves marker X.",
                "analysis_status": "completed",
                "verdict": "supported",
                "rationale": "The retrieved article supports marker X improvement.",
            },
            {
                "claim_id": "claim:conclusion",
                "source_text": "all complications are prevented",
                "canonical_claim_en": "The intervention prevents all complications.",
                "analysis_status": "completed",
                "verdict": "insufficient",
                "rationale": "The retrieved evidence does not establish complication prevention.",
            },
        ],
        "inference_steps": [
            {
                "inference_step_id": "inference:1",
                "premise_claim_ids": ["claim:premise"],
                "conclusion_claim_id": "claim:conclusion",
            }
        ],
        "articles": [
            {
                "article_node_id": "article:1",
                "claim_id": "claim:premise",
                "pmid": "1",
                "title": "Surrogate endpoint study",
                "stance": "support",
                "confidence": 0.9,
                "rationale": "The study reports marker X but not complications.",
                "applicability": {
                    "population_or_species": "MATCH",
                    "intervention_or_exposure": "MATCH",
                    "comparator": "NOT_REPORTED",
                    "outcome": "MISMATCH",
                    "direction": "MATCH",
                    "timeframe": "NOT_REPORTED",
                    "causal_strength": "MISMATCH",
                    "prevention_treatment_scope": "MISMATCH",
                },
                "applicability_issues": [
                    {
                        "dimension": "outcome",
                        "code": "SURROGATE_OUTCOME",
                        "reason": "The article reports marker X rather than complications.",
                    },
                    {
                        "dimension": "causal_strength",
                        "code": "CAUSAL_STRENGTH_MISMATCH",
                        "reason": "The article does not establish the stronger causal conclusion.",
                    },
                    {
                        "dimension": "prevention_treatment_scope",
                        "code": "PREVENTION_TREATMENT_MISMATCH",
                        "reason": "The article does not establish prevention.",
                    },
                ],
                "evidence_ids": ["evidence:1"],
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence:1",
                "section": "abstract",
                "text": "Marker X improved during follow-up.",
            }
        ],
    }


def _empty_gap() -> dict[str, object]:
    return {
        "detected": False,
        "subtype": None,
        "affected_dimensions": [],
        "supported_basis": None,
        "unsupported_extension": None,
        "reason": None,
        "closure_requirement": None,
    }


def _payload() -> dict[str, object]:
    return {
        "analyses": [
            {
                "inference_step_id": "inference:1",
                "scope_gap": {
                    "detected": True,
                    "subtype": "OUTCOME_EXPANSION",
                    "affected_dimensions": ["outcome", "quantifier"],
                    "supported_basis": "The premise establishes improvement in marker X.",
                    "unsupported_extension": "The conclusion extends this to prevention of all complications.",
                    "reason": "A surrogate result is extended to a universal clinical prevention claim.",
                    "closure_requirement": "Direct evidence connecting the intervention to the claimed complication outcomes is required.",
                },
                "causal_gap": _empty_gap(),
            }
        ]
    }


def test_phase9b_schema_requires_structured_gap_fields() -> None:
    analysis = response_json_schema()["properties"]["analyses"]["items"]
    scope = analysis["properties"]["scope_gap"]
    causal = analysis["properties"]["causal_gap"]

    expected_fields = {
        "detected",
        "subtype",
        "affected_dimensions",
        "supported_basis",
        "unsupported_extension",
        "reason",
        "closure_requirement",
    }
    assert set(scope["required"]) == expected_fields
    assert set(scope["properties"]["subtype"]["enum"][:-1]) == set(
        SCOPE_GAP_SUBTYPES
    )
    assert set(causal["properties"]["subtype"]["enum"][:-1]) == set(
        CAUSAL_GAP_SUBTYPES
    )
    assert set(scope["properties"]["affected_dimensions"]["items"]["enum"]) == set(
        GAP_AFFECTED_DIMENSIONS
    )


def test_phase9b_gap_input_includes_phase9a_applicability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gap_module, "validate_statement_bundle", lambda _: None)

    payload = build_gap_analysis_input(_statement_bundle())
    article = payload["claims"][0]["articles"][0]

    assert article["applicability"]["outcome"] == "MISMATCH"
    assert article["applicability_issues"][0]["code"] == "SURROGATE_OUTCOME"


def test_phase9b_validates_and_normalizes_structured_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gap_module, "validate_statement_bundle", lambda _: None)

    result = validate_response_payload(
        _payload(), statement_bundle=_statement_bundle()
    )[0]

    assert result["scope_gap"]["subtype"] == "OUTCOME_EXPANSION"
    assert result["scope_gap"]["affected_dimensions"] == ["outcome", "quantifier"]
    assert result["scope_gap"]["closure_requirement"].startswith("Direct evidence")
    assert result["causal_gap"] == _empty_gap()


def test_phase9b_rejects_inconsistent_or_invalid_structured_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gap_module, "validate_statement_bundle", lambda _: None)

    inconsistent = copy.deepcopy(_payload())
    inconsistent["analyses"][0]["causal_gap"]["reason"] = "Unexpected text"  # type: ignore[index]
    with pytest.raises(_ProviderError, match="null or empty"):
        validate_response_payload(
            inconsistent, statement_bundle=_statement_bundle()
        )

    invalid_subtype = copy.deepcopy(_payload())
    invalid_subtype["analyses"][0]["scope_gap"]["subtype"] = "SURROGATE_TO_CLINICAL_OUTCOME"  # type: ignore[index]
    with pytest.raises(_ProviderError, match="subtype must be one of"):
        validate_response_payload(
            invalid_subtype, statement_bundle=_statement_bundle()
        )


def test_phase9b_provider_boundary_ignores_extra_top_level_fields() -> None:
    normalized = gap_module._normalize_provider_response_payload(
        {
            "analyses": _payload()["analyses"],
            "summary": "Harmless provider-added text",
            "metadata": {"model_note": "extra"},
        }
    )

    assert normalized == {"analyses": _payload()["analyses"]}


def test_phase9b_provider_boundary_still_requires_analyses() -> None:
    with pytest.raises(_ProviderError, match="contain the analyses field"):
        gap_module._normalize_provider_response_payload({"summary": "missing"})
