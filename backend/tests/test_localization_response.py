from __future__ import annotations

import pytest

from evidencegap_backend.output.presentation import (
    LOCALIZATION_PROMPT_VERSION,
    build_localization_prompt,
    validate_localization_response,
)
from evidencegap_backend.stance.llm_judge import _ProviderError


UNITS = [
    {"text_id": "claim:1:text", "text": "Claim one."},
    {"text_id": "evidence:1", "text": "Evidence one."},
]
ROWS = [
    {"text_id": "claim:1:text", "text": "主張一。"},
    {"text_id": "evidence:1", "text": "證據一。"},
]


def test_localization_prompt_shows_exact_output_shape() -> None:
    prompt = build_localization_prompt(UNITS, "繁體中文（台灣）")

    assert LOCALIZATION_PROMPT_VERSION == "phase077_output_localization_v3"
    assert '"translations"' in prompt
    assert '"translation_units"' in prompt
    assert "Do not return a top-level texts" in prompt


def test_validator_accepts_exact_shape() -> None:
    assert validate_localization_response({"translations": ROWS}, UNITS) == {
        "claim:1:text": "主張一。",
        "evidence:1": "證據一。",
    }


def test_validator_safely_normalizes_deepseek_mirrored_texts_shape() -> None:
    assert validate_localization_response({"texts": ROWS}, UNITS) == {
        "claim:1:text": "主張一。",
        "evidence:1": "證據一。",
    }


def test_validator_safely_normalizes_single_wrapper() -> None:
    assert validate_localization_response(
        {"result": {"translations": ROWS}}, UNITS
    ) == {
        "claim:1:text": "主張一。",
        "evidence:1": "證據一。",
    }


def test_validator_rejects_ambiguous_or_extra_top_level_keys() -> None:
    with pytest.raises(_ProviderError, match="received top-level keys"):
        validate_localization_response(
            {"translations": ROWS, "target_language": "zh-TW"}, UNITS
        )
