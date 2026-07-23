from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.stance.contracts import EVIDENCE_TYPES, StanceInput
from evidencegap.stance.llm_judge import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    ProviderResponse,
    _ProviderError,
    _anthropic_thinking_config,
    _call_anthropic,
    _call_deepseek,
    _call_with_retries,
    _model_identity,
    _user_prompt,
    _validate_result_payload,
)


def fixture_input() -> StanceInput:
    return StanceInput(
        input_id="fixture-1",
        dataset="fixture",
        split="dev",
        claim_id="claim-1",
        query_id="query-1",
        claim_text="The treatment reduces mortality.",
        paper_id="paper-1",
        sentence_index=2,
        sentence_type="normal paragraph",
        evidence_rank=1,
        evidence_text="Mortality was significantly lower in the treatment group.",
        evidence_unit="sentence",
    )


def fixture_output() -> dict[str, object]:
    return {
        "results": [
            {
                "input_id": "fixture-1",
                "label": "support",
                "probabilities": {
                    "support": 0.9,
                    "refute": 0.02,
                    "insufficient": 0.08,
                },
                "rationale": "The reported reduction directly supports the claim.",
                "evidence_type": "direct_result",
                "requires_context": False,
            }
        ]
    }


class LlmJudgeContractTests(unittest.TestCase):
    def test_valid_payload(self) -> None:
        result = _validate_result_payload(fixture_output(), [fixture_input()])
        self.assertEqual(result[0]["label"], "support")
        self.assertAlmostEqual(sum(result[0]["probabilities"].values()), 1.0)

    def test_prompt_version_is_v2(self) -> None:
        self.assertEqual(PROMPT_VERSION, "phase06_llm_stance_v2")

    def test_model_identity_does_not_recurse(self) -> None:
        schema_hash, prompt_hash, model_fingerprint = _model_identity(
            provider="deepseek",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com/",
        )
        self.assertEqual(len(schema_hash), 64)
        self.assertEqual(len(prompt_hash), 64)
        self.assertEqual(len(model_fingerprint), 64)

    def test_system_prompt_requires_exact_evidence_type_choice(self) -> None:
        self.assertIn("Choose exactly one of the following seven values", SYSTEM_PROMPT)
        self.assertIn("Do not invent, rename, combine, or pluralize categories", SYSTEM_PROMPT)
        for evidence_type in EVIDENCE_TYPES:
            self.assertIn(f"- {evidence_type}:", SYSTEM_PROMPT)
        self.assertIn(
            "safety > statistical_uncertainty > population_or_scope > direct_result > method > background > mixed_or_other",
            SYSTEM_PROMPT,
        )

    def test_user_prompt_repeats_exact_evidence_type_values(self) -> None:
        prompt = _user_prompt([fixture_input()])
        self.assertIn(
            "For evidence_type, choose exactly one of: " + ", ".join(EVIDENCE_TYPES),
            prompt,
        )
        self.assertIn("Do not invent or rename categories", prompt)

    def test_study_limitation_degrades_to_mixed_or_other(self) -> None:
        payload = fixture_output()
        payload["results"][0]["evidence_type"] = "study_limitation"  # type: ignore[index]
        result = _validate_result_payload(payload, [fixture_input()])
        self.assertEqual(result[0]["evidence_type"], "mixed_or_other")

    def test_unknown_evidence_type_degrades_to_mixed_or_other(self) -> None:
        payload = fixture_output()
        payload["results"][0]["evidence_type"] = "clinical_nuance"  # type: ignore[index]
        result = _validate_result_payload(payload, [fixture_input()])
        self.assertEqual(result[0]["evidence_type"], "mixed_or_other")

    def test_wrong_input_id_is_rejected(self) -> None:
        payload = fixture_output()
        payload["results"][0]["input_id"] = "wrong-id"  # type: ignore[index]
        with self.assertRaises(_ProviderError):
            _validate_result_payload(payload, [fixture_input()])

    def test_non_argmax_probabilities_are_reconciled_to_label(self) -> None:
        payload = fixture_output()
        payload["results"][0]["label"] = "insufficient"  # type: ignore[index]
        result = _validate_result_payload(payload, [fixture_input()])
        probabilities = result[0]["probabilities"]
        self.assertEqual(result[0]["label"], "insufficient")
        self.assertEqual(
            max(probabilities, key=probabilities.get),
            "insufficient",
        )
        self.assertTrue(result[0]["probability_reconciled"])
        self.assertEqual(result[0]["probability_original_argmax"], "support")
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_blank_label_is_recovered_from_probability_argmax(self) -> None:
        payload = fixture_output()
        payload["results"][0]["label"] = ""  # type: ignore[index]
        result = _validate_result_payload(payload, [fixture_input()])
        self.assertEqual(result[0]["label"], "support")
        self.assertTrue(result[0]["label_recovered"])
        self.assertEqual(result[0]["label_original_value"], "blank")

    def test_unknown_label_is_recovered_from_probability_argmax(self) -> None:
        payload = fixture_output()
        payload["results"][0]["label"] = "partially_support"  # type: ignore[index]
        result = _validate_result_payload(payload, [fixture_input()])
        self.assertEqual(result[0]["label"], "support")
        self.assertTrue(result[0]["label_recovered"])
        self.assertEqual(result[0]["label_original_value"], "partially_support")

    def test_reconciled_cache_metadata_is_preserved(self) -> None:
        payload = fixture_output()
        payload["results"][0]["probability_reconciled"] = True  # type: ignore[index]
        payload["results"][0]["probability_original_argmax"] = "refute"  # type: ignore[index]
        result = _validate_result_payload(payload, [fixture_input()])
        self.assertTrue(result[0]["probability_reconciled"])
        self.assertEqual(result[0]["probability_original_argmax"], "refute")

    @patch("evidencegap.stance.llm_judge._call_deepseek")
    def test_probability_mismatch_does_not_retry(self, call_deepseek) -> None:
        payload = fixture_output()
        payload["results"][0]["label"] = "insufficient"  # type: ignore[index]
        call_deepseek.return_value = ProviderResponse(
            payload=payload,
            request_id="request-1",
            usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            raw_response_sha256="a" * 64,
            finish_reason="stop",
        )
        results, _response, retries = _call_with_retries(
            provider="deepseek",
            api_key="secret",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            inputs=[fixture_input()],
            max_tokens=1024,
            timeout_seconds=10,
            max_retries=4,
            thinking=False,
        )
        self.assertEqual(retries, 0)
        self.assertEqual(call_deepseek.call_count, 1)
        self.assertTrue(results[0]["probability_reconciled"])

    @patch("evidencegap.stance.llm_judge._post_json")
    def test_deepseek_uses_json_mode(self, post_json) -> None:
        output = fixture_output()
        response = {
            "id": "deepseek-request",
            "choices": [
                {
                    "message": {"content": json.dumps(output)},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        post_json.return_value = (response, json.dumps(response))
        result = _call_deepseek(
            api_key="secret",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-pro",
            inputs=[fixture_input()],
            max_tokens=1024,
            timeout_seconds=10,
            retry_note=None,
            thinking=False,
        )
        body = post_json.call_args.kwargs["body"]
        self.assertEqual(body["response_format"], {"type": "json_object"})
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(result.request_id, "deepseek-request")

    @patch("evidencegap.stance.llm_judge._post_json")
    def test_anthropic_uses_structured_outputs(self, post_json) -> None:
        output = fixture_output()
        response = {
            "id": "claude-request",
            "content": [{"type": "text", "text": json.dumps(output)}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 22},
        }
        post_json.return_value = (response, json.dumps(response))
        result = _call_anthropic(
            api_key="secret",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-4-6",
            inputs=[fixture_input()],
            max_tokens=1024,
            timeout_seconds=10,
            retry_note=None,
        )
        body = post_json.call_args.kwargs["body"]
        self.assertEqual(body["output_config"]["format"]["type"], "json_schema")
        self.assertNotIn("thinking", body)
        self.assertEqual(result.request_id, "claude-request")
        self.assertEqual(result.usage["total_tokens"], 33)

    @patch("evidencegap.stance.llm_judge._post_json")
    def test_anthropic_sonnet5_disables_default_adaptive_thinking(self, post_json) -> None:
        output = fixture_output()
        response = {
            "id": "claude-request",
            "content": [{"type": "text", "text": json.dumps(output)}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 11, "output_tokens": 22},
        }
        post_json.return_value = (response, json.dumps(response))
        _call_anthropic(
            api_key="secret",
            base_url="https://api.anthropic.com",
            model="claude-sonnet-5",
            inputs=[fixture_input()],
            max_tokens=1024,
            timeout_seconds=10,
            retry_note=None,
        )
        body = post_json.call_args.kwargs["body"]
        self.assertEqual(body["thinking"], {"type": "disabled"})
        self.assertEqual(
            _anthropic_thinking_config(" claude-sonnet-5 "),
            {"type": "disabled"},
        )
        self.assertIsNone(_anthropic_thinking_config("claude-sonnet-4-6"))


if __name__ == "__main__":
    unittest.main()
