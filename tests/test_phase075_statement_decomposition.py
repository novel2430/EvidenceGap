from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.common import EvidenceGapError
from evidencegap.pipeline.retrieval_adapters import runtime_claim_id
from evidencegap.pipeline.statement_decomposition import (
    STATEMENT_DECOMPOSITION_CONTRACT_ID,
    STATEMENT_DECOMPOSITION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    run_statement_decomposition,
    runtime_inference_step_id,
    validate_decomposition_bundle,
    validate_response_payload,
    validate_statement_decomposition_artifact,
)
from evidencegap.stance.llm_judge import ProviderResponse, _ProviderError


class StatementDecompositionPromptTests(unittest.TestCase):
    def test_prompt_requires_argument_preserving_biomedical_decomposition(self) -> None:
        self.assertEqual(
            STATEMENT_DECOMPOSITION_PROMPT_VERSION,
            "phase075_statement_decomposition_v3",
        )
        self.assertIn("argument-preserving decomposition", SYSTEM_PROMPT)
        self.assertIn(
            "independently supported or refuted by biomedical research literature",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Treat claim extraction and argument-structure preservation as equally required outputs",
            SYSTEM_PROMPT,
        )
        self.assertIn("explicit intermediate biomedical conclusions", SYSTEM_PROMPT)
        self.assertIn(
            "Do not collapse an explicit multi-step chain into a shortcut",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "partially overlaps with its premises",
            SYSTEM_PROMPT,
        )
        self.assertIn("hidden assumptions", SYSTEM_PROMPT)
        self.assertIn("policy proposals", SYSTEM_PROMPT)
        self.assertIn("commercial or institutional decisions", SYSTEM_PROMPT)
        self.assertIn("empty claims list", SYSTEM_PROMPT)
        self.assertIn("BCP 47 language tag", SYSTEM_PROMPT)
        self.assertIn(
            "exactly one independently testable biomedical outcome assertion",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "sleep insufficiency increases obesity and type 2 diabetes risk",
            SYSTEM_PROMPT,
        )
        self.assertIn("prevention separate from treatment", SYSTEM_PROMPT)
        self.assertIn(
            "same exact source_text quote may ground more than one claim",
            SYSTEM_PROMPT,
        )
        self.assertNotIn("claim_kind", SYSTEM_PROMPT)
        self.assertNotIn("analysis_eligible", SYSTEM_PROMPT)

    def test_prompt_keeps_gap_analysis_out_of_decomposition(self) -> None:
        self.assertIn(
            "Do not assess whether the claims are true, whether an inference is valid",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not classify inference type, judge logical validity, identify gaps",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "A later request will analyze the extracted inference_steps",
            SYSTEM_PROMPT,
        )

    def test_empty_claims_is_a_valid_result(self) -> None:
        bundle = validate_response_payload(
            {
                "source_language": "zh-TW",
                "claims": [],
                "inference_steps": [],
            },
            original_statement="政府應免費提供維生素D，因為健康是基本人權。",
        )
        validation = validate_decomposition_bundle(bundle)
        self.assertTrue(validation["empty_claims"])
        self.assertEqual(validation["claims"], 0)

    def test_blank_source_language_does_not_reject_empty_claims(self) -> None:
        bundle = validate_response_payload(
            {
                "source_language": "",
                "claims": [],
                "inference_steps": [],
            },
            original_statement="健康是基本人權，所以政府應免費提供所有保健品。",
        )
        self.assertEqual(bundle["source_language"], "und")
        self.assertTrue(validate_decomposition_bundle(bundle)["empty_claims"])

    def test_claims_use_phase07_compatible_ids_and_preserve_inference(self) -> None:
        statement = (
            "維生素D補充能改善免疫功能，因此維生素D補充能降低呼吸道感染風險。"
        )
        bundle = validate_response_payload(
            {
                "source_language": "zh-TW",
                "claims": [
                    {
                        "claim_ref": "C1",
                        "source_text": "維生素D補充能改善免疫功能",
                        "canonical_claim_en": "Vitamin D supplementation improves immune function.",
                    },
                    {
                        "claim_ref": "C2",
                        "source_text": "維生素D補充能降低呼吸道感染風險",
                        "canonical_claim_en": "Vitamin D supplementation reduces the risk of respiratory infections.",
                    },
                ],
                "inference_steps": [
                    {
                        "premise_claim_refs": ["C1"],
                        "conclusion_claim_ref": "C2",
                    }
                ],
            },
            original_statement=statement,
        )
        first = runtime_claim_id(
            "Vitamin D supplementation improves immune function."
        )
        second = runtime_claim_id(
            "Vitamin D supplementation reduces the risk of respiratory infections."
        )
        self.assertEqual(bundle["claims"][0]["claim_id"], first)
        self.assertEqual(
            bundle["inference_steps"],
            [
                {
                    "inference_step_id": runtime_inference_step_id(
                        [first], second
                    ),
                    "premise_claim_ids": [first],
                    "conclusion_claim_id": second,
                }
            ],
        )

    def test_intermediate_claim_can_be_conclusion_then_later_premise(self) -> None:
        statement = (
            "藥物A降低發炎，因此藥物A改善胰島素敏感性；"
            "藥物A改善胰島素敏感性，因此藥物A降低第二型糖尿病風險。"
        )
        bundle = validate_response_payload(
            {
                "source_language": "zh-TW",
                "claims": [
                    {
                        "claim_ref": "C1",
                        "source_text": "藥物A降低發炎",
                        "canonical_claim_en": "Drug A reduces inflammation.",
                    },
                    {
                        "claim_ref": "C2",
                        "source_text": "藥物A改善胰島素敏感性",
                        "canonical_claim_en": "Drug A improves insulin sensitivity.",
                    },
                    {
                        "claim_ref": "C3",
                        "source_text": "藥物A降低第二型糖尿病風險",
                        "canonical_claim_en": "Drug A reduces the risk of type 2 diabetes.",
                    },
                ],
                "inference_steps": [
                    {
                        "premise_claim_refs": ["C1"],
                        "conclusion_claim_ref": "C2",
                    },
                    {
                        "premise_claim_refs": ["C2"],
                        "conclusion_claim_ref": "C3",
                    },
                ],
            },
            original_statement=statement,
        )
        first, intermediate, final = [
            claim["claim_id"] for claim in bundle["claims"]
        ]
        self.assertEqual(
            bundle["inference_steps"],
            [
                {
                    "inference_step_id": runtime_inference_step_id(
                        [first], intermediate
                    ),
                    "premise_claim_ids": [first],
                    "conclusion_claim_id": intermediate,
                },
                {
                    "inference_step_id": runtime_inference_step_id(
                        [intermediate], final
                    ),
                    "premise_claim_ids": [intermediate],
                    "conclusion_claim_id": final,
                },
            ],
        )
        self.assertFalse(
            any(
                step["premise_claim_ids"] == [first]
                and step["conclusion_claim_id"] == final
                for step in bundle["inference_steps"]
            )
        )

    def test_decomposition_validation_rejects_tampered_inference_step_id(self) -> None:
        statement = "甲降低感染，因此乙降低死亡。"
        bundle = validate_response_payload(
            {
                "source_language": "zh-TW",
                "claims": [
                    {
                        "claim_ref": "C1",
                        "source_text": "甲降低感染",
                        "canonical_claim_en": "Treatment A reduces infections.",
                    },
                    {
                        "claim_ref": "C2",
                        "source_text": "乙降低死亡",
                        "canonical_claim_en": "Treatment B reduces mortality.",
                    },
                ],
                "inference_steps": [
                    {
                        "premise_claim_refs": ["C1"],
                        "conclusion_claim_ref": "C2",
                    }
                ],
            },
            original_statement=statement,
        )
        bundle["inference_steps"][0]["inference_step_id"] = "inference_tampered"
        with self.assertRaisesRegex(
            EvidenceGapError, "inference_step_id mismatch"
        ):
            validate_decomposition_bundle(bundle)

    def test_inference_step_id_is_stable_across_premise_order(self) -> None:
        first = runtime_claim_id("Treatment A reduces infections.")
        second = runtime_claim_id("Treatment B reduces inflammation.")
        conclusion = runtime_claim_id("Combined treatment reduces mortality.")
        self.assertEqual(
            runtime_inference_step_id([first, second], conclusion),
            runtime_inference_step_id([second, first], conclusion),
        )
        self.assertTrue(
            runtime_inference_step_id([first, second], conclusion).startswith(
                "inference_"
            )
        )

    def test_duplicate_inference_relationships_are_rejected(self) -> None:
        statement = "甲降低感染，因此乙降低死亡。"
        payload = {
            "source_language": "zh-TW",
            "claims": [
                {
                    "claim_ref": "C1",
                    "source_text": "甲降低感染",
                    "canonical_claim_en": "Treatment A reduces infections.",
                },
                {
                    "claim_ref": "C2",
                    "source_text": "乙降低死亡",
                    "canonical_claim_en": "Treatment B reduces mortality.",
                },
            ],
            "inference_steps": [
                {
                    "premise_claim_refs": ["C1"],
                    "conclusion_claim_ref": "C2",
                },
                {
                    "premise_claim_refs": ["C1"],
                    "conclusion_claim_ref": "C2",
                },
            ],
        }
        with self.assertRaisesRegex(_ProviderError, "duplicate relationships"):
            validate_response_payload(payload, original_statement=statement)

    def test_source_text_must_be_grounded_in_original_statement(self) -> None:
        with self.assertRaises(_ProviderError):
            validate_response_payload(
                {
                    "source_language": "zh-TW",
                    "claims": [
                        {
                            "claim_ref": "C1",
                            "source_text": "原文不存在的句子",
                            "canonical_claim_en": "Vitamin D prevents infection.",
                        }
                    ],
                    "inference_steps": [],
                },
                original_statement="維生素D可能有幫助。",
            )


class StatementDecompositionArtifactTests(unittest.TestCase):
    def test_run_reuses_structured_transport_and_writes_valid_artifact(self) -> None:
        statement = "維生素D能預防呼吸道感染，因此政府應免費提供維生素D。"
        response = ProviderResponse(
            payload={
                "source_language": "zh-TW",
                "claims": [
                    {
                        "claim_ref": "C1",
                        "source_text": "維生素D能預防呼吸道感染",
                        "canonical_claim_en": "Vitamin D supplementation prevents respiratory infections.",
                    }
                ],
                "inference_steps": [],
            },
            request_id="req-fixture",
            usage={"input_tokens": 80, "output_tokens": 30, "total_tokens": 110},
            raw_response_sha256="b" * 64,
            finish_reason="stop",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            with (
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture"}, clear=True),
                patch(
                    "evidencegap.pipeline.statement_decomposition.call_structured_llm",
                    return_value=response,
                ) as call,
            ):
                result = run_statement_decomposition(
                    root,
                    statement=statement,
                    provider="deepseek",
                    run_name="fixture",
                    artifact_root=root / "artifacts",
                )

            self.assertEqual(call.call_count, 1)
            self.assertEqual(result["status"], "PASS")
            output_dir = root / "artifacts/fixture"
            validation = validate_statement_decomposition_artifact(output_dir)
            self.assertEqual(validation["status"], "PASS")
            self.assertEqual(validation["claims"], 1)

            bundle = json.loads(
                (output_dir / "decomposition.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                bundle["contract_id"], STATEMENT_DECOMPOSITION_CONTRACT_ID
            )
            self.assertNotIn("claim_kind", bundle["claims"][0])
            self.assertNotIn("analysis_eligible", bundle["claims"][0])

    def test_empty_llm_result_writes_a_valid_completed_artifact(self) -> None:
        response = ProviderResponse(
            payload={
                "source_language": "zh-TW",
                "claims": [],
                "inference_steps": [],
            },
            request_id="req-empty",
            usage={"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
            raw_response_sha256="c" * 64,
            finish_reason="stop",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            with (
                patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture"}, clear=True),
                patch(
                    "evidencegap.pipeline.statement_decomposition.call_structured_llm",
                    return_value=response,
                ),
            ):
                result = run_statement_decomposition(
                    root,
                    statement="健康是基本人權，所以政府應免費提供所有保健品。",
                    provider="deepseek",
                    run_name="empty",
                    artifact_root=root / "artifacts",
                )
            self.assertEqual(result["status"], "PASS")
            self.assertTrue(result["empty_claims"])
            self.assertEqual(result["claims"], 0)


if __name__ == "__main__":
    unittest.main()
