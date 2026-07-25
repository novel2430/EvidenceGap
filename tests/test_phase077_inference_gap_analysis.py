from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evidencegap.common import EvidenceGapError, atomic_write_json  # noqa: E402
from evidencegap.pipeline.inference_gap_analysis import (  # noqa: E402
    INFERENCE_GAP_ANALYSIS_CONTRACT_ID,
    INFERENCE_GAP_ANALYSIS_PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_gap_analysis_input,
    build_inference_gap_analysis_bundle,
    run_inference_gap_analysis,
    validate_inference_gap_analysis_artifact,
    validate_inference_gap_analysis_bundle,
    validate_response_payload,
)
from evidencegap.pipeline.statement_decomposition import (  # noqa: E402
    runtime_inference_step_id,
)
from evidencegap.pipeline.statement_bundle import (  # noqa: E402
    STATEMENT_BUNDLE_CONTRACT_ID,
    STATEMENT_BUNDLE_SCHEMA_VERSION,
    validate_statement_bundle,
)
from evidencegap.stance.llm_judge import (  # noqa: E402
    ProviderResponse,
    _ProviderError,
)


def _statement_bundle(*, include_inference: bool = True) -> dict[str, object]:
    claim_ids = ["claim_1", "claim_2", "claim_3", "claim_4"]
    steps = []
    if include_inference:
        steps.append(
            {
                "inference_step_id": runtime_inference_step_id(
                    [claim_ids[1], claim_ids[2]], claim_ids[3]
                ),
                "premise_claim_ids": [claim_ids[1], claim_ids[2]],
                "conclusion_claim_id": claim_ids[3],
            }
        )

    claims = []
    articles = []
    evidence = []
    verdicts = ["supported", "supported", "mixed", "insufficient"]
    for index, (claim_id, verdict) in enumerate(
        zip(claim_ids, verdicts, strict=True), start=1
    ):
        article_node_id = f"article_{index}"
        evidence_id = f"evidence_{index}"
        stance = {
            "supported": "support",
            "mixed": "support",
            "insufficient": "insufficient",
        }[verdict]
        claims.append(
            {
                "claim_id": claim_id,
                "source_text": f"來源主張 {index}",
                "canonical_claim_en": f"Biomedical claim {index}.",
                "analysis_status": "completed",
                "verdict": verdict,
                "article_counts": {
                    "total": 1,
                    "support": int(stance == "support"),
                    "refute": 0,
                    "insufficient": int(stance == "insufficient"),
                },
                "rationale": f"Claim rationale {index}.",
                "scope": "retrieved_top_articles",
                "boundary": {
                    "is_pipeline_final_verdict": True,
                    "is_final_medical_truth": False,
                    "description": "Fixture boundary.",
                },
                "article_node_ids": [article_node_id],
                "error": None,
            }
        )
        articles.append(
            {
                "article_node_id": article_node_id,
                "claim_id": claim_id,
                "article_id": f"pmid:{index}",
                "pmid": str(index),
                "rank": 1,
                "title": f"Article {index}",
                "rationale": f"Article rationale {index}.",
                "stance": stance,
                "confidence": 0.9,
                "probabilities": {
                    "support": 0.9 if stance == "support" else 0.05,
                    "refute": 0.05,
                    "insufficient": 0.05 if stance == "support" else 0.9,
                },
                "evidence_ids": [evidence_id],
                "provider": "fixture",
                "model": "fixture",
                "model_fingerprint": "fixture",
                "prompt_version": "fixture",
            }
        )
        evidence.append(
            {
                "evidence_id": evidence_id,
                "source_node_id": f"source_{index}",
                "claim_id": claim_id,
                "article_node_id": article_node_id,
                "article_id": f"pmid:{index}",
                "pmid": str(index),
                "label": f"E{index}",
                "text": f"Evidence sentence {index}.",
                "source_evidence_id": f"source-evidence-{index}",
                "sentence_id": f"sentence-{index}",
                "sentence_index": index,
                "sentence_index_within_section": index,
                "section": "results",
                "section_index": 0,
                "character_start": 0,
                "character_end": 20,
                "source_text_fingerprint": "source-fp",
                "splitter_fingerprint": "splitter-fp",
            }
        )

    bundle = {
        "schema_version": STATEMENT_BUNDLE_SCHEMA_VERSION,
        "contract_id": STATEMENT_BUNDLE_CONTRACT_ID,
        "statement": {
            "statement_id": "statement_fixture",
            "original_text": "Fixture biomedical argument.",
            "source_language": "en",
            "analysis_status": "completed",
        },
        "claims": claims,
        "inference_steps": steps,
        "articles": articles,
        "evidence": evidence,
        "summary": {
            "total_claims": len(claims),
            "completed_claims": len(claims),
            "failed_claims": 0,
            "articles": len(articles),
            "evidence": len(evidence),
        },
    }
    validate_statement_bundle(bundle)
    return bundle


def _provider_response(payload: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(
        payload=payload,
        request_id="request-fixture",
        usage={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
        raw_response_sha256="raw-fixture",
        finish_reason="stop",
    )


class InferenceGapAnalysisTests(unittest.TestCase):
    def test_prompt_limits_task_to_two_inference_level_gaps(self) -> None:
        self.assertEqual(
            INFERENCE_GAP_ANALYSIS_PROMPT_VERSION,
            "phase077_inference_gap_analysis_v2",
        )
        self.assertIn("A gap belongs to an inference step", SYSTEM_PROMPT)
        self.assertIn("SCOPE_GAP", SYSTEM_PROMPT)
        self.assertIn("CAUSAL_GAP", SYSTEM_PROMPT)
        self.assertIn("must not be changed", SYSTEM_PROMPT)
        self.assertIn("Do not mark a gap merely because", SYSTEM_PROMPT)
        self.assertIn("Scope comparison is directional", SYSTEM_PROMPT)
        self.assertIn("A difference in population", SYSTEM_PROMPT)
        self.assertIn("scope-aware rather than scope-violating", SYSTEM_PROMPT)
        self.assertIn("Positive SCOPE_GAP example", SYSTEM_PROMPT)
        self.assertIn("Negative SCOPE_GAP example", SYSTEM_PROMPT)
        self.assertIn("unsupported scope introduced by the conclusion", SYSTEM_PROMPT)

    def test_build_input_includes_only_connected_claims_and_their_evidence(self) -> None:
        bundle = _statement_bundle()
        payload = build_gap_analysis_input(bundle)
        claim_ids = [claim["claim_id"] for claim in payload["claims"]]
        self.assertEqual(claim_ids, ["claim_2", "claim_3", "claim_4"])
        self.assertNotIn("claim_1", claim_ids)
        self.assertEqual(payload["claims"][1]["evidence_state"], "CONFLICTED")
        self.assertEqual(
            payload["claims"][0]["articles"][0]["evidence"][0]["text"],
            "Evidence sentence 2.",
        )

    def test_validate_response_requires_exact_step_coverage_and_reasons(self) -> None:
        bundle = _statement_bundle()
        step_id = bundle["inference_steps"][0]["inference_step_id"]
        analyses = validate_response_payload(
            {
                "analyses": [
                    {
                        "inference_step_id": step_id,
                        "scope_gap": {
                            "detected": True,
                            "reason": "The conclusion extends beyond the studied population.",
                        },
                        "causal_gap": {"detected": False, "reason": None},
                    }
                ]
            },
            statement_bundle=bundle,
        )
        self.assertEqual(analyses[0]["inference_step_id"], step_id)

        with self.assertRaisesRegex(_ProviderError, "reason must be non-empty"):
            validate_response_payload(
                {
                    "analyses": [
                        {
                            "inference_step_id": step_id,
                            "scope_gap": {"detected": True, "reason": None},
                            "causal_gap": {"detected": False, "reason": None},
                        }
                    ]
                },
                statement_bundle=bundle,
            )
        with self.assertRaisesRegex(_ProviderError, "Missing analyses"):
            validate_response_payload(
                {"analyses": []}, statement_bundle=bundle
            )

    def test_build_and_validate_gap_bundle(self) -> None:
        statement_bundle = _statement_bundle()
        step_id = statement_bundle["inference_steps"][0]["inference_step_id"]
        bundle = build_inference_gap_analysis_bundle(
            statement_bundle,
            [
                {
                    "inference_step_id": step_id,
                    "scope_gap": {
                        "detected": True,
                        "reason": "The population scope is broader in the conclusion.",
                    },
                    "causal_gap": {
                        "detected": True,
                        "reason": "The premises establish association rather than causation.",
                    },
                }
            ],
            source_statement_bundle_sha256="source-sha",
        )
        self.assertEqual(bundle["contract_id"], INFERENCE_GAP_ANALYSIS_CONTRACT_ID)
        self.assertEqual(bundle["summary"]["scope_gaps"], 1)
        self.assertEqual(bundle["summary"]["causal_gaps"], 1)
        validation = validate_inference_gap_analysis_bundle(
            bundle, statement_bundle=statement_bundle
        )
        self.assertEqual(validation["status"], "PASS")

    def test_run_reuses_structured_llm_and_writes_valid_artifact(self) -> None:
        statement_bundle = _statement_bundle()
        step_id = statement_bundle["inference_steps"][0]["inference_step_id"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            source_dir = root / "artifacts/source_bundle"
            source_dir.mkdir(parents=True)
            atomic_write_json(source_dir / "statement_bundle.json", statement_bundle)
            artifact_root = root / "artifacts/gaps"
            response = _provider_response(
                {
                    "analyses": [
                        {
                            "inference_step_id": step_id,
                            "scope_gap": {
                                "detected": True,
                                "reason": "The conclusion broadens the population.",
                            },
                            "causal_gap": {"detected": False, "reason": None},
                        }
                    ]
                }
            )
            with (
                patch(
                    "evidencegap.pipeline.inference_gap_analysis.validate_statement_bundle_artifact",
                    return_value={"status": "PASS"},
                ),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key"}),
                patch(
                    "evidencegap.pipeline.inference_gap_analysis.call_structured_llm",
                    return_value=response,
                ) as call_llm,
            ):
                result = run_inference_gap_analysis(
                    root,
                    statement_bundle_artifact_dir=source_dir,
                    provider="deepseek",
                    run_name="manual",
                    artifact_root=artifact_root,
                )
            self.assertEqual(call_llm.call_count, 1)
            self.assertIs(call_llm.call_args.kwargs["system_prompt"], SYSTEM_PROMPT)
            self.assertTrue(call_llm.call_args.kwargs["thinking"])
            self.assertEqual(result["scope_gaps"], 1)
            output_dir = artifact_root / "manual"
            manifest = json.loads(
                (output_dir / "run_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["parameters"]["thinking"])
            with patch(
                "evidencegap.pipeline.inference_gap_analysis.validate_statement_bundle_artifact",
                return_value={"status": "PASS"},
            ):
                validation = validate_inference_gap_analysis_artifact(output_dir)
            self.assertEqual(validation["checksums"], "PASS")

    def test_deepseek_thinking_can_be_explicitly_disabled(self) -> None:
        statement_bundle = _statement_bundle()
        step_id = statement_bundle["inference_steps"][0]["inference_step_id"]
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            source_dir = root / "artifacts/source_bundle"
            source_dir.mkdir(parents=True)
            atomic_write_json(source_dir / "statement_bundle.json", statement_bundle)
            response = _provider_response(
                {
                    "analyses": [
                        {
                            "inference_step_id": step_id,
                            "scope_gap": {"detected": False, "reason": None},
                            "causal_gap": {"detected": False, "reason": None},
                        }
                    ]
                }
            )
            with (
                patch(
                    "evidencegap.pipeline.inference_gap_analysis.validate_statement_bundle_artifact",
                    return_value={"status": "PASS"},
                ),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key"}),
                patch(
                    "evidencegap.pipeline.inference_gap_analysis.call_structured_llm",
                    return_value=response,
                ) as call_llm,
            ):
                run_inference_gap_analysis(
                    root,
                    statement_bundle_artifact_dir=source_dir,
                    provider="deepseek",
                    run_name="no-thinking",
                    thinking=False,
                    artifact_root=root / "artifacts/gaps",
                )
            self.assertFalse(call_llm.call_args.kwargs["thinking"])

    def test_no_inference_steps_skips_api_and_does_not_require_key(self) -> None:
        statement_bundle = _statement_bundle(include_inference=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            source_dir = root / "artifacts/source_bundle"
            source_dir.mkdir(parents=True)
            atomic_write_json(source_dir / "statement_bundle.json", statement_bundle)
            with (
                patch(
                    "evidencegap.pipeline.inference_gap_analysis.validate_statement_bundle_artifact",
                    return_value={"status": "PASS"},
                ),
                patch.dict("os.environ", {}, clear=True),
                patch(
                    "evidencegap.pipeline.inference_gap_analysis.call_structured_llm"
                ) as call_llm,
            ):
                result = run_inference_gap_analysis(
                    root,
                    statement_bundle_artifact_dir=source_dir,
                    provider="deepseek",
                    run_name="empty",
                    artifact_root=root / "artifacts/gaps",
                )
            call_llm.assert_not_called()
            self.assertEqual(result["api_requests"], 0)
            self.assertEqual(result["total_inference_steps"], 0)

    def test_artifact_validator_rejects_tampered_output(self) -> None:
        statement_bundle = _statement_bundle(include_inference=False)
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            source_dir = root / "artifacts/source_bundle"
            source_dir.mkdir(parents=True)
            atomic_write_json(source_dir / "statement_bundle.json", statement_bundle)
            artifact_root = root / "artifacts/gaps"
            with patch(
                "evidencegap.pipeline.inference_gap_analysis.validate_statement_bundle_artifact",
                return_value={"status": "PASS"},
            ):
                run_inference_gap_analysis(
                    root,
                    statement_bundle_artifact_dir=source_dir,
                    provider="deepseek",
                    run_name="empty",
                    artifact_root=artifact_root,
                )
            output_path = artifact_root / "empty/inference_gap_analysis.json"
            output = json.loads(output_path.read_text(encoding="utf-8"))
            output["summary"]["scope_gaps"] = 99
            atomic_write_json(output_path, output)
            with patch(
                "evidencegap.pipeline.inference_gap_analysis.validate_statement_bundle_artifact",
                return_value={"status": "PASS"},
            ):
                with self.assertRaisesRegex(EvidenceGapError, "checksum mismatch"):
                    validate_inference_gap_analysis_artifact(artifact_root / "empty")


if __name__ == "__main__":
    unittest.main()
