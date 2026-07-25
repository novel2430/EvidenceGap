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

from evidencegap.common import EvidenceGapError, atomic_write_json, sha256_file  # noqa: E402
from evidencegap.output.presentation import (  # noqa: E402
    LOCALIZATION_PROMPT_VERSION,
    SYSTEM_PROMPT,
    _translation_units,
    build_presentation_bundle,
    run_output_module,
    validate_localization_response,
    validate_output_artifact,
)
from evidencegap.pipeline.inference_gap_analysis import build_inference_gap_analysis_bundle  # noqa: E402
from evidencegap.pipeline.statement_bundle import (  # noqa: E402
    STATEMENT_BUNDLE_CONTRACT_ID,
    STATEMENT_BUNDLE_SCHEMA_VERSION,
    validate_statement_bundle,
)
from evidencegap.pipeline.statement_decomposition import runtime_inference_step_id  # noqa: E402
from evidencegap.stance.llm_judge import ProviderResponse, _ProviderError  # noqa: E402


def _statement_bundle() -> dict[str, object]:
    claim_ids = ["claim_1", "claim_2", "claim_3"]
    verdicts = ["supported", "mixed", "insufficient"]
    claims, articles, evidence = [], [], []
    for index, (claim_id, verdict) in enumerate(zip(claim_ids, verdicts, strict=True), 1):
        article_id = f"article_{index}"
        evidence_id = f"evidence_{index}"
        stance = "insufficient" if verdict == "insufficient" else "support"
        claims.append(
            {
                "claim_id": claim_id,
                "source_text": f"來源主張 {index}",
                "canonical_claim_en": f"Biomedical claim {index}.",
                "analysis_status": "completed",
                "verdict": verdict,
                "article_counts": {"total": 1, "support": int(stance == "support"), "refute": 0, "insufficient": int(stance == "insufficient")},
                "rationale": f"Claim rationale {index}.",
                "scope": "retrieved_top_articles",
                "boundary": {"is_pipeline_final_verdict": True, "is_final_medical_truth": False, "description": "Fixture boundary."},
                "article_node_ids": [article_id],
                "error": None,
            }
        )
        articles.append(
            {
                "article_node_id": article_id,
                "claim_id": claim_id,
                "article_id": f"pmid:{index}",
                "pmid": str(index),
                "rank": 1,
                "title": f"Article {index}",
                "rationale": f"Article rationale {index}.",
                "stance": stance,
                "confidence": 0.9,
                "probabilities": {"support": 0.9 if stance == "support" else 0.05, "refute": 0.05, "insufficient": 0.05 if stance == "support" else 0.9},
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
                "article_node_id": article_id,
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
    step = {
        "inference_step_id": runtime_inference_step_id([claim_ids[1]], claim_ids[2]),
        "premise_claim_ids": [claim_ids[1]],
        "conclusion_claim_id": claim_ids[2],
    }
    bundle = {
        "schema_version": STATEMENT_BUNDLE_SCHEMA_VERSION,
        "contract_id": STATEMENT_BUNDLE_CONTRACT_ID,
        "statement": {"statement_id": "statement_fixture", "original_text": "原始生醫論述。", "source_language": "zh-TW", "analysis_status": "completed"},
        "claims": claims,
        "inference_steps": [step],
        "articles": articles,
        "evidence": evidence,
        "summary": {"total_claims": 3, "completed_claims": 3, "failed_claims": 0, "articles": 3, "evidence": 3},
    }
    validate_statement_bundle(bundle)
    return bundle


def _write_sources(root: Path) -> tuple[Path, Path, dict[str, object], dict[str, object]]:
    statement_dir = root / "artifacts/statement"
    gap_dir = root / "artifacts/gaps"
    statement_dir.mkdir(parents=True)
    gap_dir.mkdir(parents=True)
    statement = _statement_bundle()
    atomic_write_json(statement_dir / "statement_bundle.json", statement)
    statement_sha = sha256_file(statement_dir / "statement_bundle.json")
    step_id = statement["inference_steps"][0]["inference_step_id"]
    gaps = build_inference_gap_analysis_bundle(
        statement,
        [{"inference_step_id": step_id, "scope_gap": {"detected": True, "reason": "The conclusion extends beyond the studied population."}, "causal_gap": {"detected": False, "reason": None}}],
        source_statement_bundle_sha256=statement_sha,
    )
    atomic_write_json(gap_dir / "inference_gap_analysis.json", gaps)
    return statement_dir, gap_dir, statement, gaps


def _response(payload: dict[str, object]) -> ProviderResponse:
    return ProviderResponse(payload=payload, request_id="request-fixture", usage={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}, raw_response_sha256="raw-fixture", finish_reason="stop")


class OutputModuleTests(unittest.TestCase):
    def test_build_maps_states_roles_and_gaps(self) -> None:
        statement = _statement_bundle()
        gaps = build_inference_gap_analysis_bundle(
            statement,
            [{"inference_step_id": statement["inference_steps"][0]["inference_step_id"], "scope_gap": {"detected": True, "reason": "Scope mismatch."}, "causal_gap": {"detected": False, "reason": None}}],
            source_statement_bundle_sha256="statement-sha",
        )
        output = build_presentation_bundle(statement, gaps, output_language="English", statement_bundle_sha256="statement-sha", gap_bundle_sha256="gap-sha")
        self.assertFalse(output["localized"])
        self.assertEqual([row["evidence_state"] for row in output["claims"]], ["SUPPORTED", "CONFLICTED", "INSUFFICIENT"])
        self.assertEqual([row["argument_role"] for row in output["claims"]], ["STANDALONE", "PREMISE", "CONCLUSION"])
        self.assertEqual(output["inference_steps"][0]["gaps"][0]["gap_type"], "SCOPE_GAP")

    def test_translation_contract_is_exact(self) -> None:
        self.assertEqual(LOCALIZATION_PROMPT_VERSION, "phase077_output_localization_v2")
        self.assertIn("Translate only the supplied text values", SYSTEM_PROMPT)
        units = [{"text_id": "a", "text": "A"}, {"text_id": "b", "text": "B"}]
        self.assertEqual(validate_localization_response({"translations": [{"text_id": "a", "text": "甲"}, {"text_id": "b", "text": "乙"}]}, units), {"a": "甲", "b": "乙"})
        with self.assertRaisesRegex(_ProviderError, "Missing translations"):
            validate_localization_response({"translations": []}, units)

    def test_english_default_skips_llm(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src/evidencegap").mkdir(parents=True)
            statement_dir, gap_dir, _, _ = _write_sources(root)
            with (
                patch("evidencegap.output.presentation.validate_statement_bundle_artifact", return_value={"status": "PASS"}),
                patch("evidencegap.output.presentation.validate_inference_gap_analysis_artifact", return_value={"status": "PASS"}),
                patch.dict("os.environ", {}, clear=True),
                patch("evidencegap.output.presentation.call_structured_llm") as call_llm,
            ):
                result = run_output_module(root, statement_bundle_artifact_dir=statement_dir, inference_gap_artifact_dir=gap_dir, run_name="english", artifact_root=root / "artifacts/output")
            call_llm.assert_not_called()
            self.assertEqual(result["api_requests"], 0)
            output = json.loads((root / "artifacts/output/english/presentation_bundle.json").read_text())
            self.assertEqual(output["claims"][0]["display_text"], "Biomedical claim 1.")

    def test_non_english_uses_one_structured_llm_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src/evidencegap").mkdir(parents=True)
            statement_dir, gap_dir, statement, gaps = _write_sources(root)
            base = build_presentation_bundle(statement, gaps, output_language="繁體中文（台灣）", statement_bundle_sha256=sha256_file(statement_dir / "statement_bundle.json"), gap_bundle_sha256=sha256_file(gap_dir / "inference_gap_analysis.json"))
            units = _translation_units(base)
            response = _response({"translations": [{"text_id": row["text_id"], "text": f"譯文 {index}"} for index, row in enumerate(units, 1)]})
            with (
                patch("evidencegap.output.presentation.validate_statement_bundle_artifact", return_value={"status": "PASS"}),
                patch("evidencegap.output.presentation.validate_inference_gap_analysis_artifact", return_value={"status": "PASS"}),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key"}),
                patch("evidencegap.output.presentation.call_structured_llm", return_value=response) as call_llm,
            ):
                result = run_output_module(root, statement_bundle_artifact_dir=statement_dir, inference_gap_artifact_dir=gap_dir, run_name="zh", language="繁體中文（台灣）", artifact_root=root / "artifacts/output")
            self.assertEqual(call_llm.call_count, 1)
            self.assertFalse(call_llm.call_args.kwargs["thinking"])
            self.assertEqual(result["api_requests"], 1)
            output = json.loads((root / "artifacts/output/zh/presentation_bundle.json").read_text())
            self.assertEqual(output["claims"][0]["display_text"], "譯文 2")
            self.assertEqual(output["claims"][0]["canonical_claim_en"], "Biomedical claim 1.")

    def test_non_english_localization_is_batched(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src/evidencegap").mkdir(parents=True)
            statement_dir, gap_dir, statement, gaps = _write_sources(root)
            base = build_presentation_bundle(
                statement,
                gaps,
                output_language="繁體中文（台灣）",
                statement_bundle_sha256=sha256_file(
                    statement_dir / "statement_bundle.json"
                ),
                gap_bundle_sha256=sha256_file(
                    gap_dir / "inference_gap_analysis.json"
                ),
            )
            units = _translation_units(base)
            batches = [units[index : index + 2] for index in range(0, len(units), 2)]
            responses = [
                _response(
                    {
                        "translations": [
                            {
                                "text_id": row["text_id"],
                                "text": f"批次譯文 {batch_index}-{row_index}",
                            }
                            for row_index, row in enumerate(batch, 1)
                        ]
                    }
                )
                for batch_index, batch in enumerate(batches, 1)
            ]
            with (
                patch(
                    "evidencegap.output.presentation.validate_statement_bundle_artifact",
                    return_value={"status": "PASS"},
                ),
                patch(
                    "evidencegap.output.presentation.validate_inference_gap_analysis_artifact",
                    return_value={"status": "PASS"},
                ),
                patch.dict("os.environ", {"DEEPSEEK_API_KEY": "fixture-key"}),
                patch(
                    "evidencegap.output.presentation.call_structured_llm",
                    side_effect=responses,
                ) as call_llm,
            ):
                result = run_output_module(
                    root,
                    statement_bundle_artifact_dir=statement_dir,
                    inference_gap_artifact_dir=gap_dir,
                    run_name="zh-batched",
                    language="繁體中文（台灣）",
                    request_batch_size=2,
                    artifact_root=root / "artifacts/output",
                )
            self.assertEqual(call_llm.call_count, len(batches))
            self.assertEqual(result["api_requests"], len(batches))
            manifest = json.loads(
                (
                    root
                    / "artifacts/output/zh-batched/run_manifest.json"
                ).read_text()
            )
            self.assertEqual(manifest["counts"]["api_requests"], len(batches))
            self.assertEqual(manifest["counts"]["translation_units"], len(units))
            self.assertEqual(len(manifest["provider_responses"]), len(batches))
            self.assertEqual(
                manifest["usage"],
                {
                    "input_tokens": 100 * len(batches),
                    "output_tokens": 50 * len(batches),
                    "total_tokens": 150 * len(batches),
                },
            )

    def test_validator_rejects_tampered_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "src/evidencegap").mkdir(parents=True)
            statement_dir, gap_dir, _, _ = _write_sources(root)
            patches = (
                patch("evidencegap.output.presentation.validate_statement_bundle_artifact", return_value={"status": "PASS"}),
                patch("evidencegap.output.presentation.validate_inference_gap_analysis_artifact", return_value={"status": "PASS"}),
            )
            with patches[0], patches[1]:
                run_output_module(root, statement_bundle_artifact_dir=statement_dir, inference_gap_artifact_dir=gap_dir, run_name="english", artifact_root=root / "artifacts/output")
            output_dir = root / "artifacts/output/english"
            path = output_dir / "presentation_bundle.json"
            value = json.loads(path.read_text())
            value["summary"]["total_claims"] = 99
            atomic_write_json(path, value)
            with (
                patch("evidencegap.output.presentation.validate_statement_bundle_artifact", return_value={"status": "PASS"}),
                patch("evidencegap.output.presentation.validate_inference_gap_analysis_artifact", return_value={"status": "PASS"}),
            ):
                with self.assertRaisesRegex(EvidenceGapError, "checksum mismatch"):
                    validate_output_artifact(output_dir)


if __name__ == "__main__":
    unittest.main()
