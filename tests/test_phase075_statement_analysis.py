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

from evidencegap.common import (  # noqa: E402
    EvidenceGapError,
    atomic_write_json,
    relative_path,
    sha256_file,
)
from evidencegap.pipeline.statement_analysis import (  # noqa: E402
    STATEMENT_ANALYSIS_CONTRACT_ID,
    run_statement_analysis,
    validate_statement_analysis_artifact,
    validate_statement_analysis_bundle,
)
from evidencegap.pipeline.statement_decomposition import (  # noqa: E402
    STATEMENT_DECOMPOSITION_CONTRACT_ID,
    STATEMENT_DECOMPOSITION_SCHEMA_VERSION,
    validate_response_payload,
)


def _write_decomposition_artifact(
    root: Path, *, statement: str, payload: dict[str, object], run_name: str = "input"
) -> tuple[Path, dict[str, object]]:
    artifact_dir = root / "artifacts/decomposition" / run_name
    artifact_dir.mkdir(parents=True)
    bundle = validate_response_payload(payload, original_statement=statement)
    output_path = artifact_dir / "decomposition.json"
    atomic_write_json(output_path, bundle)
    atomic_write_json(
        artifact_dir / "run_manifest.json",
        {
            "schema_version": STATEMENT_DECOMPOSITION_SCHEMA_VERSION,
            "contract_id": STATEMENT_DECOMPOSITION_CONTRACT_ID,
            "run_type": "fixture",
            "run_name": run_name,
            "provider": "fixture",
            "model": "fixture",
            "counts": {
                "claims": len(bundle["claims"]),
                "inference_steps": len(bundle["inference_steps"]),
            },
            "outputs": {
                "decomposition": {
                    "path": relative_path(root, output_path),
                    "sha256": sha256_file(output_path),
                }
            },
        },
    )
    return artifact_dir, bundle


class StatementAnalysisTests(unittest.TestCase):
    def test_empty_decomposition_completes_without_running_phase07(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            decomposition_dir, _ = _write_decomposition_artifact(
                root,
                statement="健康是基本人權，所以政府應免費提供所有保健品。",
                payload={
                    "source_language": "zh-TW",
                    "claims": [],
                    "inference_steps": [],
                },
            )
            with patch(
                "evidencegap.pipeline.statement_analysis.run_analysis"
            ) as run_phase07:
                result = run_statement_analysis(
                    root,
                    decomposition_artifact_dir=decomposition_dir,
                    run_name="empty",
                    provider="deepseek",
                    artifact_root=root / "artifacts/statement_analysis",
                )

            run_phase07.assert_not_called()
            self.assertEqual(result["analysis_status"], "completed")
            self.assertTrue(result["empty_claims"])
            self.assertEqual(result["total_claims"], 0)
            validation = validate_statement_analysis_artifact(
                root / "artifacts/statement_analysis/empty"
            )
            self.assertEqual(validation["status"], "PASS")
            self.assertTrue(validation["empty_claims"])

    def test_claim_failure_is_isolated_and_later_claims_continue(self) -> None:
        statement = "甲能降低感染。乙能縮短病程。丙會增加出血風險。"
        payload = {
            "source_language": "zh-TW",
            "claims": [
                {
                    "claim_ref": "C1",
                    "source_text": "甲能降低感染",
                    "canonical_claim_en": "Treatment A reduces infections.",
                },
                {
                    "claim_ref": "C2",
                    "source_text": "乙能縮短病程",
                    "canonical_claim_en": "Treatment B shortens disease duration.",
                },
                {
                    "claim_ref": "C3",
                    "source_text": "丙會增加出血風險",
                    "canonical_claim_en": "Treatment C increases bleeding risk.",
                },
            ],
            "inference_steps": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            decomposition_dir, decomposition = _write_decomposition_artifact(
                root,
                statement=statement,
                payload=payload,
            )
            statement_root = root / "artifacts/statement_analysis"
            calls: list[str] = []

            def fake_run_analysis(*args: object, **kwargs: object) -> dict[str, object]:
                claim = str(kwargs["claim"])
                claim_id = str(kwargs["run_name"])
                calls.append(claim)
                if "Treatment B" in claim:
                    raise EvidenceGapError("fixture Phase 07 failure")
                claim_dir = Path(kwargs["artifact_root"]) / claim_id
                graph_path = claim_dir / "final_graph/graph_bundle.json"
                graph_path.parent.mkdir(parents=True)
                atomic_write_json(graph_path, {"claim_id": claim_id})
                verdict = "supported" if "Treatment A" in claim else "refuted"
                return {
                    "status": "PASS",
                    "claim_id": claim_id,
                    "verdict": verdict,
                    "graph_bundle_path": relative_path(root, graph_path),
                }

            with patch(
                "evidencegap.pipeline.statement_analysis.run_analysis",
                side_effect=fake_run_analysis,
            ):
                result = run_statement_analysis(
                    root,
                    decomposition_artifact_dir=decomposition_dir,
                    run_name="partial",
                    provider="deepseek",
                    artifact_root=statement_root,
                )

            self.assertEqual(
                calls,
                [claim["canonical_claim_en"] for claim in decomposition["claims"]],
            )
            self.assertEqual(result["analysis_status"], "partial_failure")
            self.assertEqual(result["completed_claims"], 2)
            self.assertEqual(result["failed_claims"], 1)

            output_dir = statement_root / "partial"
            bundle = json.loads(
                (output_dir / "statement_result.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                [row["status"] for row in bundle["claim_results"]],
                ["completed", "failed", "completed"],
            )
            self.assertEqual(
                bundle["claim_results"][1]["error"],
                "fixture Phase 07 failure",
            )
            self.assertIsNone(bundle["claim_results"][1]["verdict"])

            phase07_by_id = {
                row["claim_id"]: row
                for row in bundle["claim_results"]
                if row["status"] == "completed"
            }

            def fake_validate_phase07(path: Path) -> dict[str, object]:
                row = phase07_by_id[path.name]
                return {
                    "status": "PASS",
                    "claim_id": row["claim_id"],
                    "verdict": row["verdict"],
                }

            with patch(
                "evidencegap.pipeline.statement_analysis.validate_analysis_artifact",
                side_effect=fake_validate_phase07,
            ) as validate_phase07:
                validation = validate_statement_analysis_artifact(output_dir)
            self.assertEqual(validate_phase07.call_count, 2)
            self.assertEqual(validation["analysis_status"], "partial_failure")
            self.assertEqual(validation["checksums"], "PASS")

    def test_all_claim_failures_produce_a_valid_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            decomposition_dir, _ = _write_decomposition_artifact(
                root,
                statement="甲能降低感染。",
                payload={
                    "source_language": "zh-TW",
                    "claims": [
                        {
                            "claim_ref": "C1",
                            "source_text": "甲能降低感染",
                            "canonical_claim_en": "Treatment A reduces infections.",
                        }
                    ],
                    "inference_steps": [],
                },
            )
            with patch(
                "evidencegap.pipeline.statement_analysis.run_analysis",
                side_effect=EvidenceGapError("fixture failure"),
            ):
                result = run_statement_analysis(
                    root,
                    decomposition_artifact_dir=decomposition_dir,
                    run_name="failed",
                    provider="deepseek",
                    artifact_root=root / "artifacts/statement_analysis",
                )
            self.assertEqual(result["status"], "FAILED")
            self.assertEqual(result["artifact_status"], "PASS")
            self.assertEqual(result["analysis_status"], "failed")
            self.assertEqual(result["failed_claims"], 1)

    def test_bundle_rejects_failed_claim_with_a_verdict(self) -> None:
        with self.assertRaises(EvidenceGapError):
            validate_statement_analysis_bundle(
                {
                    "schema_version": "1.0.0",
                    "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
                    "statement_id": "statement_fixture",
                    "analysis_status": "failed",
                    "claim_results": [
                        {
                            "claim_id": "claim_fixture",
                            "source_text": "fixture",
                            "canonical_claim_en": "A biomedical fixture claim.",
                            "status": "failed",
                            "phase07_artifact_dir": None,
                            "graph_bundle_path": None,
                            "verdict": "supported",
                            "error": "failed",
                        }
                    ],
                    "summary": {
                        "total_claims": 1,
                        "completed_claims": 0,
                        "failed_claims": 1,
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
