from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidencegap.common import EvidenceGapError, atomic_write_json
from evidencegap.pipeline.claim_aggregation import (
    CLAIM_AGGREGATION_CONTRACT_ID,
    CLAIM_AGGREGATION_SCHEMA_VERSION,
    aggregate_article_evidence_rows,
    run_claim_aggregation,
    validate_claim_aggregation_artifact,
)


class ClaimAggregationFixtureMixin:
    def _row(
        self,
        *,
        article_id: str,
        rank: int,
        label: str,
    ) -> dict[str, object]:
        evidence = []
        if label != "insufficient":
            evidence = [
                {
                    "sentence_id": f"{article_id}-sentence",
                    "sentence_text": f"Direct {label} evidence.",
                }
            ]
        return {
            "article_id": article_id,
            "claim_id": "claim-1",
            "claim_text": "Vitamin D supplementation prevents respiratory infections.",
            "final_article_rank": rank,
            "predicted_label": label,
            "selected_evidence": evidence,
        }

    def _write_source_artifact(
        self, directory: Path, rows: list[dict[str, object]]
    ) -> Path:
        source = directory / "article-evidence"
        source.mkdir(parents=True)
        with (source / "article_evidence.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        atomic_write_json(source / "run_manifest.json", {"fixture": True})
        return source


class ClaimAggregationRuleTests(unittest.TestCase, ClaimAggregationFixtureMixin):
    def test_four_deterministic_verdict_branches(self) -> None:
        cases = {
            "supported": ["support", "insufficient"],
            "refuted": ["refute", "insufficient"],
            "mixed": ["support", "refute", "insufficient"],
            "insufficient": ["insufficient", "insufficient"],
        }
        for expected, labels in cases.items():
            with self.subTest(expected=expected):
                rows = [
                    self._row(article_id=f"pmid:{index}", rank=index, label=label)
                    for index, label in enumerate(labels, start=1)
                ]
                result = aggregate_article_evidence_rows(rows)
                self.assertEqual(result["verdict"], expected)
                self.assertEqual(result["article_counts"]["total"], len(rows))
                self.assertEqual(result["scope"], "retrieved_top_articles")

    def test_mixed_result_preserves_rank_order_within_stance_groups(self) -> None:
        rows = [
            self._row(article_id="pmid:30", rank=3, label="support"),
            self._row(article_id="pmid:10", rank=1, label="refute"),
            self._row(article_id="pmid:20", rank=2, label="support"),
        ]
        result = aggregate_article_evidence_rows(rows)
        self.assertEqual(result["verdict"], "mixed")
        self.assertEqual(result["support_article_ids"], ["pmid:20", "pmid:30"])
        self.assertEqual(result["refute_article_ids"], ["pmid:10"])
        self.assertEqual(result["insufficient_article_ids"], [])

    def test_rejects_multiple_claims_and_invalid_evidence_shape(self) -> None:
        rows = [
            self._row(article_id="pmid:1", rank=1, label="support"),
            self._row(article_id="pmid:2", rank=2, label="refute"),
        ]
        rows[1]["claim_id"] = "claim-2"
        with self.assertRaisesRegex(EvidenceGapError, "multiple claims"):
            aggregate_article_evidence_rows(rows)

        bad = self._row(article_id="pmid:3", rank=1, label="support")
        bad["selected_evidence"] = []
        with self.assertRaisesRegex(EvidenceGapError, "must contain selected evidence"):
            aggregate_article_evidence_rows([bad])


class ClaimAggregationArtifactTests(unittest.TestCase, ClaimAggregationFixtureMixin):
    def test_run_and_validate_claim_aggregation_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        rows = [
            self._row(article_id="pmid:1", rank=1, label="support"),
            self._row(article_id="pmid:2", rank=2, label="refute"),
            self._row(article_id="pmid:3", rank=3, label="insufficient"),
        ]
        with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
            temp = Path(temp_dir)
            source = self._write_source_artifact(temp, rows)
            artifact_root = temp / "aggregation-output"
            with patch(
                "evidencegap.pipeline.claim_aggregation.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                run = run_claim_aggregation(
                    repo_root,
                    article_evidence_artifact_dir=source,
                    run_name="fixture-mixed",
                    artifact_root=artifact_root,
                )
            self.assertEqual(run["verdict"], "mixed")
            target = artifact_root / "fixture-mixed"
            result = json.loads((target / "claim_result.json").read_text())
            self.assertEqual(result["contract_id"], CLAIM_AGGREGATION_CONTRACT_ID)
            self.assertEqual(result["schema_version"], CLAIM_AGGREGATION_SCHEMA_VERSION)
            self.assertEqual(
                result["article_counts"],
                {"total": 3, "support": 1, "refute": 1, "insufficient": 1},
            )
            with patch(
                "evidencegap.pipeline.claim_aggregation.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                validated = validate_claim_aggregation_artifact(target)
            self.assertEqual(validated["status"], "PASS")
            self.assertEqual(validated["verdict"], "mixed")
            self.assertEqual(validated["checksums"], "PASS")

    def test_validation_rejects_tampered_claim_result(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        rows = [self._row(article_id="pmid:1", rank=1, label="support")]
        with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
            temp = Path(temp_dir)
            source = self._write_source_artifact(temp, rows)
            artifact_root = temp / "aggregation-output"
            with patch(
                "evidencegap.pipeline.claim_aggregation.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                run_claim_aggregation(
                    repo_root,
                    article_evidence_artifact_dir=source,
                    run_name="fixture-supported",
                    artifact_root=artifact_root,
                )
            target = artifact_root / "fixture-supported"
            result_path = target / "claim_result.json"
            result_path.write_text(result_path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceGapError, "checksum mismatch"):
                validate_claim_aggregation_artifact(target)


if __name__ == "__main__":
    unittest.main()
