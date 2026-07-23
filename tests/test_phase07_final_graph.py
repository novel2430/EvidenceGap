from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidencegap.common import EvidenceGapError, atomic_write_json
from evidencegap.pipeline.claim_aggregation import (
    aggregate_article_evidence_rows,
    run_claim_aggregation,
)
from evidencegap.pipeline.final_graph import (
    FINAL_GRAPH_CONTRACT_ID,
    FINAL_GRAPH_SCHEMA_VERSION,
    build_final_graph_bundle,
    run_final_graph,
    validate_final_graph_artifact,
)


class FinalGraphFixtureMixin:
    def _evidence(self, article_id: str, rank: int, label: str) -> dict[str, object]:
        sentence_id = f"{article_id}-sentence-{rank}"
        return {
            "evidence_id": f"evidence-{article_id}-{rank}",
            "sentence_alias": f"S{rank:02d}",
            "sentence_id": sentence_id,
            "sentence_index": rank,
            "sentence_index_within_section": rank - 1,
            "section": "results",
            "section_index": 2,
            "sentence_text": f"Direct {label} evidence from {article_id}.",
            "character_start": rank * 100,
            "character_end": rank * 100 + 40,
            "source_text_fingerprint": f"source-{article_id}",
            "splitter_fingerprint": "splitter-fixture",
        }

    def _row(
        self,
        *,
        article_id: str,
        rank: int,
        label: str,
        evidence_count: int = 1,
    ) -> dict[str, object]:
        selected = []
        if label != "insufficient":
            selected = [
                self._evidence(article_id, index, label)
                for index in range(1, evidence_count + 1)
            ]
        probabilities = {
            "support": 0.9 if label == "support" else 0.05,
            "refute": 0.9 if label == "refute" else 0.05,
            "insufficient": 0.9 if label == "insufficient" else 0.05,
        }
        if label != "insufficient":
            probabilities["insufficient"] = 0.05
        return {
            "article_id": article_id,
            "claim_id": "claim-1",
            "claim_text": "Vitamin D supplementation prevents respiratory infections.",
            "pmid": article_id.split(":", 1)[-1],
            "title": f"Article {article_id}",
            "final_article_rank": rank,
            "predicted_label": label,
            "probabilities": probabilities,
            "confidence": probabilities[label],
            "rationale": f"Article-level {label} rationale.",
            "selected_evidence": selected,
            "provider": "deepseek",
            "model": "deepseek-v4-pro",
            "model_fingerprint": "model-fixture",
            "prompt_version": "phase07_article_evidence_v3",
        }

    def _write_article_artifact(
        self, directory: Path, rows: list[dict[str, object]]
    ) -> Path:
        source = directory / "article-evidence"
        source.mkdir(parents=True)
        with (source / "article_evidence.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        atomic_write_json(source / "run_manifest.json", {"fixture": True})
        return source


class FinalGraphContractTests(unittest.TestCase, FinalGraphFixtureMixin):
    def test_bundle_contains_only_selected_grounded_evidence(self) -> None:
        rows = [
            self._row(article_id="pmid:1", rank=1, label="support"),
            self._row(article_id="pmid:2", rank=2, label="refute"),
            self._row(article_id="pmid:3", rank=3, label="insufficient"),
        ]
        result = aggregate_article_evidence_rows(rows)
        bundle = build_final_graph_bundle(rows, result)

        self.assertEqual(bundle["contract_id"], FINAL_GRAPH_CONTRACT_ID)
        self.assertEqual(bundle["schema_version"], FINAL_GRAPH_SCHEMA_VERSION)
        self.assertEqual(bundle["verdict"], "mixed")
        self.assertTrue(bundle["boundary"]["is_pipeline_final_verdict"])
        self.assertFalse(bundle["boundary"]["is_final_medical_truth"])

        node_types: dict[str, int] = {}
        for node in bundle["nodes"]:
            node_types[node["node_type"]] = node_types.get(node["node_type"], 0) + 1
        self.assertEqual(
            node_types,
            {"claim": 1, "article": 3, "evidence": 2},
        )

        relations: dict[str, int] = {}
        for edge in bundle["edges"]:
            relations[edge["relation"]] = relations.get(edge["relation"], 0) + 1
        self.assertEqual(relations["article_supports"], 1)
        self.assertEqual(relations["article_refutes"], 1)
        self.assertEqual(relations["article_insufficient"], 1)
        self.assertEqual(relations["contains_evidence"], 2)
        self.assertEqual(
            set(relations),
            {
                "article_supports",
                "article_refutes",
                "article_insufficient",
                "contains_evidence",
            },
        )

        evidence_nodes = [
            node for node in bundle["nodes"] if node["node_type"] == "evidence"
        ]
        self.assertEqual(
            {node["article_id"] for node in evidence_nodes},
            {"pmid:1", "pmid:2"},
        )
        self.assertTrue(all(node["sentence_id"] for node in evidence_nodes))
        self.assertTrue(all(node["character_start"] >= 0 for node in evidence_nodes))
        self.assertTrue(all("stance" not in node for node in evidence_nodes))

    def test_rejects_claim_result_not_derived_from_article_rows(self) -> None:
        rows = [self._row(article_id="pmid:1", rank=1, label="support")]
        result = aggregate_article_evidence_rows(rows)
        result["verdict"] = "refuted"
        with self.assertRaisesRegex(EvidenceGapError, "does not match"):
            build_final_graph_bundle(rows, result)


class FinalGraphArtifactTests(unittest.TestCase, FinalGraphFixtureMixin):
    def test_run_and_validate_final_graph_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        rows = [
            self._row(article_id="pmid:1", rank=1, label="support"),
            self._row(article_id="pmid:2", rank=2, label="refute", evidence_count=2),
            self._row(article_id="pmid:3", rank=3, label="insufficient"),
        ]
        with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
            temp = Path(temp_dir)
            article_source = self._write_article_artifact(temp, rows)
            aggregation_root = temp / "aggregation-output"
            with patch(
                "evidencegap.pipeline.claim_aggregation.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                run_claim_aggregation(
                    repo_root,
                    article_evidence_artifact_dir=article_source,
                    run_name="fixture-aggregation",
                    artifact_root=aggregation_root,
                )
            aggregation_dir = aggregation_root / "fixture-aggregation"
            graph_root = temp / "graph-output"
            with patch(
                "evidencegap.pipeline.final_graph.validate_claim_aggregation_artifact",
                return_value={"status": "PASS"},
            ), patch(
                "evidencegap.pipeline.final_graph.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                run = run_final_graph(
                    repo_root,
                    claim_aggregation_artifact_dir=aggregation_dir,
                    run_name="fixture-graph",
                    artifact_root=graph_root,
                )
            self.assertEqual(run["status"], "PASS")
            self.assertEqual(run["verdict"], "mixed")
            self.assertEqual(run["node_counts"]["evidence"], 3)

            target = graph_root / "fixture-graph"
            bundle = json.loads((target / "graph_bundle.json").read_text())
            self.assertEqual(bundle["contract_id"], FINAL_GRAPH_CONTRACT_ID)
            self.assertEqual(bundle["summary"]["article_counts"]["total"], 3)

            with patch(
                "evidencegap.pipeline.final_graph.validate_claim_aggregation_artifact",
                return_value={"status": "PASS"},
            ), patch(
                "evidencegap.pipeline.final_graph.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                validated = validate_final_graph_artifact(target)
            self.assertEqual(validated["status"], "PASS")
            self.assertEqual(validated["checksums"], "PASS")
            self.assertEqual(validated["node_counts"]["article"], 3)

    def test_validation_rejects_tampered_graph_bundle(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        rows = [self._row(article_id="pmid:1", rank=1, label="support")]
        with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
            temp = Path(temp_dir)
            article_source = self._write_article_artifact(temp, rows)
            aggregation_root = temp / "aggregation-output"
            with patch(
                "evidencegap.pipeline.claim_aggregation.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                run_claim_aggregation(
                    repo_root,
                    article_evidence_artifact_dir=article_source,
                    run_name="fixture-aggregation",
                    artifact_root=aggregation_root,
                )
            aggregation_dir = aggregation_root / "fixture-aggregation"
            graph_root = temp / "graph-output"
            with patch(
                "evidencegap.pipeline.final_graph.validate_claim_aggregation_artifact",
                return_value={"status": "PASS"},
            ), patch(
                "evidencegap.pipeline.final_graph.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ):
                run_final_graph(
                    repo_root,
                    claim_aggregation_artifact_dir=aggregation_dir,
                    run_name="fixture-graph",
                    artifact_root=graph_root,
                )
            target = graph_root / "fixture-graph"
            bundle_path = target / "graph_bundle.json"
            bundle_path.write_text(bundle_path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(EvidenceGapError, "checksum mismatch"):
                validate_final_graph_artifact(target)


if __name__ == "__main__":
    unittest.main()
