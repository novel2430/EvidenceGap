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

from evidencegap.common import EvidenceGapError
from evidencegap.stance.graph_contracts import (
    GRAPH_CONTRACT_ID,
    aggregate_prediction_rows,
    build_graph_bundle,
    build_graph_rows,
    validate_graph_rows,
)


def prediction_row(
    *,
    rank: int,
    label: str,
    probabilities: tuple[float, float, float],
    evidence_type: str = "direct_result",
    query_id: str = "evidencebench:train_4",
    paper_id: str = "pmc_1343553",
) -> dict[str, object]:
    support, refute, insufficient = probabilities
    confidence = {
        "support": support,
        "refute": refute,
        "insufficient": insufficient,
    }[label]
    ordered = sorted(probabilities, reverse=True)
    return {
        "schema_version": "1.0.0",
        "task_id": "STANCE-EVIDENCE-3",
        "record_type": "StancePredictionRecord",
        "input_id": f"stance:phase05:{query_id}:{rank - 1}",
        "dataset": "evidencebench_100k",
        "split": "dev",
        "claim_id": query_id,
        "query_id": query_id,
        "claim_text": "The intervention improves the outcome.",
        "paper_id": paper_id,
        "sentence_index": rank - 1,
        "sentence_type": "normal paragraph",
        "evidence_rank": rank,
        "evidence_text": f"Evidence sentence {rank}.",
        "evidence_unit": "sentence",
        "context_before": None,
        "context_after": None,
        "retrieval_model": "rrf:bmretriever+medcpt",
        "retrieval_score": 0.15 - rank * 0.01,
        "cross_encoder_score": None,
        "source_run_name": "phase05-frozen",
        "source_artifact_sha256": "a" * 64,
        "source_locator_json": "{}",
        "run_name": "phase06-predictions",
        "model_name": "deepseek:deepseek-v4-pro",
        "model_fingerprint": "b" * 64,
        "stance_input_artifact_sha256": "c" * 64,
        "predicted_label": label,
        "probability_support": support,
        "probability_refute": refute,
        "probability_insufficient": insufficient,
        "confidence": confidence,
        "probability_margin": ordered[0] - ordered[1],
        "abstained": False,
        "rationale": "The sentence is classified from the supplied evidence.",
        "evidence_type": evidence_type,
        "requires_context": False,
        "provider": "deepseek",
        "provider_request_id": "request-1",
        "raw_response_sha256": "d" * 64,
        "prompt_version": "phase06_llm_stance_v2",
    }


class Phase06GraphExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            prediction_row(
                rank=1,
                label="support",
                probabilities=(0.90, 0.02, 0.08),
            ),
            prediction_row(
                rank=2,
                label="insufficient",
                probabilities=(0.10, 0.10, 0.80),
                evidence_type="background",
            ),
            prediction_row(
                rank=3,
                label="refute",
                probabilities=(0.05, 0.70, 0.25),
                evidence_type="statistical_uncertainty",
            ),
        ]

    def test_aggregation_is_transparent_and_conflict_aware(self) -> None:
        query_summaries, paper_summaries, groups = aggregate_prediction_rows(self.rows)
        self.assertEqual(len(query_summaries), 1)
        self.assertEqual(len(paper_summaries), 1)
        self.assertEqual(len(groups), 1)
        summary = query_summaries[0]
        self.assertEqual(summary["directional_evidence_pattern"], "mixed")
        self.assertTrue(summary["has_conflict"])
        self.assertEqual(summary["support_count"], 1)
        self.assertEqual(summary["refute_count"], 1)
        self.assertEqual(summary["insufficient_count"], 1)
        self.assertAlmostEqual(summary["support_mass"], 0.90 + 0.05 + 0.05 / 3)
        self.assertAlmostEqual(summary["refute_mass"], 0.02 + 0.05 + 0.70 / 3)
        self.assertAlmostEqual(summary["insufficient_mass"], 0.08 + 0.40 + 0.25 / 3)
        directional_total = summary["support_mass"] + summary["refute_mass"]
        total_mass = directional_total + summary["insufficient_mass"]
        self.assertAlmostEqual(
            summary["directional_mass_share"], directional_total / total_mass
        )
        self.assertEqual(summary["top_support_input_id"], self.rows[0]["input_id"])
        self.assertEqual(summary["top_refute_input_id"], self.rows[2]["input_id"])

    def test_directional_pattern_does_not_imply_overall_support(self) -> None:
        rows = [
            prediction_row(
                rank=1,
                label="insufficient",
                probabilities=(0.10, 0.05, 0.85),
                evidence_type="background",
            ),
            prediction_row(
                rank=2,
                label="support",
                probabilities=(0.60, 0.05, 0.35),
            ),
            prediction_row(
                rank=3,
                label="insufficient",
                probabilities=(0.05, 0.05, 0.90),
                evidence_type="method",
            ),
        ]
        summary = aggregate_prediction_rows(rows)[0][0]
        self.assertEqual(summary["directional_evidence_pattern"], "support_only")
        self.assertEqual(summary["mass_leader"], "insufficient")
        self.assertLess(summary["directional_mass_share"], 0.5)
        self.assertNotIn("evidence_landscape", summary)

    def test_graph_has_expected_nodes_edges_and_provenance(self) -> None:
        graph_id = str(self.rows[0]["query_id"])
        nodes, edges = build_graph_rows(graph_id, self.rows)
        self.assertEqual(len(nodes), 5)  # claim + article + 3 evidence
        self.assertEqual(len(edges), 7)  # retrieved_from + 3 contains + 3 stance
        relation_counts = {}
        for edge in edges:
            relation_counts[edge["relation"]] = relation_counts.get(edge["relation"], 0) + 1
        self.assertEqual(relation_counts["retrieved_from"], 1)
        self.assertEqual(relation_counts["contains"], 3)
        self.assertEqual(relation_counts["supports"], 1)
        self.assertEqual(relation_counts["refutes"], 1)
        self.assertEqual(relation_counts["insufficient"], 1)
        article_edge = next(edge for edge in edges if edge["relation"] == "retrieved_from")
        self.assertIsNone(article_edge["retrieval_model"])
        self.assertIsNone(article_edge["retrieval_score"])
        evidence_nodes = [node for node in nodes if node["node_type"] == "evidence"]
        self.assertEqual({node["sentence_index"] for node in evidence_nodes}, {0, 1, 2})
        stance_edges = [edge for edge in edges if edge["stance_label"] is not None]
        self.assertTrue(all(edge["model_fingerprint"] == "b" * 64 for edge in stance_edges))
        validation = validate_graph_rows(
            [aggregate_prediction_rows(self.rows)[0][0]],
            [aggregate_prediction_rows(self.rows)[1][0]],
            nodes,
            edges,
        )
        self.assertEqual(validation["status"], "PASS")
        self.assertEqual(validation["evidence_nodes"], 3)

    def test_bundle_explicitly_is_not_final_verdict(self) -> None:
        summaries, papers, groups = aggregate_prediction_rows(self.rows)
        graph_id = next(iter(groups))
        nodes, edges = build_graph_rows(graph_id, groups[graph_id])
        bundle = build_graph_bundle(summaries[0], papers, nodes, edges)
        self.assertEqual(bundle["contract_id"], GRAPH_CONTRACT_ID)
        self.assertFalse(bundle["boundary"]["is_final_medical_verdict"])
        self.assertNotIn("verdict", bundle["summary"])

    def test_rank_gap_is_rejected(self) -> None:
        rows = [self.rows[0], self.rows[2]]
        with self.assertRaises(EvidenceGapError):
            aggregate_prediction_rows(rows)

    def test_missing_stance_edge_is_rejected(self) -> None:
        summaries, papers, groups = aggregate_prediction_rows(self.rows)
        graph_id = next(iter(groups))
        nodes, edges = build_graph_rows(graph_id, groups[graph_id])
        stance_edge_index = next(
            index
            for index, edge in enumerate(edges)
            if edge["relation"] in {"supports", "refutes", "insufficient"}
        )
        del edges[stance_edge_index]
        with self.assertRaises(EvidenceGapError):
            validate_graph_rows(summaries, papers, nodes, edges)


class Phase06GraphExportIntegrationTests(unittest.TestCase):
    @patch("evidencegap.stance.graph_export.validate_prediction_artifact")
    @patch("evidencegap.stance.graph_export.iter_prediction_rows")
    @patch("evidencegap.stance.graph_export.write_summary_rows_atomic")
    @patch("evidencegap.stance.graph_export.write_node_rows_atomic")
    @patch("evidencegap.stance.graph_export.write_edge_rows_atomic")
    @patch("evidencegap.stance.graph_export.write_jsonl_atomic")
    def test_export_is_offline_and_preserves_partial_source(
        self,
        write_jsonl,
        write_edges,
        write_nodes,
        write_summaries,
        iter_rows,
        validate_predictions,
    ) -> None:
        from evidencegap.stance.graph_export import export_graph_ready_stance

        rows = [
            prediction_row(
                rank=1,
                label="support",
                probabilities=(0.9, 0.02, 0.08),
            )
        ]

        def fake_writer(path: Path, values: object) -> int:
            path.write_text("fixture", encoding="utf-8")
            return len(list(values))  # type: ignore[arg-type]

        write_summaries.side_effect = fake_writer
        write_nodes.side_effect = fake_writer
        write_edges.side_effect = fake_writer
        write_jsonl.side_effect = fake_writer
        iter_rows.return_value = iter(rows)
        validate_predictions.return_value = {
            "sha256": "e" * 64,
            "run_name": "phase06-predictions",
            "model_name": "deepseek:deepseek-v4-pro",
            "model_fingerprint": "b" * 64,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            prediction_path = prediction_dir / "stance_predictions.parquet"
            prediction_path.write_text("fixture", encoding="utf-8")
            (prediction_dir / "run_manifest.json").write_text(
                json.dumps({"partial": True, "coverage": {"query_percent": 57.6}}),
                encoding="utf-8",
            )
            result = export_graph_ready_stance(
                root,
                prediction_path=prediction_path,
                run_name="fixture-graph",
            )

            self.assertEqual(result["api_requests"], 0)
            self.assertTrue(result["source_partial"])
            self.assertEqual(result["queries"], 1)
            manifest = json.loads(
                (
                    root
                    / "artifacts/v1/stance_verification/graph_ready/fixture-graph/run_manifest.json"
                ).read_text(encoding="utf-8")
            )
            self.assertFalse(manifest["aggregation"]["is_final_medical_verdict"])
            self.assertEqual(manifest["source_coverage"]["query_percent"], 57.6)



if __name__ == "__main__":
    unittest.main()
