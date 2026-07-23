from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path, sha256_file
from evidencegap.pipeline.analysis import (
    ANALYSIS_CONTRACT_ID,
    ANALYSIS_SCHEMA_VERSION,
    run_analysis,
    validate_analysis_artifact,
)
from evidencegap.pipeline.retrieval_adapters import (
    RUNTIME_RETRIEVAL_CONTRACT_ID,
    RUNTIME_RETRIEVAL_SCHEMA_VERSION,
)


class AnalysisOrchestrationTests(unittest.TestCase):
    def test_run_analysis_wires_only_new_runtime_path(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
            artifact_root = Path(temp_dir) / "analysis-output"
            target = artifact_root / "fixture"

            def article_stage(*args, **kwargs):
                stage = kwargs["artifact_dir"]
                stage.mkdir(parents=True)
                atomic_write_json(stage / "run_manifest.json", {"stage": "article"})
                (stage / "runtime_articles.jsonl").write_text("{}\n", encoding="utf-8")
                return {
                    "top_articles": 10,
                    "outputs": {
                        "runtime_articles_input": {
                            "path": relative_path(
                                repo_root, stage / "runtime_articles.jsonl"
                            )
                        }
                    },
                }

            def sentence_stage(*args, **kwargs):
                stage = kwargs["artifact_root"] / kwargs["run_name"]
                stage.mkdir(parents=True)
                atomic_write_json(stage / "run_manifest.json", {"stage": "sentence"})
                return {"sentences": 120}

            def article_evidence_stage(*args, **kwargs):
                stage = kwargs["artifact_root"] / kwargs["run_name"]
                stage.mkdir(parents=True)
                atomic_write_json(stage / "run_manifest.json", {"stage": "llm"})
                return {
                    "articles": 10,
                    "evidence_selections": 6,
                }

            def aggregation_stage(*args, **kwargs):
                stage = kwargs["artifact_root"] / kwargs["run_name"]
                stage.mkdir(parents=True)
                atomic_write_json(stage / "run_manifest.json", {"stage": "aggregation"})
                return {
                    "verdict": "mixed",
                    "support_articles": 1,
                    "refute_articles": 2,
                    "insufficient_articles": 7,
                }

            def graph_stage(*args, **kwargs):
                stage = kwargs["artifact_root"] / kwargs["run_name"]
                stage.mkdir(parents=True)
                atomic_write_json(stage / "run_manifest.json", {"stage": "graph"})
                atomic_write_json(
                    stage / "graph_bundle.json",
                    {
                        "claim_id": "claim-fixture",
                        "claim_text": "Fixture claim.",
                        "verdict": "mixed",
                    },
                )
                return {"verdict": "mixed"}

            with patch(
                "evidencegap.pipeline.analysis.runtime_claim_id",
                return_value="claim-fixture",
            ), patch(
                "evidencegap.pipeline.analysis.retrieve_runtime_articles",
                side_effect=article_stage,
            ) as article_mock, patch(
                "evidencegap.pipeline.analysis.materialize_runtime_sentences",
                side_effect=sentence_stage,
            ), patch(
                "evidencegap.pipeline.analysis.run_article_evidence_extractor",
                side_effect=article_evidence_stage,
            ) as llm_mock, patch(
                "evidencegap.pipeline.analysis.run_claim_aggregation",
                side_effect=aggregation_stage,
            ), patch(
                "evidencegap.pipeline.analysis.run_final_graph",
                side_effect=graph_stage,
            ), patch(
                "evidencegap.pipeline.analysis.validate_analysis_artifact",
                return_value={"status": "PASS"},
            ):
                result = run_analysis(
                    repo_root,
                    claim="Fixture claim.",
                    run_name="fixture",
                    provider="deepseek",
                    model="deepseek-v4-pro",
                    artifact_root=artifact_root,
                )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["verdict"], "mixed")
            self.assertFalse((target / "evidence_retrieval").exists())
            manifest = json.loads((target / "run_manifest.json").read_text())
            self.assertNotIn(
                "phase05_runtime_sentence_retrieval", manifest["pipeline"]
            )
            self.assertIn(
                "phase05_runtime_sentence_retrieval",
                manifest["excluded_pipeline_stages"],
            )
            self.assertEqual(set(manifest["stages"]), {
                "article_retrieval",
                "sentence_materialization",
                "article_evidence",
                "claim_aggregation",
                "final_graph",
            })
            self.assertEqual(article_mock.call_count, 1)
            self.assertEqual(
                llm_mock.call_args.kwargs["retrieval_artifact_dir"], target
            )
            self.assertEqual(llm_mock.call_args.kwargs["request_batch_size"], 2)


class AnalysisArtifactValidationTests(unittest.TestCase):
    def _write_fixture(self, root: Path, target: Path) -> None:
        target.mkdir(parents=True)
        request = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "contract_id": ANALYSIS_CONTRACT_ID,
            "run_name": "fixture",
            "claim_id": "claim-fixture",
            "claim_text": "Fixture claim.",
        }
        atomic_write_json(target / "request.json", request)

        stage_names = (
            "article_retrieval",
            "sentence_materialization",
            "article_evidence",
            "claim_aggregation",
            "final_graph",
        )
        for name in stage_names:
            stage = target / name
            stage.mkdir()
            atomic_write_json(stage / "run_manifest.json", {"stage": name})
        atomic_write_json(
            target / "article_retrieval/run_manifest.json",
            {
                "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
                "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
                "claim_id": "claim-fixture",
                "outputs": {"top_articles": {"rows": 10}},
            },
        )
        atomic_write_json(
            target / "final_graph/graph_bundle.json",
            {
                "claim_id": "claim-fixture",
                "claim_text": "Fixture claim.",
                "verdict": "mixed",
            },
        )

        manifest = {
            "schema_version": ANALYSIS_SCHEMA_VERSION,
            "contract_id": ANALYSIS_CONTRACT_ID,
            "run_name": "fixture",
            "claim_id": "claim-fixture",
            "pipeline": [
                "phase04_runtime_article_retrieval",
                "phase07_runtime_sentence_materialization",
                "phase07_article_llm_evidence_extractor",
                "phase07_claim_aggregation",
                "phase07_final_graph",
            ],
            "verdict": "mixed",
            "request": {
                "path": relative_path(root, target / "request.json"),
                "sha256": sha256_file(target / "request.json"),
            },
            "stages": {
                name: {
                    "artifact_dir": relative_path(root, target / name),
                    "manifest": {
                        "path": relative_path(root, target / name / "run_manifest.json"),
                        "sha256": sha256_file(target / name / "run_manifest.json"),
                    },
                }
                for name in stage_names
            },
            "output": {
                "graph_bundle": {
                    "path": relative_path(
                        root, target / "final_graph/graph_bundle.json"
                    ),
                    "sha256": sha256_file(
                        target / "final_graph/graph_bundle.json"
                    ),
                }
            },
        }
        atomic_write_json(target / "run_manifest.json", manifest)

    def test_validate_analysis_checks_complete_chain(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
            target = Path(temp_dir) / "fixture"
            self._write_fixture(repo_root, target)
            with patch(
                "evidencegap.pipeline.analysis.validate_runtime_sentence_artifact",
                return_value={"status": "PASS"},
            ), patch(
                "evidencegap.pipeline.analysis.load_article_prompt_inputs",
                return_value=(
                    {"claim_id": "claim-fixture", "claim_text": "Fixture claim."},
                    [object()] * 10,
                    {},
                ),
            ), patch(
                "evidencegap.pipeline.analysis.validate_article_evidence_artifact",
                return_value={"status": "PASS"},
            ), patch(
                "evidencegap.pipeline.analysis.validate_claim_aggregation_artifact",
                return_value={"status": "PASS", "verdict": "mixed"},
            ), patch(
                "evidencegap.pipeline.analysis.validate_final_graph_artifact",
                return_value={
                    "status": "PASS",
                    "verdict": "mixed",
                    "node_counts": {"claim": 1, "article": 10, "evidence": 7},
                    "relation_counts": {
                        "article_supports": 1,
                        "article_refutes": 2,
                        "article_insufficient": 7,
                        "contains_evidence": 7,
                    },
                },
            ):
                result = validate_analysis_artifact(target)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["top_articles"], 10)
            self.assertFalse(result["phase05_sentence_retrieval_used"])
            self.assertEqual(result["checksums"], "PASS")

    def test_validate_analysis_rejects_phase05_artifact(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo_root) as temp_dir:
            target = Path(temp_dir) / "fixture"
            self._write_fixture(repo_root, target)
            (target / "evidence_retrieval").mkdir()
            with self.assertRaisesRegex(EvidenceGapError, "must not contain Phase 05"):
                validate_analysis_artifact(target)


if __name__ == "__main__":
    unittest.main()
