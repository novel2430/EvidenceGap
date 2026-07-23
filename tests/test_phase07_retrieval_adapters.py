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

from evidencegap.common import EvidenceGapError, sha256_file
from evidencegap.pipeline.retrieval_adapters import (
    RUNTIME_RETRIEVAL_CONTRACT_ID,
    RUNTIME_RETRIEVAL_SCHEMA_VERSION,
    evidence_sentence_exclusion_reason,
    fuse_article_rankings,
    fuse_sentence_rankings,
    partition_evidence_sentences,
    run_retrieval_adapters,
    runtime_claim_id,
    validate_runtime_evidence_rows,
)


class Phase07ArticleFusionTests(unittest.TestCase):
    def test_frozen_three_way_rrf_and_tie_breaks(self) -> None:
        sources = {
            "bm25": [
                {"article_id": "a", "doc_idx": 0, "rank": 1, "score": 9.0},
                {"article_id": "b", "doc_idx": 1, "rank": 2, "score": 8.0},
            ],
            "medcpt": [
                {"article_id": "b", "doc_idx": 1, "rank": 1, "score": 0.9},
                {"article_id": "a", "doc_idx": 0, "rank": 2, "score": 0.8},
            ],
            "bmretriever": [
                {"article_id": "a", "doc_idx": 0, "rank": 1, "score": 0.7},
                {"article_id": "c", "doc_idx": 2, "rank": 2, "score": 0.6},
            ],
        }
        rows = fuse_article_rankings(sources, rrf_k=60)
        self.assertEqual([row["article_id"] for row in rows], ["a", "b", "c"])
        self.assertEqual(rows[0]["source_count"], 3)
        self.assertEqual(rows[0]["fusion_rank"], 1)
        self.assertEqual(rows[1]["bmretriever_rank"], None)

    def test_article_index_drift_is_rejected(self) -> None:
        with self.assertRaises(EvidenceGapError):
            fuse_article_rankings(
                {
                    "bm25": [
                        {"article_id": "a", "doc_idx": 1, "rank": 1, "score": 1.0}
                    ],
                    "medcpt": [
                        {"article_id": "a", "doc_idx": 2, "rank": 1, "score": 1.0}
                    ],
                    "bmretriever": [],
                }
            )


class Phase07SentenceFusionTests(unittest.TestCase):
    def test_sentence_rrf_is_paper_local_and_preserves_source_scores(self) -> None:
        fused = fuse_sentence_rankings(
            {
                "bmretriever": [
                    {
                        "sentence_id": "s0",
                        "sentence_index": 0,
                        "retrieval_rank": 1,
                        "retrieval_score": 0.8,
                    },
                    {
                        "sentence_id": "s1",
                        "sentence_index": 1,
                        "retrieval_rank": 2,
                        "retrieval_score": 0.7,
                    },
                ],
                "medcpt": [
                    {
                        "sentence_id": "s1",
                        "sentence_index": 1,
                        "retrieval_rank": 1,
                        "retrieval_score": 0.9,
                    },
                    {
                        "sentence_id": "s0",
                        "sentence_index": 0,
                        "retrieval_rank": 2,
                        "retrieval_score": 0.6,
                    },
                ],
            },
            source_depth=2,
            rrf_k=10,
        )
        self.assertEqual([row["sentence_id"] for row in fused], ["s0", "s1"])
        self.assertEqual(fused[0]["evidence_rank_within_article"], 1)
        self.assertEqual(fused[0]["bmretriever_score"], 0.8)
        self.assertEqual(fused[0]["medcpt_score"], 0.6)


class Phase07EvidenceContractTests(unittest.TestCase):
    def _fixtures(self):
        articles = [
            {
                "article_id": f"a{index}",
                "final_article_rank": index + 1,
                "cross_encoder_score": 10.0 - index,
            }
            for index in range(10)
        ]
        sentences = []
        evidence = []
        for index, article in enumerate(articles):
            sentence = {
                "sentence_id": f"s{index}",
                "article_id": article["article_id"],
                "sentence_index": 0,
                "sentence_type": "abstract",
                "section": "abstract",
                "sentence_text": f"Sentence {index}.",
            }
            sentences.append(sentence)
            evidence.append(
                {
                    "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
                    "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
                    "evidence_id": f"e{index}",
                    "article_id": article["article_id"],
                    "sentence_id": sentence["sentence_id"],
                    "sentence_index": 0,
                    "sentence_type": sentence["sentence_type"],
                    "section": sentence["section"],
                    "sentence_text": sentence["sentence_text"],
                    "rrf_score": 0.1,
                    "bmretriever_rank": 1,
                    "medcpt_rank": 1,
                    "evidence_rank_within_article": 1,
                }
            )
        return articles, sentences, evidence

    def test_evidence_validation_requires_every_top_article(self) -> None:
        articles, sentences, evidence = self._fixtures()
        report = validate_runtime_evidence_rows(
            evidence, sentences=sentences, top_articles=articles
        )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["articles"], 10)
        self.assertEqual(report["title_evidence_count"], 0)
        with self.assertRaises(EvidenceGapError):
            validate_runtime_evidence_rows(
                evidence[:-1], sentences=sentences, top_articles=articles
            )

    def test_evidence_validation_rejects_title_sentences(self) -> None:
        articles, sentences, evidence = self._fixtures()
        sentences[0]["sentence_type"] = "title"
        evidence[0]["sentence_type"] = "title"
        with self.assertRaises(EvidenceGapError):
            validate_runtime_evidence_rows(
                evidence, sentences=sentences, top_articles=articles
            )

    def test_structured_unterminated_fragments_are_not_evidence_eligible(self) -> None:
        rows = [
            {
                "sentence_id": "title",
                "sentence_type": "title",
                "section": "title",
                "sentence_text": "Trial title",
            },
            {
                "sentence_id": "intervention_fragment",
                "sentence_type": "abstract",
                "section": "intervention",
                "sentence_text": "The high-dose group received monthly supplement of vitamin D",
            },
            {
                "sentence_id": "conclusion_fragment",
                "sentence_type": "abstract",
                "section": "conclusion",
                "sentence_text": "Monthly high-dose vitamin D",
            },
            {
                "sentence_id": "complete",
                "sentence_type": "abstract",
                "section": "results",
                "sentence_text": "The intervention reduced ARI incidence.",
            },
            {
                "sentence_id": "plain",
                "sentence_type": "abstract",
                "section": "abstract",
                "sentence_text": "An unstructured abstract without final punctuation",
            },
        ]
        eligible, excluded = partition_evidence_sentences(rows)
        self.assertEqual(
            [row["sentence_id"] for row in eligible],
            ["complete", "plain"],
        )
        self.assertEqual(
            excluded,
            {
                "title": 1,
                "unterminated_structured_section_fragment": 2,
            },
        )
        self.assertEqual(
            evidence_sentence_exclusion_reason(rows[1]),
            "unterminated_structured_section_fragment",
        )
        self.assertEqual(
            evidence_sentence_exclusion_reason(rows[2]),
            "unterminated_structured_section_fragment",
        )
        self.assertIsNone(evidence_sentence_exclusion_reason(rows[3]))
        self.assertIsNone(evidence_sentence_exclusion_reason(rows[4]))

    def test_evidence_validation_rejects_unterminated_structured_fragment(self) -> None:
        articles, sentences, evidence = self._fixtures()
        sentences[0]["section"] = "conclusion"
        sentences[0]["sentence_text"] = "Monthly high-dose vitamin D"
        evidence[0]["section"] = "conclusion"
        evidence[0]["sentence_text"] = "Monthly high-dose vitamin D"
        with self.assertRaisesRegex(
            EvidenceGapError,
            "unterminated_structured_section_fragment",
        ):
            validate_runtime_evidence_rows(
                evidence, sentences=sentences, top_articles=articles
            )


class Phase07RetrievalOrchestrationTests(unittest.TestCase):
    def test_orchestrator_links_three_reusable_stage_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)

            def fake_article(_root, **kwargs):
                stage = Path(kwargs["artifact_dir"])
                stage.mkdir(parents=True)
                (stage / "run_manifest.json").write_text("{}\n", encoding="utf-8")
                (stage / "top_articles.parquet").write_bytes(b"top")
                (stage / "runtime_articles.jsonl").write_text(
                    json.dumps({"article_id": "a", "abstract": "Text."}) + "\n",
                    encoding="utf-8",
                )
                return {
                    "top_articles": 10,
                    "outputs": {
                        "top_articles": {
                            "path": str(stage / "top_articles.parquet"),
                            "sha256": sha256_file(stage / "top_articles.parquet"),
                            "rows": 10,
                        },
                        "runtime_articles_input": {
                            "path": str(stage / "runtime_articles.jsonl"),
                            "sha256": sha256_file(stage / "runtime_articles.jsonl"),
                            "rows": 10,
                        },
                    },
                }

            def fake_materialize(_root, **kwargs):
                stage = Path(kwargs["artifact_root"]) / kwargs["run_name"]
                stage.mkdir(parents=True)
                (stage / "run_manifest.json").write_text("{}\n", encoding="utf-8")
                (stage / "runtime_sentences.parquet").write_bytes(b"sentences")
                return {
                    "sentences": 42,
                    "outputs": {
                        "runtime_sentences": {
                            "path": str(stage / "runtime_sentences.parquet"),
                            "sha256": sha256_file(stage / "runtime_sentences.parquet"),
                            "rows": 42,
                        }
                    },
                }

            def fake_evidence(_root, **kwargs):
                stage = Path(kwargs["artifact_dir"])
                stage.mkdir(parents=True)
                (stage / "run_manifest.json").write_text("{}\n", encoding="utf-8")
                (stage / "evidence_candidates.parquet").write_bytes(b"evidence")
                (stage / "evidence_candidates.jsonl").write_text("{}\n", encoding="utf-8")
                return {
                    "evidence_candidates": 50,
                    "outputs": {
                        "evidence_candidates": {
                            "path": str(stage / "evidence_candidates.parquet"),
                            "sha256": sha256_file(stage / "evidence_candidates.parquet"),
                            "rows": 50,
                        },
                        "evidence_candidates_preview": {
                            "path": str(stage / "evidence_candidates.jsonl"),
                            "sha256": sha256_file(stage / "evidence_candidates.jsonl"),
                            "rows": 50,
                        },
                    },
                }

            with (
                patch(
                    "evidencegap.pipeline.retrieval_adapters.retrieve_runtime_articles",
                    side_effect=fake_article,
                ),
                patch(
                    "evidencegap.pipeline.retrieval_adapters.materialize_runtime_sentences",
                    side_effect=fake_materialize,
                ),
                patch(
                    "evidencegap.pipeline.retrieval_adapters.retrieve_runtime_evidence",
                    side_effect=fake_evidence,
                ),
                patch(
                    "evidencegap.pipeline.retrieval_adapters.validate_retrieval_adapter_artifact",
                    return_value={"status": "PASS"},
                ),
            ):
                result = run_retrieval_adapters(
                    root,
                    claim="Vitamin D reduces infection risk.",
                    run_name="fixture",
                )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["top_articles"], 10)
            manifest = json.loads(
                (root / "artifacts/v1/pipeline/retrieval_adapters/fixture/run_manifest.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["pipeline"],
                [
                    "phase04_runtime_article_retrieval",
                    "phase07_runtime_sentence_materialization",
                    "phase05_runtime_sentence_retrieval",
                ],
            )
            self.assertEqual(manifest["counts"]["evidence_candidates"], 50)
            self.assertEqual(
                runtime_claim_id("  Vitamin   D reduces infection risk. "),
                runtime_claim_id("Vitamin D reduces infection risk."),
            )


if __name__ == "__main__":
    unittest.main()
