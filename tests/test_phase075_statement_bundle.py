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
)
from evidencegap.pipeline.final_graph import (  # noqa: E402
    FINAL_GRAPH_CONTRACT_ID,
    FINAL_GRAPH_SCHEMA_VERSION,
)
from evidencegap.pipeline.statement_analysis import (  # noqa: E402
    STATEMENT_ANALYSIS_CONTRACT_ID,
    STATEMENT_ANALYSIS_SCHEMA_VERSION,
    validate_statement_analysis_bundle,
)
from evidencegap.pipeline.statement_bundle import (  # noqa: E402
    STATEMENT_BUNDLE_CONTRACT_ID,
    build_statement_bundle,
    run_statement_bundle,
    validate_statement_bundle,
    validate_statement_bundle_artifact,
)
from evidencegap.pipeline.statement_decomposition import (  # noqa: E402
    runtime_inference_step_id,
    validate_response_payload,
)


def _graph(
    *,
    claim_id: str,
    claim_text: str,
    verdict: str = "supported",
    article_id: str = "pmid:1",
    article_node_id: str | None = None,
    evidence_node_id: str = "evidence:shared",
) -> dict[str, object]:
    article_node_id = article_node_id or f"article:{claim_id[-8:]}"
    claim_node_id = f"claim-node:{claim_id}"
    stance = {
        "supported": "support",
        "refuted": "refute",
        "insufficient": "insufficient",
    }.get(verdict, "support")
    evidence_nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = [
        {
            "edge_id": "edge:article-claim",
            "source_node_id": article_node_id,
            "target_node_id": claim_node_id,
            "relation": {
                "support": "article_supports",
                "refute": "article_refutes",
                "insufficient": "article_insufficient",
            }[stance],
            "claim_id": claim_id,
            "article_id": article_id,
            "evidence_id": None,
            "stance": stance,
        }
    ]
    if stance != "insufficient":
        evidence_nodes.append(
            {
                "node_id": evidence_node_id,
                "node_type": "evidence",
                "label": "A1",
                "text": "The intervention reduced infections.",
                "claim_id": claim_id,
                "article_id": article_id,
                "pmid": "1",
                "evidence_id": "evidence-source-1",
                "sentence_id": "sentence-1",
                "sentence_index": 1,
                "sentence_index_within_section": 1,
                "section": "abstract",
                "section_index": 0,
                "character_start": 10,
                "character_end": 46,
                "source_text_fingerprint": "source-fp",
                "splitter_fingerprint": "splitter-fp",
            }
        )
        edges.append(
            {
                "edge_id": "edge:contains",
                "source_node_id": article_node_id,
                "target_node_id": evidence_node_id,
                "relation": "contains_evidence",
                "claim_id": claim_id,
                "article_id": article_id,
                "evidence_id": "evidence-source-1",
                "stance": None,
            }
        )

    support = int(stance == "support")
    refute = int(stance == "refute")
    insufficient = int(stance == "insufficient")
    return {
        "schema_version": FINAL_GRAPH_SCHEMA_VERSION,
        "contract_id": FINAL_GRAPH_CONTRACT_ID,
        "graph_id": f"graph:{claim_id}",
        "claim_id": claim_id,
        "claim_text": claim_text,
        "verdict": verdict,
        "summary": {
            "claim_id": claim_id,
            "claim_text": claim_text,
            "verdict": verdict,
            "article_counts": {
                "total": 1,
                "support": support,
                "refute": refute,
                "insufficient": insufficient,
            },
            "rationale": "Fixture rationale.",
            "scope": "retrieved_top_articles",
        },
        "nodes": [
            {
                "node_id": claim_node_id,
                "node_type": "claim",
                "label": "Claim",
                "text": claim_text,
                "claim_id": claim_id,
            },
            {
                "node_id": article_node_id,
                "node_type": "article",
                "label": "Fixture article",
                "text": "Fixture article rationale.",
                "claim_id": claim_id,
                "article_id": article_id,
                "pmid": "1",
                "final_article_rank": 1,
                "stance": stance,
                "confidence": 0.9,
                "probabilities": {
                    "support": float(support),
                    "refute": float(refute),
                    "insufficient": float(insufficient),
                },
                "selected_evidence_count": len(evidence_nodes),
                "provider": "fixture",
                "model": "fixture",
                "model_fingerprint": "model-fp",
                "prompt_version": "prompt-v1",
            },
            *evidence_nodes,
        ],
        "edges": edges,
        "boundary": {
            "is_pipeline_final_verdict": True,
            "is_final_medical_truth": False,
            "description": "Fixture boundary.",
        },
    }


def _analysis_result(
    decomposition: dict[str, object],
    *,
    statuses: list[str],
    verdicts: list[str | None],
    graph_paths: list[str | None],
) -> dict[str, object]:
    rows = []
    for claim, status, verdict, graph_path in zip(
        decomposition["claims"], statuses, verdicts, graph_paths, strict=True
    ):
        assert isinstance(claim, dict)
        rows.append(
            {
                **claim,
                "status": status,
                "phase07_artifact_dir": (
                    f"artifacts/phase07/{claim['claim_id']}"
                    if status == "completed"
                    else None
                ),
                "graph_bundle_path": graph_path,
                "verdict": verdict,
                "error": None if status == "completed" else "fixture failure",
            }
        )
    completed = statuses.count("completed")
    failed = statuses.count("failed")
    analysis_status = (
        "completed"
        if failed == 0
        else "failed"
        if completed == 0
        else "partial_failure"
    )
    bundle = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "statement_id": decomposition["statement_id"],
        "original_statement": decomposition["original_statement"],
        "source_language": decomposition["source_language"],
        "analysis_status": analysis_status,
        "claim_results": rows,
        "summary": {
            "total_claims": len(rows),
            "completed_claims": completed,
            "failed_claims": failed,
        },
    }
    validate_statement_analysis_bundle(bundle)
    return bundle


class StatementBundleTests(unittest.TestCase):
    def test_build_bundle_merges_graph_data_and_preserves_failed_claim(self) -> None:
        statement = "甲能降低感染，因此乙能降低死亡。"
        decomposition = validate_response_payload(
            {
                "source_language": "zh-TW",
                "claims": [
                    {
                        "claim_ref": "C1",
                        "source_text": "甲能降低感染",
                        "canonical_claim_en": "Treatment A reduces infections.",
                    },
                    {
                        "claim_ref": "C2",
                        "source_text": "乙能降低死亡",
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
        first_id = decomposition["claims"][0]["claim_id"]
        analysis = _analysis_result(
            decomposition,
            statuses=["completed", "failed"],
            verdicts=["supported", None],
            graph_paths=["artifacts/graph-1.json", None],
        )
        bundle = build_statement_bundle(
            decomposition,
            analysis,
            {
                first_id: _graph(
                    claim_id=first_id,
                    claim_text="Treatment A reduces infections.",
                )
            },
        )

        self.assertEqual(bundle["contract_id"], STATEMENT_BUNDLE_CONTRACT_ID)
        self.assertEqual(bundle["statement"]["analysis_status"], "partial_failure")
        self.assertEqual(len(bundle["claims"]), 2)
        self.assertEqual(bundle["claims"][0]["verdict"], "supported")
        self.assertEqual(bundle["claims"][1]["analysis_status"], "failed")
        self.assertEqual(bundle["claims"][1]["article_node_ids"], [])
        self.assertEqual(len(bundle["articles"]), 1)
        self.assertEqual(len(bundle["evidence"]), 1)
        self.assertNotIn("stance", bundle["evidence"][0])
        self.assertTrue(bundle["evidence"][0]["evidence_id"].startswith(first_id))
        inference_step = bundle["inference_steps"][0]
        self.assertEqual(
            inference_step["conclusion_claim_id"],
            decomposition["claims"][1]["claim_id"],
        )
        self.assertEqual(
            inference_step["inference_step_id"],
            runtime_inference_step_id(
                inference_step["premise_claim_ids"],
                inference_step["conclusion_claim_id"],
            ),
        )
        self.assertEqual(validate_statement_bundle(bundle)["status"], "PASS")

    def test_empty_claims_produce_an_empty_valid_bundle(self) -> None:
        decomposition = validate_response_payload(
            {"source_language": "zh-TW", "claims": [], "inference_steps": []},
            original_statement="健康是基本人權。",
        )
        analysis = _analysis_result(
            decomposition, statuses=[], verdicts=[], graph_paths=[]
        )
        bundle = build_statement_bundle(decomposition, analysis, {})
        self.assertEqual(bundle["claims"], [])
        self.assertEqual(bundle["articles"], [])
        self.assertEqual(bundle["evidence"], [])
        self.assertTrue(validate_statement_bundle(bundle)["empty_claims"])

    def test_same_source_evidence_is_unique_for_each_claim(self) -> None:
        statement = "甲降低感染，乙降低感染。"
        decomposition = validate_response_payload(
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
                        "source_text": "乙降低感染",
                        "canonical_claim_en": "Treatment B reduces infections.",
                    },
                ],
                "inference_steps": [],
            },
            original_statement=statement,
        )
        claim_ids = [claim["claim_id"] for claim in decomposition["claims"]]
        analysis = _analysis_result(
            decomposition,
            statuses=["completed", "completed"],
            verdicts=["supported", "supported"],
            graph_paths=["graph-1.json", "graph-2.json"],
        )
        bundle = build_statement_bundle(
            decomposition,
            analysis,
            {
                claim_ids[0]: _graph(
                    claim_id=claim_ids[0],
                    claim_text="Treatment A reduces infections.",
                    article_node_id="article:claim-1",
                ),
                claim_ids[1]: _graph(
                    claim_id=claim_ids[1],
                    claim_text="Treatment B reduces infections.",
                    article_node_id="article:claim-2",
                ),
            },
        )
        evidence_ids = [item["evidence_id"] for item in bundle["evidence"]]
        self.assertEqual(len(evidence_ids), 2)
        self.assertEqual(len(set(evidence_ids)), 2)

    def test_artifact_can_be_rebuilt_and_validated_from_sources(self) -> None:
        statement = "甲能降低感染。"
        decomposition = validate_response_payload(
            {
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
            original_statement=statement,
        )
        claim_id = decomposition["claims"][0]["claim_id"]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "src/evidencegap").mkdir(parents=True)
            decomposition_dir = root / "artifacts/decomposition/input"
            analysis_dir = root / "artifacts/statement_analysis/input"
            graph_path = root / "artifacts/phase07/graph_bundle.json"
            output_root = root / "artifacts/statement_bundle"
            decomposition_dir.mkdir(parents=True)
            analysis_dir.mkdir(parents=True)
            graph_path.parent.mkdir(parents=True)

            analysis = _analysis_result(
                decomposition,
                statuses=["completed"],
                verdicts=["supported"],
                graph_paths=[relative_path(root, graph_path)],
            )
            atomic_write_json(decomposition_dir / "decomposition.json", decomposition)
            atomic_write_json(analysis_dir / "statement_result.json", analysis)
            atomic_write_json(
                analysis_dir / "request.json",
                {
                    "decomposition_artifact_dir": relative_path(
                        root, decomposition_dir
                    )
                },
            )
            atomic_write_json(
                graph_path,
                _graph(
                    claim_id=claim_id,
                    claim_text="Treatment A reduces infections.",
                ),
            )

            with patch(
                "evidencegap.pipeline.statement_bundle."
                "validate_statement_analysis_artifact",
                return_value={"status": "PASS"},
            ):
                result = run_statement_bundle(
                    root,
                    statement_analysis_artifact_dir=analysis_dir,
                    run_name="bundle",
                    artifact_root=output_root,
                )
                validation = validate_statement_bundle_artifact(
                    output_root / "bundle"
                )

            self.assertEqual(result["status"], "PASS")
            self.assertEqual(validation["checksums"], "PASS")
            bundle = json.loads(
                (output_root / "bundle/statement_bundle.json").read_text(
                    encoding="utf-8"
                )
            )
            bundle["claims"][0]["verdict"] = "refuted"
            atomic_write_json(output_root / "bundle/statement_bundle.json", bundle)
            with patch(
                "evidencegap.pipeline.statement_bundle."
                "validate_statement_analysis_artifact",
                return_value={"status": "PASS"},
            ):
                with self.assertRaises(EvidenceGapError):
                    validate_statement_bundle_artifact(output_root / "bundle")


if __name__ == "__main__":
    unittest.main()
