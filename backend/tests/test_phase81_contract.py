from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from evidencegap_backend.common import EvidenceGapError
from evidencegap_backend.config import PipelineConfig
from evidencegap_backend.output.presentation import build_presentation_bundle
from evidencegap_backend.pipeline.article_evidence import build_retrieval_trace
from evidencegap_backend.pipeline.claim_aggregation import aggregate_article_evidence_rows
from evidencegap_backend.pipeline.final_graph import build_final_graph_bundle
from evidencegap_backend.pipeline.inference_gap_analysis import (
    build_inference_gap_analysis_bundle,
)
from evidencegap_backend.pipeline.statement_analysis import (
    STATEMENT_ANALYSIS_CONTRACT_ID,
    STATEMENT_ANALYSIS_SCHEMA_VERSION,
)
from evidencegap_backend.pipeline.statement_bundle import (
    build_statement_bundle,
    inference_impacts,
    validate_statement_bundle,
)
from evidencegap_backend.pipeline.statement_decomposition import (
    exact_source_spans,
    runtime_inference_step_id,
    validate_decomposition_bundle,
    validate_response_payload,
)
from evidencegap_backend.pipeline.statement_run import (
    _STAGE_NAMES,
    build_execution_summary,
    validate_execution_summary,
)


def _retrieval_source(rank: int) -> dict[str, Any]:
    return {
        "final_article_rank": rank,
        "bm25_rank": rank + 4,
        "bm25_score": 6.5,
        "medcpt_rank": rank,
        "medcpt_score": 0.82,
        "bmretriever_rank": None,
        "bmretriever_score": None,
        "fusion_rank": rank + 1,
        "rrf_score": 0.047,
        "cross_encoder_score": 0.91,
    }


def _article_row(
    *, claim_id: str, claim_text: str, article_id: str, label: str
) -> dict[str, Any]:
    retrieval_source = _retrieval_source(1)
    evidence = []
    if label != "insufficient":
        evidence = [
            {
                "evidence_id": f"evidence_{article_id}",
                "sentence_alias": "S01",
                "sentence_id": f"sentence_{article_id}",
                "sentence_index": 1,
                "sentence_index_within_section": 0,
                "section": "abstract",
                "section_index": 1,
                "sentence_text": "The study reports a direct result.",
                "character_start": 10,
                "character_end": 44,
                "source_text_fingerprint": "f" * 64,
                "splitter_fingerprint": "s" * 64,
            }
        ]
    probabilities = {
        "support": 0.9 if label == "support" else 0.05,
        "refute": 0.9 if label == "refute" else 0.05,
        "insufficient": 0.9 if label == "insufficient" else 0.05,
    }
    return {
        "claim_id": claim_id,
        "claim_text": claim_text,
        "article_id": article_id,
        "pmid": article_id.removeprefix("pmid:"),
        "final_article_rank": 1,
        "retrieval_trace": build_retrieval_trace(retrieval_source),
        "title": f"Article for {claim_id}",
        "predicted_label": label,
        "probabilities": probabilities,
        "confidence": probabilities[label],
        "probability_margin": 0.85,
        "rationale": "Grounded article-level rationale.",
        "selected_evidence": evidence,
        "provider": "deepseek",
        "model": "test-model",
        "model_fingerprint": "m" * 64,
        "prompt_version": "test-prompt",
    }


def _phase81_bundles() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    statement = (
        "Vitamin D improves marker X. Vitamin D improves marker X. "
        "Therefore marker X reduces infections."
    )
    decomposition = validate_response_payload(
        {
            "source_language": "en",
            "claims": [
                {
                    "claim_ref": "C1",
                    "source_text": "Vitamin D improves marker X",
                    "canonical_claim_en": "Vitamin D improves marker X.",
                },
                {
                    "claim_ref": "C2",
                    "source_text": "marker X reduces infections",
                    "canonical_claim_en": "Marker X reduces infections.",
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
    claim_results = []
    graphs: dict[str, dict[str, Any]] = {}
    for index, claim in enumerate(decomposition["claims"], start=1):
        label = "support" if index == 1 else "insufficient"
        article_rows = [
            _article_row(
                claim_id=claim["claim_id"],
                claim_text=claim["canonical_claim_en"],
                article_id=f"pmid:{index}",
                label=label,
            )
        ]
        claim_result = aggregate_article_evidence_rows(article_rows)
        graphs[claim["claim_id"]] = build_final_graph_bundle(
            article_rows, claim_result
        )
        claim_results.append(
            {
                "claim_id": claim["claim_id"],
                "source_text": claim["source_text"],
                "source_spans": claim["source_spans"],
                "canonical_claim_en": claim["canonical_claim_en"],
                "status": "completed",
                "phase07_artifact_dir": f"artifacts/{claim['claim_id']}",
                "graph_bundle_path": (
                    f"artifacts/{claim['claim_id']}/graph_bundle.json"
                ),
                "verdict": claim_result["verdict"],
                "error": None,
            }
        )

    analysis_context = PipelineConfig(
        source_depth=75,
        rerank_depth=60,
        final_article_top_k=8,
        max_evidence_sentences=4,
    ).analysis_context()
    statement_result = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "statement_id": decomposition["statement_id"],
        "original_statement": decomposition["original_statement"],
        "source_language": decomposition["source_language"],
        "analysis_status": "completed",
        "analysis_context": analysis_context,
        "claim_results": claim_results,
        "summary": {
            "total_claims": 2,
            "completed_claims": 2,
            "failed_claims": 0,
        },
    }
    statement_bundle = build_statement_bundle(
        decomposition, statement_result, graphs
    )
    step_id = statement_bundle["inference_steps"][0]["inference_step_id"]
    gap_bundle = build_inference_gap_analysis_bundle(
        statement_bundle,
        [
            {
                "inference_step_id": step_id,
                "scope_gap": {
                    "detected": True,
                    "reason": "The conclusion uses a broader outcome.",
                },
                "causal_gap": {"detected": False, "reason": None},
            }
        ],
        source_statement_bundle_sha256="a" * 64,
    )
    presentation = build_presentation_bundle(
        statement_bundle,
        gap_bundle,
        output_language="English",
        statement_bundle_sha256="a" * 64,
        gap_bundle_sha256="b" * 64,
    )
    return decomposition, statement_bundle, presentation


def test_phase81_contract_enrichments_propagate_to_presentation() -> None:
    decomposition, statement_bundle, presentation = _phase81_bundles()

    first_claim = decomposition["claims"][0]
    assert first_claim["source_spans"] == exact_source_spans(
        decomposition["original_statement"], first_claim["source_text"]
    )
    assert len(first_claim["source_spans"]) == 2

    article = statement_bundle["articles"][0]
    assert article["retrieval_trace"]["medcpt"]["rank"] == 1
    assert article["retrieval_trace"]["bmretriever"] == {
        "rank": None,
        "score": None,
    }

    impact = statement_bundle["inference_steps"][0]["impact"]
    conclusion_id = statement_bundle["inference_steps"][0]["conclusion_claim_id"]
    assert impact == {
        "direct_conclusion_claim_id": conclusion_id,
        "downstream_claim_ids": [conclusion_id],
        "downstream_inference_step_ids": [],
        "terminal_claim_ids": [conclusion_id],
        "affects_terminal_conclusion": True,
        "cycle_detected": False,
    }

    assert presentation["analysis_context"] == statement_bundle["analysis_context"]
    assert presentation["analysis_context"]["article_top_k"] == 8
    assert presentation["articles"][0]["retrieval_trace"] == article[
        "retrieval_trace"
    ]
    assert presentation["inference_steps"][0]["impact"] == impact


def test_phase81_validators_reject_tampered_enrichments() -> None:
    decomposition, statement_bundle, _ = _phase81_bundles()

    bad_decomposition = copy.deepcopy(decomposition)
    bad_decomposition["claims"][0]["source_spans"][0]["character_start"] += 1
    with pytest.raises(EvidenceGapError, match="source_spans"):
        validate_decomposition_bundle(bad_decomposition)

    bad_trace = copy.deepcopy(statement_bundle)
    bad_trace["articles"][0]["retrieval_trace"]["final_article_rank"] = 9
    with pytest.raises(EvidenceGapError, match="rank mismatch"):
        validate_statement_bundle(bad_trace)

    bad_impact = copy.deepcopy(statement_bundle)
    bad_impact["inference_steps"][0]["impact"][
        "affects_terminal_conclusion"
    ] = False
    with pytest.raises(EvidenceGapError, match="impact mismatch"):
        validate_statement_bundle(bad_impact)


def test_inference_impact_cycle_is_finite_and_explicit() -> None:
    claims = [{"claim_id": "A"}, {"claim_id": "B"}]
    step_ab = {
        "inference_step_id": runtime_inference_step_id(["A"], "B"),
        "premise_claim_ids": ["A"],
        "conclusion_claim_id": "B",
    }
    step_ba = {
        "inference_step_id": runtime_inference_step_id(["B"], "A"),
        "premise_claim_ids": ["B"],
        "conclusion_claim_id": "A",
    }

    impacts = inference_impacts(claims, [step_ab, step_ba])

    assert impacts[step_ab["inference_step_id"]]["cycle_detected"] is True
    assert impacts[step_ab["inference_step_id"]]["terminal_claim_ids"] == []
    assert (
        impacts[step_ab["inference_step_id"]]["affects_terminal_conclusion"]
        is False
    )


def test_execution_summary_reads_and_validates_nested_stage_timings(
    tmp_path: Path,
) -> None:
    stage_dirs: dict[str, Path] = {}
    for index, key in enumerate(_STAGE_NAMES, start=1):
        stage_dir = tmp_path / _STAGE_NAMES[key]
        stage_dir.mkdir()
        (stage_dir / "run_manifest.json").write_text(
            json.dumps({"seconds": index / 10}), encoding="utf-8"
        )
        stage_dirs[key] = stage_dir

    summary = build_execution_summary(stage_dirs, total_seconds=2.75)

    assert summary["total_seconds"] == 2.75
    assert summary["stages"]["statement_decomposition"]["seconds"] == 0.1
    assert summary["stages"]["output_generation"]["seconds"] == 0.5
    assert validate_execution_summary(
        summary, stage_dirs=stage_dirs, total_seconds=2.75
    ) == summary

    tampered = copy.deepcopy(summary)
    tampered["stages"]["claim_analysis"]["seconds"] = 99.0
    with pytest.raises(EvidenceGapError, match="execution summary mismatch"):
        validate_execution_summary(
            tampered, stage_dirs=stage_dirs, total_seconds=2.75
        )


def test_statement_pipeline_persists_execution_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from evidencegap_backend.pipeline.statement_run import run_statement_pipeline

    stage_seconds = {
        "decomposition": 0.11,
        "analysis": 0.22,
        "bundle": 0.03,
        "gaps": 0.14,
        "output": 0.05,
    }

    def write_stage(artifact_root: Path, run_name: str, filename: str, data: dict[str, Any]) -> Path:
        stage_dir = artifact_root / run_name
        stage_dir.mkdir()
        (stage_dir / filename).write_text(json.dumps(data), encoding="utf-8")
        (stage_dir / "run_manifest.json").write_text(
            json.dumps({"seconds": stage_seconds[run_name]}), encoding="utf-8"
        )
        return stage_dir

    decomposition = {
        "statement_id": "statement_test",
        "claims": [],
        "inference_steps": [],
    }
    statement_bundle = {"statement": {"statement_id": "statement_test"}}
    gap_bundle = {"statement_id": "statement_test"}
    presentation = {"contract_id": "phase077.presentation-bundle.v1"}

    def fake_decomposition(*args: Any, **kwargs: Any) -> dict[str, Any]:
        write_stage(
            kwargs["artifact_root"],
            kwargs["run_name"],
            "decomposition.json",
            decomposition,
        )
        return {
            "statement_id": "statement_test",
            "decomposition": decomposition,
        }

    def fake_analysis(*args: Any, **kwargs: Any) -> dict[str, Any]:
        write_stage(
            kwargs["artifact_root"],
            kwargs["run_name"],
            "statement_result.json",
            {},
        )
        return {
            "analysis_status": "completed",
            "decomposition": decomposition,
            "statement_result": {},
            "claim_graph_bundles": {},
        }

    def fake_bundle(*args: Any, **kwargs: Any) -> dict[str, Any]:
        write_stage(
            kwargs["artifact_root"],
            kwargs["run_name"],
            "statement_bundle.json",
            statement_bundle,
        )
        return {
            "statement_bundle": statement_bundle,
            "total_claims": 0,
            "completed_claims": 0,
            "failed_claims": 0,
            "articles": 0,
            "evidence": 0,
        }

    def fake_gaps(*args: Any, **kwargs: Any) -> dict[str, Any]:
        write_stage(
            kwargs["artifact_root"],
            kwargs["run_name"],
            "inference_gap_analysis.json",
            gap_bundle,
        )
        return {
            "inference_gap_bundle": gap_bundle,
            "total_inference_steps": 0,
            "scope_gaps": 0,
            "causal_gaps": 0,
            "api_requests": 0,
        }

    def fake_output(*args: Any, **kwargs: Any) -> dict[str, Any]:
        write_stage(
            kwargs["artifact_root"],
            kwargs["run_name"],
            "presentation_bundle.json",
            presentation,
        )
        return {
            "presentation_bundle": presentation,
            "output_language": kwargs["language"],
            "localized": False,
            "api_requests": 0,
        }

    monkeypatch.setattr(
        "evidencegap_backend.pipeline.statement_run.run_statement_decomposition",
        fake_decomposition,
    )
    monkeypatch.setattr(
        "evidencegap_backend.pipeline.statement_run.run_statement_analysis",
        fake_analysis,
    )
    monkeypatch.setattr(
        "evidencegap_backend.pipeline.statement_run.run_statement_bundle",
        fake_bundle,
    )
    monkeypatch.setattr(
        "evidencegap_backend.pipeline.statement_run.run_inference_gap_analysis",
        fake_gaps,
    )
    monkeypatch.setattr(
        "evidencegap_backend.pipeline.statement_run.run_output_module",
        fake_output,
    )

    result = run_statement_pipeline(
        tmp_path,
        statement="A biomedical statement.",
        run_name="phase81",
        provider="deepseek",
        artifact_root=tmp_path / "artifacts",
    )

    assert result["execution_summary"]["stages"] == {
        "statement_decomposition": {"seconds": 0.11},
        "claim_analysis": {"seconds": 0.22},
        "statement_bundle": {"seconds": 0.03},
        "inference_gap_analysis": {"seconds": 0.14},
        "output_generation": {"seconds": 0.05},
    }
    manifest = json.loads(
        (tmp_path / "artifacts/phase81/run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["execution_summary"] == result["execution_summary"]
    assert manifest["execution_summary"]["total_seconds"] == manifest["seconds"]
