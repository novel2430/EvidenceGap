from __future__ import annotations

import copy
from pathlib import Path

import pytest

from evidencegap_backend.common import EvidenceGapError, sha256_text
from evidencegap_backend.config import BackendConfig
from evidencegap_backend.engine import EvidenceGapEngine
from evidencegap_backend.pipeline.contracts import RuntimeArticle
from evidencegap_backend.pipeline.sentence_materialization import (
    canonicalize_article_text,
)


class ArticleOnlyResources:
    def __init__(self, article: dict[str, object]) -> None:
        self.loaded = True
        self.article = article

    def status(self) -> dict[str, object]:
        return {"loaded": True, "load_count": 1, "analysis_runs": 0}

    def fetch_article_texts(self, article_ids: list[str]):
        assert article_ids == ["pmid:123"]
        return {"pmid:123": dict(self.article)}

    def close(self) -> None:
        self.loaded = False


def _presentation(article: dict[str, object]) -> dict[str, object]:
    runtime = RuntimeArticle.from_mapping(article)
    canonical = canonicalize_article_text(runtime, section_mode="none")
    evidence_text = "Vitamin D reduced infections."
    start = canonical.source_text.index(evidence_text)
    end = start + len(evidence_text)
    return {
        "articles": [
            {
                "article_node_id": "article_1",
                "article_id": "pmid:123",
                "claim_id": "claim_1",
                "rank": 1,
            }
        ],
        "evidence": [
            {
                "evidence_id": "evidence_1",
                "article_node_id": "article_1",
                "claim_id": "claim_1",
                "section": "abstract",
                "section_index": 1,
                "sentence_index": 0,
                "character_start": start,
                "character_end": end,
                "text": evidence_text,
                "source_text_fingerprint": sha256_text(canonical.source_text),
            }
        ],
    }


def test_article_context_rebuilds_and_verifies_exact_offsets(
    tmp_path: Path,
) -> None:
    article = {
        "article_id": "pmid:123",
        "pmid": "123",
        "doc_idx": 1,
        "title": "Vitamin D trial",
        "abstract": "Vitamin D reduced infections.",
    }
    engine = EvidenceGapEngine(
        BackendConfig(workspace_root=tmp_path, provider="deepseek", section_mode="none"),
        resources=ArticleOnlyResources(article),
    )

    context = engine.get_article_context(
        presentation_bundle=_presentation(article),
        article_node_id="article_1",
    )

    assert context["fingerprint_verified"] is True
    assert context["canonical_text"] == (
        "Vitamin D trial\n\nVitamin D reduced infections."
    )
    span = context["evidence_spans"][0]
    assert context["canonical_text"][
        span["character_start"] : span["character_end"]
    ] == span["text"]
    assert [section["section"] for section in context["sections"]] == [
        "title",
        "abstract",
    ]


def test_article_context_rejects_changed_source_fingerprint(
    tmp_path: Path,
) -> None:
    article = {
        "article_id": "pmid:123",
        "pmid": "123",
        "doc_idx": 1,
        "title": "Vitamin D trial",
        "abstract": "Vitamin D reduced infections.",
    }
    engine = EvidenceGapEngine(
        BackendConfig(workspace_root=tmp_path, provider="deepseek", section_mode="none"),
        resources=ArticleOnlyResources(article),
    )
    presentation = _presentation(article)
    bad = copy.deepcopy(presentation)
    bad["evidence"][0]["source_text_fingerprint"] = "0" * 64

    with pytest.raises(EvidenceGapError, match="fingerprint changed"):
        engine.get_article_context(
            presentation_bundle=bad,
            article_node_id="article_1",
        )
