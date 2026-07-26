from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from evidencegap_backend.common import EvidenceGapError, load_json
from evidencegap_backend.retrieval.contracts import SearchHit, SparseRetriever
from evidencegap_backend.retrieval.tokenization import (
    build_tokenizer,
    normalize_for_search,
)


def _bm25s():
    try:
        import bm25s
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing bm25s. Install the backend runtime dependencies"
        ) from exc
    return bm25s


class BM25SBackend(SparseRetriever):
    """Load and query an existing Phase 02 BM25S article index."""

    def __init__(self, index_dir: Path, *, mmap: bool = True) -> None:
        bm25s = _bm25s()
        self.index_dir = Path(index_dir).resolve()
        self.manifest = load_json(self.index_dir / "index_manifest.json")
        self.retriever = bm25s.BM25.load(
            self.index_dir / "index",
            mmap=mmap,
            load_corpus=False,
        )
        self.tokenizer = build_tokenizer()
        self.tokenizer.load_vocab(self.index_dir / "index")
        self.tokenizer.load_stopwords(self.index_dir / "index")
        self.article_ids = np.load(
            self.index_dir / "article_ids.npy",
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )
        if len(self.article_ids) != self.retriever.scores["num_docs"]:
            raise EvidenceGapError("BM25 index and article ID map have different sizes")

    @property
    def loaded(self) -> bool:
        return self.retriever is not None

    def close(self) -> None:
        self.retriever = None
        self.tokenizer = None
        self.article_ids = None

    def _require_loaded(self) -> None:
        if self.retriever is None or self.tokenizer is None or self.article_ids is None:
            raise EvidenceGapError("BM25 backend is closed")

    def _query_ids(self, query: str) -> list[int]:
        self._require_loaded()
        tokenized = self.tokenizer.tokenize(
            [normalize_for_search(query)],
            update_vocab="never",
            return_as="ids",
            show_progress=False,
            allow_empty=True,
        )
        return list(tokenized[0])

    def search(
        self,
        query: str,
        *,
        top_k: int,
        exclude_doc_indices: Sequence[int] = (),
    ) -> list[SearchHit]:
        self._require_loaded()
        if top_k <= 0:
            return []
        excluded = {int(value) for value in exclude_doc_indices}
        requested = min(len(self.article_ids), top_k + len(excluded) + 8)
        result = self.retriever.retrieve(
            [self._query_ids(query)],
            k=requested,
            sorted=True,
            show_progress=False,
            return_as="tuple",
        )
        hits: list[SearchHit] = []
        for doc_idx, score in zip(result.documents[0], result.scores[0]):
            index = int(doc_idx)
            if index in excluded:
                continue
            hits.append(
                SearchHit(
                    doc_idx=index,
                    article_id=str(self.article_ids[index]),
                    score=float(score),
                    rank=len(hits) + 1,
                )
            )
            if len(hits) >= top_k:
                break
        return hits

    def score_documents(
        self,
        query: str,
        doc_indices: Sequence[int],
    ) -> list[float]:
        self._require_loaded()
        if not doc_indices:
            return []
        candidates = np.asarray([int(value) for value in doc_indices], dtype=np.int64)
        if candidates.min() < 0 or candidates.max() >= len(self.article_ids):
            raise EvidenceGapError("Candidate doc_idx outside BM25 index")

        order = np.argsort(candidates, kind="stable")
        sorted_candidates = candidates[order]
        sorted_scores = np.zeros(len(candidates), dtype=np.float64)
        query_ids = self._query_ids(query)

        data = self.retriever.scores["data"]
        indices = self.retriever.scores["indices"]
        indptr = self.retriever.scores["indptr"]
        for token_id in query_ids:
            if token_id < 0 or token_id + 1 >= len(indptr):
                continue
            start = int(indptr[token_id])
            end = int(indptr[token_id + 1])
            posting_docs = indices[start:end]
            positions = np.searchsorted(posting_docs, sorted_candidates)
            valid = positions < len(posting_docs)
            valid_positions = positions[valid]
            matched = np.zeros(len(candidates), dtype=bool)
            matched[valid] = posting_docs[valid_positions] == sorted_candidates[valid]
            if matched.any():
                sorted_scores[matched] += data[start + positions[matched]]

        nonoccurrence = getattr(self.retriever, "nonoccurrence_array", None)
        if nonoccurrence is not None and query_ids:
            sorted_scores += float(np.asarray(nonoccurrence)[query_ids].sum())

        scores = np.empty(len(candidates), dtype=np.float64)
        scores[order] = sorted_scores
        return scores.tolist()
