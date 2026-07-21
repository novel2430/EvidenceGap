from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    sha256_file,
)
from evidencegap.retrieval.contracts import SearchHit, SparseRetriever
from evidencegap.retrieval.tokenization import (
    TOKENIZER_CONTRACT,
    ParquetTextStream,
    build_tokenizer,
    normalize_for_search,
)

INDEX_SCHEMA_VERSION = "1.0.0"
DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")
DEFAULT_INDEX_DIR = Path("artifacts/v1/bm25_index")


def _bm25s():
    try:
        import bm25s
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing bm25s. Install requirements/v1-phase02.txt"
        ) from exc
    return bm25s


def build_bm25s_index(
    root: Path,
    *,
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
    force: bool = False,
    k1: float = 1.2,
    b: float = 0.75,
    method: str = "lucene",
    backend: str = "auto",
    csc_backend: str = "auto",
    batch_size: int = 8192,
    max_docs: int | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    index_dir = (root / (index_dir or DEFAULT_INDEX_DIR)).resolve()
    articles_path = corpus_dir / "articles.parquet"
    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    if not articles_path.exists() or not corpus_manifest_path.exists():
        raise EvidenceGapError("Build the Phase 02 article corpus first")

    corpus_manifest = load_json(corpus_manifest_path)
    corpus_rows = pq.read_metadata(articles_path).num_rows
    num_docs = min(corpus_rows, max_docs) if max_docs else corpus_rows
    if num_docs <= 0:
        raise EvidenceGapError("Article corpus is empty")

    bm25s = _bm25s()
    with atomic_directory(index_dir, force=force) as staging:
        model_dir = staging / "index"
        model_dir.mkdir(parents=True, exist_ok=True)

        text_stream = ParquetTextStream(
            articles_path,
            batch_size=batch_size,
            limit=max_docs,
        )
        tokenizer = build_tokenizer()
        tokenized = tokenizer.tokenize(
            text_stream,
            update_vocab=True,
            return_as="tuple",
            show_progress=True,
            length=num_docs,
            allow_empty=True,
        )

        retriever = bm25s.BM25(
            k1=k1,
            b=b,
            method=method,
            backend=backend,
            csc_backend=csc_backend,
        )
        retriever.index(tokenized, show_progress=True)
        retriever.save(model_dir, show_progress=True)
        tokenizer.save_vocab(model_dir)
        tokenizer.save_stopwords(model_dir)

        article_table = pq.read_table(
            articles_path,
            columns=["article_id"],
        ).slice(0, num_docs)
        article_ids_list = [str(v) for v in article_table["article_id"].to_pylist()]
        max_len = max(len(value) for value in article_ids_list)
        article_ids = np.asarray(article_ids_list, dtype=f"<U{max_len}")
        article_ids_path = staging / "article_ids.npy"
        np.save(article_ids_path, article_ids, allow_pickle=False)

        model_files = sorted(path for path in model_dir.rglob("*") if path.is_file())
        manifest = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "backend": "bm25s",
            "bm25s_version": getattr(bm25s, "__version__", "unknown"),
            "parameters": {
                "method": method,
                "k1": k1,
                "b": b,
                "backend": retriever.backend,
                "csc_backend": retriever.csc_backend,
            },
            "tokenizer": TOKENIZER_CONTRACT,
            "corpus": {
                "path": relative_path(root, articles_path),
                "rows_available": corpus_rows,
                "rows_indexed": num_docs,
                "articles_sha256": sha256_file(articles_path),
                "corpus_manifest_sha256": sha256_file(corpus_manifest_path),
                "corpus_schema_version": corpus_manifest.get("schema_version"),
            },
            "index": {
                "vocab_size": len(tokenizer.get_vocab_dict()),
                "files": {
                    relative_path(staging, path): {
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in model_files
                },
                "article_ids": {
                    "path": article_ids_path.name,
                    "bytes": article_ids_path.stat().st_size,
                    "sha256": sha256_file(article_ids_path),
                },
            },
        }
        atomic_write_json(staging / "index_manifest.json", manifest)

    return manifest


class BM25SBackend(SparseRetriever):
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

    def _query_ids(self, query: str) -> list[int]:
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
        if top_k <= 0:
            return []
        excluded = {int(value) for value in exclude_doc_indices}
        requested = min(
            len(self.article_ids),
            top_k + len(excluded) + 8,
        )
        query_ids = [self._query_ids(query)]
        result = self.retriever.retrieve(
            query_ids,
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
        """Score a small candidate set directly against CSC postings.

        This avoids allocating a full corpus-sized score vector for judged
        candidate ranking. BM25S stores document IDs sorted within each token
        posting list, so searchsorted provides deterministic sparse lookup.
        """
        if not doc_indices:
            return []
        candidates = np.asarray([int(v) for v in doc_indices], dtype=np.int64)
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


def validate_bm25s_index(
    root: Path,
    *,
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    index_dir = (root / (index_dir or DEFAULT_INDEX_DIR)).resolve()
    manifest = load_json(index_dir / "index_manifest.json")
    errors: list[str] = []

    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        errors.append("index schema version mismatch")
    articles_path = corpus_dir / "articles.parquet"
    if manifest.get("corpus", {}).get("articles_sha256") != sha256_file(articles_path):
        errors.append("article corpus fingerprint mismatch")

    article_ids_path = index_dir / "article_ids.npy"
    article_meta = manifest.get("index", {}).get("article_ids", {})
    if not article_ids_path.exists():
        errors.append("missing article_ids.npy")
    elif article_meta.get("sha256") != sha256_file(article_ids_path):
        errors.append("article_ids.npy fingerprint mismatch")

    for relative, metadata in manifest.get("index", {}).get("files", {}).items():
        path = index_dir / relative
        if not path.exists():
            errors.append(f"missing index file: {relative}")
        elif metadata.get("sha256") != sha256_file(path):
            errors.append(f"index fingerprint mismatch: {relative}")

    if not errors:
        backend = BM25SBackend(index_dir, mmap=True)
        if len(backend.article_ids) != manifest["corpus"]["rows_indexed"]:
            errors.append("loaded index row count mismatch")

    report = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "index": relative_path(root, index_dir),
    }
    if errors:
        raise EvidenceGapError("; ".join(errors))
    return report
