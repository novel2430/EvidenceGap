from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    require_empty_or_force,
    sha256_file,
    sha256_text,
)
from evidencegap.dense.encoders import DenseEncoder, encoder_spec, model_fingerprint
from evidencegap.pipeline.sentence_materialization import (
    DEFAULT_STANZA_MODEL_DIR,
    SentenceSplitter,
    materialize_runtime_sentences,
    validate_runtime_sentence_artifact,
)

RUNTIME_RETRIEVAL_SCHEMA_VERSION = "1.0.0"
RUNTIME_RETRIEVAL_CONTRACT_ID = "phase07.retrieval-adapters.v1"
ARTICLE_RECORD_TYPE = "RuntimeArticleCandidate"
SENTENCE_RANKING_RECORD_TYPE = "RuntimeSentenceRanking"
EVIDENCE_RECORD_TYPE = "RuntimeEvidenceCandidate"

DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/retrieval_adapters")
DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")
DEFAULT_ARTICLE_INPUT_DIR = Path("artifacts/v1/dense/article_inputs")
DEFAULT_BM25_INDEX_DIR = Path("artifacts/v1/bm25_index")
DEFAULT_MEDCPT_INDEX_DIR = Path("artifacts/v1/dense/medcpt/faiss_index")
DEFAULT_BMRETRIEVER_INDEX_DIR = Path("artifacts/v1/dense/bmretriever/faiss_index")
DEFAULT_CROSS_ENCODER_MODEL_DIR = Path("models/v1/medcpt-cross")

ARTICLE_SOURCE_DEPTH = 100
ARTICLE_DENSE_NPROBE = 1024
ARTICLE_RRF_K = 60
ARTICLE_RERANK_DEPTH = 100
FINAL_ARTICLE_TOP_K = 10
SENTENCE_SOURCE_DEPTH = 20
SENTENCE_RRF_K = 10
EVIDENCE_TOP_K = 5
EVIDENCE_SENTENCE_ELIGIBILITY_POLICY_ID = (
    "phase07.evidence-sentence-eligibility.v1"
)
_SENTENCE_FINAL_PUNCTUATION = re.compile(r"[.!?][\"'’”\)\]]*\s*$")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty")
    return cleaned


def _clean_claim(value: str) -> str:
    claim = " ".join(str(value).strip().split())
    if not claim:
        raise EvidenceGapError("claim cannot be empty")
    return claim


def runtime_claim_id(claim_text: str) -> str:
    return "claim_" + sha256_text(_clean_claim(claim_text))[:24]


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing pyarrow. Install requirements/v1-phase07.txt and the Phase 02-05 requirements"
        ) from exc
    return pa, pq


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing duckdb. Install requirements/v1-phase02.txt"
        ) from exc
    return duckdb


def _write_parquet_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    if not rows:
        raise EvidenceGapError(f"Refusing to write empty Parquet artifact: {path}")
    pa, pq = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, temp, compression="zstd")
    os.replace(temp, path)
    return int(table.num_rows)


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    os.replace(temp, path)
    return count


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    _pa, pq = _pyarrow()
    try:
        return [dict(row) for row in pq.read_table(path).to_pylist()]
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing Parquet artifact: {path}") from exc


def _resolve(root: Path, value: Path | None, default: Path) -> Path:
    return (root / (value or default)).resolve()


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _release_dense_encoder(encoder: DenseEncoder) -> None:
    torch = getattr(encoder, "torch", None)
    for attribute in ("_query_bundle", "_article_bundle"):
        bundle = getattr(encoder, attribute, None)
        if bundle is not None:
            try:
                bundle[1].to("cpu")
            except Exception:
                pass
            setattr(encoder, attribute, None)
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def fuse_article_rankings(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    aliases: Sequence[str] = ("bm25", "medcpt", "bmretriever"),
    rrf_k: int = ARTICLE_RRF_K,
) -> list[dict[str, Any]]:
    """Equal-weight Phase 04 RRF with the frozen deterministic tie-breaks."""
    if rrf_k < 0:
        raise EvidenceGapError("article rrf_k must be non-negative")
    candidates: dict[str, dict[str, Any]] = {}
    for source_index, alias in enumerate(aliases):
        rows = source_rows.get(alias)
        if rows is None:
            raise EvidenceGapError(f"Missing article retrieval source: {alias}")
        seen: set[str] = set()
        for fallback_rank, raw in enumerate(rows, start=1):
            article_id = str(raw.get("article_id", "")).strip()
            if not article_id:
                raise EvidenceGapError(f"{alias}: article_id cannot be empty")
            if article_id in seen:
                raise EvidenceGapError(f"{alias}: duplicate article_id {article_id}")
            seen.add(article_id)
            rank = int(raw.get("rank", fallback_rank))
            if rank <= 0:
                raise EvidenceGapError(f"{alias}: rank must be positive")
            doc_idx = int(raw["doc_idx"])
            score = float(raw["score"])
            if not math.isfinite(score):
                raise EvidenceGapError(f"{alias}: non-finite retrieval score")
            item = candidates.setdefault(
                article_id,
                {
                    "article_id": article_id,
                    "doc_idx": doc_idx,
                    "source_mask": 0,
                    "source_count": 0,
                    "best_rank": rank,
                    "rank_sum": 0,
                    "rrf_score": 0.0,
                },
            )
            if int(item["doc_idx"]) != doc_idx:
                raise EvidenceGapError(
                    f"Article index drift for {article_id}: {item['doc_idx']} != {doc_idx}"
                )
            if f"{alias}_rank" in item:
                raise EvidenceGapError(f"Duplicate {alias} row for {article_id}")
            item[f"{alias}_rank"] = rank
            item[f"{alias}_score"] = score
            item["source_mask"] |= 1 << source_index
            item["source_count"] += 1
            item["best_rank"] = min(int(item["best_rank"]), rank)
            item["rank_sum"] += rank
            item["rrf_score"] += 1.0 / (rrf_k + rank)

    values = list(candidates.values())
    values.sort(
        key=lambda row: (
            -float(row["rrf_score"]),
            -int(row["source_count"]),
            int(row["best_rank"]),
            int(row["rank_sum"]),
            str(row["article_id"]),
        )
    )
    for fusion_rank, row in enumerate(values, start=1):
        row["fusion_rank"] = fusion_rank
        for alias in aliases:
            row.setdefault(f"{alias}_rank", None)
            row.setdefault(f"{alias}_score", None)
    return values


def fuse_sentence_rankings(
    source_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    aliases: Sequence[str] = ("bmretriever", "medcpt"),
    source_depth: int = SENTENCE_SOURCE_DEPTH,
    rrf_k: int = SENTENCE_RRF_K,
) -> list[dict[str, Any]]:
    """Paper-local Phase 05 equal-weight RRF over RuntimeSentence IDs."""
    if source_depth <= 0 or rrf_k < 0:
        raise EvidenceGapError("sentence source_depth must be positive and rrf_k non-negative")
    ranks: dict[str, dict[str, Any]] = {}
    for alias in aliases:
        rows = source_rows.get(alias)
        if rows is None:
            raise EvidenceGapError(f"Missing sentence retrieval source: {alias}")
        seen: set[str] = set()
        for fallback_rank, raw in enumerate(rows[:source_depth], start=1):
            sentence_id = str(raw.get("sentence_id", "")).strip()
            if not sentence_id:
                raise EvidenceGapError(f"{alias}: sentence_id cannot be empty")
            if sentence_id in seen:
                raise EvidenceGapError(f"{alias}: duplicate sentence_id {sentence_id}")
            seen.add(sentence_id)
            rank = int(raw.get("retrieval_rank", fallback_rank))
            score = float(raw["retrieval_score"])
            if not math.isfinite(score):
                raise EvidenceGapError(f"{alias}: non-finite sentence score")
            item = ranks.setdefault(
                sentence_id,
                {
                    "sentence_id": sentence_id,
                    "sentence_index": int(raw["sentence_index"]),
                    "sentence_pool_fingerprint": raw.get("sentence_pool_fingerprint"),
                    "source_count": 0,
                    "best_rank": rank,
                    "rrf_score": 0.0,
                },
            )
            if int(item["sentence_index"]) != int(raw["sentence_index"]):
                raise EvidenceGapError(f"Sentence index drift for {sentence_id}")
            if item.get("sentence_pool_fingerprint") != raw.get(
                "sentence_pool_fingerprint"
            ):
                raise EvidenceGapError(f"Sentence pool drift for {sentence_id}")
            item[f"{alias}_rank"] = rank
            item[f"{alias}_score"] = score
            item["source_count"] += 1
            item["best_rank"] = min(int(item["best_rank"]), rank)
            item["rrf_score"] += 1.0 / (rrf_k + rank)

    values = list(ranks.values())
    values.sort(
        key=lambda row: (
            -float(row["rrf_score"]),
            -int(row["source_count"]),
            int(row["best_rank"]),
            int(row["sentence_index"]),
        )
    )
    for rank, row in enumerate(values, start=1):
        row["evidence_rank_within_article"] = rank
        for alias in aliases:
            row.setdefault(f"{alias}_rank", None)
            row.setdefault(f"{alias}_score", None)
    return values


def _query_dense_source(
    root: Path,
    *,
    model_key: str,
    claim_text: str,
    device: str,
    amp: str,
    index_dir: Path,
    article_ids: Sequence[str],
    expected_article_input_sha256: str,
    nprobe: int,
    top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    encoder = DenseEncoder(root, model_key, device=device, amp=amp)
    try:
        query = encoder.encode_queries([claim_text])
        from evidencegap.dense.faiss_backend import DenseFaissBackend

        backend = DenseFaissBackend(root, index_dir, nprobe=nprobe)
        if backend.manifest.get("model_key") != model_key:
            raise EvidenceGapError(
                f"{model_key} FAISS manifest model mismatch: "
                f"{backend.manifest.get('model_key')}"
            )
        index_path = index_dir / "index.faiss"
        expected_index_sha256 = str(
            backend.manifest.get("index", {}).get("sha256", "")
        )
        if not expected_index_sha256 or sha256_file(index_path) != expected_index_sha256:
            raise EvidenceGapError(f"{model_key} FAISS index checksum mismatch")
        if int(backend.index.ntotal) != len(article_ids):
            raise EvidenceGapError(
                f"{model_key} FAISS/article ID row mismatch: "
                f"{backend.index.ntotal} != {len(article_ids)}"
            )
        embedding_manifest_value = backend.manifest.get(
            "article_embedding_manifest"
        )
        if not embedding_manifest_value:
            raise EvidenceGapError(
                f"{model_key} FAISS manifest has no article embedding manifest"
            )
        embedding_manifest_path = Path(str(embedding_manifest_value))
        if not embedding_manifest_path.is_absolute():
            embedding_manifest_path = root / embedding_manifest_path
        embedding_manifest = load_json(embedding_manifest_path)
        if embedding_manifest.get("article_input_sha256") != expected_article_input_sha256:
            raise EvidenceGapError(
                f"{model_key} FAISS index was built from a different article input"
            )
        scores, ids = backend.search(query, top_k=top_k)
        rows: list[dict[str, Any]] = []
        for rank, (doc_idx_value, score_value) in enumerate(
            zip(ids[0], scores[0], strict=True), start=1
        ):
            doc_idx = int(doc_idx_value)
            if doc_idx < 0:
                continue
            if doc_idx >= len(article_ids):
                raise EvidenceGapError(
                    f"{model_key} FAISS returned out-of-range doc_idx {doc_idx}"
                )
            rows.append(
                {
                    "rank": rank,
                    "doc_idx": doc_idx,
                    "article_id": str(article_ids[doc_idx]),
                    "score": float(score_value),
                }
            )
        metadata = {
            "model_key": model_key,
            "query_model_path": relative_path(root, encoder.spec.query_model),
            "query_model_fingerprint": model_fingerprint(encoder.spec, article=False),
            "index_path": relative_path(root, index_dir),
            "index_manifest_sha256": sha256_file(index_dir / "index_manifest.json"),
            "article_embedding_manifest_path": relative_path(
                root, embedding_manifest_path
            ),
            "article_embedding_manifest_sha256": sha256_file(
                embedding_manifest_path
            ),
            "article_input_sha256": expected_article_input_sha256,
            "requested_nprobe": nprobe,
            "actual_nprobe": backend.nprobe,
            "device": device,
            "amp": amp,
            "rows": len(rows),
            "seconds": round(time.perf_counter() - started, 6),
        }
        return rows, metadata
    finally:
        _release_dense_encoder(encoder)


def _load_article_texts(
    *,
    article_ids: Sequence[str],
    article_input_path: Path,
    corpus_articles_path: Path,
) -> dict[str, dict[str, Any]]:
    if not article_input_path.exists():
        raise EvidenceGapError(f"Missing Phase 03 article inputs: {article_input_path}")
    if not corpus_articles_path.exists():
        raise EvidenceGapError(f"Missing Phase 02 article corpus: {corpus_articles_path}")
    pa, _pq = _pyarrow()
    candidate_table = pa.Table.from_pylist(
        [{"article_id": str(article_id)} for article_id in article_ids]
    )
    connection = _duckdb().connect()
    try:
        connection.register("runtime_candidates", candidate_table)
        rows = connection.execute(
            f"""
            SELECT
                CAST(i.article_id AS VARCHAR) AS article_id,
                CAST(i.doc_idx AS BIGINT) AS doc_idx,
                CASE WHEN a.pmid IS NULL THEN NULL ELSE CAST(a.pmid AS VARCHAR) END AS pmid,
                coalesce(CAST(i.title AS VARCHAR), '') AS title,
                coalesce(CAST(i.abstract AS VARCHAR), '') AS abstract
            FROM read_parquet('{_quote(article_input_path)}') i
            JOIN runtime_candidates c
              ON CAST(i.article_id AS VARCHAR) = c.article_id
            LEFT JOIN read_parquet('{_quote(corpus_articles_path)}') a
              ON CAST(i.article_id AS VARCHAR) = CAST(a.article_id AS VARCHAR)
            """
        ).fetch_arrow_table().to_pylist()
    finally:
        connection.close()
    result = {str(row["article_id"]): dict(row) for row in rows}
    missing = [article_id for article_id in article_ids if article_id not in result]
    if missing:
        raise EvidenceGapError(
            f"Article input join missed {len(missing)} candidates; first={missing[0]}"
        )
    return result


def retrieve_runtime_articles(
    root: Path,
    *,
    claim_id: str,
    claim_text: str,
    artifact_dir: Path,
    device: str = "cuda:0",
    amp: str = "fp16",
    corpus_dir: Path | None = None,
    article_input_dir: Path | None = None,
    bm25_index_dir: Path | None = None,
    medcpt_index_dir: Path | None = None,
    bmretriever_index_dir: Path | None = None,
    cross_encoder_model_dir: Path | None = None,
    cross_encoder_batch_size: int = 16,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    claim_text = _clean_claim(claim_text)
    corpus_dir = _resolve(root, corpus_dir, DEFAULT_CORPUS_DIR)
    article_input_dir = _resolve(root, article_input_dir, DEFAULT_ARTICLE_INPUT_DIR)
    bm25_index_dir = _resolve(root, bm25_index_dir, DEFAULT_BM25_INDEX_DIR)
    medcpt_index_dir = _resolve(root, medcpt_index_dir, DEFAULT_MEDCPT_INDEX_DIR)
    bmretriever_index_dir = _resolve(
        root, bmretriever_index_dir, DEFAULT_BMRETRIEVER_INDEX_DIR
    )
    cross_encoder_model_dir = _resolve(
        root, cross_encoder_model_dir, DEFAULT_CROSS_ENCODER_MODEL_DIR
    )
    artifact_dir = artifact_dir.resolve()

    started = time.perf_counter()
    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    article_input_manifest_path = article_input_dir / "article_inputs_manifest.json"
    corpus_manifest = load_json(corpus_manifest_path)
    article_input_manifest = load_json(article_input_manifest_path)
    expected_corpus_articles_sha256 = str(
        corpus_manifest.get("files", {})
        .get("articles.parquet", {})
        .get("sha256", "")
    )
    expected_article_input_sha256 = str(
        article_input_manifest.get("output", {}).get("sha256", "")
    )
    if not expected_corpus_articles_sha256 or not expected_article_input_sha256:
        raise EvidenceGapError("Article corpus/input manifests are incomplete")
    if article_input_manifest.get("source_corpus_manifest_sha256") != sha256_file(
        corpus_manifest_path
    ):
        raise EvidenceGapError("Phase 03 article inputs do not match the Phase 02 corpus")

    from evidencegap.retrieval.bm25s_backend import BM25SBackend

    bm25_backend = BM25SBackend(bm25_index_dir, mmap=True)
    article_ids_metadata = bm25_backend.manifest.get("index", {}).get("article_ids", {})
    if article_ids_metadata.get("sha256") != sha256_file(
        bm25_index_dir / "article_ids.npy"
    ):
        raise EvidenceGapError("BM25 article ID map checksum mismatch")
    if bm25_backend.manifest.get("corpus", {}).get(
        "articles_sha256"
    ) != expected_corpus_articles_sha256:
        raise EvidenceGapError("BM25 index does not match the Phase 02 article corpus")
    bm25_hits = bm25_backend.search(claim_text, top_k=ARTICLE_SOURCE_DEPTH)
    bm25_rows = [
        {
            "rank": int(hit.rank),
            "doc_idx": int(hit.doc_idx),
            "article_id": str(hit.article_id),
            "score": float(hit.score),
        }
        for hit in bm25_hits
    ]
    article_ids = bm25_backend.article_ids
    if len(bm25_rows) != ARTICLE_SOURCE_DEPTH:
        raise EvidenceGapError(
            f"BM25 returned {len(bm25_rows)} rows, expected {ARTICLE_SOURCE_DEPTH}"
        )

    medcpt_rows, medcpt_meta = _query_dense_source(
        root,
        model_key="medcpt",
        claim_text=claim_text,
        device=device,
        amp=amp,
        index_dir=medcpt_index_dir,
        article_ids=article_ids,
        expected_article_input_sha256=expected_article_input_sha256,
        nprobe=ARTICLE_DENSE_NPROBE,
        top_k=ARTICLE_SOURCE_DEPTH,
    )
    bmretriever_rows, bmretriever_meta = _query_dense_source(
        root,
        model_key="bmretriever",
        claim_text=claim_text,
        device=device,
        amp=amp,
        index_dir=bmretriever_index_dir,
        article_ids=article_ids,
        expected_article_input_sha256=expected_article_input_sha256,
        nprobe=ARTICLE_DENSE_NPROBE,
        top_k=ARTICLE_SOURCE_DEPTH,
    )
    for label, rows in (("medcpt", medcpt_rows), ("bmretriever", bmretriever_rows)):
        if len(rows) != ARTICLE_SOURCE_DEPTH:
            raise EvidenceGapError(
                f"{label} returned {len(rows)} rows, expected {ARTICLE_SOURCE_DEPTH}"
            )

    fused = fuse_article_rankings(
        {"bm25": bm25_rows, "medcpt": medcpt_rows, "bmretriever": bmretriever_rows},
        rrf_k=ARTICLE_RRF_K,
    )[:ARTICLE_RERANK_DEPTH]
    if len(fused) != ARTICLE_RERANK_DEPTH:
        raise EvidenceGapError(
            f"Article RRF produced {len(fused)} candidates, expected {ARTICLE_RERANK_DEPTH}"
        )
    article_text = _load_article_texts(
        article_ids=[str(row["article_id"]) for row in fused],
        article_input_path=article_input_dir / "article_inputs.parquet",
        corpus_articles_path=corpus_dir / "articles.parquet",
    )
    for row in fused:
        source = article_text[str(row["article_id"])]
        if int(row["doc_idx"]) != int(source["doc_idx"]):
            raise EvidenceGapError(
                f"Article doc_idx mismatch for {row['article_id']}: "
                f"{row['doc_idx']} != {source['doc_idx']}"
            )
        row.update(
            {
                "pmid": source.get("pmid"),
                "title": str(source.get("title") or ""),
                "abstract": str(source.get("abstract") or ""),
            }
        )

    from evidencegap.reranking.cross_encoder import score_runtime_article_pairs

    reranked = score_runtime_article_pairs(
        root,
        claim_text=claim_text,
        articles=fused,
        model_dir=cross_encoder_model_dir,
        device=device,
        batch_size=cross_encoder_batch_size,
        max_length=512,
        amp=amp,
    )
    ce_scores = {
        str(row["article_id"]): float(row["cross_encoder_score"])
        for row in reranked["scores"]
    }
    if set(ce_scores) != {str(row["article_id"]) for row in fused}:
        raise EvidenceGapError("Cross-encoder output does not cover the RRF Top-100")
    for row in fused:
        row["cross_encoder_score"] = ce_scores[str(row["article_id"])]
    fused.sort(
        key=lambda row: (
            -float(row["cross_encoder_score"]),
            int(row["fusion_rank"]),
            str(row["article_id"]),
        )
    )

    candidate_rows: list[dict[str, Any]] = []
    for final_rank, row in enumerate(fused, start=1):
        candidate_rows.append(
            {
                "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
                "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
                "record_type": ARTICLE_RECORD_TYPE,
                "claim_id": claim_id,
                "claim_text": claim_text,
                **row,
                "final_article_rank": final_rank,
            }
        )
    runtime_rows = candidate_rows[:FINAL_ARTICLE_TOP_K]

    with atomic_directory(artifact_dir, force=force) as staging:
        candidates_path = staging / "article_candidates.parquet"
        selected_path = staging / "top_articles.parquet"
        input_path = staging / "runtime_articles.jsonl"
        candidate_count = _write_parquet_atomic(candidates_path, candidate_rows)
        selected_count = _write_parquet_atomic(selected_path, runtime_rows)
        runtime_input_rows = []
        for row in runtime_rows:
            runtime_input_rows.append(
                {
                    "article_id": row["article_id"],
                    "pmid": row.get("pmid"),
                    "title": row.get("title") or None,
                    "abstract": row.get("abstract") or None,
                    "final_article_rank": row["final_article_rank"],
                    "bm25_rank": row.get("bm25_rank"),
                    "bm25_score": row.get("bm25_score"),
                    "medcpt_rank": row.get("medcpt_rank"),
                    "medcpt_score": row.get("medcpt_score"),
                    "bmretriever_rank": row.get("bmretriever_rank"),
                    "bmretriever_score": row.get("bmretriever_score"),
                    "rrf_score": row["rrf_score"],
                    "fusion_rank": row["fusion_rank"],
                    "cross_encoder_score": row["cross_encoder_score"],
                }
            )
        input_count = _write_jsonl_atomic(input_path, runtime_input_rows)
        if input_count != FINAL_ARTICLE_TOP_K:
            raise EvidenceGapError("Runtime article input depth mismatch")

        outputs = {
            "article_candidates": {
                "path": relative_path(root, artifact_dir / candidates_path.name),
                "sha256": sha256_file(candidates_path),
                "rows": candidate_count,
            },
            "top_articles": {
                "path": relative_path(root, artifact_dir / selected_path.name),
                "sha256": sha256_file(selected_path),
                "rows": selected_count,
            },
            "runtime_articles_input": {
                "path": relative_path(root, artifact_dir / input_path.name),
                "sha256": sha256_file(input_path),
                "rows": input_count,
            },
        }
        manifest = {
            "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
            "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
            "run_type": "runtime_article_retrieval_adapter",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claim_id": claim_id,
            "claim_text_sha256": sha256_text(claim_text),
            "parameters": {
                "source_depth": ARTICLE_SOURCE_DEPTH,
                "dense_nprobe": ARTICLE_DENSE_NPROBE,
                "rrf_k": ARTICLE_RRF_K,
                "rerank_depth": ARTICLE_RERANK_DEPTH,
                "final_article_top_k": FINAL_ARTICLE_TOP_K,
                "fusion": "equal_weight_rrf",
                "reranker": "medcpt_cross_encoder",
                "max_length": 512,
                "amp": amp,
            },
            "sources": {
                "bm25": {
                    "index_path": relative_path(root, bm25_index_dir),
                    "index_manifest_sha256": sha256_file(
                        bm25_index_dir / "index_manifest.json"
                    ),
                    "rows": len(bm25_rows),
                },
                "medcpt": medcpt_meta,
                "bmretriever": bmretriever_meta,
                "article_inputs": {
                    "path": relative_path(
                        root, article_input_dir / "article_inputs.parquet"
                    ),
                    "manifest_sha256": sha256_file(
                        article_input_manifest_path
                    ),
                    "article_input_sha256": expected_article_input_sha256,
                    "source_corpus_manifest_sha256": sha256_file(
                        corpus_manifest_path
                    ),
                },
                "cross_encoder": reranked["metadata"],
            },
            "outputs": outputs,
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_write_json(staging / "run_manifest.json", manifest)

    return {
        "status": "PASS",
        "artifact_dir": relative_path(root, artifact_dir),
        "claim_id": claim_id,
        "article_candidates": len(candidate_rows),
        "top_articles": len(runtime_rows),
        "outputs": outputs,
        "manifest": manifest,
    }


def _rank_sentences_for_model(
    root: Path,
    *,
    model_key: str,
    claim_id: str,
    claim_text: str,
    device: str,
    amp: str,
    batch_size: int,
    sentences: Sequence[Mapping[str, Any]],
    article_order: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    encoder = DenseEncoder(root, model_key, device=device, amp=amp)
    try:
        query_vector = encoder.encode_queries([claim_text], batch_size=batch_size)[0]
        matrix = encoder.encode_articles(
            [("", str(row["sentence_text"])) for row in sentences],
            batch_size=batch_size,
        )
        scores = matrix @ query_vector
        by_article_positions: dict[str, list[int]] = defaultdict(list)
        for position, row in enumerate(sentences):
            by_article_positions[str(row["article_id"])].append(position)
        pool_fingerprint_by_article = {
            article_id: sha256_text(
                json.dumps(
                    [str(sentences[position]["sentence_id"]) for position in positions],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            for article_id, positions in by_article_positions.items()
        }
        rows: list[dict[str, Any]] = []
        for article_id in article_order:
            positions = by_article_positions.get(article_id, [])
            if not positions:
                raise EvidenceGapError(f"No runtime sentences for article {article_id}")
            ranked = sorted(
                positions,
                key=lambda position: (
                    -float(scores[position]),
                    int(sentences[position]["sentence_index"]),
                ),
            )[: min(SENTENCE_SOURCE_DEPTH, len(positions))]
            for retrieval_rank, position in enumerate(ranked, start=1):
                sentence = sentences[position]
                score = float(scores[position])
                if not math.isfinite(score):
                    raise EvidenceGapError(
                        f"{model_key}: non-finite sentence score for {sentence['sentence_id']}"
                    )
                rows.append(
                    {
                        "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
                        "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
                        "record_type": SENTENCE_RANKING_RECORD_TYPE,
                        "claim_id": claim_id,
                        "claim_text": claim_text,
                        "article_id": article_id,
                        "sentence_pool_fingerprint": pool_fingerprint_by_article[
                            article_id
                        ],
                        "sentence_id": str(sentence["sentence_id"]),
                        "sentence_index": int(sentence["sentence_index"]),
                        "sentence_type": str(sentence["sentence_type"]),
                        "section": str(sentence["section"]),
                        "sentence_text": str(sentence["sentence_text"]),
                        "retrieval_model": model_key,
                        "retrieval_score": score,
                        "retrieval_rank": retrieval_rank,
                    }
                )
        spec = encoder_spec(root, model_key)
        metadata = {
            "model_key": model_key,
            "query_model_path": relative_path(root, spec.query_model),
            "article_model_path": relative_path(root, spec.article_model),
            "query_model_fingerprint": model_fingerprint(spec, article=False),
            "article_model_fingerprint": model_fingerprint(spec, article=True),
            "query_format": (
                "raw_claim"
                if model_key == "medcpt"
                else "BMR_TASK + Query: claim"
            ),
            "candidate_format": (
                "medcpt_pair(empty_title, exact_runtime_sentence)"
                if model_key == "medcpt"
                else "Represent this passage + exact_runtime_sentence"
            ),
            "device": device,
            "amp": amp,
            "batch_size": batch_size,
            "rows": len(rows),
            "seconds": round(time.perf_counter() - started, 6),
        }
        return rows, metadata
    finally:
        _release_dense_encoder(encoder)


def _evidence_id(claim_id: str, sentence_id: str) -> str:
    return "evidence_" + sha256_text(
        f"{RUNTIME_RETRIEVAL_CONTRACT_ID}\0{claim_id}\0{sentence_id}"
    )[:24]


def evidence_sentence_exclusion_reason(
    row: Mapping[str, Any],
) -> str | None:
    """Return the frozen reason a RuntimeSentence cannot enter retrieval.

    Titles remain materialized for provenance and display, but they are not
    evidence. Structured-abstract sections are recovered from inline MedFact
    headers before Stanza. When such a section ends without sentence-final
    punctuation, the source is treated as an unterminated fragment rather than
    a complete evidence statement. Plain ``section=abstract`` text is not
    filtered by this heuristic because no reliable section boundary exists.
    """

    if str(row.get("sentence_type", "")) == "title":
        return "title"
    sentence_text = str(row.get("sentence_text", "")).strip()
    if not sentence_text:
        return "empty_sentence"
    section = str(row.get("section", "")).strip().lower()
    if (
        str(row.get("sentence_type", "")) == "abstract"
        and section not in {"", "abstract"}
        and _SENTENCE_FINAL_PUNCTUATION.search(sentence_text) is None
    ):
        return "unterminated_structured_section_fragment"
    return None


def _is_evidence_eligible_sentence(row: Mapping[str, Any]) -> bool:
    return evidence_sentence_exclusion_reason(row) is None


def partition_evidence_sentences(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    eligible: list[dict[str, Any]] = []
    excluded_reason_counts: dict[str, int] = defaultdict(int)
    for raw in rows:
        row = dict(raw)
        reason = evidence_sentence_exclusion_reason(row)
        if reason is None:
            eligible.append(row)
        else:
            excluded_reason_counts[reason] += 1
    return eligible, dict(sorted(excluded_reason_counts.items()))


def retrieve_runtime_evidence(
    root: Path,
    *,
    claim_id: str,
    claim_text: str,
    runtime_sentences_path: Path,
    top_articles_path: Path,
    artifact_dir: Path,
    device: str = "cuda:0",
    amp: str = "fp16",
    medcpt_batch_size: int = 64,
    bmretriever_batch_size: int = 8,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    claim_text = _clean_claim(claim_text)
    runtime_sentences_path = runtime_sentences_path.resolve()
    top_articles_path = top_articles_path.resolve()
    artifact_dir = artifact_dir.resolve()
    sentences = _read_parquet(runtime_sentences_path)
    top_articles = sorted(
        _read_parquet(top_articles_path),
        key=lambda row: int(row["final_article_rank"]),
    )
    article_order = [str(row["article_id"]) for row in top_articles]
    article_map = {str(row["article_id"]): row for row in top_articles}
    if len(article_order) != FINAL_ARTICLE_TOP_K or len(set(article_order)) != len(article_order):
        raise EvidenceGapError("Top article artifact must contain exactly 10 unique articles")
    sentence_map = {str(row["sentence_id"]): row for row in sentences}
    if len(sentence_map) != len(sentences):
        raise EvidenceGapError("Runtime sentence artifact contains duplicate sentence IDs")
    extra_articles = sorted({str(row["article_id"]) for row in sentences} - set(article_order))
    if extra_articles:
        raise EvidenceGapError(
            f"Runtime sentence artifact contains unexpected articles: {extra_articles[:3]}"
        )
    eligible_sentences, excluded_reason_counts = partition_evidence_sentences(
        sentences
    )
    eligible_counts: dict[str, int] = defaultdict(int)
    for row in eligible_sentences:
        eligible_counts[str(row["article_id"])] += 1
    missing_eligible = [
        article_id for article_id in article_order if eligible_counts.get(article_id, 0) == 0
    ]
    if missing_eligible:
        raise EvidenceGapError(
            "Top articles must expose at least one non-title runtime sentence for evidence retrieval: "
            f"{missing_eligible[:3]}"
        )

    started = time.perf_counter()
    bmretriever_rows, bmretriever_meta = _rank_sentences_for_model(
        root,
        model_key="bmretriever",
        claim_id=claim_id,
        claim_text=claim_text,
        device=device,
        amp=amp,
        batch_size=bmretriever_batch_size,
        sentences=eligible_sentences,
        article_order=article_order,
    )
    medcpt_rows, medcpt_meta = _rank_sentences_for_model(
        root,
        model_key="medcpt",
        claim_id=claim_id,
        claim_text=claim_text,
        device=device,
        amp=amp,
        batch_size=medcpt_batch_size,
        sentences=eligible_sentences,
        article_order=article_order,
    )
    by_model_article: dict[str, dict[str, list[dict[str, Any]]]] = {
        "bmretriever": defaultdict(list),
        "medcpt": defaultdict(list),
    }
    for row in bmretriever_rows:
        by_model_article["bmretriever"][str(row["article_id"])].append(row)
    for row in medcpt_rows:
        by_model_article["medcpt"][str(row["article_id"])].append(row)

    evidence_rows: list[dict[str, Any]] = []
    for article_id in article_order:
        fused = fuse_sentence_rankings(
            {
                "bmretriever": by_model_article["bmretriever"][article_id],
                "medcpt": by_model_article["medcpt"][article_id],
            },
            source_depth=min(
                SENTENCE_SOURCE_DEPTH,
                eligible_counts[article_id],
            ),
            rrf_k=SENTENCE_RRF_K,
        )
        selected = fused[: min(EVIDENCE_TOP_K, len(fused))]
        article = article_map[article_id]
        for fused_row in selected:
            sentence = sentence_map[str(fused_row["sentence_id"])]
            evidence_rows.append(
                {
                    "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
                    "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
                    "record_type": EVIDENCE_RECORD_TYPE,
                    "evidence_id": _evidence_id(claim_id, str(sentence["sentence_id"])),
                    "claim_id": claim_id,
                    "claim_text": claim_text,
                    "article_id": article_id,
                    "pmid": article.get("pmid"),
                    "final_article_rank": int(article["final_article_rank"]),
                    "article_cross_encoder_score": float(article["cross_encoder_score"]),
                    "sentence_id": str(sentence["sentence_id"]),
                    "sentence_index": int(sentence["sentence_index"]),
                    "sentence_index_within_section": int(
                        sentence["sentence_index_within_section"]
                    ),
                    "sentence_type": str(sentence["sentence_type"]),
                    "section": str(sentence["section"]),
                    "section_index": int(sentence["section_index"]),
                    "sentence_text": str(sentence["sentence_text"]),
                    "character_start": int(sentence["character_start"]),
                    "character_end": int(sentence["character_end"]),
                    "source_text_fingerprint": str(
                        sentence["source_text_fingerprint"]
                    ),
                    "splitter_fingerprint": str(sentence["splitter_fingerprint"]),
                    **fused_row,
                }
            )

    validation = validate_runtime_evidence_rows(
        evidence_rows,
        sentences=sentences,
        top_articles=top_articles,
    )
    source_rows = [*bmretriever_rows, *medcpt_rows]
    with atomic_directory(artifact_dir, force=force) as staging:
        source_path = staging / "sentence_rankings.parquet"
        evidence_path = staging / "evidence_candidates.parquet"
        preview_path = staging / "evidence_candidates.jsonl"
        source_count = _write_parquet_atomic(source_path, source_rows)
        evidence_count = _write_parquet_atomic(evidence_path, evidence_rows)
        preview_count = _write_jsonl_atomic(preview_path, evidence_rows)
        if evidence_count != preview_count:
            raise EvidenceGapError("Evidence Parquet and JSONL counts disagree")
        outputs = {
            "sentence_rankings": {
                "path": relative_path(root, artifact_dir / source_path.name),
                "sha256": sha256_file(source_path),
                "rows": source_count,
            },
            "evidence_candidates": {
                "path": relative_path(root, artifact_dir / evidence_path.name),
                "sha256": sha256_file(evidence_path),
                "rows": evidence_count,
            },
            "evidence_candidates_preview": {
                "path": relative_path(root, artifact_dir / preview_path.name),
                "sha256": sha256_file(preview_path),
                "rows": preview_count,
            },
        }
        manifest = {
            "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
            "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
            "run_type": "runtime_sentence_retrieval_adapter",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "claim_id": claim_id,
            "claim_text_sha256": sha256_text(claim_text),
            "inputs": {
                "runtime_sentences": {
                    "path": relative_path(root, runtime_sentences_path),
                    "sha256": sha256_file(runtime_sentences_path),
                },
                "top_articles": {
                    "path": relative_path(root, top_articles_path),
                    "sha256": sha256_file(top_articles_path),
                },
            },
            "parameters": {
                "retrievers": ["bmretriever", "medcpt"],
                "source_depth_per_article": SENTENCE_SOURCE_DEPTH,
                "fusion": "equal_weight_rrf",
                "rrf_k": SENTENCE_RRF_K,
                "evidence_top_k_per_article": EVIDENCE_TOP_K,
                "pooling": "per_article_independent",
                "cross_encoder": None,
                "sentence_eligibility": {
                    "policy_id": EVIDENCE_SENTENCE_ELIGIBILITY_POLICY_ID,
                    "exclude_titles": True,
                    "exclude_unterminated_structured_sections": True,
                    "plain_abstract_policy": "no_fragment_heuristic",
                },
            },
            "models": {
                "bmretriever": bmretriever_meta,
                "medcpt": medcpt_meta,
            },
            "counts": {
                "articles": len(article_order),
                "runtime_sentences": len(sentences),
                "eligible_runtime_sentences": len(eligible_sentences),
                "excluded_runtime_sentences": sum(excluded_reason_counts.values()),
                "excluded_sentence_reasons": dict(
                    sorted(excluded_reason_counts.items())
                ),
                "source_ranking_rows": len(source_rows),
                "evidence_candidates": len(evidence_rows),
            },
            "validation": validation,
            "outputs": outputs,
            "seconds": round(time.perf_counter() - started, 6),
        }
        atomic_write_json(staging / "run_manifest.json", manifest)

    return {
        "status": "PASS",
        "artifact_dir": relative_path(root, artifact_dir),
        "articles": len(article_order),
        "runtime_sentences": len(sentences),
        "evidence_candidates": len(evidence_rows),
        "validation": validation,
        "outputs": outputs,
        "manifest": manifest,
    }


def validate_runtime_evidence_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sentences: Sequence[Mapping[str, Any]],
    top_articles: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not rows:
        raise EvidenceGapError("Evidence candidate artifact cannot be empty")
    sentence_map = {str(row["sentence_id"]): row for row in sentences}
    article_ids = [str(row["article_id"]) for row in top_articles]
    by_article_sentences: dict[str, int] = defaultdict(int)
    for sentence in sentences:
        if _is_evidence_eligible_sentence(sentence):
            by_article_sentences[str(sentence["article_id"])] += 1
    by_article: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    evidence_ids: set[str] = set()
    sentence_ids: set[str] = set()
    for row in rows:
        if row.get("schema_version") != RUNTIME_RETRIEVAL_SCHEMA_VERSION:
            raise EvidenceGapError("Unexpected evidence schema_version")
        if row.get("contract_id") != RUNTIME_RETRIEVAL_CONTRACT_ID:
            raise EvidenceGapError("Unexpected evidence contract_id")
        evidence_id = str(row.get("evidence_id", ""))
        sentence_id = str(row.get("sentence_id", ""))
        article_id = str(row.get("article_id", ""))
        if not evidence_id or evidence_id in evidence_ids:
            raise EvidenceGapError(f"Duplicate or empty evidence_id: {evidence_id}")
        if not sentence_id or sentence_id in sentence_ids:
            raise EvidenceGapError(f"Duplicate or empty selected sentence_id: {sentence_id}")
        evidence_ids.add(evidence_id)
        sentence_ids.add(sentence_id)
        source = sentence_map.get(sentence_id)
        if source is None:
            raise EvidenceGapError(f"Evidence references unknown sentence: {sentence_id}")
        if article_id != str(source["article_id"]):
            raise EvidenceGapError(f"Evidence/article mismatch for {sentence_id}")
        if int(row["sentence_index"]) != int(source["sentence_index"]):
            raise EvidenceGapError(f"Evidence sentence index mismatch for {sentence_id}")
        if str(row["sentence_text"]) != str(source["sentence_text"]):
            raise EvidenceGapError(f"Evidence text mismatch for {sentence_id}")
        if str(row.get("sentence_type", "")) != str(
            source.get("sentence_type", "")
        ):
            raise EvidenceGapError(f"Evidence sentence type mismatch for {sentence_id}")
        if str(row.get("section", "")) != str(source.get("section", "")):
            raise EvidenceGapError(f"Evidence section mismatch for {sentence_id}")
        exclusion_reason = evidence_sentence_exclusion_reason(source)
        if exclusion_reason is not None:
            raise EvidenceGapError(
                f"Ineligible sentence selected as evidence ({exclusion_reason}): {sentence_id}"
            )
        if not math.isfinite(float(row["rrf_score"])):
            raise EvidenceGapError(f"Non-finite RRF score for {sentence_id}")
        if row.get("bmretriever_rank") is None and row.get("medcpt_rank") is None:
            raise EvidenceGapError(f"Evidence has no retrieval source: {sentence_id}")
        by_article[article_id].append(row)

    if set(by_article) != set(article_ids):
        raise EvidenceGapError(
            "Evidence article coverage mismatch: "
            f"expected={len(article_ids)}, actual={len(by_article)}"
        )
    for article_id in article_ids:
        article_rows = sorted(
            by_article[article_id],
            key=lambda row: int(row["evidence_rank_within_article"]),
        )
        expected = min(EVIDENCE_TOP_K, by_article_sentences[article_id])
        ranks = [int(row["evidence_rank_within_article"]) for row in article_rows]
        if ranks != list(range(1, expected + 1)):
            raise EvidenceGapError(
                f"Evidence rank/depth mismatch for {article_id}: {ranks}, expected {expected}"
            )
    return {
        "status": "PASS",
        "articles": len(by_article),
        "evidence_candidates": len(rows),
        "unique_evidence_ids": len(evidence_ids),
        "unique_sentence_ids": len(sentence_ids),
        "title_evidence_count": 0,
        "unterminated_fragment_evidence_count": 0,
        "sentence_eligibility_policy_id": EVIDENCE_SENTENCE_ELIGIBILITY_POLICY_ID,
        "per_article_ranking": True,
        "source_sentence_identity_preserved": True,
    }


def _resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def run_retrieval_adapters(
    root: Path,
    *,
    claim: str,
    run_name: str,
    device: str = "cuda:0",
    amp: str = "fp16",
    artifact_root: Path | None = None,
    corpus_dir: Path | None = None,
    article_input_dir: Path | None = None,
    bm25_index_dir: Path | None = None,
    medcpt_index_dir: Path | None = None,
    bmretriever_index_dir: Path | None = None,
    cross_encoder_model_dir: Path | None = None,
    stanza_model_dir: Path | None = None,
    stanza_package: str = "genia",
    stanza_batch_size: int = 32,
    cross_encoder_batch_size: int = 16,
    medcpt_sentence_batch_size: int = 64,
    bmretriever_sentence_batch_size: int = 8,
    section_mode: str = "auto",
    allow_cpu_fallback: bool = False,
    force: bool = False,
    splitter: SentenceSplitter | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    claim_text = _clean_claim(claim)
    claim_id = runtime_claim_id(claim_text)
    name = _safe_name(run_name)
    base = (
        artifact_root.resolve()
        if artifact_root is not None
        else (root / DEFAULT_ARTIFACT_ROOT).resolve()
    )
    target = base / name
    require_empty_or_force(target, force=force)
    target.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    request = {
        "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
        "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
        "run_name": name,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "claim_text_sha256": sha256_text(claim_text),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(target / "request.json", request)

    article_result = retrieve_runtime_articles(
        root,
        claim_id=claim_id,
        claim_text=claim_text,
        artifact_dir=target / "article_retrieval",
        device=device,
        amp=amp,
        corpus_dir=corpus_dir,
        article_input_dir=article_input_dir,
        bm25_index_dir=bm25_index_dir,
        medcpt_index_dir=medcpt_index_dir,
        bmretriever_index_dir=bmretriever_index_dir,
        cross_encoder_model_dir=cross_encoder_model_dir,
        cross_encoder_batch_size=cross_encoder_batch_size,
        force=False,
    )
    runtime_articles_input = _resolve_manifest_path(
        root,
        article_result["outputs"]["runtime_articles_input"]["path"],
    )
    sentence_result = materialize_runtime_sentences(
        root,
        input_path=runtime_articles_input,
        run_name="sentence_materialization",
        model_dir=(
            stanza_model_dir.resolve()
            if stanza_model_dir is not None
            else (root / DEFAULT_STANZA_MODEL_DIR).resolve()
        ),
        device=device,
        package=stanza_package,
        batch_size=stanza_batch_size,
        section_mode=section_mode,
        allow_cpu_fallback=allow_cpu_fallback,
        artifact_root=target,
        force=False,
        splitter=splitter,
    )
    runtime_sentences_path = _resolve_manifest_path(
        root,
        sentence_result["outputs"]["runtime_sentences"]["path"],
    )
    # Stanza is no longer needed after materialization. Release its PyTorch
    # allocations before loading BMRetriever/MedCPT sentence encoders.
    _clear_cuda_cache()
    top_articles_path = _resolve_manifest_path(
        root,
        article_result["outputs"]["top_articles"]["path"],
    )
    evidence_result = retrieve_runtime_evidence(
        root,
        claim_id=claim_id,
        claim_text=claim_text,
        runtime_sentences_path=runtime_sentences_path,
        top_articles_path=top_articles_path,
        artifact_dir=target / "evidence_retrieval",
        device=device,
        amp=amp,
        medcpt_batch_size=medcpt_sentence_batch_size,
        bmretriever_batch_size=bmretriever_sentence_batch_size,
        force=False,
    )

    outputs = {
        "request": {
            "path": relative_path(root, target / "request.json"),
            "sha256": sha256_file(target / "request.json"),
        },
        "article_retrieval_manifest": {
            "path": relative_path(root, target / "article_retrieval/run_manifest.json"),
            "sha256": sha256_file(target / "article_retrieval/run_manifest.json"),
        },
        "sentence_materialization_manifest": {
            "path": relative_path(root, target / "sentence_materialization/run_manifest.json"),
            "sha256": sha256_file(target / "sentence_materialization/run_manifest.json"),
        },
        "evidence_retrieval_manifest": {
            "path": relative_path(root, target / "evidence_retrieval/run_manifest.json"),
            "sha256": sha256_file(target / "evidence_retrieval/run_manifest.json"),
        },
        "top_articles": article_result["outputs"]["top_articles"],
        "runtime_sentences": sentence_result["outputs"]["runtime_sentences"],
        "evidence_candidates": evidence_result["outputs"]["evidence_candidates"],
        "evidence_candidates_preview": evidence_result["outputs"][
            "evidence_candidates_preview"
        ],
    }
    manifest = {
        "schema_version": RUNTIME_RETRIEVAL_SCHEMA_VERSION,
        "contract_id": RUNTIME_RETRIEVAL_CONTRACT_ID,
        "run_type": "phase07_retrieval_adapters",
        "run_name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "claim_id": claim_id,
        "claim_text_sha256": sha256_text(claim_text),
        "pipeline": [
            "phase04_runtime_article_retrieval",
            "phase07_runtime_sentence_materialization",
            "phase05_runtime_sentence_retrieval",
        ],
        "frozen_parameters": {
            "article_source_depth": ARTICLE_SOURCE_DEPTH,
            "article_dense_nprobe": ARTICLE_DENSE_NPROBE,
            "article_rrf_k": ARTICLE_RRF_K,
            "article_rerank_depth": ARTICLE_RERANK_DEPTH,
            "final_article_top_k": FINAL_ARTICLE_TOP_K,
            "sentence_source_depth_per_article": SENTENCE_SOURCE_DEPTH,
            "sentence_rrf_k": SENTENCE_RRF_K,
            "evidence_top_k_per_article": EVIDENCE_TOP_K,
        },
        "execution": {
            "device": device,
            "amp": amp,
            "cross_encoder_batch_size": cross_encoder_batch_size,
            "stanza_batch_size": stanza_batch_size,
            "medcpt_sentence_batch_size": medcpt_sentence_batch_size,
            "bmretriever_sentence_batch_size": bmretriever_sentence_batch_size,
            "section_mode": section_mode,
            "allow_cpu_fallback": allow_cpu_fallback,
        },
        "counts": {
            "top_articles": article_result["top_articles"],
            "runtime_sentences": sentence_result["sentences"],
            "evidence_candidates": evidence_result["evidence_candidates"],
        },
        "outputs": outputs,
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_write_json(target / "run_manifest.json", manifest)
    validation = validate_retrieval_adapter_artifact(target)
    return {
        "status": "PASS",
        "run_name": name,
        "claim_id": claim_id,
        "artifact_dir": relative_path(root, target),
        "top_articles": article_result["top_articles"],
        "runtime_sentences": sentence_result["sentences"],
        "evidence_candidates": evidence_result["evidence_candidates"],
        "validation": validation,
        "outputs": outputs,
    }


def validate_retrieval_adapter_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest = load_json(artifact_dir / "run_manifest.json")
    if manifest.get("schema_version") != RUNTIME_RETRIEVAL_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected retrieval-adapter schema_version")
    if manifest.get("contract_id") != RUNTIME_RETRIEVAL_CONTRACT_ID:
        raise EvidenceGapError("Unexpected retrieval-adapter contract_id")
    root_guess = artifact_dir
    while root_guess.parent != root_guess and not (root_guess / "src/evidencegap").exists():
        root_guess = root_guess.parent
    root = root_guess if (root_guess / "src/evidencegap").exists() else artifact_dir

    for label, metadata in manifest.get("outputs", {}).items():
        if not isinstance(metadata, Mapping) or "path" not in metadata or "sha256" not in metadata:
            continue
        path = _resolve_manifest_path(root, str(metadata["path"]))
        if not path.exists():
            raise EvidenceGapError(f"Missing retrieval-adapter output {label}: {path}")
        if sha256_file(path) != metadata["sha256"]:
            raise EvidenceGapError(f"Checksum mismatch for retrieval-adapter output {label}")

    sentence_validation = validate_runtime_sentence_artifact(
        artifact_dir / "sentence_materialization"
    )
    article_candidates = _read_parquet(
        artifact_dir / "article_retrieval/article_candidates.parquet"
    )
    top_articles = _read_parquet(
        artifact_dir / "article_retrieval/top_articles.parquet"
    )
    sentences = _read_parquet(
        artifact_dir / "sentence_materialization/runtime_sentences.parquet"
    )
    sentence_rankings = _read_parquet(
        artifact_dir / "evidence_retrieval/sentence_rankings.parquet"
    )
    evidence = _read_parquet(
        artifact_dir / "evidence_retrieval/evidence_candidates.parquet"
    )

    if len(article_candidates) != ARTICLE_RERANK_DEPTH:
        raise EvidenceGapError(
            f"Article candidate depth mismatch: {len(article_candidates)}"
        )
    candidate_ids = [str(row["article_id"]) for row in article_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise EvidenceGapError("Article candidate artifact contains duplicates")
    final_ranks = [int(row["final_article_rank"]) for row in article_candidates]
    if final_ranks != list(range(1, ARTICLE_RERANK_DEPTH + 1)):
        raise EvidenceGapError("Article candidate final ranks are not contiguous")
    fusion_ranks = sorted(int(row["fusion_rank"]) for row in article_candidates)
    if fusion_ranks != list(range(1, ARTICLE_RERANK_DEPTH + 1)):
        raise EvidenceGapError("Article candidate fusion ranks are not a fixed Top-100")
    for row in article_candidates:
        if not math.isfinite(float(row["cross_encoder_score"])):
            raise EvidenceGapError(
                f"Non-finite article cross-encoder score: {row['article_id']}"
            )

    article_ranks = [int(row["final_article_rank"]) for row in top_articles]
    if article_ranks != list(range(1, FINAL_ARTICLE_TOP_K + 1)):
        raise EvidenceGapError(f"Top article ranks are not contiguous: {article_ranks}")
    expected_top_ids = candidate_ids[:FINAL_ARTICLE_TOP_K]
    actual_top_ids = [str(row["article_id"]) for row in top_articles]
    if actual_top_ids != expected_top_ids:
        raise EvidenceGapError("Top article artifact is not the final-ranked Top-10")

    sentence_counts: dict[str, int] = defaultdict(int)
    for row in sentences:
        if _is_evidence_eligible_sentence(row):
            sentence_counts[str(row["article_id"])] += 1
    ranking_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in sentence_rankings:
        key = (str(row["article_id"]), str(row["retrieval_model"]))
        ranking_groups[key].append(row)
    for article_id in actual_top_ids:
        for model_key in ("bmretriever", "medcpt"):
            rows = sorted(
                ranking_groups.get((article_id, model_key), []),
                key=lambda row: int(row["retrieval_rank"]),
            )
            for row in rows:
                exclusion_reason = evidence_sentence_exclusion_reason(row)
                if exclusion_reason is not None:
                    raise EvidenceGapError(
                        f"{model_key} ranking contains ineligible sentence "
                        f"({exclusion_reason}): {row['sentence_id']}"
                    )
            expected_depth = min(SENTENCE_SOURCE_DEPTH, sentence_counts[article_id])
            ranks = [int(row["retrieval_rank"]) for row in rows]
            if ranks != list(range(1, expected_depth + 1)):
                raise EvidenceGapError(
                    f"{model_key} sentence ranking depth mismatch for {article_id}"
                )
            sentence_ids = [str(row["sentence_id"]) for row in rows]
            if len(sentence_ids) != len(set(sentence_ids)):
                raise EvidenceGapError(
                    f"{model_key} sentence ranking has duplicates for {article_id}"
                )

    evidence_validation = validate_runtime_evidence_rows(
        evidence,
        sentences=sentences,
        top_articles=top_articles,
    )
    return {
        "status": "PASS",
        "article_candidates": len(article_candidates),
        "top_articles": len(top_articles),
        "runtime_sentences": len(sentences),
        "sentence_ranking_rows": len(sentence_rankings),
        "evidence_candidates": len(evidence),
        "article_candidate_set_preserved": True,
        "article_ranks_contiguous": True,
        "per_article_sentence_depths_valid": True,
        "sentence_materialization": sentence_validation,
        "evidence_retrieval": evidence_validation,
    }
