from __future__ import annotations

import gc
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    require_empty_or_force,
    sha256_file,
    sha256_text,
)
from evidencegap_backend.dense.encoders import (
    DenseEncoder,
    encoder_spec,
    model_fingerprint,
)

if TYPE_CHECKING:
    from evidencegap_backend.resources import RuntimeResources

RUNTIME_RETRIEVAL_SCHEMA_VERSION = "1.0.0"
RUNTIME_RETRIEVAL_CONTRACT_ID = "phase07.retrieval-adapters.v1"
ARTICLE_RECORD_TYPE = "RuntimeArticleCandidate"

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
            handle.write(
                json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n"
            )
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
        from evidencegap_backend.dense.faiss_backend import DenseFaissBackend

        backend = DenseFaissBackend(root, index_dir, nprobe=nprobe)
        if backend.manifest.get("model_key") != model_key:
            raise EvidenceGapError(
                f"{model_key} FAISS manifest model mismatch: "
                f"{backend.manifest.get('model_key')}"
            )
        index_path = index_dir / "index.faiss"
        expected_index_sha256 = str(backend.manifest.get("index", {}).get("sha256", ""))
        if (
            not expected_index_sha256
            or sha256_file(index_path) != expected_index_sha256
        ):
            raise EvidenceGapError(f"{model_key} FAISS index checksum mismatch")
        if int(backend.index.ntotal) != len(article_ids):
            raise EvidenceGapError(
                f"{model_key} FAISS/article ID row mismatch: "
                f"{backend.index.ntotal} != {len(article_ids)}"
            )
        embedding_manifest_value = backend.manifest.get("article_embedding_manifest")
        if not embedding_manifest_value:
            raise EvidenceGapError(
                f"{model_key} FAISS manifest has no article embedding manifest"
            )
        embedding_manifest_path = Path(str(embedding_manifest_value))
        if not embedding_manifest_path.is_absolute():
            embedding_manifest_path = root / embedding_manifest_path
        embedding_manifest = load_json(embedding_manifest_path)
        if (
            embedding_manifest.get("article_input_sha256")
            != expected_article_input_sha256
        ):
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
            "article_embedding_manifest_sha256": sha256_file(embedding_manifest_path),
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
        raise EvidenceGapError(
            f"Missing Phase 02 article corpus: {corpus_articles_path}"
        )
    pa, _pq = _pyarrow()
    candidate_table = pa.Table.from_pylist(
        [{"article_id": str(article_id)} for article_id in article_ids]
    )
    connection = _duckdb().connect()
    try:
        connection.register("runtime_candidates", candidate_table)
        rows = (
            connection.execute(
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
            )
            .fetch_arrow_table()
            .to_pylist()
        )
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
    query_text: str | None = None,
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
    source_depth: int = ARTICLE_SOURCE_DEPTH,
    dense_nprobe: int = ARTICLE_DENSE_NPROBE,
    rrf_k: int = ARTICLE_RRF_K,
    rerank_depth: int = ARTICLE_RERANK_DEPTH,
    final_article_top_k: int = FINAL_ARTICLE_TOP_K,
    runtime_resources: "RuntimeResources | None" = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    claim_text = _clean_claim(claim_text)
    query_text = _clean_claim(query_text or claim_text)
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
    if any(
        value <= 0
        for value in (
            source_depth,
            dense_nprobe,
            rerank_depth,
            final_article_top_k,
        )
    ):
        raise EvidenceGapError("Retrieval depths and nprobe must be positive")
    if rrf_k < 0:
        raise EvidenceGapError("article rrf_k must be non-negative")
    if final_article_top_k > rerank_depth:
        raise EvidenceGapError("final_article_top_k cannot exceed rerank_depth")
    if (
        runtime_resources is not None
        and dense_nprobe != runtime_resources.config.pipeline.dense_nprobe
    ):
        raise EvidenceGapError(
            "dense_nprobe must match the value used to preload RuntimeResources"
        )

    started = time.perf_counter()
    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    article_input_manifest_path = article_input_dir / "article_inputs_manifest.json"
    if runtime_resources is not None:
        if not runtime_resources.loaded:
            raise EvidenceGapError("Runtime resources must be loaded before retrieval")
        corpus_manifest = runtime_resources.corpus_manifest
        article_input_manifest = runtime_resources.article_input_manifest
        expected_corpus_articles_sha256 = (
            runtime_resources.expected_corpus_articles_sha256
        )
        expected_article_input_sha256 = runtime_resources.expected_article_input_sha256
        bm25_backend = runtime_resources.bm25
        if bm25_backend is None:
            raise EvidenceGapError("Runtime BM25 resource is unavailable")
    else:
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
            raise EvidenceGapError(
                "Phase 03 article inputs do not match the Phase 02 corpus"
            )
        from evidencegap_backend.retrieval.bm25s_backend import BM25SBackend

        bm25_backend = BM25SBackend(bm25_index_dir, mmap=True)
        article_ids_metadata = bm25_backend.manifest.get("index", {}).get(
            "article_ids", {}
        )
        if article_ids_metadata.get("sha256") != sha256_file(
            bm25_index_dir / "article_ids.npy"
        ):
            raise EvidenceGapError("BM25 article ID map checksum mismatch")
        if (
            bm25_backend.manifest.get("corpus", {}).get("articles_sha256")
            != expected_corpus_articles_sha256
        ):
            raise EvidenceGapError(
                "BM25 index does not match the Phase 02 article corpus"
            )
    bm25_hits = bm25_backend.search(query_text, top_k=source_depth)
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
    if len(bm25_rows) != source_depth:
        raise EvidenceGapError(
            f"BM25 returned {len(bm25_rows)} rows, expected {source_depth}"
        )

    if runtime_resources is not None:
        medcpt_rows, medcpt_meta = runtime_resources.query_dense(
            "medcpt", query_text, top_k=source_depth
        )
        bmretriever_rows, bmretriever_meta = runtime_resources.query_dense(
            "bmretriever", query_text, top_k=source_depth
        )
    else:
        medcpt_rows, medcpt_meta = _query_dense_source(
            root,
            model_key="medcpt",
            claim_text=query_text,
            device=device,
            amp=amp,
            index_dir=medcpt_index_dir,
            article_ids=article_ids,
            expected_article_input_sha256=expected_article_input_sha256,
            nprobe=dense_nprobe,
            top_k=source_depth,
        )
        bmretriever_rows, bmretriever_meta = _query_dense_source(
            root,
            model_key="bmretriever",
            claim_text=query_text,
            device=device,
            amp=amp,
            index_dir=bmretriever_index_dir,
            article_ids=article_ids,
            expected_article_input_sha256=expected_article_input_sha256,
            nprobe=dense_nprobe,
            top_k=source_depth,
        )
    for label, rows in (("medcpt", medcpt_rows), ("bmretriever", bmretriever_rows)):
        if len(rows) != source_depth:
            raise EvidenceGapError(
                f"{label} returned {len(rows)} rows, expected {source_depth}"
            )

    fused = fuse_article_rankings(
        {"bm25": bm25_rows, "medcpt": medcpt_rows, "bmretriever": bmretriever_rows},
        rrf_k=rrf_k,
    )[:rerank_depth]
    if len(fused) != rerank_depth:
        raise EvidenceGapError(
            f"Article RRF produced {len(fused)} candidates, expected {rerank_depth}"
        )
    fused_article_ids = [str(row["article_id"]) for row in fused]
    article_text = (
        runtime_resources.fetch_article_texts(fused_article_ids)
        if runtime_resources is not None
        else _load_article_texts(
            article_ids=fused_article_ids,
            article_input_path=article_input_dir / "article_inputs.parquet",
            corpus_articles_path=corpus_dir / "articles.parquet",
        )
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

    from evidencegap_backend.reranking.cross_encoder import score_runtime_article_pairs

    reranked = (
        runtime_resources.score_articles(
            claim_text=query_text,
            articles=fused,
            batch_size=cross_encoder_batch_size,
        )
        if runtime_resources is not None
        else score_runtime_article_pairs(
            root,
            claim_text=query_text,
            articles=fused,
            model_dir=cross_encoder_model_dir,
            device=device,
            batch_size=cross_encoder_batch_size,
            max_length=512,
            amp=amp,
        )
    )
    ce_scores = {
        str(row["article_id"]): float(row["cross_encoder_score"])
        for row in reranked["scores"]
    }
    if set(ce_scores) != {str(row["article_id"]) for row in fused}:
        raise EvidenceGapError(
            "Cross-encoder output does not cover the configured RRF candidate set"
        )
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
                "retrieval_query": query_text,
                **row,
                "final_article_rank": final_rank,
            }
        )
    runtime_rows = candidate_rows[:final_article_top_k]

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
        if input_count != final_article_top_k:
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
            "retrieval_query": query_text,
            "retrieval_query_sha256": sha256_text(query_text),
            "parameters": {
                "source_depth": source_depth,
                "dense_nprobe": dense_nprobe,
                "rrf_k": rrf_k,
                "rerank_depth": rerank_depth,
                "final_article_top_k": final_article_top_k,
                "fusion": "equal_weight_rrf",
                "reranker": "medcpt_cross_encoder",
                "max_length": 512,
                "amp": amp,
                "resource_lifecycle": (
                    "engine_resident" if runtime_resources is not None else "per_call"
                ),
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
                    "manifest_sha256": sha256_file(article_input_manifest_path),
                    "article_input_sha256": expected_article_input_sha256,
                    "source_corpus_manifest_sha256": sha256_file(corpus_manifest_path),
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
        "top_article_rows": runtime_rows,
    }
