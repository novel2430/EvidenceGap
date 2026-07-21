from __future__ import annotations

import os
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow.parquet as pq

from evidencegap.common import EvidenceGapError, atomic_write_json, load_json, relative_path
from evidencegap.dense.embeddings import (
    DEFAULT_CORPUS_DIR,
    DEFAULT_DENSE_DIR,
    QueryEmbeddingStore,
    ShardedEmbeddingStore,
)
from evidencegap.dense.encoders import DenseEncoder
from evidencegap.dense.faiss_backend import DenseFaissBackend
from evidencegap.evaluation.article_retrieval import run_article_retrieval

TRACKS = ("independent", "origin", "overall")
DEFAULT_RUN_DIR = Path("artifacts/v1/article_retrieval_runs")
DEFAULT_REPORT_DIR = Path("reports/v1")


def _duckdb() -> Any:
    try:
        import duckdb
    except ImportError as exc:
        raise EvidenceGapError("Missing duckdb dependency") from exc
    return duckdb


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _load_queries(corpus_dir: Path, split: str) -> list[dict[str, Any]]:
    connection = _duckdb().connect()
    try:
        table = connection.execute(
            f"""
            SELECT
                c.claim_id,
                c.claim_text,
                j.article_id,
                a.doc_idx,
                j.relevance_grade,
                j.stance_label,
                j.is_origin_source
            FROM read_parquet('{_quote(corpus_dir / 'claims.parquet')}') c
            JOIN read_parquet('{_quote(corpus_dir / 'judgments.parquet')}') j
              USING(claim_id)
            JOIN read_parquet('{_quote(corpus_dir / 'articles.parquet')}') a
              USING(article_id)
            WHERE c.split = ? AND j.eligible_for_qrels
            ORDER BY c.claim_id, j.article_id
            """,
            [split],
        ).fetch_arrow_table()
    finally:
        connection.close()
    grouped: dict[str, dict[str, Any]] = {}
    for row in table.to_pylist():
        claim_id = str(row["claim_id"])
        query = grouped.setdefault(
            claim_id,
            {
                "claim_id": claim_id,
                "claim_text": str(row["claim_text"]),
                "candidates": [],
            },
        )
        query["candidates"].append(
            {
                "article_id": str(row["article_id"]),
                "doc_idx": int(row["doc_idx"]),
                "relevance_grade": int(row["relevance_grade"]),
                "stance_label": int(row["stance_label"]),
                "is_origin_source": bool(row["is_origin_source"]),
            }
        )
    return [grouped[key] for key in sorted(grouped)]


def _track_candidates(rows: list[dict[str, Any]], track: str) -> list[dict[str, Any]]:
    if track == "independent":
        return [row for row in rows if not row["is_origin_source"]]
    if track == "origin":
        return [row for row in rows if row["is_origin_source"]]
    return list(rows)


def _run_path(run_dir: Path, split: str, kind: str, track: str, run_name: str) -> Path:
    return run_dir / f"{split}_{kind}_{track}_{run_name}.trec"


def _trec_line(
    query_id: str,
    article_id: str,
    rank: int,
    score: float,
    run_name: str,
) -> str:
    return f"{query_id} Q0 {article_id} {rank} {score:.8f} {run_name}\n"


def _article_ids(corpus_dir: Path) -> list[str]:
    table = pq.read_table(
        corpus_dir / "articles.parquet",
        columns=["doc_idx", "article_id"],
    ).sort_by([("doc_idx", "ascending")])
    doc_indices = table["doc_idx"].to_numpy(zero_copy_only=False)
    if len(doc_indices) and not np.array_equal(
        doc_indices, np.arange(len(doc_indices), dtype=doc_indices.dtype)
    ):
        raise EvidenceGapError("article doc_idx is not contiguous")
    return [str(value) for value in table["article_id"].to_pylist()]


def run_dense_article_retrieval(
    root: Path,
    *,
    model_key: str,
    split: str,
    corpus_dir: Path | None = None,
    embedding_dir: Path | None = None,
    query_dir: Path | None = None,
    index_dir: Path | None = None,
    run_dir: Path | None = None,
    report_dir: Path | None = None,
    top_k: int = 100,
    nprobe: int | None = None,
    search_batch_size: int = 256,
    max_queries: int | None = None,
    run_name: str | None = None,
    reuse_run: bool = False,
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise EvidenceGapError("split must be dev or test")
    if top_k < 100:
        raise EvidenceGapError("top_k must be at least 100")
    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    embedding_dir = (
        (root / embedding_dir).resolve()
        if embedding_dir is not None
        else root / DEFAULT_DENSE_DIR / model_key / "article_embeddings"
    )
    query_dir = (
        (root / query_dir).resolve()
        if query_dir is not None
        else root / DEFAULT_DENSE_DIR / model_key / "query_embeddings" / split
    )
    index_dir = (
        (root / index_dir).resolve()
        if index_dir is not None
        else root / DEFAULT_DENSE_DIR / model_key / "faiss_index"
    )
    run_dir = (root / (run_dir or DEFAULT_RUN_DIR)).resolve()
    report_dir = (root / (report_dir or DEFAULT_REPORT_DIR)).resolve()
    index_manifest = load_json(index_dir / "index_manifest.json")
    effective_nprobe = (
        nprobe
        if index_manifest.get("nlist") is not None
        else None
    )
    if effective_nprobe is None and index_manifest.get("nlist") is not None:
        effective_nprobe = int(index_manifest["default_nprobe"])
    run_name = run_name or (
        f"{model_key}_{index_manifest['index_type']}"
        + (f"_nprobe{effective_nprobe}" if effective_nprobe is not None else "")
    )

    if not reuse_run:
        article_store = ShardedEmbeddingStore(
            root, embedding_dir / "embedding_manifest.json"
        )
        query_store = QueryEmbeddingStore(root, query_dir)
        backend = DenseFaissBackend(root, index_dir, nprobe=effective_nprobe)
        article_ids = _article_ids(corpus_dir)
        queries = _load_queries(corpus_dir, split)
        if max_queries is not None:
            queries = queries[:max_queries]
        run_dir.mkdir(parents=True, exist_ok=True)
        final_paths = {
            (kind, track): _run_path(run_dir, split, kind, track, run_name)
            for kind in ("judged", "open")
            for track in TRACKS
        }
        temporary_paths = {
            key: path.with_name(path.name + ".tmp")
            for key, path in final_paths.items()
        }
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)

        try:
            with ExitStack() as stack:
                handles = {
                    key: stack.enter_context(path.open("w", encoding="utf-8"))
                    for key, path in temporary_paths.items()
                }
                for batch_start in range(0, len(queries), search_batch_size):
                    batch = queries[batch_start : batch_start + search_batch_size]
                    query_matrix = query_store.rows_for_claims(
                        [query["claim_id"] for query in batch]
                    )
                    maximum_origins = max(
                        (
                            sum(
                                row["is_origin_source"]
                                for row in query["candidates"]
                            )
                            for query in batch
                        ),
                        default=0,
                    )
                    open_scores, open_doc_indices = backend.search(
                        query_matrix,
                        top_k=top_k + maximum_origins,
                    )
                    for offset, query in enumerate(batch):
                        qvec = query_matrix[offset]
                        candidates = query["candidates"]
                        candidate_vectors = article_store.get_rows(
                            [row["doc_idx"] for row in candidates]
                        )
                        candidate_scores = candidate_vectors @ qvec
                        scored = [
                            dict(row, score=float(score))
                            for row, score in zip(candidates, candidate_scores)
                        ]
                        origin_doc_indices = {
                            row["doc_idx"]
                            for row in candidates
                            if row["is_origin_source"]
                        }
                        open_hits: list[tuple[int, str, float]] = []
                        for doc_idx, score in zip(
                            open_doc_indices[offset], open_scores[offset]
                        ):
                            value = int(doc_idx)
                            if value < 0:
                                continue
                            open_hits.append(
                                (value, article_ids[value], float(score))
                            )

                        for track in TRACKS:
                            track_rows = _track_candidates(scored, track)
                            if not any(
                                row["relevance_grade"] > 0 for row in track_rows
                            ):
                                continue
                            ranked = sorted(
                                track_rows,
                                key=lambda row: (-row["score"], row["article_id"]),
                            )
                            for rank, row in enumerate(ranked, start=1):
                                handles[("judged", track)].write(
                                    _trec_line(
                                        query["claim_id"],
                                        row["article_id"],
                                        rank,
                                        row["score"],
                                        run_name,
                                    )
                                )
                            if track == "independent":
                                filtered = [
                                    hit
                                    for hit in open_hits
                                    if hit[0] not in origin_doc_indices
                                ][:top_k]
                            else:
                                filtered = open_hits[:top_k]
                            for rank, (_doc_idx, article_id, score) in enumerate(
                                filtered, start=1
                            ):
                                handles[("open", track)].write(
                                    _trec_line(
                                        query["claim_id"],
                                        article_id,
                                        rank,
                                        score,
                                        run_name,
                                    )
                                )
                    processed = min(len(queries), batch_start + len(batch))
                    if processed % 1000 < len(batch):
                        print(
                            f"  Dense retrieval: {processed:,}/{len(queries):,} queries",
                            flush=True,
                        )
            for key, temporary in temporary_paths.items():
                os.replace(temporary, final_paths[key])
        except Exception:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)
            raise

    report = run_article_retrieval(
        root,
        split=split,
        corpus_dir=corpus_dir,
        index_dir=index_dir,
        run_dir=run_dir,
        report_dir=report_dir,
        top_k=top_k,
        max_queries=max_queries,
        run_name=run_name,
        reuse_run=True,
    )
    report["retriever"] = {
        "family": "dense",
        "model_key": model_key,
        "similarity": "inner_product",
        "article_scoring": "exact dot product from stored embeddings",
        "open_corpus_backend": "faiss",
        "index_type": index_manifest["index_type"],
        "nprobe": effective_nprobe,
        "index_manifest": relative_path(root, index_dir / "index_manifest.json"),
        "query_embedding_manifest": relative_path(
            root, query_dir / "query_embedding_manifest.json"
        ),
        "article_embedding_manifest": relative_path(
            root, embedding_dir / "embedding_manifest.json"
        ),
    }
    report["created_at"] = datetime.now(timezone.utc).isoformat()
    stem = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in run_name
    )
    json_path = report_dir / f"article_retrieval_{stem}_{split}.json"
    atomic_write_json(json_path, report)
    md_path = report_dir / f"article_retrieval_{stem}_{split}.md"
    md_path.write_text(_dense_markdown(report), encoding="utf-8")
    print(f"Dense report: {json_path}")
    return report


def _dense_markdown(report: dict[str, Any]) -> str:
    retriever = report["retriever"]
    lines = [
        f"# Dense Article Retrieval — {report['split']}",
        "",
        f"Run: `{report['run_name']}`  ",
        f"Model: `{retriever['model_key']}`  ",
        f"Index: `{retriever['index_type']}`  ",
        f"nprobe: `{retriever['nprobe']}`",
        "",
        "| Track | Eligible | MRR | nDCG@5 | Top-1 positive | Pairwise acc. | KP Recall@10 | KP Recall@50 | KP Recall@100 | HitRate@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track in TRACKS:
        data = report["tracks"][track]
        judged = data["judged_candidate_ranking"]
        opened = data["open_corpus_incomplete_qrels"]
        pairwise = judged.get("pairwise_ordering_accuracy")
        lines.append(
            "| {track} | {eligible} | {mrr:.4f} | {ndcg:.4f} | {top1:.4f} | "
            "{pairwise} | {r10:.4f} | {r50:.4f} | {r100:.4f} | {h100:.4f} |".format(
                track=track,
                eligible=data["eligible_queries"],
                mrr=judged.get("mrr", 0.0),
                ndcg=judged.get("ndcg@5", 0.0),
                top1=judged.get("top1_positive_rate", 0.0),
                pairwise="n/a" if pairwise is None else f"{pairwise:.4f}",
                r10=opened.get("known_positive_recall@10", 0.0),
                r50=opened.get("known_positive_recall@50", 0.0),
                r100=opened.get("known_positive_recall@100", 0.0),
                h100=opened.get("hitrate@100", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "Judged candidate scores are exact dot products. Open-corpus scores use the recorded FAISS ANN configuration.",
            "Independent-source is the primary track. Open-corpus judgments are incomplete.",
            "",
        ]
    )
    return "\n".join(lines)


def query_dense_index(
    root: Path,
    *,
    model_key: str,
    claim: str,
    device: str,
    index_dir: Path | None = None,
    corpus_dir: Path | None = None,
    nprobe: int | None = None,
    top_k: int = 10,
    amp: str = "fp16",
) -> list[dict[str, Any]]:
    root = root.resolve()
    index_dir = (
        (root / index_dir).resolve()
        if index_dir is not None
        else root / DEFAULT_DENSE_DIR / model_key / "faiss_index"
    )
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    encoder = DenseEncoder(root, model_key, device=device, amp=amp)
    query = encoder.encode_queries([claim])
    backend = DenseFaissBackend(root, index_dir, nprobe=nprobe)
    scores, ids = backend.search(query, top_k=top_k)
    article_ids = _article_ids(corpus_dir)
    return [
        {
            "rank": rank,
            "doc_idx": int(doc_idx),
            "article_id": article_ids[int(doc_idx)],
            "score": float(score),
        }
        for rank, (doc_idx, score) in enumerate(zip(ids[0], scores[0]), start=1)
        if int(doc_idx) >= 0
    ]


def compare_retrieval_reports(
    root: Path,
    *,
    split: str,
    report_paths: Sequence[Path],
    output_stem: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    reports = [load_json((root / path).resolve()) for path in report_paths]
    rows = []
    for report in reports:
        independent = report["tracks"]["independent"]
        judged = independent["judged_candidate_ranking"]
        opened = independent["open_corpus_incomplete_qrels"]
        rows.append(
            {
                "run_name": report["run_name"],
                "mrr": judged.get("mrr"),
                "ndcg@5": judged.get("ndcg@5"),
                "top1_positive_rate": judged.get("top1_positive_rate"),
                "pairwise_ordering_accuracy": judged.get(
                    "pairwise_ordering_accuracy"
                ),
                "known_positive_recall@10": opened.get(
                    "known_positive_recall@10"
                ),
                "known_positive_recall@50": opened.get(
                    "known_positive_recall@50"
                ),
                "known_positive_recall@100": opened.get(
                    "known_positive_recall@100"
                ),
                "hitrate@100": opened.get("hitrate@100"),
            }
        )
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "track": "independent",
        "runs": rows,
    }
    stem = output_stem or f"article_retrieval_comparison_{split}"
    output_dir = root / DEFAULT_REPORT_DIR
    atomic_write_json(output_dir / f"{stem}.json", result)
    lines = [
        f"# Article Retrieval Comparison — {split}",
        "",
        "Primary track: `independent`",
        "",
        "| Run | MRR | nDCG@5 | Top-1 positive | Pairwise acc. | KP Recall@10 | KP Recall@50 | KP Recall@100 | HitRate@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {run_name} | {mrr:.4f} | {ndcg:.4f} | {top1:.4f} | {pairwise:.4f} | "
            "{r10:.4f} | {r50:.4f} | {r100:.4f} | {h100:.4f} |".format(
                run_name=row["run_name"],
                mrr=row["mrr"],
                ndcg=row["ndcg@5"],
                top1=row["top1_positive_rate"],
                pairwise=row["pairwise_ordering_accuracy"],
                r10=row["known_positive_recall@10"],
                r50=row["known_positive_recall@50"],
                r100=row["known_positive_recall@100"],
                h100=row["hitrate@100"],
            )
        )
    (output_dir / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result
