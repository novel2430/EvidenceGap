from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path
from evidencegap.retrieval.bm25s_backend import BM25SBackend

DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")
DEFAULT_INDEX_DIR = Path("artifacts/v1/bm25_index")
DEFAULT_RUN_DIR = Path("artifacts/v1/article_retrieval_runs")
DEFAULT_REPORT_DIR = Path("reports/v1")
TRACKS = ("independent", "origin", "overall")


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise EvidenceGapError("Missing duckdb dependency") from exc
    return duckdb


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _load_eval_rows(corpus_dir: Path, split: str) -> list[dict[str, Any]]:
    duckdb = _duckdb()
    connection = duckdb.connect()
    claims = corpus_dir / "claims.parquet"
    judgments = corpus_dir / "judgments.parquet"
    articles = corpus_dir / "articles.parquet"
    table = connection.execute(
        f"""
        SELECT
            c.claim_id,
            c.claim_text,
            c.claim_pmid,
            j.article_id,
            a.doc_idx,
            j.relevance_grade,
            j.stance_label,
            j.is_origin_source
        FROM read_parquet('{_quote(claims)}') c
        JOIN read_parquet('{_quote(judgments)}') j USING(claim_id)
        JOIN read_parquet('{_quote(articles)}') a USING(article_id)
        WHERE c.split = ? AND j.eligible_for_qrels
        ORDER BY c.claim_id, j.article_id
        """,
        [split],
    ).fetch_arrow_table()
    connection.close()
    return table.to_pylist()


def _group_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        claim_id = str(row["claim_id"])
        item = grouped.setdefault(
            claim_id,
            {
                "claim_id": claim_id,
                "claim_text": str(row["claim_text"]),
                "claim_pmid": row.get("claim_pmid"),
                "candidates": [],
            },
        )
        item["candidates"].append(
            {
                "article_id": str(row["article_id"]),
                "doc_idx": int(row["doc_idx"]),
                "relevance_grade": int(row["relevance_grade"]),
                "stance_label": int(row["stance_label"]),
                "is_origin_source": bool(row["is_origin_source"]),
            }
        )
    return [grouped[key] for key in sorted(grouped)]


def _track_candidates(candidates: list[dict[str, Any]], track: str) -> list[dict[str, Any]]:
    if track == "independent":
        return [row for row in candidates if not row["is_origin_source"]]
    if track == "origin":
        return [row for row in candidates if row["is_origin_source"]]
    return list(candidates)


def _dcg(relevances: list[int], k: int) -> float:
    return sum(
        (2**rel - 1) / math.log2(rank + 2)
        for rank, rel in enumerate(relevances[:k])
    )


def _judged_metrics(ranked: list[dict[str, Any]]) -> dict[str, float]:
    rels = [int(row["relevance_grade"]) for row in ranked]
    positive_count = sum(rel > 0 for rel in rels)
    first = next((idx + 1 for idx, rel in enumerate(rels) if rel > 0), None)
    ideal = sorted(rels, reverse=True)
    ideal_dcg = _dcg(ideal, 10)
    return {
        "mrr": 0.0 if first is None else 1.0 / first,
        "ndcg@10": 0.0 if ideal_dcg == 0 else _dcg(rels, 10) / ideal_dcg,
        "recall@10": 0.0 if positive_count == 0 else sum(r > 0 for r in rels[:10]) / positive_count,
        "recall@50": 0.0 if positive_count == 0 else sum(r > 0 for r in rels[:50]) / positive_count,
    }


def _mean_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    return {
        key: round(fmean(item[key] for item in items), 8)
        for key in items[0]
    }


def _write_trec(path: Path, rows: list[tuple[str, str, int, float]], run_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, article_id, rank, score in rows:
            handle.write(
                f"{query_id} Q0 {article_id} {rank} {score:.8f} {run_name}\n"
            )


def run_article_retrieval(
    root: Path,
    *,
    split: str,
    corpus_dir: Path | None = None,
    index_dir: Path | None = None,
    run_dir: Path | None = None,
    report_dir: Path | None = None,
    top_k: int = 100,
    max_queries: int | None = None,
    run_name: str = "bm25s_default",
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise EvidenceGapError("split must be dev or test")
    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    index_dir = (root / (index_dir or DEFAULT_INDEX_DIR)).resolve()
    run_dir = (root / (run_dir or DEFAULT_RUN_DIR)).resolve()
    report_dir = (root / (report_dir or DEFAULT_REPORT_DIR)).resolve()

    queries = _group_rows(_load_eval_rows(corpus_dir, split))
    if max_queries is not None:
        queries = queries[:max_queries]
    backend = BM25SBackend(index_dir, mmap=True)

    judged_rows: dict[str, list[tuple[str, str, int, float]]] = {
        track: [] for track in TRACKS
    }
    open_rows: dict[str, list[tuple[str, str, int, float]]] = {
        track: [] for track in TRACKS
    }
    judged_metric_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    open_metric_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    eligible_counts = {track: 0 for track in TRACKS}

    for query_number, query in enumerate(queries, start=1):
        candidates = query["candidates"]
        scores = backend.score_documents(
            query["claim_text"],
            [row["doc_idx"] for row in candidates],
        )
        scored = [dict(row, score=float(score)) for row, score in zip(candidates, scores)]

        # One open-corpus search is sufficient. Track-specific outputs are derived
        # without running the 1.32M-document search three times.
        origin_doc_indices = [row["doc_idx"] for row in candidates if row["is_origin_source"]]
        open_hits = backend.search(
            query["claim_text"],
            top_k=top_k + 1,
        )

        for track in TRACKS:
            track_rows = _track_candidates(scored, track)
            positives = {row["article_id"] for row in track_rows if row["relevance_grade"] > 0}
            if not positives:
                continue
            eligible_counts[track] += 1

            ranked = sorted(
                track_rows,
                key=lambda row: (-row["score"], row["article_id"]),
            )
            judged_metric_rows[track].append(_judged_metrics(ranked))
            for rank, row in enumerate(ranked, start=1):
                judged_rows[track].append(
                    (query["claim_id"], row["article_id"], rank, row["score"])
                )

            if track == "independent":
                filtered_hits = [
                    hit for hit in open_hits if hit.doc_idx not in set(origin_doc_indices)
                ][:top_k]
            else:
                filtered_hits = open_hits[:top_k]
            retrieved_ids = [hit.article_id for hit in filtered_hits]
            found = len(positives & set(retrieved_ids))
            open_metric_rows[track].append(
                {
                    "known_positive_recall@10": len(positives & set(retrieved_ids[:10])) / len(positives),
                    "known_positive_recall@50": len(positives & set(retrieved_ids[:50])) / len(positives),
                    "known_positive_recall@100": found / len(positives),
                    "hitrate@10": float(bool(positives & set(retrieved_ids[:10]))),
                    "hitrate@100": float(bool(positives & set(retrieved_ids))),
                }
            )
            for rank, hit in enumerate(filtered_hits, start=1):
                open_rows[track].append(
                    (query["claim_id"], hit.article_id, rank, hit.score)
                )

        if query_number % 1000 == 0:
            print(f"  Retrieval evaluation: {query_number:,}/{len(queries):,} queries", flush=True)

    files: dict[str, str] = {}
    for track in TRACKS:
        judged_path = run_dir / f"{split}_judged_{track}_{run_name}.trec"
        open_path = run_dir / f"{split}_open_{track}_{run_name}.trec"
        _write_trec(judged_path, judged_rows[track], run_name)
        _write_trec(open_path, open_rows[track], run_name)
        files[f"judged_{track}"] = relative_path(root, judged_path)
        files[f"open_{track}"] = relative_path(root, open_path)

    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "run_name": run_name,
        "top_k": top_k,
        "query_limit": max_queries,
        "queries_loaded": len(queries),
        "tracks": {
            track: {
                "eligible_queries": eligible_counts[track],
                "judged_candidate_ranking": _mean_metrics(judged_metric_rows[track]),
                "open_corpus_incomplete_qrels": _mean_metrics(open_metric_rows[track]),
            }
            for track in TRACKS
        },
        "files": files,
        "caveats": [
            "Open-corpus metrics use incomplete judgments and treat only known positives as gold.",
            "Unjudged articles are not counted as confirmed negatives.",
            "Independent-source is the primary track; origin-source is a sanity check.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"article_retrieval_bm25_{split}.json"
    atomic_write_json(json_path, report)
    md_path = report_dir / f"article_retrieval_bm25_{split}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"Report: {json_path}")
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# BM25 Article Retrieval — {report['split']}",
        "",
        f"Run: `{report['run_name']}`",
        "",
        "| Track | Queries | MRR | nDCG@10 | Recall@10 | KP Recall@100 | HitRate@100 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for track in TRACKS:
        data = report["tracks"][track]
        judged = data["judged_candidate_ranking"]
        opened = data["open_corpus_incomplete_qrels"]
        lines.append(
            "| {track} | {q} | {mrr:.4f} | {ndcg:.4f} | {r10:.4f} | {kr:.4f} | {hit:.4f} |".format(
                track=track,
                q=data["eligible_queries"],
                mrr=judged.get("mrr", 0.0),
                ndcg=judged.get("ndcg@10", 0.0),
                r10=judged.get("recall@10", 0.0),
                kr=opened.get("known_positive_recall@100", 0.0),
                hit=opened.get("hitrate@100", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Judged Candidate Ranking is evaluated only on annotated MedFact pairs.",
            "Open-Corpus results report known-positive recall under incomplete judgments; unjudged articles are not confirmed negatives.",
            "The independent-source track is the primary result.",
            "",
        ]
    )
    return "\n".join(lines)
