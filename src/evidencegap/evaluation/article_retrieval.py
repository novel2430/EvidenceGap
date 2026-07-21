from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Iterator

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path
from evidencegap.retrieval.bm25s_backend import BM25SBackend

DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")
DEFAULT_INDEX_DIR = Path("artifacts/v1/bm25_index")
DEFAULT_RUN_DIR = Path("artifacts/v1/article_retrieval_runs")
DEFAULT_REPORT_DIR = Path("reports/v1")
TRACKS = ("independent", "origin", "overall")
REPORT_SCHEMA_VERSION = "1.1.0"


def _duckdb():
    try:
        import duckdb
    except ImportError as exc:
        raise EvidenceGapError("Missing duckdb dependency") from exc
    return duckdb


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _load_split_query_count(corpus_dir: Path, split: str) -> int:
    duckdb = _duckdb()
    connection = duckdb.connect()
    try:
        claims = corpus_dir / "claims.parquet"
        return int(
            connection.execute(
                f"SELECT COUNT(*) FROM read_parquet('{_quote(claims)}') WHERE split = ?",
                [split],
            ).fetchone()[0]
        )
    finally:
        connection.close()


def _load_eval_rows(corpus_dir: Path, split: str) -> list[dict[str, Any]]:
    duckdb = _duckdb()
    connection = duckdb.connect()
    claims = corpus_dir / "claims.parquet"
    judgments = corpus_dir / "judgments.parquet"
    articles = corpus_dir / "articles.parquet"
    try:
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
    finally:
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


def _track_query_stats(queries: list[dict[str, Any]], track: str) -> dict[str, int]:
    no_candidates = 0
    no_positive = 0
    eligible = 0
    for query in queries:
        candidates = _track_candidates(query["candidates"], track)
        if not candidates:
            no_candidates += 1
            continue
        if not any(row["relevance_grade"] > 0 for row in candidates):
            no_positive += 1
            continue
        eligible += 1
    return {
        "queries_with_eligible_judgments": len(queries),
        "eligible_queries": eligible,
        "excluded_no_candidates": no_candidates,
        "excluded_no_positive": no_positive,
    }


def _dcg(relevances: list[int], k: int) -> float:
    return sum(
        (2**rel - 1) / math.log2(rank + 2)
        for rank, rel in enumerate(relevances[:k])
    )


def _ndcg(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = _dcg(ideal, k)
    return 0.0 if ideal_dcg == 0 else _dcg(relevances, k) / ideal_dcg


def _pairwise_ordering_accuracy(ranked: list[dict[str, Any]]) -> float | None:
    correct = 0.0
    comparable = 0
    for left in range(len(ranked)):
        for right in range(left + 1, len(ranked)):
            left_relevance = int(ranked[left]["relevance_grade"])
            right_relevance = int(ranked[right]["relevance_grade"])
            if left_relevance == right_relevance:
                continue
            comparable += 1
            left_score = float(ranked[left]["score"])
            right_score = float(ranked[right]["score"])
            if left_relevance > right_relevance:
                higher_score, lower_score = left_score, right_score
            else:
                higher_score, lower_score = right_score, left_score
            if higher_score > lower_score:
                correct += 1.0
            elif higher_score == lower_score:
                correct += 0.5
    if comparable == 0:
        return None
    return correct / comparable


def _judged_metrics(ranked: list[dict[str, Any]]) -> dict[str, float | None]:
    rels = [int(row["relevance_grade"]) for row in ranked]
    positive_count = sum(rel > 0 for rel in rels)
    first = next((idx + 1 for idx, rel in enumerate(rels) if rel > 0), None)
    top_relevance = rels[0] if rels else 0
    return {
        "mrr": 0.0 if first is None else 1.0 / first,
        "ndcg@3": _ndcg(rels, 3),
        "ndcg@5": _ndcg(rels, 5),
        "ndcg@10": _ndcg(rels, 10),
        "top1_positive_rate": float(top_relevance > 0),
        "mean_top1_relevance_grade": float(top_relevance),
        "pairwise_ordering_accuracy": _pairwise_ordering_accuracy(ranked),
        # Retained for backward compatibility. Candidate pools contain at most
        # five documents, so these metrics are expected to saturate.
        "recall@10": (
            0.0
            if positive_count == 0
            else sum(r > 0 for r in rels[:10]) / positive_count
        ),
        "recall@50": (
            0.0
            if positive_count == 0
            else sum(r > 0 for r in rels[:50]) / positive_count
        ),
    }


def _open_metrics(
    positives: set[str],
    retrieved_ids: list[str],
) -> tuple[dict[str, float], int | None]:
    retrieved_sets = {
        10: set(retrieved_ids[:10]),
        50: set(retrieved_ids[:50]),
        100: set(retrieved_ids[:100]),
    }
    first_positive_rank = next(
        (rank for rank, article_id in enumerate(retrieved_ids, start=1) if article_id in positives),
        None,
    )
    return (
        {
            "known_positive_recall@10": len(positives & retrieved_sets[10]) / len(positives),
            "known_positive_recall@50": len(positives & retrieved_sets[50]) / len(positives),
            "known_positive_recall@100": len(positives & retrieved_sets[100]) / len(positives),
            "hitrate@10": float(bool(positives & retrieved_sets[10])),
            "hitrate@100": float(bool(positives & retrieved_sets[100])),
        },
        first_positive_rank,
    )


def _mean_metrics(items: list[dict[str, float | None]]) -> dict[str, float | None]:
    if not items:
        return {}
    keys = sorted({key for item in items for key in item})
    result: dict[str, float | None] = {}
    for key in keys:
        values = [float(item[key]) for item in items if item.get(key) is not None]
        result[key] = round(fmean(values), 8) if values else None
    return result


def _write_trec(path: Path, rows: list[tuple[str, str, int, float]], run_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id, article_id, rank, score in rows:
            handle.write(
                f"{query_id} Q0 {article_id} {rank} {score:.8f} {run_name}\n"
            )


def _iter_trec_groups(path: Path) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    if not path.exists():
        raise EvidenceGapError(f"Saved run does not exist: {path}")
    current_query: str | None = None
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n").split()
            if len(parts) != 6:
                raise EvidenceGapError(
                    f"Malformed TREC row in {path}:{line_number}: expected 6 fields"
                )
            query_id, _, article_id, rank_text, score_text, _ = parts
            if current_query is not None and query_id != current_query:
                yield current_query, rows
                rows = []
            current_query = query_id
            rows.append(
                {
                    "article_id": article_id,
                    "rank": int(rank_text),
                    "score": float(score_text),
                }
            )
    if current_query is not None:
        yield current_query, rows


def _run_paths(
    run_dir: Path,
    *,
    split: str,
    track: str,
    run_name: str,
) -> tuple[Path, Path]:
    return (
        run_dir / f"{split}_judged_{track}_{run_name}.trec",
        run_dir / f"{split}_open_{track}_{run_name}.trec",
    )


def _failure_record(
    query: dict[str, Any],
    *,
    ranked: list[dict[str, Any]],
    open_hits: list[dict[str, Any]],
    positives: set[str],
    pairwise_accuracy: float | None,
    first_positive_rank: int | None,
) -> dict[str, Any] | None:
    failure_types: list[str] = []
    if ranked and ranked[0]["relevance_grade"] == 0:
        failure_types.append("judged_top1_not_positive")
    if pairwise_accuracy is not None and pairwise_accuracy < 1.0:
        failure_types.append("judged_relevance_inversion")
    if first_positive_rank is None:
        failure_types.append("open_no_known_positive_in_topk")
    elif first_positive_rank > 50:
        failure_types.append("open_first_known_positive_after_50")
    if not failure_types:
        return None

    return {
        "claim_id": query["claim_id"],
        "claim_text": query["claim_text"][:1000],
        "track": "independent",
        "failure_types": failure_types,
        "known_positive_article_ids": sorted(positives),
        "first_known_positive_rank": first_positive_rank,
        "pairwise_ordering_accuracy": pairwise_accuracy,
        "judged_top5": [
            {
                "article_id": row["article_id"],
                "relevance_grade": row["relevance_grade"],
                "score": round(float(row["score"]), 8),
            }
            for row in ranked[:5]
        ],
        "open_top10": [
            {
                "article_id": row["article_id"],
                "rank": row["rank"],
                "score": round(float(row["score"]), 8),
                "is_known_positive": row["article_id"] in positives,
            }
            for row in open_hits[:10]
        ],
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _evaluate_saved_track(
    *,
    track: str,
    queries_by_id: dict[str, dict[str, Any]],
    judged_path: Path,
    open_path: Path,
) -> tuple[
    list[dict[str, float | None]],
    list[dict[str, float]],
    list[dict[str, Any]],
    int,
]:
    judged_metrics: list[dict[str, float | None]] = []
    open_metrics: list[dict[str, float]] = []
    failures: list[dict[str, Any]] = []
    evaluated = 0

    judged_groups = _iter_trec_groups(judged_path)
    open_groups = _iter_trec_groups(open_path)
    for judged_group, open_group in zip_longest(judged_groups, open_groups):
        if judged_group is None or open_group is None:
            raise EvidenceGapError(
                f"Saved judged/open runs have different query counts for {track}"
            )
        judged_query_id, judged_rows = judged_group
        open_query_id, open_rows = open_group
        if judged_query_id != open_query_id:
            raise EvidenceGapError(
                "Saved judged/open runs are not aligned: "
                f"{judged_query_id} != {open_query_id}"
            )
        if judged_query_id not in queries_by_id:
            raise EvidenceGapError(f"Saved run contains unknown query: {judged_query_id}")

        query = queries_by_id[judged_query_id]
        candidates = _track_candidates(query["candidates"], track)
        candidate_by_id = {row["article_id"]: row for row in candidates}
        positives = {
            row["article_id"] for row in candidates if row["relevance_grade"] > 0
        }
        if not positives:
            raise EvidenceGapError(
                f"Saved run contains ineligible query for {track}: {judged_query_id}"
            )

        ranked: list[dict[str, Any]] = []
        for trec_row in judged_rows:
            article_id = trec_row["article_id"]
            if article_id not in candidate_by_id:
                raise EvidenceGapError(
                    f"Saved judged run contains non-candidate article {article_id} "
                    f"for {judged_query_id}"
                )
            ranked.append(
                dict(
                    candidate_by_id[article_id],
                    rank=trec_row["rank"],
                    score=trec_row["score"],
                )
            )
        if len(ranked) != len(candidates):
            raise EvidenceGapError(
                f"Saved judged run is incomplete for {judged_query_id}: "
                f"{len(ranked)} rows, expected {len(candidates)}"
            )

        judged = _judged_metrics(ranked)
        retrieved_ids = [row["article_id"] for row in open_rows]
        opened, first_positive_rank = _open_metrics(positives, retrieved_ids)
        judged_metrics.append(judged)
        open_metrics.append(opened)
        evaluated += 1

        if track == "independent":
            failure = _failure_record(
                query,
                ranked=ranked,
                open_hits=open_rows,
                positives=positives,
                pairwise_accuracy=judged["pairwise_ordering_accuracy"],
                first_positive_rank=first_positive_rank,
            )
            if failure is not None:
                failures.append(failure)

    return judged_metrics, open_metrics, failures, evaluated


def _report_stem(split: str, run_name: str) -> str:
    if run_name == "bm25s_default":
        return f"article_retrieval_bm25_{split}"
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in run_name
    )
    return f"article_retrieval_{safe_name}_{split}"


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
    reuse_run: bool = False,
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise EvidenceGapError("split must be dev or test")
    if top_k < 100:
        raise EvidenceGapError(
            "top_k must be at least 100 because the evaluation reports Recall@100"
        )

    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    index_dir = (root / (index_dir or DEFAULT_INDEX_DIR)).resolve()
    run_dir = (root / (run_dir or DEFAULT_RUN_DIR)).resolve()
    report_dir = (root / (report_dir or DEFAULT_REPORT_DIR)).resolve()

    queries_in_split = _load_split_query_count(corpus_dir, split)
    all_queries = _group_rows(_load_eval_rows(corpus_dir, split))
    queries_with_eligible_judgments = len(all_queries)
    queries = all_queries[:max_queries] if max_queries is not None else all_queries
    queries_by_id = {query["claim_id"]: query for query in queries}
    query_stats = {
        track: _track_query_stats(queries, track) for track in TRACKS
    }

    files: dict[str, str] = {}
    judged_metric_rows: dict[str, list[dict[str, float | None]]] = defaultdict(list)
    open_metric_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []

    if reuse_run:
        for track in TRACKS:
            judged_path, open_path = _run_paths(
                run_dir,
                split=split,
                track=track,
                run_name=run_name,
            )
            judged, opened, track_failures, evaluated = _evaluate_saved_track(
                track=track,
                queries_by_id=queries_by_id,
                judged_path=judged_path,
                open_path=open_path,
            )
            expected = query_stats[track]["eligible_queries"]
            if evaluated != expected:
                raise EvidenceGapError(
                    f"Saved run query count mismatch for {track}: "
                    f"{evaluated} evaluated, expected {expected}"
                )
            judged_metric_rows[track] = judged
            open_metric_rows[track] = opened
            failures.extend(track_failures)
            files[f"judged_{track}"] = relative_path(root, judged_path)
            files[f"open_{track}"] = relative_path(root, open_path)
    else:
        backend = BM25SBackend(index_dir, mmap=True)
        judged_rows: dict[str, list[tuple[str, str, int, float]]] = {
            track: [] for track in TRACKS
        }
        open_rows: dict[str, list[tuple[str, str, int, float]]] = {
            track: [] for track in TRACKS
        }

        for query_number, query in enumerate(queries, start=1):
            candidates = query["candidates"]
            scores = backend.score_documents(
                query["claim_text"],
                [row["doc_idx"] for row in candidates],
            )
            scored = [
                dict(row, score=float(score))
                for row, score in zip(candidates, scores)
            ]

            origin_doc_indices = {
                row["doc_idx"] for row in candidates if row["is_origin_source"]
            }
            open_hits = backend.search(
                query["claim_text"],
                top_k=top_k + len(origin_doc_indices),
            )

            for track in TRACKS:
                track_rows = _track_candidates(scored, track)
                positives = {
                    row["article_id"]
                    for row in track_rows
                    if row["relevance_grade"] > 0
                }
                if not positives:
                    continue

                ranked = sorted(
                    track_rows,
                    key=lambda row: (-row["score"], row["article_id"]),
                )
                judged = _judged_metrics(ranked)
                judged_metric_rows[track].append(judged)
                for rank, row in enumerate(ranked, start=1):
                    judged_rows[track].append(
                        (query["claim_id"], row["article_id"], rank, row["score"])
                    )

                if track == "independent":
                    filtered_hits = [
                        hit for hit in open_hits if hit.doc_idx not in origin_doc_indices
                    ][:top_k]
                else:
                    filtered_hits = open_hits[:top_k]
                retrieved_ids = [hit.article_id for hit in filtered_hits]
                opened, first_positive_rank = _open_metrics(positives, retrieved_ids)
                open_metric_rows[track].append(opened)
                for rank, hit in enumerate(filtered_hits, start=1):
                    open_rows[track].append(
                        (query["claim_id"], hit.article_id, rank, hit.score)
                    )

                if track == "independent":
                    failure = _failure_record(
                        query,
                        ranked=ranked,
                        open_hits=[
                            {
                                "article_id": hit.article_id,
                                "rank": rank,
                                "score": hit.score,
                            }
                            for rank, hit in enumerate(filtered_hits, start=1)
                        ],
                        positives=positives,
                        pairwise_accuracy=judged["pairwise_ordering_accuracy"],
                        first_positive_rank=first_positive_rank,
                    )
                    if failure is not None:
                        failures.append(failure)

            if query_number % 1000 == 0:
                print(
                    f"  Retrieval evaluation: {query_number:,}/{len(queries):,} queries",
                    flush=True,
                )

        for track in TRACKS:
            judged_path, open_path = _run_paths(
                run_dir,
                split=split,
                track=track,
                run_name=run_name,
            )
            _write_trec(judged_path, judged_rows[track], run_name)
            _write_trec(open_path, open_rows[track], run_name)
            files[f"judged_{track}"] = relative_path(root, judged_path)
            files[f"open_{track}"] = relative_path(root, open_path)

    for track in TRACKS:
        query_stats[track]["pairwise_evaluable_queries"] = sum(
            metric.get("pairwise_ordering_accuracy") is not None
            for metric in judged_metric_rows[track]
        )

    report_stem = _report_stem(split, run_name)
    failure_path = report_dir / f"{report_stem}_failures.jsonl"
    _write_jsonl(failure_path, failures)
    files["independent_failure_cases"] = relative_path(root, failure_path)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "run_name": run_name,
        "run_source": "saved_trec" if reuse_run else "live_retrieval",
        "top_k": top_k,
        "query_limit": max_queries,
        "queries_in_split": queries_in_split,
        "queries_with_eligible_judgments": queries_with_eligible_judgments,
        "excluded_no_eligible_judgments": max(
            0, queries_in_split - queries_with_eligible_judgments
        ),
        "queries_evaluated": len(queries),
        "tracks": {
            track: {
                **query_stats[track],
                "judged_candidate_ranking": _mean_metrics(
                    judged_metric_rows[track]
                ),
                "open_corpus_incomplete_qrels": _mean_metrics(
                    open_metric_rows[track]
                ),
            }
            for track in TRACKS
        },
        "failure_cases": {
            "track": "independent",
            "count": len(failures),
            "path": relative_path(root, failure_path),
        },
        "files": files,
        "caveats": [
            "Open-corpus metrics use incomplete judgments and treat only known positives as gold.",
            "Unjudged articles are not counted as confirmed negatives.",
            "Independent-source is the primary track; origin-source is a sanity check.",
            "Judged Recall@10 and Recall@50 are retained only for backward compatibility; candidate pools contain at most five articles and these metrics saturate.",
            "Pairwise ordering accuracy excludes queries whose judged candidates all have the same relevance grade.",
        ],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / f"{report_stem}.json"
    atomic_write_json(json_path, report)
    md_path = report_dir / f"{report_stem}.md"
    md_path.write_text(_markdown_report(report), encoding="utf-8")
    print(f"Report: {json_path}")
    return report


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        f"# BM25 Article Retrieval — {report['split']}",
        "",
        f"Run: `{report['run_name']}`  ",
        f"Source: `{report['run_source']}`",
        "",
        f"Queries in split: {report['queries_in_split']:,}",
        f"Queries with eligible judgments: {report['queries_with_eligible_judgments']:,}",
        f"Queries evaluated in this run: {report['queries_evaluated']:,}",
        f"Excluded without eligible judgments: {report['excluded_no_eligible_judgments']:,}",
        "",
        "| Track | Eligible | Excl. no candidates | Excl. no positive | MRR | nDCG@3 | nDCG@5 | Top-1 positive | Mean Top-1 grade | Pairwise acc. | KP Recall@10 | KP Recall@100 | HitRate@100 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for track in TRACKS:
        data = report["tracks"][track]
        judged = data["judged_candidate_ranking"]
        opened = data["open_corpus_incomplete_qrels"]
        pairwise = judged.get("pairwise_ordering_accuracy")
        pairwise_text = "n/a" if pairwise is None else f"{pairwise:.4f}"
        lines.append(
            "| {track} | {eligible} | {no_candidates} | {no_positive} | "
            "{mrr:.4f} | {ndcg3:.4f} | {ndcg5:.4f} | {top1:.4f} | "
            "{top_grade:.4f} | {pairwise} | {kr10:.4f} | {kr100:.4f} | "
            "{hit100:.4f} |".format(
                track=track,
                eligible=data["eligible_queries"],
                no_candidates=data["excluded_no_candidates"],
                no_positive=data["excluded_no_positive"],
                mrr=judged.get("mrr", 0.0),
                ndcg3=judged.get("ndcg@3", 0.0),
                ndcg5=judged.get("ndcg@5", 0.0),
                top1=judged.get("top1_positive_rate", 0.0),
                top_grade=judged.get("mean_top1_relevance_grade", 0.0),
                pairwise=pairwise_text,
                kr10=opened.get("known_positive_recall@10", 0.0),
                kr100=opened.get("known_positive_recall@100", 0.0),
                hit100=opened.get("hitrate@100", 0.0),
            )
        )
    lines.extend(
        [
            "",
            "## Failure cases",
            "",
            f"Independent-track failures: {report['failure_cases']['count']:,}",
            "",
            f"Path: `{report['failure_cases']['path']}`",
            "",
            "A failure record is emitted when judged Top-1 is not positive, "
            "graded relevance is inverted, no known positive appears in Top-K, "
            "or the first known positive appears after rank 50.",
            "",
            "## Interpretation boundary",
            "",
            "Judged Candidate Ranking is evaluated only on annotated MedFact pairs.",
            "Open-Corpus results report known-positive recall under incomplete judgments; unjudged articles are not confirmed negatives.",
            "The independent-source track is the primary result.",
            "Judged Recall@10/50 remain in JSON only for compatibility and should not be used as headline metrics because candidate pools contain at most five documents.",
            "",
        ]
    )
    return "\n".join(lines)
