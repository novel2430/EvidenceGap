#!/usr/bin/env python3
"""Check MedCPT TREC depth, recall plateau, IVF misses, and FAISS provenance.

The script only reads existing Phase 03 artifacts. It does not encode models,
rebuild embeddings, or rebuild the FAISS index.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

NPROBE_RE = re.compile(r"(?:^|_)nprobe(\d+)(?:_|$)")


class CheckError(RuntimeError):
    pass


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose MedCPT Recall@10/50/100 saturation and IVF nprobe behavior."
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--run-name",
        action="append",
        help="Run name without TREC prefix. Repeat to compare runs; omitted means auto-discover.",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--verify-faiss-sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--tolerance", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    value = parser.parse_args()
    if value.top_k < 100:
        parser.error("--top-k must be at least 100")
    return value


def nprobe(run_name: str) -> int | None:
    match = NPROBE_RE.search(run_name)
    return int(match.group(1)) if match else None


def discover(run_dir: Path, split: str) -> list[str]:
    prefix = f"{split}_open_independent_"
    names = [
        path.name[len(prefix) : -5]
        for path in run_dir.glob(f"{prefix}medcpt_*.trec")
    ]
    return sorted(set(names), key=lambda name: (nprobe(name) is None, nprobe(name) or 0))


def read_trec(path: Path, run_name: str, top_k: int) -> dict[str, list[tuple[str, int, float]]]:
    if not path.exists():
        raise CheckError(f"Missing TREC run: {path}")
    groups: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    closed: set[str] = set()
    previous: str | None = None
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            parts = line.split()
            if len(parts) != 6:
                raise CheckError(f"Malformed TREC row {path}:{line_number}")
            query_id, q0, article_id, rank_text, score_text, tag = parts
            if q0 != "Q0" or tag != run_name:
                raise CheckError(f"TREC tag/Q0 mismatch {path}:{line_number}")
            if previous is not None and query_id != previous:
                closed.add(previous)
                if query_id in closed:
                    raise CheckError(f"Non-contiguous query block: {query_id}")
            previous = query_id
            groups[query_id].append((article_id, int(rank_text), float(score_text)))

    for query_id, hits in groups.items():
        ranks = [rank for _, rank, _ in hits]
        article_ids = [article_id for article_id, _, _ in hits]
        if len(hits) != top_k:
            raise CheckError(f"{run_name}/{query_id}: expected {top_k} hits, got {len(hits)}")
        if ranks != list(range(1, top_k + 1)):
            raise CheckError(f"{run_name}/{query_id}: ranks are not contiguous 1..{top_k}")
        if len(article_ids) != len(set(article_ids)):
            raise CheckError(f"{run_name}/{query_id}: duplicate article IDs")
    return dict(groups)


def load_gold(
    corpus_dir: Path,
    split: str,
    query_ids: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[int]]]:
    try:
        import duckdb
    except ImportError as exc:
        raise CheckError("duckdb is required") from exc

    judgments = corpus_dir / "judgments.parquet"
    articles = corpus_dir / "articles.parquet"
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT j.claim_id, j.article_id, j.relevance_grade,
                   j.is_origin_source, a.doc_idx
            FROM read_parquet(?) j
            JOIN read_parquet(?) a USING(article_id)
            WHERE j.split = ? AND j.eligible_for_qrels
            """,
            [str(judgments), str(articles), split],
        ).fetchall()
    finally:
        con.close()

    positives: dict[str, set[str]] = defaultdict(set)
    origins: dict[str, set[int]] = defaultdict(set)
    for claim_id_raw, article_id_raw, relevance, is_origin, doc_idx in rows:
        claim_id = str(claim_id_raw)
        if claim_id not in query_ids:
            continue
        if bool(is_origin):
            origins[claim_id].add(int(doc_idx))
        elif int(relevance) > 0:
            positives[claim_id].add(str(article_id_raw))

    missing = sorted(query_ids - set(positives))
    if missing:
        raise CheckError(
            f"{len(missing)} run queries have no independent positive judgments: {missing[:5]}"
        )
    return dict(positives), dict(origins)


def analyze(
    hits_by_query: dict[str, list[tuple[str, int, float]]],
    positives: dict[str, set[str]],
) -> dict[str, Any]:
    buckets = Counter({"01-10": 0, "11-50": 0, "51-100": 0, "missing": 0})
    recalls = {10: [], 50: [], 100: []}
    hitrates = {10: [], 100: []}

    for query_id, hits in hits_by_query.items():
        ranks = {article_id: rank for article_id, rank, _ in hits}
        gold = positives[query_id]
        for article_id in gold:
            rank = ranks.get(article_id)
            if rank is None:
                buckets["missing"] += 1
            elif rank <= 10:
                buckets["01-10"] += 1
            elif rank <= 50:
                buckets["11-50"] += 1
            elif rank <= 100:
                buckets["51-100"] += 1
        for cutoff in (10, 50, 100):
            recalls[cutoff].append(
                sum(ranks.get(article_id, 10**9) <= cutoff for article_id in gold) / len(gold)
            )
        for cutoff in (10, 100):
            hitrates[cutoff].append(
                float(any(ranks.get(article_id, 10**9) <= cutoff for article_id in gold))
            )

    metric = lambda values: round(fmean(values), 8) if values else 0.0
    result = {
        "queries": len(hits_by_query),
        "known_positives": sum(len(positives[q]) for q in hits_by_query),
        "positive_rank_buckets": dict(buckets),
        "known_positive_recall@10": metric(recalls[10]),
        "known_positive_recall@50": metric(recalls[50]),
        "known_positive_recall@100": metric(recalls[100]),
        "hitrate@10": metric(hitrates[10]),
        "hitrate@100": metric(hitrates[100]),
    }
    result["recall_plateau"] = (
        math.isclose(result["known_positive_recall@10"], result["known_positive_recall@50"])
        and math.isclose(result["known_positive_recall@50"], result["known_positive_recall@100"])
    )
    result["all_recovered_positives_are_top10"] = (
        buckets["11-50"] == 0 and buckets["51-100"] == 0
    )
    return result


def metrics_on_shared(
    hits_by_query: dict[str, list[tuple[str, int, float]]],
    positives: dict[str, set[str]],
    shared: set[str],
) -> dict[str, float]:
    subset = {query_id: hits_by_query[query_id] for query_id in shared}
    return {
        key: value
        for key, value in analyze(subset, positives).items()
        if key.startswith("known_positive_") or key.startswith("hitrate@")
    }


def compare(
    current: dict[str, list[tuple[str, int, float]]],
    reference: dict[str, list[tuple[str, int, float]]],
    positives: dict[str, set[str]],
) -> dict[str, Any]:
    shared = set(current) & set(reference)
    current_metrics = metrics_on_shared(current, positives, shared)
    reference_metrics = metrics_on_shared(reference, positives, shared)
    gained = Counter({"01-10": 0, "11-50": 0, "51-100": 0})
    reference_found_current_missing = 0

    for query_id in shared:
        current_ranks = {article_id: rank for article_id, rank, _ in current[query_id]}
        reference_ranks = {article_id: rank for article_id, rank, _ in reference[query_id]}
        for article_id in positives[query_id]:
            if article_id not in current_ranks and article_id in reference_ranks:
                reference_found_current_missing += 1
                rank = reference_ranks[article_id]
                gained["01-10" if rank <= 10 else "11-50" if rank <= 50 else "51-100"] += 1

    top10_fraction = (
        gained["01-10"] / reference_found_current_missing
        if reference_found_current_missing
        else None
    )
    return {
        "shared_queries": len(shared),
        "current_metrics": current_metrics,
        "reference_metrics": reference_metrics,
        "reference_minus_current": {
            key: round(reference_metrics[key] - current_metrics[key], 8)
            for key in reference_metrics
        },
        "reference_found_current_missing": reference_found_current_missing,
        "reference_gain_rank_buckets": dict(gained),
        "fraction_of_reference_gains_entering_top10": (
            None if top10_fraction is None else round(top10_fraction, 8)
        ),
        "consistent_with_ivf_cluster_gating": (
            reference_found_current_missing > 0
            and top10_fraction is not None
            and top10_fraction >= 0.9
        ),
    }


def article_ids(corpus_dir: Path) -> list[str]:
    import numpy as np
    import pyarrow.parquet as pq

    table = pq.read_table(corpus_dir / "articles.parquet", columns=["doc_idx", "article_id"])
    table = table.sort_by([("doc_idx", "ascending")])
    indices = table["doc_idx"].to_numpy(zero_copy_only=False)
    if len(indices) and not np.array_equal(indices, np.arange(len(indices), dtype=indices.dtype)):
        raise CheckError("articles.parquet doc_idx is not contiguous")
    return [str(value) for value in table["article_id"].to_pylist()]


def verify_faiss(
    root: Path,
    split: str,
    run_name: str,
    hits_by_query: dict[str, list[tuple[str, int, float]]],
    origins: dict[str, set[int]],
    sample_size: int,
    seed: int,
    top_k: int,
) -> dict[str, Any]:
    if sample_size == 0:
        return {"status": "SKIPPED"}
    value = nprobe(run_name)
    if value is None:
        return {"status": "SKIPPED", "reason": "run name has no nprobe"}

    from evidencegap.dense.embeddings import QueryEmbeddingStore
    from evidencegap.dense.faiss_backend import DenseFaissBackend

    dense_dir = root / "artifacts/v1/dense/medcpt"
    query_store = QueryEmbeddingStore(root, dense_dir / "query_embeddings" / split)
    backend = DenseFaissBackend(root, dense_dir / "faiss_index", nprobe=value)
    ids = article_ids(root / "artifacts/v1/article_corpus")

    query_ids = sorted(hits_by_query)
    randomizer = random.Random(seed + value)
    selected = query_ids if sample_size >= len(query_ids) else randomizer.sample(query_ids, sample_size)
    matrix = query_store.rows_for_claims(selected)
    extra = max((len(origins.get(query_id, set())) for query_id in selected), default=0)
    _, doc_indices = backend.search(matrix, top_k=top_k + extra)

    mismatches: list[dict[str, Any]] = []
    for offset, query_id in enumerate(selected):
        reproduced: list[str] = []
        for doc_idx_raw in doc_indices[offset]:
            doc_idx = int(doc_idx_raw)
            if doc_idx < 0 or doc_idx in origins.get(query_id, set()):
                continue
            reproduced.append(ids[doc_idx])
            if len(reproduced) == top_k:
                break
        saved = [article_id for article_id, _, _ in hits_by_query[query_id]]
        if reproduced != saved:
            first = next(
                (i + 1 for i, (left, right) in enumerate(zip(reproduced, saved)) if left != right),
                min(len(reproduced), len(saved)) + 1,
            )
            mismatches.append({"claim_id": query_id, "first_mismatch_rank": first})
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "sample_size": len(selected),
        "mismatch_queries": len(mismatches),
        "mismatch_examples": mismatches[:10],
    }


def main() -> None:
    value = args()
    root = value.root.resolve()
    run_dir = root / "artifacts/v1/article_retrieval_runs"
    corpus_dir = root / "artifacts/v1/article_corpus"
    run_names = value.run_name or discover(run_dir, value.split)
    if not run_names:
        raise CheckError("No MedCPT runs found")

    runs: dict[str, dict[str, list[tuple[str, int, float]]]] = {}
    for run_name in run_names:
        path = run_dir / f"{value.split}_open_independent_{run_name}.trec"
        runs[run_name] = read_trec(path, run_name, value.top_k)

    query_ids = set().union(*(set(run) for run in runs.values()))
    positives, origins = load_gold(corpus_dir, value.split, query_ids)
    reference_name = max(run_names, key=lambda name: nprobe(name) or -1)
    reference = runs[reference_name]

    reports: list[dict[str, Any]] = []
    for run_name in run_names:
        item = {
            "run_name": run_name,
            "nprobe": nprobe(run_name),
            "analysis": analyze(runs[run_name], positives),
            "faiss_verification": verify_faiss(
                root,
                value.split,
                run_name,
                runs[run_name],
                origins,
                value.verify_faiss_sample,
                value.seed,
                value.top_k,
            ),
        }
        if run_name != reference_name:
            item["comparison_to_reference"] = compare(runs[run_name], reference, positives)
        reports.append(item)

    shared_all = set.intersection(*(set(run) for run in runs.values()))
    ref_metrics = metrics_on_shared(reference, positives, shared_all)
    recommendation = None
    candidates = []
    for run_name in sorted(run_names, key=lambda name: nprobe(name) or 10**9):
        metrics = metrics_on_shared(runs[run_name], positives, shared_all)
        qualifies = (
            ref_metrics["known_positive_recall@100"] - metrics["known_positive_recall@100"] <= value.tolerance
            and ref_metrics["hitrate@100"] - metrics["hitrate@100"] <= value.tolerance
        )
        candidates.append({"run_name": run_name, "nprobe": nprobe(run_name), "metrics": metrics, "qualifies": qualifies})
        if recommendation is None and qualifies:
            recommendation = nprobe(run_name)

    reference_analysis = next(item["analysis"] for item in reports if item["run_name"] == reference_name)
    conclusions = [
        "TREC depth/ranks are valid for all selected runs.",
        (
            "Reference Recall@10/50/100 plateau is real: every recovered known positive is already in top 10."
            if reference_analysis["recall_plateau"] and reference_analysis["all_recovered_positives_are_top10"]
            else "Reference run has recovered positives outside top 10; the recall plateau is not strict."
        ),
    ]
    comparisons = [item.get("comparison_to_reference") for item in reports if item.get("comparison_to_reference")]
    if any(item["consistent_with_ivf_cluster_gating"] for item in comparisons):
        conclusions.append(
            "Higher nprobe recovers missing positives mostly directly into top 10, consistent with IVF cluster-gating misses."
        )
    elif comparisons:
        conclusions.append("Current comparisons do not show a strong IVF cluster-gating signature.")
    else:
        conclusions.append("Only one run was checked; add a higher-nprobe run to diagnose IVF misses.")
    if all(item["faiss_verification"]["status"] in {"PASS", "SKIPPED"} for item in reports):
        conclusions.append("FAISS spot checks show no evidence of positive injection into saved TREC runs.")
    else:
        conclusions.append("At least one saved run could not be reproduced from current FAISS assets.")

    report = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": value.split,
        "reference_run_name": reference_name,
        "shared_queries_for_nprobe_selection": len(shared_all),
        "recommended_nprobe": recommendation,
        "nprobe_candidates": candidates,
        "runs": reports,
        "conclusions": conclusions,
        "caveat": "Known-positive metrics use incomplete qrels; unjudged articles are not confirmed negatives.",
    }
    output = value.output or root / "reports/v1" / f"medcpt_retrieval_diagnostics_{value.split}.json"
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Report: {output}")
    print(json.dumps({
        "reference_run_name": reference_name,
        "recommended_nprobe": recommendation,
        "conclusions": conclusions,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except CheckError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
