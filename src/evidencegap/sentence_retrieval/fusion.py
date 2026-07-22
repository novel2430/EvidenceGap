from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path, sha256_file
from evidencegap.sentence_retrieval.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    RUN_SCHEMA_VERSION,
    read_rows_by_query,
    safe_run_name,
    validate_ranking_rows,
    write_rows_atomic,
)
from evidencegap.sentence_retrieval.contracts import EvidenceQuery, SCHEMA_VERSION, TASK_ID
from evidencegap.sentence_retrieval.evaluation import evaluate_sentence_run
from evidencegap.sentence_retrieval.evidencebench import load_canonical_queries


def _ordered_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted((dict(row) for row in rows), key=lambda row: int(row["final_rank"]))


def _validate_source_rows(
    query: EvidenceQuery,
    rows: Sequence[Mapping[str, Any]],
    *,
    source_label: str,
) -> list[dict[str, Any]]:
    ordered = _ordered_rows(rows)
    if not ordered:
        raise EvidenceGapError(f"{source_label} has no rows for {query.query_id}")
    seen: set[int] = set()
    for expected_rank, row in enumerate(ordered, start=1):
        if int(row["final_rank"]) != expected_rank:
            raise EvidenceGapError(
                f"{source_label} has non-contiguous final_rank for {query.query_id}"
            )
        if str(row["paper_id"]) != query.paper_id:
            raise EvidenceGapError(f"{source_label} paper mismatch for {query.query_id}")
        if str(row["pool_fingerprint"]) != query.pool_fingerprint:
            raise EvidenceGapError(f"{source_label} pool mismatch for {query.query_id}")
        index = int(row["sentence_index"])
        if index in seen:
            raise EvidenceGapError(
                f"{source_label} repeats sentence {query.query_id}:{index}"
            )
        seen.add(index)
        if index < 0 or index >= len(query.candidate_sentences):
            raise EvidenceGapError(
                f"{source_label} has invalid sentence {query.query_id}:{index}"
            )
        if str(row["sentence_text"]) != query.candidate_sentences[index]:
            raise EvidenceGapError(
                f"{source_label} changed sentence text for {query.query_id}:{index}"
            )
    return ordered


def _covered_aspects(query: EvidenceQuery, indices: Iterable[int]) -> set[str]:
    predicted = {int(index) for index in indices}
    return {
        aspect_id
        for aspect_id in query.coverable_aspect_ids
        if predicted.intersection(query.aspect_to_sentence_indices.get(aspect_id, ()))
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def analyze_sentence_run_complementarity(
    *,
    canonical_dir: Path,
    left_path: Path,
    right_path: Path,
    depths: Sequence[int] = (5, 10, 20, 50),
    left_name: str | None = None,
    right_name: str | None = None,
    report_path: Path | None = None,
) -> dict[str, Any]:
    depths = tuple(sorted(set(int(depth) for depth in depths)))
    if not depths or any(depth <= 0 for depth in depths):
        raise EvidenceGapError("Complementarity depths must be positive")
    queries, canonical_manifest = load_canonical_queries(canonical_dir)
    left = read_rows_by_query(left_path)
    right = read_rows_by_query(right_path)
    expected = {query.query_id for query in queries}
    for label, rows in (("left", left), ("right", right)):
        if set(rows) != expected:
            raise EvidenceGapError(
                f"{label} run query coverage mismatch: "
                f"missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}"
            )

    left_label = left_name or left_path.parent.name
    right_label = right_name or right_path.parent.name
    by_depth: dict[str, Any] = {}
    for depth in depths:
        overlap_counts: list[float] = []
        jaccards: list[float] = []
        union_sizes: list[float] = []
        left_gold_counts: list[float] = []
        right_gold_counts: list[float] = []
        union_gold_counts: list[float] = []
        left_only_gold_counts: list[float] = []
        right_only_gold_counts: list[float] = []
        left_recalls: list[float] = []
        right_recalls: list[float] = []
        union_recalls: list[float] = []
        oracle_recalls: list[float] = []
        union_better_left = 0
        union_better_right = 0
        union_better_both = 0
        eligible = 0

        for query in queries:
            left_rows = _validate_source_rows(
                query, left[query.query_id], source_label=left_label
            )
            right_rows = _validate_source_rows(
                query, right[query.query_id], source_label=right_label
            )
            if len(left_rows) < depth or len(right_rows) < depth:
                raise EvidenceGapError(
                    f"Depth {depth} exceeds available rows for {query.query_id}: "
                    f"{left_label}={len(left_rows)}, {right_label}={len(right_rows)}"
                )
            left_set = {int(row["sentence_index"]) for row in left_rows[:depth]}
            right_set = {int(row["sentence_index"]) for row in right_rows[:depth]}
            union = left_set | right_set
            overlap = left_set & right_set
            overlap_counts.append(float(len(overlap)))
            union_sizes.append(float(len(union)))
            jaccards.append(len(overlap) / len(union) if union else 1.0)

            gold = {
                int(index)
                for index, aspects in query.sentence_to_aspects.items()
                if aspects
            }
            left_gold_counts.append(float(len(left_set & gold)))
            right_gold_counts.append(float(len(right_set & gold)))
            union_gold_counts.append(float(len(union & gold)))
            left_only_gold_counts.append(float(len((left_set - right_set) & gold)))
            right_only_gold_counts.append(float(len((right_set - left_set) & gold)))

            if not query.coverable_aspect_ids:
                continue
            eligible += 1
            denominator = len(query.coverable_aspect_ids)
            left_recall = len(_covered_aspects(query, left_set)) / denominator
            right_recall = len(_covered_aspects(query, right_set)) / denominator
            union_recall = len(_covered_aspects(query, union)) / denominator
            oracle_recall = max(left_recall, right_recall)
            left_recalls.append(left_recall)
            right_recalls.append(right_recall)
            union_recalls.append(union_recall)
            oracle_recalls.append(oracle_recall)
            union_better_left += int(union_recall > left_recall)
            union_better_right += int(union_recall > right_recall)
            union_better_both += int(
                union_recall > left_recall and union_recall > right_recall
            )

        left_mean = _mean(left_recalls)
        right_mean = _mean(right_recalls)
        union_mean = _mean(union_recalls)
        oracle_mean = _mean(oracle_recalls)
        best_single = (
            None
            if left_mean is None or right_mean is None
            else max(left_mean, right_mean)
        )
        by_depth[str(depth)] = {
            "depth_per_source": depth,
            "eligible_queries": eligible,
            "candidate_overlap": {
                "mean_overlap_count": _mean(overlap_counts),
                "mean_union_size": _mean(union_sizes),
                "mean_jaccard": _mean(jaccards),
            },
            "gold_sentence_candidates": {
                f"{left_label}_mean": _mean(left_gold_counts),
                f"{right_label}_mean": _mean(right_gold_counts),
                "union_mean": _mean(union_gold_counts),
                f"{left_label}_only_mean": _mean(left_only_gold_counts),
                f"{right_label}_only_mean": _mean(right_only_gold_counts),
            },
            "aspect_recall_at_candidate_depth": {
                left_label: left_mean,
                right_label: right_mean,
                "union": union_mean,
                "oracle_best_single_per_query": oracle_mean,
                "union_minus_best_global_single": (
                    None
                    if union_mean is None or best_single is None
                    else union_mean - best_single
                ),
                "union_minus_oracle_best_single_per_query": (
                    None
                    if union_mean is None or oracle_mean is None
                    else union_mean - oracle_mean
                ),
            },
            "query_counts": {
                f"union_better_than_{left_label}": union_better_left,
                f"union_better_than_{right_label}": union_better_right,
                "union_better_than_both": union_better_both,
            },
        }

    result = {
        "schema_version": "1.0.0",
        "task_id": TASK_ID,
        "analysis": "sentence_candidate_complementarity",
        "split": canonical_manifest["split"],
        "selected_queries": canonical_manifest["selected_queries"],
        "left_name": left_label,
        "right_name": right_label,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "depths": by_depth,
    }
    if report_path is not None:
        atomic_write_json(report_path.resolve(), result)
        result["report_path"] = str(report_path.resolve())
    return result


def _rrf_candidates(
    *,
    query: EvidenceQuery,
    left_rows: Sequence[Mapping[str, Any]],
    right_rows: Sequence[Mapping[str, Any]],
    left_depth: int,
    right_depth: int,
    rrf_k: int,
) -> list[tuple[int, float, int, int]]:
    if left_depth <= 0 or right_depth <= 0 or rrf_k < 0:
        raise EvidenceGapError("Fusion depths must be positive and rrf_k non-negative")
    left = _validate_source_rows(query, left_rows, source_label="left")
    right = _validate_source_rows(query, right_rows, source_label="right")
    if len(left) < left_depth or len(right) < right_depth:
        raise EvidenceGapError(
            f"Fusion source depth exceeds available rows for {query.query_id}: "
            f"left={len(left)}/{left_depth}, right={len(right)}/{right_depth}"
        )
    ranks: dict[int, list[int | None]] = {}
    for rank, row in enumerate(left[:left_depth], start=1):
        ranks.setdefault(int(row["sentence_index"]), [None, None])[0] = rank
    for rank, row in enumerate(right[:right_depth], start=1):
        ranks.setdefault(int(row["sentence_index"]), [None, None])[1] = rank

    candidates: list[tuple[int, float, int, int]] = []
    sentinel = max(left_depth, right_depth) + 1
    for index, (left_rank, right_rank) in ranks.items():
        score = 0.0
        if left_rank is not None:
            score += 1.0 / (rrf_k + left_rank)
        if right_rank is not None:
            score += 1.0 / (rrf_k + right_rank)
        best_rank = min(
            left_rank if left_rank is not None else sentinel,
            right_rank if right_rank is not None else sentinel,
        )
        source_count = int(left_rank is not None) + int(right_rank is not None)
        candidates.append((index, score, best_rank, source_count))
    candidates.sort(key=lambda item: (-item[1], -item[3], item[2], item[0]))
    return candidates


def run_sentence_rrf_fusion(
    root: Path,
    *,
    split: str,
    canonical_dir: Path,
    left_path: Path,
    right_path: Path,
    left_name: str | None = None,
    right_name: str | None = None,
    left_depth: int = 20,
    right_depth: int = 20,
    output_depth: int | None = None,
    rrf_k: int = 60,
    run_name: str | None = None,
    artifact_root: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    canonical_dir = canonical_dir.resolve()
    left_path = left_path.resolve()
    right_path = right_path.resolve()
    queries, canonical_manifest = load_canonical_queries(canonical_dir)
    if canonical_manifest["split"] != split:
        raise EvidenceGapError(
            f"Canonical split mismatch: expected {split}, got {canonical_manifest['split']}"
        )
    left_rows = read_rows_by_query(left_path)
    right_rows = read_rows_by_query(right_path)
    expected = {query.query_id for query in queries}
    for label, rows in (("left", left_rows), ("right", right_rows)):
        if set(rows) != expected:
            raise EvidenceGapError(
                f"{label} fusion source query coverage mismatch: "
                f"missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}"
            )
    if output_depth is not None and output_depth <= 0:
        raise EvidenceGapError("output_depth must be positive when provided")

    left_label = left_name or left_path.parent.name
    right_label = right_name or right_path.parent.name
    default_name = (
        f"rrf_{left_label}_d{left_depth}_{right_label}_d{right_depth}_"
        f"{'union' if output_depth is None else f'top{output_depth}'}_k{rrf_k}"
    )
    name = safe_run_name(run_name or default_name)
    base_root = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    run_base = base_root / "fusion" / name
    output_path = run_base / "ranked_sentences.parquet"
    manifest_path = run_base / "run_manifest.json"
    if (output_path.exists() or manifest_path.exists()) and not force:
        raise EvidenceGapError(f"Fusion run already exists: {run_base}; use --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    expected_depths: dict[str, int] = {}

    def rows() -> Iterable[dict[str, Any]]:
        for query in queries:
            candidates = _rrf_candidates(
                query=query,
                left_rows=left_rows[query.query_id],
                right_rows=right_rows[query.query_id],
                left_depth=left_depth,
                right_depth=right_depth,
                rrf_k=rrf_k,
            )
            selected = candidates if output_depth is None else candidates[:output_depth]
            expected_depths[query.query_id] = len(selected)
            for rank, (index, score, _best_rank, _source_count) in enumerate(
                selected, start=1
            ):
                yield {
                    "schema_version": SCHEMA_VERSION,
                    "task_id": TASK_ID,
                    "split": split,
                    "run_name": name,
                    "query_id": query.query_id,
                    "paper_id": query.paper_id,
                    "pool_fingerprint": query.pool_fingerprint,
                    "sentence_index": index,
                    "sentence_type": query.sentence_types[index],
                    "sentence_text": query.candidate_sentences[index],
                    "retrieval_model": f"rrf:{left_label}+{right_label}",
                    "retrieval_score": float(score),
                    "retrieval_rank": rank,
                    "cross_encoder_score": None,
                    "final_score": float(score),
                    "final_rank": rank,
                }

    row_count = write_rows_atomic(output_path, rows())
    validation = validate_ranking_rows(
        output_path,
        expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
        expected_depths=expected_depths,
        expected_run_name=name,
    )
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_name": name,
        "split": split,
        "run_type": "equal_weight_rrf_sentence_fusion",
        "canonical_dir": relative_path(root, canonical_dir),
        "canonical_sha256": canonical_manifest["canonical_sha256"],
        "sources": [
            {
                "name": left_label,
                "path": relative_path(root, left_path),
                "sha256": sha256_file(left_path),
                "depth": left_depth,
            },
            {
                "name": right_label,
                "path": relative_path(root, right_path),
                "sha256": sha256_file(right_path),
                "depth": right_depth,
            },
        ],
        "parameters": {
            "rrf_k": rrf_k,
            "output_depth": output_depth,
            "score_semantics": "equal_weight_rrf_higher_is_better",
            "union_contract": (
                "all unique source Top-N sentences are retained when output_depth is null; "
                "RRF only provides deterministic ordering before full cross-encoder reranking"
            ),
        },
        "queries": len(queries),
        "rows": row_count,
        "output_path": relative_path(root, output_path),
        "output_sha256": validation["sha256"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(manifest_path, manifest)
    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_path = report_root / f"evidence_sentence_retrieval_{name}_{split}.json"
    evaluation = evaluate_sentence_run(
        root,
        canonical_dir=canonical_dir,
        run_path=output_path,
        report_path=report_path,
    )
    return {
        "run_name": name,
        "run_path": relative_path(root, output_path),
        "manifest_path": relative_path(root, manifest_path),
        "report_path": relative_path(root, report_path),
        "validation": validation,
        "metrics": evaluation["metrics"],
        "union_size": {
            "min": min(expected_depths.values()),
            "max": max(expected_depths.values()),
            "mean": sum(expected_depths.values()) / len(expected_depths),
        },
    }


def _per_query_metrics(
    query: EvidenceQuery,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    ordered = _ordered_rows(rows)
    indices = [int(row["sentence_index"]) for row in ordered]
    if len(indices) != len(set(indices)):
        raise EvidenceGapError(f"Duplicate sentence indices for {query.query_id}")
    if len(indices) < min(5, len(query.candidate_sentences)):
        raise EvidenceGapError(f"Run is too shallow for @5 on {query.query_id}")
    top5 = indices[:5]
    if not query.coverable_aspect_ids:
        return {
            "aspect_recall_at_5": None,
            "aspect_recall_at_optimal": None,
            "sentence_precision_at_5": None,
            "first_hit_mrr": None,
        }
    denominator = len(query.coverable_aspect_ids)
    aspect_at_5 = len(_covered_aspects(query, top5)) / denominator
    if query.optimal_sentence_budget is None:
        aspect_at_optimal = None
    else:
        if len(indices) < query.optimal_sentence_budget:
            raise EvidenceGapError(f"Run is too shallow for @Optimal on {query.query_id}")
        aspect_at_optimal = (
            len(_covered_aspects(query, indices[: query.optimal_sentence_budget]))
            / denominator
        )
    precision = sum(
        bool(query.sentence_to_aspects.get(str(index), ())) for index in top5
    ) / len(top5)
    reciprocal_rank = 0.0
    for rank, index in enumerate(indices, start=1):
        if query.sentence_to_aspects.get(str(index), ()):
            reciprocal_rank = 1.0 / rank
            break
    return {
        "aspect_recall_at_5": aspect_at_5,
        "aspect_recall_at_optimal": aspect_at_optimal,
        "sentence_precision_at_5": precision,
        "first_hit_mrr": reciprocal_rank,
    }


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise EvidenceGapError("Cannot compute percentile of an empty sequence")
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def compare_sentence_runs_paired(
    *,
    canonical_dir: Path,
    baseline_path: Path,
    challenger_path: Path,
    baseline_name: str | None = None,
    challenger_name: str | None = None,
    bootstrap_samples: int = 5000,
    seed: int = 20260722,
    bootstrap_unit: str = "systematic_review",
    report_path: Path | None = None,
) -> dict[str, Any]:
    if bootstrap_samples <= 0:
        raise EvidenceGapError("bootstrap_samples must be positive")
    queries, canonical_manifest = load_canonical_queries(canonical_dir)
    baseline = read_rows_by_query(baseline_path)
    challenger = read_rows_by_query(challenger_path)
    expected = {query.query_id for query in queries}
    for label, rows in (("baseline", baseline), ("challenger", challenger)):
        if set(rows) != expected:
            raise EvidenceGapError(
                f"{label} run query coverage mismatch: "
                f"missing={len(expected-set(rows))}, extra={len(set(rows)-expected)}"
            )
    base_label = baseline_name or baseline_path.parent.name
    challenge_label = challenger_name or challenger_path.parent.name
    if bootstrap_unit not in {"query", "paper", "systematic_review"}:
        raise EvidenceGapError(
            "bootstrap_unit must be query, paper, or systematic_review"
        )
    metric_pairs: dict[str, list[tuple[str, float, float]]] = {
        "aspect_recall_at_5": [],
        "aspect_recall_at_optimal": [],
        "sentence_precision_at_5": [],
        "first_hit_mrr": [],
    }
    for query in queries:
        base_metrics = _per_query_metrics(query, baseline[query.query_id])
        challenge_metrics = _per_query_metrics(query, challenger[query.query_id])
        for metric in metric_pairs:
            base_value = base_metrics[metric]
            challenge_value = challenge_metrics[metric]
            if base_value is None or challenge_value is None:
                continue
            if bootstrap_unit == "query":
                cluster_id = query.query_id
            elif bootstrap_unit == "paper":
                cluster_id = query.paper_id
            else:
                cluster_id = query.systematic_review_id or f"paper:{query.paper_id}"
            metric_pairs[metric].append(
                (str(cluster_id), float(base_value), float(challenge_value))
            )

    rng = random.Random(seed)
    comparisons: dict[str, Any] = {}
    for metric, pairs in metric_pairs.items():
        if not pairs:
            comparisons[metric] = None
            continue
        deltas = [challenger_value - baseline_value for _cluster, baseline_value, challenger_value in pairs]
        clusters: dict[str, list[tuple[float, float]]] = {}
        for cluster_id, baseline_value, challenger_value in pairs:
            clusters.setdefault(cluster_id, []).append((baseline_value, challenger_value))
        cluster_ids = sorted(clusters)
        bootstrap_deltas: list[float] = []
        for _ in range(bootstrap_samples):
            sampled_pairs: list[tuple[float, float]] = []
            for _cluster_index in range(len(cluster_ids)):
                sampled_cluster = cluster_ids[rng.randrange(len(cluster_ids))]
                sampled_pairs.extend(clusters[sampled_cluster])
            sample_delta = sum(
                challenger_value - baseline_value
                for baseline_value, challenger_value in sampled_pairs
            ) / len(sampled_pairs)
            bootstrap_deltas.append(sample_delta)
        mean_baseline = sum(value[1] for value in pairs) / len(pairs)
        mean_challenger = sum(value[2] for value in pairs) / len(pairs)
        mean_delta = sum(deltas) / len(deltas)
        comparisons[metric] = {
            "eligible_queries": len(pairs),
            "baseline_mean": mean_baseline,
            "challenger_mean": mean_challenger,
            "mean_delta": mean_delta,
            "relative_delta": (
                None if mean_baseline == 0 else mean_delta / mean_baseline
            ),
            "paired_query_wins": sum(delta > 0 for delta in deltas),
            "paired_query_ties": sum(delta == 0 for delta in deltas),
            "paired_query_losses": sum(delta < 0 for delta in deltas),
            "bootstrap": {
                "samples": bootstrap_samples,
                "seed": seed,
                "unit": bootstrap_unit,
                "clusters": len(clusters),
                "ci95": [
                    _percentile(bootstrap_deltas, 0.025),
                    _percentile(bootstrap_deltas, 0.975),
                ],
                "probability_mean_delta_gt_0": (
                    sum(delta > 0 for delta in bootstrap_deltas)
                    / len(bootstrap_deltas)
                ),
            },
        }

    result = {
        "schema_version": "1.0.0",
        "task_id": TASK_ID,
        "analysis": "paired_sentence_run_comparison",
        "split": canonical_manifest["split"],
        "selected_queries": canonical_manifest["selected_queries"],
        "baseline_name": base_label,
        "challenger_name": challenge_label,
        "baseline_path": str(baseline_path),
        "challenger_path": str(challenger_path),
        "primary_metric": "aspect_recall_at_5",
        "comparisons": comparisons,
    }
    if report_path is not None:
        atomic_write_json(report_path.resolve(), result)
        result["report_path"] = str(report_path.resolve())
    return result
