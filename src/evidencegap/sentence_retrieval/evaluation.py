from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path
from evidencegap.sentence_retrieval.artifacts import read_rows_by_query
from evidencegap.sentence_retrieval.contracts import EvidenceQuery
from evidencegap.sentence_retrieval.evidencebench import load_canonical_queries


def _covered_aspects(
    query: EvidenceQuery,
    predicted_indices: Iterable[int],
    allowed_aspects: Sequence[str],
) -> set[str]:
    predicted = set(int(index) for index in predicted_indices)
    return {
        aspect_id
        for aspect_id in allowed_aspects
        if predicted.intersection(query.aspect_to_sentence_indices.get(aspect_id, ()))
    }


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_rankings(
    queries: Sequence[EvidenceQuery],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    expected_ids = {query.query_id for query in queries}
    extra = set(rankings) - expected_ids
    missing = expected_ids - set(rankings)
    if extra:
        raise EvidenceGapError(f"Ranking contains unexpected queries: {sorted(extra)[:5]}")
    if missing:
        raise EvidenceGapError(f"Ranking misses queries: {sorted(missing)[:5]}")

    aspect_at_5: list[float] = []
    aspect_at_optimal: list[float] = []
    results_aspect_at_5: list[float] = []
    sentence_precision_at_5: list[float] = []
    first_hit_rr: list[float] = []
    candidate_lt5 = 0
    empty_predictions = 0
    empty_aspects = 0
    empty_results = 0
    optimal_unavailable = 0

    for query in queries:
        rows = sorted(rankings[query.query_id], key=lambda row: int(row["final_rank"]))
        indices = [int(row["sentence_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            raise EvidenceGapError(f"Duplicate sentence indices for {query.query_id}")
        if any(index < 0 or index >= len(query.candidate_sentences) for index in indices):
            raise EvidenceGapError(f"Invalid sentence index for {query.query_id}")
        required_at_5 = min(5, len(query.candidate_sentences))
        if len(query.candidate_sentences) < 5:
            candidate_lt5 += 1
        if not rows:
            empty_predictions += 1
        if len(indices) < required_at_5:
            raise EvidenceGapError(
                f"Ranking depth is too small for @5 on {query.query_id}: "
                f"need {required_at_5}, got {len(indices)}"
            )

        top5 = indices[:5]
        if query.coverable_aspect_ids:
            aspect_at_5.append(
                len(_covered_aspects(query, top5, query.coverable_aspect_ids))
                / len(query.coverable_aspect_ids)
            )
            if query.optimal_sentence_budget is None:
                optimal_unavailable += 1
            else:
                if len(indices) < min(query.optimal_sentence_budget, len(query.candidate_sentences)):
                    raise EvidenceGapError(
                        f"Ranking depth is too small for @Optimal on {query.query_id}: "
                        f"need {query.optimal_sentence_budget}, got {len(indices)}"
                    )
                top_optimal = indices[: query.optimal_sentence_budget]
                aspect_at_optimal.append(
                    len(
                        _covered_aspects(
                            query, top_optimal, query.coverable_aspect_ids
                        )
                    )
                    / len(query.coverable_aspect_ids)
                )
        else:
            empty_aspects += 1

        coverable_results_aspects = tuple(
            aspect_id
            for aspect_id in (query.results_aspect_ids or ())
            if aspect_id in set(query.coverable_aspect_ids)
        )
        if coverable_results_aspects:
            results_aspect_at_5.append(
                len(_covered_aspects(query, top5, coverable_results_aspects))
                / len(coverable_results_aspects)
            )
        else:
            empty_results += 1

        if query.coverable_aspect_ids:
            denominator = len(top5)
            if denominator == 0:
                sentence_precision_at_5.append(0.0)
            else:
                relevant = sum(
                    bool(query.sentence_to_aspects.get(str(index), ()))
                    for index in top5
                )
                sentence_precision_at_5.append(relevant / denominator)

            reciprocal_rank = 0.0
            for rank, index in enumerate(indices, start=1):
                if query.sentence_to_aspects.get(str(index), ()):
                    reciprocal_rank = 1.0 / rank
                    break
            first_hit_rr.append(reciprocal_rank)

    return {
        "metrics": {
            "aspect_recall_at_5": _mean(aspect_at_5),
            "aspect_recall_at_optimal": _mean(aspect_at_optimal),
            "results_aspect_recall_at_5": _mean(results_aspect_at_5),
            "sentence_precision_at_5": _mean(sentence_precision_at_5),
            "first_hit_mrr": _mean(first_hit_rr),
        },
        "eligible_queries": {
            "aspect_recall_at_5": len(aspect_at_5),
            "aspect_recall_at_optimal": len(aspect_at_optimal),
            "results_aspect_recall_at_5": len(results_aspect_at_5),
            "sentence_precision_at_5": len(sentence_precision_at_5),
            "first_hit_mrr": len(first_hit_rr),
        },
        "skipped_or_special": {
            "empty_coverable_aspect_queries": empty_aspects,
            "optimal_unavailable_queries": optimal_unavailable,
            "null_or_empty_results_aspect_queries": empty_results,
            "candidate_pool_lt5_queries": candidate_lt5,
            "empty_prediction_queries": empty_predictions,
        },
        "denominator_semantics": {
            "aspect_recall_at_5": (
                "macro mean over queries with at least one coverable aspect; "
                "the denominator is the union of aspects appearing in "
                "sentence_index2aspects, matching the official evaluator"
            ),
            "aspect_recall_at_optimal": (
                "macro mean using each query's official optimal budget; "
                "when the raw precomputed field is absent, the same minimum "
                "set-cover budget is derived exactly from gold aspect mappings"
            ),
            "results_aspect_recall_at_5": (
                "macro mean only over queries with at least one results aspect "
                "that appears in sentence_index2aspects"
            ),
            "sentence_precision_at_5": (
                "macro mean over queries with non-empty coverable_aspect_ids; relevant returned "
                "sentences divided by min(5, returned rows), zero for an empty prediction"
            ),
            "first_hit_mrr": (
                "macro mean over queries with non-empty coverable_aspect_ids; 1/rank of first "
                "sentence linked to any gold aspect, else 0"
            ),
        },
    }


def evaluate_sentence_run(
    root: Path,
    *,
    canonical_dir: Path,
    run_path: Path,
    report_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    queries, canonical_manifest = load_canonical_queries(canonical_dir)
    rankings = read_rows_by_query(run_path)
    result = evaluate_rankings(queries, rankings)
    result.update(
        {
            "schema_version": "1.0.0",
            "task_id": "ESR-EVIDENCEBENCH",
            "split": canonical_manifest["split"],
            "selected_queries": canonical_manifest["selected_queries"],
            "canonical_path": relative_path(root, canonical_dir),
            "run_path": relative_path(root, run_path),
        }
    )
    if report_path is not None:
        atomic_write_json(report_path.resolve(), result)
        result["report_path"] = relative_path(root, report_path.resolve())
    return result


def score_direction_diagnostics(
    queries: Sequence[EvidenceQuery],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    score_field: str,
    pair_limit_per_query: int = 5000,
) -> dict[str, Any]:
    gold_scores: list[float] = []
    non_gold_scores: list[float] = []
    pairs_correct = 0
    pairs_total = 0
    for query in queries:
        rows = rankings.get(query.query_id, ())
        positives: list[float] = []
        negatives: list[float] = []
        for row in rows:
            value = row.get(score_field)
            if value is None:
                continue
            score = float(value)
            index = int(row["sentence_index"])
            if query.sentence_to_aspects.get(str(index), ()):
                positives.append(score)
                gold_scores.append(score)
            else:
                negatives.append(score)
                non_gold_scores.append(score)
        compared = 0
        for positive in positives:
            for negative in negatives:
                if compared >= pair_limit_per_query:
                    break
                pairs_correct += int(positive > negative)
                pairs_total += 1
                compared += 1
            if compared >= pair_limit_per_query:
                break
    gold_mean = _mean(gold_scores)
    non_gold_mean = _mean(non_gold_scores)
    accuracy = pairs_correct / pairs_total if pairs_total else None
    warning = bool(
        gold_mean is None
        or non_gold_mean is None
        or accuracy is None
        or gold_mean <= non_gold_mean
        or accuracy <= 0.5
    )
    return {
        "score_field": score_field,
        "gold_sentence_count": len(gold_scores),
        "non_gold_sentence_count": len(non_gold_scores),
        "gold_mean_score": gold_mean,
        "non_gold_mean_score": non_gold_mean,
        "mean_score_difference": (
            None if gold_mean is None or non_gold_mean is None else gold_mean - non_gold_mean
        ),
        "pairwise_comparisons": pairs_total,
        "pairwise_gold_above_non_gold_accuracy": accuracy,
        "warning": warning,
        "warning_reason": (
            "Check input format, pair direction, pooling, truncation, and model loading"
            if warning
            else None
        ),
    }


def diagnose_sentence_run(
    *,
    canonical_dir: Path,
    run_path: Path,
    score_field: str,
    pair_limit_per_query: int = 5000,
) -> dict[str, Any]:
    queries, manifest = load_canonical_queries(canonical_dir)
    rankings = read_rows_by_query(run_path)
    value = score_direction_diagnostics(
        queries,
        rankings,
        score_field=score_field,
        pair_limit_per_query=pair_limit_per_query,
    )
    value["split"] = manifest["split"]
    value["selected_queries"] = manifest["selected_queries"]
    return value
