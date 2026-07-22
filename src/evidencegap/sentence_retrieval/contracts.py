from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from evidencegap.common import EvidenceGapError

SCHEMA_VERSION = "1.1.0"
TASK_ID = "ESR-EVIDENCEBENCH"
SCORE_SEMANTICS = "higher_is_more_relevant"


@dataclass(frozen=True)
class EvidenceModelInput:
    """Gold-free projection passed to sentence scorers."""

    query_id: str
    hypothesis: str
    paper_id: str
    candidate_sentences: tuple[str, ...]
    sentence_types: tuple[str, ...]
    pool_fingerprint: str


@dataclass(frozen=True)
class EvidenceQuery:
    query_id: str
    contract_version: str
    split: str
    official_split: str
    hypothesis: str
    paper_id: str
    systematic_review_id: str | None
    candidate_sentences: tuple[str, ...]
    sentence_types: tuple[str, ...]
    aspect_ids: tuple[str, ...]
    coverable_aspect_ids: tuple[str, ...]
    unmapped_aspect_ids: tuple[str, ...]
    aspect_text: Mapping[str, str]
    aspect_to_sentence_indices: Mapping[str, tuple[int, ...]]
    sentence_to_aspects: Mapping[str, tuple[str, ...]]
    results_aspect_ids: tuple[str, ...] | None
    optimal_sentence_budget: int | None
    optimal_sentence_budget_source: str | None
    raw_locator: Mapping[str, Any]
    pool_fingerprint: str

    def model_input(self) -> EvidenceModelInput:
        return EvidenceModelInput(
            query_id=self.query_id,
            hypothesis=self.hypothesis,
            paper_id=self.paper_id,
            candidate_sentences=self.candidate_sentences,
            sentence_types=self.sentence_types,
            pool_fingerprint=self.pool_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "record_type": "EvidenceQueryRecord",
            "dataset": "evidencebench_100k",
            "contract_version": self.contract_version,
            "query_id": self.query_id,
            "split": self.split,
            "official_split": self.official_split,
            "hypothesis": self.hypothesis,
            "paper_id": self.paper_id,
            "systematic_review_id": self.systematic_review_id,
            "candidate_sentences": list(self.candidate_sentences),
            "sentence_types": list(self.sentence_types),
            "aspect_ids": list(self.aspect_ids),
            "coverable_aspect_ids": list(self.coverable_aspect_ids),
            "unmapped_aspect_ids": list(self.unmapped_aspect_ids),
            "aspect_text": dict(self.aspect_text),
            "aspect_to_sentence_indices": {
                key: list(value) for key, value in self.aspect_to_sentence_indices.items()
            },
            "sentence_to_aspects": {
                key: list(value) for key, value in self.sentence_to_aspects.items()
            },
            "results_aspect_ids": (
                None if self.results_aspect_ids is None else list(self.results_aspect_ids)
            ),
            "optimal_sentence_budget": self.optimal_sentence_budget,
            "optimal_sentence_budget_source": self.optimal_sentence_budget_source,
            "raw_locator": dict(self.raw_locator),
            "pool_fingerprint": self.pool_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceQuery":
        aspect_ids = tuple(str(v) for v in value["aspect_ids"])
        sentence_to_aspects = {
            str(k): tuple(str(v) for v in aspects)
            for k, aspects in value["sentence_to_aspects"].items()
        }
        derived_coverable = tuple(
            aspect_id
            for aspect_id in aspect_ids
            if any(aspect_id in aspects for aspects in sentence_to_aspects.values())
        )
        coverable_aspect_ids = tuple(
            str(v) for v in value.get("coverable_aspect_ids", derived_coverable)
        )
        unmapped_aspect_ids = tuple(
            str(v)
            for v in value.get(
                "unmapped_aspect_ids",
                [aspect_id for aspect_id in aspect_ids if aspect_id not in set(coverable_aspect_ids)],
            )
        )
        return cls(
            query_id=str(value["query_id"]),
            contract_version=str(value.get("contract_version", "1.0.0")),
            split=str(value["split"]),
            official_split=str(value["official_split"]),
            hypothesis=str(value["hypothesis"]),
            paper_id=str(value["paper_id"]),
            systematic_review_id=(
                None
                if value.get("systematic_review_id") is None
                else str(value["systematic_review_id"])
            ),
            candidate_sentences=tuple(str(v) for v in value["candidate_sentences"]),
            sentence_types=tuple(str(v) for v in value["sentence_types"]),
            aspect_ids=aspect_ids,
            coverable_aspect_ids=coverable_aspect_ids,
            unmapped_aspect_ids=unmapped_aspect_ids,
            aspect_text={str(k): str(v) for k, v in value["aspect_text"].items()},
            aspect_to_sentence_indices={
                str(k): tuple(int(index) for index in indices)
                for k, indices in value["aspect_to_sentence_indices"].items()
            },
            sentence_to_aspects=sentence_to_aspects,
            results_aspect_ids=(
                None
                if value.get("results_aspect_ids") is None
                else tuple(str(v) for v in value["results_aspect_ids"])
            ),
            optimal_sentence_budget=(
                None
                if value.get("optimal_sentence_budget") is None
                else int(value["optimal_sentence_budget"])
            ),
            optimal_sentence_budget_source=(
                None
                if value.get("optimal_sentence_budget") is None
                else str(value.get("optimal_sentence_budget_source", "legacy_canonical"))
            ),
            raw_locator=dict(value["raw_locator"]),
            pool_fingerprint=str(value["pool_fingerprint"]),
        )


def pool_fingerprint(paper_id: str, sentences: Sequence[str]) -> str:
    payload = json.dumps(
        {"paper_id": paper_id, "sentences": list(sentences)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _list_of_strings(value: Any, *, field: str, allow_none: bool = False) -> list[str] | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, list):
        raise EvidenceGapError(f"EvidenceBench field {field} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise EvidenceGapError(f"EvidenceBench field {field} must contain only strings")
    return list(value)


def _deduplicate_preserving_order(values: Sequence[Any]) -> tuple[Any, ...]:
    """Normalize set-like raw lists without changing their first-seen order."""

    seen: set[Any] = set()
    unique: list[Any] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)


def _aspect_index_mapping(value: Any, *, field: str) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, dict):
        raise EvidenceGapError(f"EvidenceBench field {field} must be an object")
    result: dict[str, tuple[int, ...]] = {}
    for key, indices in value.items():
        if not isinstance(indices, list):
            raise EvidenceGapError(f"{field}[{key!r}] must be a list")
        try:
            parsed = tuple(int(index) for index in indices)
        except (TypeError, ValueError) as exc:
            raise EvidenceGapError(f"{field}[{key!r}] has a non-integer index") from exc
        # EvidenceBench occasionally repeats the same sentence index inside an
        # aspect list. The mapping is set-valued, so exact duplicates carry no
        # additional annotation information. Normalize them here while keeping
        # deterministic first-seen order; later bounds and forward/reverse
        # consistency validation still catch real data errors.
        result[str(key)] = _deduplicate_preserving_order(parsed)
    return result


def _sentence_aspect_mapping(value: Any, *, field: str) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, dict):
        raise EvidenceGapError(f"EvidenceBench field {field} must be an object")
    result: dict[str, tuple[str, ...]] = {}
    for key, aspects in value.items():
        try:
            index_key = str(int(key))
        except (TypeError, ValueError) as exc:
            raise EvidenceGapError(f"{field} has invalid sentence index key {key!r}") from exc
        if not isinstance(aspects, list):
            raise EvidenceGapError(f"{field}[{key!r}] must be a list")
        # The reverse mapping is also set-valued. Normalize redundant aspect
        # IDs symmetrically so the two projections compare by meaning rather
        # than raw list multiplicity.
        result[index_key] = _deduplicate_preserving_order(
            tuple(str(item) for item in aspects)
        )
    return result


def _derive_optimal_sentence_budget(
    *,
    query_id: str,
    coverable_aspect_ids: Sequence[str],
    sentence_to_aspects: Mapping[str, Sequence[str]],
) -> int:
    """Derive the exact ER@Optimal cutoff from the gold aspect mapping.

    EvidenceBench defines this cutoff as the minimum number of sentences needed
    to cover every coverable aspect. Some EvidenceBench-100k snapshots omit the
    precomputed ``evidence_retrieval_at_optimal_evaluation`` object, even though
    the aspect mappings are present. This exact branch-and-bound set-cover solver
    reconstructs the same evaluator-only quantity without exposing gold to any
    scorer.
    """

    aspect_position = {
        aspect_id: index for index, aspect_id in enumerate(coverable_aspect_ids)
    }
    full_mask = (1 << len(aspect_position)) - 1
    coverage_masks: set[int] = set()
    covered_union = 0
    for aspects in sentence_to_aspects.values():
        mask = 0
        for aspect_id in aspects:
            position = aspect_position.get(aspect_id)
            if position is not None:
                mask |= 1 << position
        if mask:
            coverage_masks.add(mask)
            covered_union |= mask

    if covered_union != full_mask:
        missing = [
            aspect_id
            for aspect_id, position in aspect_position.items()
            if not (covered_union & (1 << position))
        ]
        raise EvidenceGapError(
            f"{query_id} has inconsistent coverable-aspect projection: {missing[:5]}"
        )

    # A sentence whose aspect set is a strict subset of another sentence's set
    # can never improve a minimum-cardinality cover. Removing those masks makes
    # exact search cheap for the small aspect sets used by EvidenceBench.
    ordered_masks = sorted(coverage_masks, key=lambda mask: (-mask.bit_count(), mask))
    non_dominated: list[int] = []
    for mask in ordered_masks:
        if any(mask | kept == kept for kept in non_dominated):
            continue
        non_dominated.append(mask)
    masks = tuple(non_dominated)

    remaining = full_mask
    greedy_upper_bound = 0
    while remaining:
        best_mask = max(masks, key=lambda mask: (mask & remaining).bit_count())
        gain = (best_mask & remaining).bit_count()
        if gain == 0:
            raise EvidenceGapError(f"{query_id} cannot derive a complete optimal cover")
        remaining &= ~best_mask
        greedy_upper_bound += 1

    best = greedy_upper_bound
    masks_by_aspect: list[tuple[int, ...]] = []
    for position in range(len(aspect_position)):
        bit = 1 << position
        masks_by_aspect.append(tuple(mask for mask in masks if mask & bit))

    seen_depth: dict[int, int] = {}

    def search(covered: int, depth: int) -> None:
        nonlocal best
        if covered == full_mask:
            best = min(best, depth)
            return
        if depth >= best:
            return
        previous = seen_depth.get(covered)
        if previous is not None and previous <= depth:
            return
        seen_depth[covered] = depth

        uncovered = full_mask & ~covered
        max_gain = max((mask & uncovered).bit_count() for mask in masks)
        lower_bound = (uncovered.bit_count() + max_gain - 1) // max_gain
        if depth + lower_bound >= best:
            return

        uncovered_positions = [
            position
            for position in range(len(aspect_position))
            if uncovered & (1 << position)
        ]
        branch_position = min(
            uncovered_positions,
            key=lambda position: sum(
                1 for mask in masks_by_aspect[position] if mask & uncovered
            ),
        )
        options = sorted(
            masks_by_aspect[branch_position],
            key=lambda mask: (-(mask & uncovered).bit_count(), mask),
        )
        for mask in options:
            updated = covered | mask
            if updated != covered:
                search(updated, depth + 1)

    search(0, 0)
    return best


def canonicalize_raw_record(
    *,
    manifest_record: Mapping[str, Any],
    raw_record_id: str,
    raw_record: Mapping[str, Any],
) -> EvidenceQuery:
    expected_query_id = str(manifest_record.get("query_id", ""))
    expected_record_id = str(manifest_record.get("raw_locator", {}).get("record_id", ""))
    if raw_record_id != expected_record_id:
        raise EvidenceGapError(
            f"Raw record ID mismatch for {expected_query_id}: "
            f"expected {expected_record_id!r}, got {raw_record_id!r}"
        )
    if expected_query_id != f"evidencebench:{raw_record_id}":
        raise EvidenceGapError(
            f"Manifest query_id does not match raw record_id: {expected_query_id}"
        )

    hypothesis = raw_record.get("hypothesis")
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise EvidenceGapError(f"{expected_query_id} has empty hypothesis")
    paper_id_value = raw_record.get("paper_id")
    paper_id = "" if paper_id_value is None else str(paper_id_value)
    expected_paper_id = manifest_record.get("paper_id")
    if expected_paper_id is not None and paper_id != str(expected_paper_id):
        raise EvidenceGapError(f"{expected_query_id} paper_id mismatch")
    if not paper_id:
        raise EvidenceGapError(f"{expected_query_id} has empty paper_id")

    candidates = _list_of_strings(
        raw_record.get("paper_as_candidate_pool"), field="paper_as_candidate_pool"
    )
    sentence_types = _list_of_strings(
        raw_record.get("sentence_types_in_candidate_pool"),
        field="sentence_types_in_candidate_pool",
    )
    assert candidates is not None and sentence_types is not None
    if not candidates:
        raise EvidenceGapError(f"{expected_query_id} has an empty candidate pool")
    if len(candidates) != len(sentence_types):
        raise EvidenceGapError(
            f"{expected_query_id} candidate/sentence_type length mismatch: "
            f"{len(candidates)} != {len(sentence_types)}"
        )
    manifest_count = manifest_record.get("candidate_sentence_count")
    if manifest_count is not None and int(manifest_count) != len(candidates):
        raise EvidenceGapError(f"{expected_query_id} candidate count differs from manifest")

    aspect_ids = _list_of_strings(raw_record.get("aspect_list_ids"), field="aspect_list_ids")
    results_ids = _list_of_strings(
        raw_record.get("results_aspect_list_ids"),
        field="results_aspect_list_ids",
        allow_none=True,
    )
    assert aspect_ids is not None
    aspect_text_value = raw_record.get("aspect_id2aspect")
    if not isinstance(aspect_text_value, dict):
        raise EvidenceGapError(f"{expected_query_id} aspect_id2aspect must be an object")
    aspect_text = {str(key): str(value) for key, value in aspect_text_value.items()}
    aspect_to_indices = _aspect_index_mapping(
        raw_record.get("aspect2sentence_indices"), field="aspect2sentence_indices"
    )
    sentence_to_aspects = _sentence_aspect_mapping(
        raw_record.get("sentence_index2aspects"), field="sentence_index2aspects"
    )

    missing_text = set(aspect_ids) - set(aspect_text)
    if missing_text:
        raise EvidenceGapError(
            f"{expected_query_id} aspects missing text: {sorted(missing_text)[:5]}"
        )
    missing_mapping = set(aspect_ids) - set(aspect_to_indices)
    if missing_mapping:
        raise EvidenceGapError(
            f"{expected_query_id} aspects missing sentence mapping: "
            f"{sorted(missing_mapping)[:5]}"
        )
    extra_mapping = set(aspect_to_indices) - set(aspect_ids)
    if extra_mapping:
        raise EvidenceGapError(
            f"{expected_query_id} unknown aspects in aspect2sentence_indices: "
            f"{sorted(extra_mapping)[:5]}"
        )
    if results_ids is not None and not set(results_ids).issubset(aspect_ids):
        raise EvidenceGapError(f"{expected_query_id} results aspects are not a subset")

    forward_pairs: set[tuple[int, str]] = set()
    for aspect_id, indices in aspect_to_indices.items():
        for index in indices:
            if index < 0 or index >= len(candidates):
                raise EvidenceGapError(
                    f"{expected_query_id} gold index {index} outside candidate pool"
                )
            forward_pairs.add((index, aspect_id))

    reverse_pairs: set[tuple[int, str]] = set()
    for index_key, aspects in sentence_to_aspects.items():
        index = int(index_key)
        if index < 0 or index >= len(candidates):
            raise EvidenceGapError(
                f"{expected_query_id} reverse index {index} outside candidate pool"
            )
        unknown = set(aspects) - set(aspect_ids)
        if unknown:
            raise EvidenceGapError(
                f"{expected_query_id} sentence {index} references unknown aspects"
            )
        reverse_pairs.update((index, aspect_id) for aspect_id in aspects)
    if forward_pairs != reverse_pairs:
        only_forward = sorted(forward_pairs - reverse_pairs)[:5]
        only_reverse = sorted(reverse_pairs - forward_pairs)[:5]
        raise EvidenceGapError(
            f"{expected_query_id} aspect mappings disagree; "
            f"forward_only={only_forward}, reverse_only={only_reverse}"
        )

    # EvidenceBench-100k may declare aspects that have no supporting sentence
    # in the candidate pool. The official evaluator defines the denominator as
    # the union of aspects that actually occur in sentence_index2aspects, not
    # every ID in aspect_list_ids. Preserve both sets so the data issue remains
    # visible while metrics remain achievable and official-compatible.
    mapped_aspects = {
        aspect_id
        for aspects in sentence_to_aspects.values()
        for aspect_id in aspects
    }
    coverable_aspect_ids = tuple(
        aspect_id for aspect_id in aspect_ids if aspect_id in mapped_aspects
    )
    unmapped_aspect_ids = tuple(
        aspect_id for aspect_id in aspect_ids if aspect_id not in mapped_aspects
    )

    optimal_info = raw_record.get("evidence_retrieval_at_optimal_evaluation")
    optimal: int | None
    optimal_source: str | None
    if not coverable_aspect_ids:
        optimal = None
        optimal_source = None
    elif optimal_info is None:
        optimal = _derive_optimal_sentence_budget(
            query_id=expected_query_id,
            coverable_aspect_ids=coverable_aspect_ids,
            sentence_to_aspects=sentence_to_aspects,
        )
        optimal_source = "derived_exact_set_cover"
    else:
        if not isinstance(optimal_info, dict):
            raise EvidenceGapError(
                f"{expected_query_id} evidence_retrieval_at_optimal_evaluation "
                "must be an object when present"
            )
        try:
            optimal = int(optimal_info.get("optimal"))
        except (TypeError, ValueError) as exc:
            raise EvidenceGapError(f"{expected_query_id} has invalid optimal budget") from exc
        if optimal <= 0 or optimal > len(candidates):
            raise EvidenceGapError(
                f"{expected_query_id} optimal budget {optimal} is outside 1..{len(candidates)}"
            )
        optimal_source = "raw_evidence_retrieval_at_optimal_evaluation"

    expected_review = manifest_record.get("systematic_review_id")
    raw_review_value = raw_record.get("systematic_review_id")
    raw_review = None if raw_review_value is None else str(raw_review_value)
    if expected_review is not None and raw_review != str(expected_review):
        raise EvidenceGapError(f"{expected_query_id} systematic_review_id mismatch")

    return EvidenceQuery(
        query_id=expected_query_id,
        contract_version=str(manifest_record.get("contract_version", "1.0.0")),
        split=str(manifest_record["split"]),
        official_split=str(manifest_record["official_split"]),
        hypothesis=hypothesis,
        paper_id=paper_id,
        systematic_review_id=raw_review,
        candidate_sentences=tuple(candidates),
        sentence_types=tuple(sentence_types),
        aspect_ids=tuple(aspect_ids),
        coverable_aspect_ids=coverable_aspect_ids,
        unmapped_aspect_ids=unmapped_aspect_ids,
        aspect_text=aspect_text,
        aspect_to_sentence_indices=aspect_to_indices,
        sentence_to_aspects=sentence_to_aspects,
        results_aspect_ids=None if results_ids is None else tuple(results_ids),
        optimal_sentence_budget=optimal,
        optimal_sentence_budget_source=optimal_source,
        raw_locator=dict(manifest_record["raw_locator"]),
        pool_fingerprint=pool_fingerprint(paper_id, candidates),
    )
