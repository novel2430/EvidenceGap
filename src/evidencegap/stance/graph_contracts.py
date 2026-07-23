from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

from evidencegap.common import EvidenceGapError, sha256_text
from evidencegap.stance.contracts import EVIDENCE_TYPES, STANCE_LABELS

GRAPH_SCHEMA_VERSION = "1.1.0"
GRAPH_CONTRACT_ID = "phase06.graph-ready.v1.1"
DIRECTIONAL_EVIDENCE_PATTERNS = (
    "support_only",
    "refute_only",
    "mixed",
    "none",
)
STANCE_RELATIONS = {
    "support": "supports",
    "refute": "refutes",
    "insufficient": "insufficient",
}

def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty after sanitization")
    return cleaned


def _node_token(value: str) -> str:
    return sha256_text(value)[:20]


def _claim_node_id(graph_id: str) -> str:
    return f"claim:{_node_token(graph_id)}"


def _article_node_id(graph_id: str, paper_id: str) -> str:
    return f"article:{_node_token(graph_id + chr(0) + paper_id)}"


def _evidence_node_id(input_id: str) -> str:
    return f"evidence:{_node_token(input_id)}"


def _edge_id(relation: str, source_id: str, target_id: str) -> str:
    return f"edge:{relation}:{_node_token(source_id + chr(0) + target_id)}"


def _query_id(row: Mapping[str, Any]) -> str:
    value = row.get("query_id") or row.get("claim_id")
    text = str(value or "").strip()
    if not text:
        raise EvidenceGapError(
            f"Prediction {row.get('input_id')!r} has neither query_id nor claim_id"
        )
    return text


def _paper_id(row: Mapping[str, Any]) -> str:
    value = str(row.get("paper_id") or "").strip()
    if not value:
        raise EvidenceGapError(
            f"Prediction {row.get('input_id')!r} has no paper_id; "
            "Phase 06.7 graph export requires sentence evidence linked to a paper"
        )
    return value


def _rank(row: Mapping[str, Any]) -> int:
    value = row.get("evidence_rank")
    if value is None:
        raise EvidenceGapError(
            f"Prediction {row.get('input_id')!r} has no evidence_rank"
        )
    rank = int(value)
    if rank <= 0:
        raise EvidenceGapError(
            f"Prediction {row.get('input_id')!r} has invalid evidence_rank {rank}"
        )
    return rank


def _probabilities(row: Mapping[str, Any]) -> dict[str, float]:
    result = {
        "support": float(row["probability_support"]),
        "refute": float(row["probability_refute"]),
        "insufficient": float(row["probability_insufficient"]),
    }
    if any(not math.isfinite(value) for value in result.values()):
        raise EvidenceGapError(
            f"Prediction {row.get('input_id')!r} has non-finite probabilities"
        )
    return result


def _rank_weight(row: Mapping[str, Any]) -> float:
    """Transparent display weight: reciprocal Phase 05 rank."""

    return 1.0 / float(_rank(row))


def _stance_mass(row: Mapping[str, Any], label: str | None = None) -> float:
    probabilities = _probabilities(row)
    selected = label or str(row["predicted_label"])
    return _rank_weight(row) * probabilities[selected]


def _directional_pattern(counts: Mapping[str, int]) -> str:
    has_support = int(counts.get("support", 0)) > 0
    has_refute = int(counts.get("refute", 0)) > 0
    if has_support and has_refute:
        return "mixed"
    if has_support:
        return "support_only"
    if has_refute:
        return "refute_only"
    return "none"


def _mass_leader(masses: Mapping[str, float]) -> str:
    # STANCE_LABELS provides a stable tie order.
    return max(STANCE_LABELS, key=lambda label: (float(masses[label]), -STANCE_LABELS.index(label)))


def _top_input(rows: Sequence[Mapping[str, Any]], label: str) -> str | None:
    candidates = [row for row in rows if row["predicted_label"] == label]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (
            _stance_mass(row),
            -_rank(row),
            str(row["input_id"]),
        ),
    )
    return str(best["input_id"])


def _assert_group_consistency(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_id: str,
) -> None:
    if not rows:
        raise EvidenceGapError(f"Empty graph group: {graph_id}")
    claim_ids = {str(row["claim_id"]) for row in rows}
    claim_texts = {str(row["claim_text"]) for row in rows}
    datasets = {str(row["dataset"]) for row in rows}
    splits = {str(row["split"]) for row in rows}
    if len(claim_ids) != 1 or len(claim_texts) != 1:
        raise EvidenceGapError(f"Query {graph_id} mixes claim identity or text")
    if len(datasets) != 1 or len(splits) != 1:
        raise EvidenceGapError(f"Query {graph_id} mixes dataset or split")

    ranks_by_paper: dict[str, list[int]] = defaultdict(list)
    indices_by_paper: dict[str, set[int]] = defaultdict(set)
    for row in rows:
        if str(row.get("evidence_unit")) != "sentence":
            raise EvidenceGapError(
                f"Prediction {row.get('input_id')!r} is not sentence evidence"
            )
        paper_id = _paper_id(row)
        rank = _rank(row)
        ranks_by_paper[paper_id].append(rank)
        sentence_index = int(row["sentence_index"])
        if sentence_index in indices_by_paper[paper_id]:
            raise EvidenceGapError(
                f"Query {graph_id}, paper {paper_id} repeats sentence_index {sentence_index}"
            )
        indices_by_paper[paper_id].add(sentence_index)

    for paper_id, ranks in ranks_by_paper.items():
        ordered = sorted(ranks)
        expected = list(range(1, len(ordered) + 1))
        if ordered != expected:
            raise EvidenceGapError(
                f"Query {graph_id}, paper {paper_id} has rank gaps: {ordered}"
            )


def _summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    graph_id: str,
    paper_id: str | None,
) -> dict[str, Any]:
    counts = Counter(str(row["predicted_label"]) for row in rows)
    evidence_type_counts = Counter(str(row.get("evidence_type") or "") for row in rows)
    masses = {
        label: sum(_rank_weight(row) * _probabilities(row)[label] for row in rows)
        for label in STANCE_LABELS
    }
    directional_total = masses["support"] + masses["refute"]
    directional_margin = (
        abs(masses["support"] - masses["refute"]) / directional_total
        if directional_total > 0.0
        else 0.0
    )
    total_mass = directional_total + masses["insufficient"]
    directional_mass_share = (
        directional_total / total_mass if total_mass > 0.0 else 0.0
    )
    first = rows[0]
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "contract_id": GRAPH_CONTRACT_ID,
        "record_type": "PaperStanceSummary" if paper_id is not None else "QueryStanceSummary",
        "graph_id": graph_id,
        "dataset": str(first["dataset"]),
        "split": str(first["split"]),
        "claim_id": str(first["claim_id"]),
        "query_id": graph_id,
        "claim_text": str(first["claim_text"]),
        "paper_id": paper_id,
        "paper_count": len({_paper_id(row) for row in rows}),
        "evidence_count": len(rows),
        "support_count": counts["support"],
        "refute_count": counts["refute"],
        "insufficient_count": counts["insufficient"],
        "support_mass": masses["support"],
        "refute_mass": masses["refute"],
        "insufficient_mass": masses["insufficient"],
        "mass_leader": _mass_leader(masses),
        "directional_margin": directional_margin,
        "directional_mass_share": directional_mass_share,
        "directional_evidence_pattern": _directional_pattern(counts),
        "has_conflict": counts["support"] > 0 and counts["refute"] > 0,
        "requires_context_count": sum(bool(row.get("requires_context")) for row in rows),
        "direct_result_count": evidence_type_counts["direct_result"],
        "background_count": evidence_type_counts["background"],
        "method_count": evidence_type_counts["method"],
        "top_support_input_id": _top_input(rows, "support"),
        "top_refute_input_id": _top_input(rows, "refute"),
        "top_insufficient_input_id": _top_input(rows, "insufficient"),
        "source_prediction_run": str(first["run_name"]),
        "source_model_name": str(first["model_name"]),
        "source_model_fingerprint": str(first["model_fingerprint"]),
        "source_prompt_version": (
            None if first.get("prompt_version") is None else str(first["prompt_version"])
        ),
    }


def _aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_input_ids: set[str] = set()
    for raw in rows:
        row = dict(raw)
        input_id = str(row.get("input_id") or "").strip()
        if not input_id or input_id in seen_input_ids:
            raise EvidenceGapError(f"Missing or duplicate input_id in graph export: {input_id!r}")
        seen_input_ids.add(input_id)
        label = str(row.get("predicted_label") or "")
        if label not in STANCE_LABELS:
            raise EvidenceGapError(f"Invalid predicted_label {label!r} for {input_id}")
        evidence_type = str(row.get("evidence_type") or "")
        if evidence_type not in EVIDENCE_TYPES:
            raise EvidenceGapError(f"Invalid evidence_type {evidence_type!r} for {input_id}")
        _probabilities(row)
        by_query[_query_id(row)].append(row)

    query_summaries: list[dict[str, Any]] = []
    paper_summaries: list[dict[str, Any]] = []
    ordered_groups: dict[str, list[dict[str, Any]]] = {}
    for graph_id in sorted(by_query):
        group = sorted(
            by_query[graph_id],
            key=lambda row: (_paper_id(row), _rank(row), int(row["sentence_index"])),
        )
        _assert_group_consistency(group, graph_id=graph_id)
        ordered_groups[graph_id] = group
        query_summaries.append(_summary(group, graph_id=graph_id, paper_id=None))
        paper_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            paper_groups[_paper_id(row)].append(row)
        for paper_id in sorted(paper_groups):
            paper_summaries.append(
                _summary(paper_groups[paper_id], graph_id=graph_id, paper_id=paper_id)
            )
    return query_summaries, paper_summaries, ordered_groups


def _node_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    locator = row.get("source_locator_json")
    source_locator: Any = None
    if locator:
        try:
            source_locator = json.loads(str(locator))
        except json.JSONDecodeError:
            source_locator = str(locator)
    return {
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "sentence_type": row.get("sentence_type"),
        "context_before": row.get("context_before"),
        "context_after": row.get("context_after"),
        "rationale": row.get("rationale"),
        "probability_margin": row.get("probability_margin"),
        "provider": row.get("provider"),
        "provider_request_id": row.get("provider_request_id"),
        "prompt_version": row.get("prompt_version"),
        "source_locator": source_locator,
    }


def _build_graph_rows(
    graph_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _assert_group_consistency(rows, graph_id=graph_id)
    first = rows[0]
    claim_node = _claim_node_id(graph_id)
    nodes: list[dict[str, Any]] = [
        {
            "schema_version": GRAPH_SCHEMA_VERSION,
            "contract_id": GRAPH_CONTRACT_ID,
            "record_type": "GraphNode",
            "graph_id": graph_id,
            "node_id": claim_node,
            "node_type": "claim",
            "label": "Medical Claim",
            "text": str(first["claim_text"]),
            "claim_id": str(first["claim_id"]),
            "query_id": graph_id,
            "paper_id": None,
            "input_id": None,
            "sentence_index": None,
            "evidence_rank": None,
            "stance_label": None,
            "evidence_type": None,
            "confidence": None,
            "retrieval_score": None,
            "rank_weight": None,
            "stance_mass": None,
            "requires_context": None,
            "metadata_json": json.dumps(
                {"dataset": first["dataset"], "split": first["split"]},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
    ]
    edges: list[dict[str, Any]] = []
    article_nodes: set[str] = set()
    for row in rows:
        paper_id = _paper_id(row)
        article_node = _article_node_id(graph_id, paper_id)
        if article_node not in article_nodes:
            article_nodes.add(article_node)
            nodes.append(
                {
                    "schema_version": GRAPH_SCHEMA_VERSION,
                    "contract_id": GRAPH_CONTRACT_ID,
                    "record_type": "GraphNode",
                    "graph_id": graph_id,
                    "node_id": article_node,
                    "node_type": "article",
                    "label": paper_id,
                    "text": None,
                    "claim_id": str(row["claim_id"]),
                    "query_id": graph_id,
                    "paper_id": paper_id,
                    "input_id": None,
                    "sentence_index": None,
                    "evidence_rank": None,
                    "stance_label": None,
                    "evidence_type": None,
                    "confidence": None,
                    "retrieval_score": None,
                    "rank_weight": None,
                    "stance_mass": None,
                    "requires_context": None,
                    "metadata_json": json.dumps(
                        {
                            "source_run_name": row.get("source_run_name"),
                            "source_artifact_sha256": row.get("source_artifact_sha256"),
                            "article_relevance_score_available": False,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
            edges.append(
                _edge_row(
                    graph_id=graph_id,
                    relation="retrieved_from",
                    source_node_id=claim_node,
                    target_node_id=article_node,
                    row=row,
                    stance_label=None,
                    stance_probability=None,
                    rank_weight=None,
                    stance_mass=None,
                )
            )

        evidence_node = _evidence_node_id(str(row["input_id"]))
        label = str(row["predicted_label"])
        rank_weight = _rank_weight(row)
        stance_probability = _probabilities(row)[label]
        stance_mass = rank_weight * stance_probability
        nodes.append(
            {
                "schema_version": GRAPH_SCHEMA_VERSION,
                "contract_id": GRAPH_CONTRACT_ID,
                "record_type": "GraphNode",
                "graph_id": graph_id,
                "node_id": evidence_node,
                "node_type": "evidence",
                "label": f"Evidence #{_rank(row)}",
                "text": str(row["evidence_text"]),
                "claim_id": str(row["claim_id"]),
                "query_id": graph_id,
                "paper_id": paper_id,
                "input_id": str(row["input_id"]),
                "sentence_index": int(row["sentence_index"]),
                "evidence_rank": _rank(row),
                "stance_label": label,
                "evidence_type": str(row["evidence_type"]),
                "confidence": float(row["confidence"]),
                "retrieval_score": (
                    None if row.get("retrieval_score") is None else float(row["retrieval_score"])
                ),
                "rank_weight": rank_weight,
                "stance_mass": stance_mass,
                "requires_context": bool(row.get("requires_context")),
                "metadata_json": json.dumps(
                    _node_metadata(row), ensure_ascii=False, sort_keys=True
                ),
            }
        )
        edges.append(
            _edge_row(
                graph_id=graph_id,
                relation="contains",
                source_node_id=article_node,
                target_node_id=evidence_node,
                row=row,
                stance_label=None,
                stance_probability=None,
                rank_weight=None,
                stance_mass=None,
            )
        )
        edges.append(
            _edge_row(
                graph_id=graph_id,
                relation=STANCE_RELATIONS[label],
                source_node_id=evidence_node,
                target_node_id=claim_node,
                row=row,
                stance_label=label,
                stance_probability=stance_probability,
                rank_weight=rank_weight,
                stance_mass=stance_mass,
            )
        )
    return nodes, edges


def _edge_row(
    *,
    graph_id: str,
    relation: str,
    source_node_id: str,
    target_node_id: str,
    row: Mapping[str, Any],
    stance_label: str | None,
    stance_probability: float | None,
    rank_weight: float | None,
    stance_mass: float | None,
) -> dict[str, Any]:
    source_reference: dict[str, Any] = {
        "source_run_name": row.get("source_run_name"),
        "source_artifact_sha256": row.get("source_artifact_sha256"),
    }
    if relation != "retrieved_from":
        source_reference.update(
            {
                "input_id": row.get("input_id"),
                "sentence_index": row.get("sentence_index"),
                "source_locator_json": row.get("source_locator_json"),
            }
        )
    if stance_label is not None:
        source_reference.update(
            {
                "stance_prediction_run": row.get("run_name"),
                "stance_input_artifact_sha256": row.get(
                    "stance_input_artifact_sha256"
                ),
                "raw_response_sha256": row.get("raw_response_sha256"),
                "prompt_version": row.get("prompt_version"),
            }
        )
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "contract_id": GRAPH_CONTRACT_ID,
        "record_type": "GraphEdge",
        "graph_id": graph_id,
        "edge_id": _edge_id(relation, source_node_id, target_node_id),
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "relation": relation,
        "claim_id": str(row["claim_id"]),
        "query_id": graph_id,
        "paper_id": _paper_id(row),
        "input_id": None if relation == "retrieved_from" else str(row["input_id"]),
        "sentence_index": (
            None if relation == "retrieved_from" else int(row["sentence_index"])
        ),
        "evidence_rank": (
            None if relation == "retrieved_from" else _rank(row)
        ),
        "retrieval_model": (
            None
            if relation == "retrieved_from" or row.get("retrieval_model") is None
            else str(row["retrieval_model"])
        ),
        "retrieval_score": (
            None
            if relation == "retrieved_from" or row.get("retrieval_score") is None
            else float(row["retrieval_score"])
        ),
        "stance_label": stance_label,
        "stance_probability": stance_probability,
        "rank_weight": rank_weight,
        "stance_mass": stance_mass,
        "model_name": str(row["model_name"]),
        "model_fingerprint": str(row["model_fingerprint"]),
        "source_reference_json": json.dumps(
            source_reference, ensure_ascii=False, sort_keys=True
        ),
    }


def _bundle(
    summary: Mapping[str, Any],
    paper_summaries: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    bundle_nodes: list[dict[str, Any]] = []
    for value in nodes:
        node = dict(value)
        metadata_json = node.pop("metadata_json", None)
        node["metadata"] = (
            None if metadata_json is None else json.loads(str(metadata_json))
        )
        bundle_nodes.append(node)
    bundle_edges: list[dict[str, Any]] = []
    for value in edges:
        edge = dict(value)
        reference_json = edge.pop("source_reference_json", None)
        edge["source_reference"] = (
            None if reference_json is None else json.loads(str(reference_json))
        )
        bundle_edges.append(edge)
    return {
        "schema_version": GRAPH_SCHEMA_VERSION,
        "contract_id": GRAPH_CONTRACT_ID,
        "graph_id": summary["graph_id"],
        "query_id": summary["query_id"],
        "claim_id": summary["claim_id"],
        "claim_text": summary["claim_text"],
        "summary": dict(summary),
        "papers": [dict(value) for value in paper_summaries],
        "nodes": bundle_nodes,
        "edges": bundle_edges,
        "boundary": {
            "is_final_medical_verdict": False,
            "description": (
                "This graph represents retrieved evidence stances. Cross-article "
                "quality weighting and final medical verdict aggregation are downstream tasks."
            ),
        },
    }

def _validate_graph_rows(
    query_summaries: Sequence[Mapping[str, Any]],
    paper_summaries: Sequence[Mapping[str, Any]],
    nodes: Sequence[Mapping[str, Any]],
    edges: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    graph_ids = {str(row["graph_id"]) for row in query_summaries}
    if len(graph_ids) != len(query_summaries):
        raise EvidenceGapError("Duplicate query summary graph_id")

    paper_keys = {
        (str(row["graph_id"]), str(row["paper_id"])) for row in paper_summaries
    }
    if len(paper_keys) != len(paper_summaries):
        raise EvidenceGapError("Duplicate paper summary graph_id/paper_id")
    if {graph_id for graph_id, _paper_id_value in paper_keys} != graph_ids:
        raise EvidenceGapError("Paper summaries do not cover exactly the query graphs")

    node_ids_by_graph: dict[str, set[str]] = defaultdict(set)
    node_types_by_graph: dict[str, Counter[str]] = defaultdict(Counter)
    evidence_node_ids: set[tuple[str, str]] = set()
    article_node_ids: set[tuple[str, str]] = set()
    for node in nodes:
        graph_id = str(node["graph_id"])
        node_id = str(node["node_id"])
        if graph_id not in graph_ids:
            raise EvidenceGapError(f"Node {node_id} references unknown graph {graph_id}")
        if node_id in node_ids_by_graph[graph_id]:
            raise EvidenceGapError(f"Duplicate node_id {node_id} in graph {graph_id}")
        node_ids_by_graph[graph_id].add(node_id)
        node_type = str(node["node_type"])
        node_types_by_graph[graph_id][node_type] += 1
        if node_type == "evidence":
            evidence_node_ids.add((graph_id, node_id))
        elif node_type == "article":
            article_node_ids.add((graph_id, node_id))
    for graph_id in graph_ids:
        if node_types_by_graph[graph_id]["claim"] != 1:
            raise EvidenceGapError(
                f"Graph {graph_id} must contain exactly one claim node"
            )
    if len(article_node_ids) != len(paper_summaries):
        raise EvidenceGapError(
            "Article node count does not match paper summary count"
        )

    edge_ids_by_graph: dict[str, set[str]] = defaultdict(set)
    relation_counts: Counter[str] = Counter()
    contains_by_evidence: Counter[tuple[str, str]] = Counter()
    stance_by_evidence: Counter[tuple[str, str]] = Counter()
    retrieved_by_article: Counter[tuple[str, str]] = Counter()
    for edge in edges:
        graph_id = str(edge["graph_id"])
        edge_id = str(edge["edge_id"])
        if graph_id not in graph_ids:
            raise EvidenceGapError(f"Edge {edge_id} references unknown graph {graph_id}")
        if edge_id in edge_ids_by_graph[graph_id]:
            raise EvidenceGapError(f"Duplicate edge_id {edge_id} in graph {graph_id}")
        edge_ids_by_graph[graph_id].add(edge_id)
        source = str(edge["source_node_id"])
        target = str(edge["target_node_id"])
        if source not in node_ids_by_graph[graph_id]:
            raise EvidenceGapError(f"Missing source node for edge {edge_id}")
        if target not in node_ids_by_graph[graph_id]:
            raise EvidenceGapError(f"Missing target node for edge {edge_id}")
        relation = str(edge["relation"])
        relation_counts[relation] += 1
        if relation == "contains":
            contains_by_evidence[(graph_id, target)] += 1
        elif relation in STANCE_RELATIONS.values():
            stance_by_evidence[(graph_id, source)] += 1
        elif relation == "retrieved_from":
            retrieved_by_article[(graph_id, target)] += 1
        else:
            raise EvidenceGapError(f"Unknown graph relation {relation!r}")

    for key in evidence_node_ids:
        if contains_by_evidence[key] != 1:
            raise EvidenceGapError(
                f"Evidence node {key[1]} must have exactly one contains edge"
            )
        if stance_by_evidence[key] != 1:
            raise EvidenceGapError(
                f"Evidence node {key[1]} must have exactly one stance edge"
            )
    for key in article_node_ids:
        if retrieved_by_article[key] != 1:
            raise EvidenceGapError(
                f"Article node {key[1]} must have exactly one retrieved_from edge"
            )

    return {
        "status": "PASS",
        "schema_version": GRAPH_SCHEMA_VERSION,
        "contract_id": GRAPH_CONTRACT_ID,
        "graphs": len(graph_ids),
        "paper_summaries": len(paper_summaries),
        "nodes": len(nodes),
        "edges": len(edges),
        "evidence_nodes": len(evidence_node_ids),
        "article_nodes": len(article_node_ids),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


aggregate_prediction_rows = _aggregate_rows
build_graph_rows = _build_graph_rows
build_graph_bundle = _bundle
validate_graph_rows = _validate_graph_rows
