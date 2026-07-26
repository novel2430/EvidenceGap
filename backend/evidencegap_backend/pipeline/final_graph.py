from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
    sha256_text,
    find_workspace_root,
)
from evidencegap_backend.pipeline.article_evidence import (
    validate_article_evidence_artifact,
    validate_retrieval_trace,
)
from evidencegap_backend.pipeline.claim_aggregation import (
    aggregate_article_evidence_rows,
    validate_claim_aggregation_artifact,
)

FINAL_GRAPH_SCHEMA_VERSION = "2.1.0"
FINAL_GRAPH_CONTRACT_ID = "phase07.final-graph.v2"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/final_graph")

ARTICLE_RELATIONS = {
    "support": "article_supports",
    "refute": "article_refutes",
    "insufficient": "article_insufficient",
}
def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid name: {value!r}")
    return cleaned


def _token(value: str) -> str:
    return sha256_text(value)[:20]


def _node_id(node_type: str, identity: str) -> str:
    return f"{node_type}:{_token(identity)}"


def _edge_id(relation: str, source_id: str, target_id: str) -> str:
    return f"edge:{relation}:{_token(source_id + chr(0) + target_id)}"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(value, Mapping):
        raise EvidenceGapError(f"Expected JSON object: {path}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing JSONL artifact: {path}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceGapError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, Mapping):
            raise EvidenceGapError(f"Expected JSON object at {path}:{line_number}")
        rows.append(dict(value))
    return rows


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _find_repo_root(start: Path) -> Path:
    return find_workspace_root(start)


def _edge(
    *,
    relation: str,
    source_node_id: str,
    target_node_id: str,
    claim_id: str,
    article_id: str | None = None,
    evidence_id: str | None = None,
    stance: str | None = None,
) -> dict[str, Any]:
    return {
        "edge_id": _edge_id(relation, source_node_id, target_node_id),
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "relation": relation,
        "claim_id": claim_id,
        "article_id": article_id,
        "evidence_id": evidence_id,
        "stance": stance,
    }


def build_final_graph_bundle(
    article_rows: Sequence[Mapping[str, Any]],
    claim_result: Mapping[str, Any],
) -> dict[str, Any]:
    rows = [dict(value) for value in article_rows]
    expected_result = aggregate_article_evidence_rows(rows)
    if dict(claim_result) != expected_result:
        raise EvidenceGapError(
            "Claim aggregation result does not match article evidence input"
        )

    claim_id = str(expected_result["claim_id"])
    claim_text = str(expected_result["claim_text"])
    verdict = str(expected_result["verdict"])
    graph_id = f"graph:{_token(claim_id)}"
    claim_node_id = _node_id("claim", claim_id)
    nodes: list[dict[str, Any]] = [
        {
            "node_id": claim_node_id,
            "node_type": "claim",
            "label": "Claim",
            "text": claim_text,
            "claim_id": claim_id,
        }
    ]
    edges: list[dict[str, Any]] = []

    seen_evidence_nodes: set[str] = set()
    for row in sorted(rows, key=lambda value: int(value["final_article_rank"])):
        article_id = str(row["article_id"])
        stance = str(row["predicted_label"])
        article_node_id = _node_id("article", claim_id + chr(0) + article_id)
        selected_evidence = [dict(value) for value in row.get("selected_evidence") or []]
        retrieval_trace = row.get("retrieval_trace")
        if not isinstance(retrieval_trace, Mapping):
            raise EvidenceGapError(f"Missing retrieval trace for {article_id}")
        validate_retrieval_trace(retrieval_trace)
        nodes.append(
            {
                "node_id": article_node_id,
                "node_type": "article",
                "label": str(row.get("title") or article_id),
                "text": str(row.get("rationale") or ""),
                "claim_id": claim_id,
                "article_id": article_id,
                "pmid": None if row.get("pmid") is None else str(row["pmid"]),
                "final_article_rank": int(row["final_article_rank"]),
                "retrieval_trace": dict(retrieval_trace),
                "stance": stance,
                "confidence": float(row.get("confidence") or 0.0),
                "probabilities": dict(row.get("probabilities") or {}),
                "selected_evidence_count": len(selected_evidence),
                "provider": row.get("provider"),
                "model": row.get("model"),
                "model_fingerprint": row.get("model_fingerprint"),
                "prompt_version": row.get("prompt_version"),
            }
        )
        edges.append(
            _edge(
                relation=ARTICLE_RELATIONS[stance],
                source_node_id=article_node_id,
                target_node_id=claim_node_id,
                claim_id=claim_id,
                article_id=article_id,
                stance=stance,
            )
        )

        for selected in selected_evidence:
            sentence_id = str(selected.get("sentence_id") or "").strip()
            if not sentence_id:
                raise EvidenceGapError(
                    f"Selected evidence in {article_id} is missing sentence_id"
                )
            source_evidence_id = str(selected.get("evidence_id") or "").strip() or None
            evidence_identity = source_evidence_id or sentence_id
            evidence_node_id = _node_id("evidence", evidence_identity)
            if evidence_node_id in seen_evidence_nodes:
                raise EvidenceGapError(
                    f"Duplicate evidence node identity in final graph: {evidence_identity}"
                )
            seen_evidence_nodes.add(evidence_node_id)
            nodes.append(
                {
                    "node_id": evidence_node_id,
                    "node_type": "evidence",
                    "label": str(selected.get("sentence_alias") or sentence_id),
                    "text": str(selected.get("sentence_text") or ""),
                    "claim_id": claim_id,
                    "article_id": article_id,
                    "pmid": None if row.get("pmid") is None else str(row["pmid"]),
                    "evidence_id": source_evidence_id,
                    "sentence_id": sentence_id,
                    "sentence_index": int(selected.get("sentence_index", -1)),
                    "sentence_index_within_section": int(
                        selected.get("sentence_index_within_section", -1)
                    ),
                    "section": selected.get("section"),
                    "section_index": int(selected.get("section_index", -1)),
                    "character_start": int(selected.get("character_start", -1)),
                    "character_end": int(selected.get("character_end", -1)),
                    "source_text_fingerprint": selected.get(
                        "source_text_fingerprint"
                    ),
                    "splitter_fingerprint": selected.get("splitter_fingerprint"),
                }
            )
            edges.append(
                _edge(
                    relation="contains_evidence",
                    source_node_id=article_node_id,
                    target_node_id=evidence_node_id,
                    claim_id=claim_id,
                    article_id=article_id,
                    evidence_id=source_evidence_id,
                )
            )

    return {
        "schema_version": FINAL_GRAPH_SCHEMA_VERSION,
        "contract_id": FINAL_GRAPH_CONTRACT_ID,
        "graph_id": graph_id,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "verdict": verdict,
        "summary": dict(expected_result),
        "nodes": nodes,
        "edges": edges,
        "boundary": {
            "is_pipeline_final_verdict": True,
            "is_final_medical_truth": False,
            "description": (
                "The verdict summarizes the retrieved Top Articles and their grounded "
                "article-level evidence. It is not a systematic review or a clinical "
                "recommendation."
            ),
        },
    }


def _validation_summary(bundle: Mapping[str, Any]) -> dict[str, Any]:
    node_counts: dict[str, int] = {}
    for node in bundle["nodes"]:
        node_type = str(node["node_type"])
        node_counts[node_type] = node_counts.get(node_type, 0) + 1
    relation_counts: dict[str, int] = {}
    for edge in bundle["edges"]:
        relation = str(edge["relation"])
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
    return {
        "verdict": bundle["verdict"],
        "nodes": len(bundle["nodes"]),
        "edges": len(bundle["edges"]),
        "node_counts": dict(sorted(node_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
    }


def _load_sources_from_aggregation(
    root: Path, claim_aggregation_artifact_dir: Path
) -> tuple[Path, Path, Path, dict[str, Any], list[dict[str, Any]]]:
    aggregation_manifest_path = claim_aggregation_artifact_dir / "run_manifest.json"
    claim_result_path = claim_aggregation_artifact_dir / "claim_result.json"
    aggregation_manifest = _read_json(aggregation_manifest_path)
    claim_result = _read_json(claim_result_path)
    try:
        source_dir_value = str(
            aggregation_manifest["source"]["article_evidence_artifact_dir"]
        )
    except (KeyError, TypeError) as exc:
        raise EvidenceGapError(
            "Claim aggregation manifest has no article evidence source directory"
        ) from exc
    article_evidence_dir = _resolve(root, source_dir_value).resolve()
    article_evidence_path = article_evidence_dir / "article_evidence.jsonl"
    article_rows = _read_jsonl(article_evidence_path)
    return (
        aggregation_manifest_path,
        claim_result_path,
        article_evidence_dir,
        claim_result,
        article_rows,
    )


def run_final_graph(
    root: Path,
    *,
    claim_aggregation_artifact_dir: Path,
    run_name: str | None = None,
    artifact_root: Path | None = None,
    claim_result: Mapping[str, Any] | None = None,
    article_rows: Sequence[Mapping[str, Any]] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    aggregation_dir = claim_aggregation_artifact_dir.resolve()
    if (claim_result is None) != (article_rows is None):
        raise EvidenceGapError(
            "claim_result and article_rows must be provided together"
        )
    if claim_result is None or article_rows is None:
        validate_claim_aggregation_artifact(aggregation_dir)
        (
            aggregation_manifest_path,
            claim_result_path,
            article_evidence_dir,
            loaded_claim_result,
            loaded_article_rows,
        ) = _load_sources_from_aggregation(root, aggregation_dir)
        validate_article_evidence_artifact(article_evidence_dir)
        claim_result_value = loaded_claim_result
        article_rows_value = loaded_article_rows
        source_handoff = "artifact_reload"
    else:
        aggregation_manifest_path = aggregation_dir / "run_manifest.json"
        claim_result_path = aggregation_dir / "claim_result.json"
        aggregation_manifest = _read_json(aggregation_manifest_path)
        source_dir_value = str(
            aggregation_manifest["source"]["article_evidence_artifact_dir"]
        )
        article_evidence_dir = _resolve(root, source_dir_value).resolve()
        claim_result_value = dict(claim_result)
        article_rows_value = [dict(row) for row in article_rows]
        source_handoff = "in_memory_handoff"

    article_evidence_path = article_evidence_dir / "article_evidence.jsonl"
    article_evidence_manifest_path = article_evidence_dir / "run_manifest.json"
    bundle = build_final_graph_bundle(article_rows_value, claim_result_value)
    validation = _validation_summary(bundle)

    name = _safe_name(run_name or aggregation_dir.name)
    target = (
        artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    ) / name
    with atomic_directory(target, force=force) as staging:
        bundle_path = staging / "graph_bundle.json"
        atomic_write_json(bundle_path, bundle)
        atomic_write_json(
            staging / "run_manifest.json",
            {
                "schema_version": FINAL_GRAPH_SCHEMA_VERSION,
                "contract_id": FINAL_GRAPH_CONTRACT_ID,
                "run_type": "phase07_final_graph",
                "run_name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_handoff": source_handoff,
                "source": {
                    "claim_aggregation_artifact_dir": relative_path(
                        root, aggregation_dir
                    ),
                    "claim_result": {
                        "path": relative_path(root, claim_result_path),
                        "sha256": sha256_file(claim_result_path),
                    },
                    "claim_aggregation_manifest": {
                        "path": relative_path(root, aggregation_manifest_path),
                        "sha256": sha256_file(aggregation_manifest_path),
                    },
                    "article_evidence_artifact_dir": relative_path(
                        root, article_evidence_dir
                    ),
                    "article_evidence": {
                        "path": relative_path(root, article_evidence_path),
                        "sha256": sha256_file(article_evidence_path),
                    },
                    "article_evidence_manifest": {
                        "path": relative_path(root, article_evidence_manifest_path),
                        "sha256": sha256_file(article_evidence_manifest_path),
                    },
                },
                "output": {
                    "graph_bundle": {
                        "path": relative_path(root, target / bundle_path.name),
                        "sha256": sha256_file(bundle_path),
                    }
                },
                "validation": validation,
            },
        )

    return {
        "status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "graph_bundle_path": relative_path(root, target / "graph_bundle.json"),
        **validation,
        "graph_bundle": bundle,
    }


def validate_final_graph_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest = _read_json(artifact_dir / "run_manifest.json")
    if manifest.get("schema_version") != FINAL_GRAPH_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected final graph manifest schema_version")
    if manifest.get("contract_id") != FINAL_GRAPH_CONTRACT_ID:
        raise EvidenceGapError("Unexpected final graph manifest contract_id")

    root = _find_repo_root(artifact_dir)
    output_meta = manifest["output"]["graph_bundle"]
    bundle_path = _resolve(root, str(output_meta["path"]))
    if sha256_file(bundle_path) != str(output_meta["sha256"]):
        raise EvidenceGapError("Final graph bundle checksum mismatch")
    bundle = _read_json(bundle_path)

    source = manifest["source"]
    source_checks = (
        ("claim_result", "Claim result"),
        ("claim_aggregation_manifest", "Claim aggregation manifest"),
        ("article_evidence", "Article evidence"),
        ("article_evidence_manifest", "Article evidence manifest"),
    )
    for key, label in source_checks:
        meta = source[key]
        path = _resolve(root, str(meta["path"]))
        if sha256_file(path) != str(meta["sha256"]):
            raise EvidenceGapError(f"{label} checksum mismatch")

    aggregation_dir = _resolve(
        root, str(source["claim_aggregation_artifact_dir"])
    ).resolve()
    article_evidence_dir = _resolve(
        root, str(source["article_evidence_artifact_dir"])
    ).resolve()
    validate_claim_aggregation_artifact(aggregation_dir)
    validate_article_evidence_artifact(article_evidence_dir)

    claim_result = _read_json(aggregation_dir / "claim_result.json")
    article_rows = _read_jsonl(article_evidence_dir / "article_evidence.jsonl")
    expected = build_final_graph_bundle(article_rows, claim_result)
    if bundle != expected:
        raise EvidenceGapError("Final graph bundle does not match source artifacts")

    return {
        "status": "PASS",
        "run_name": manifest.get("run_name"),
        **_validation_summary(bundle),
        "checksums": "PASS",
    }
