from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
)
from evidencegap.pipeline.final_graph import (
    FINAL_GRAPH_CONTRACT_ID,
    FINAL_GRAPH_SCHEMA_VERSION,
)
from evidencegap.pipeline.statement_analysis import (
    validate_statement_analysis_artifact,
    validate_statement_analysis_bundle,
)
from evidencegap.pipeline.statement_decomposition import (
    runtime_inference_step_id,
    validate_decomposition_bundle,
)

STATEMENT_BUNDLE_SCHEMA_VERSION = "1.1.0"
STATEMENT_BUNDLE_CONTRACT_ID = "phase075.statement-bundle.v1"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/statement_bundle")
_VALID_VERDICTS = {"supported", "refuted", "mixed", "insufficient"}
_VALID_STANCES = {"support", "refute", "insufficient"}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty")
    return cleaned


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing required JSON artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceGapError(f"Expected JSON object in {path}")
    return value


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current.parent != current:
        if (current / "src/evidencegap").exists():
            return current
        current = current.parent
    return start.resolve()


def _merged_evidence_id(claim_id: str, source_node_id: str) -> str:
    return f"{claim_id}:{source_node_id}"


def _flatten_graph(
    claim_id: str, graph: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if (
        graph.get("schema_version") != FINAL_GRAPH_SCHEMA_VERSION
        or graph.get("contract_id") != FINAL_GRAPH_CONTRACT_ID
        or graph.get("claim_id") != claim_id
    ):
        raise EvidenceGapError(f"Invalid Phase 07 final graph for {claim_id}")

    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise EvidenceGapError(f"Final graph collections are invalid for {claim_id}")

    article_nodes = {
        str(node["node_id"]): node
        for node in nodes
        if isinstance(node, Mapping) and node.get("node_type") == "article"
    }
    evidence_nodes = {
        str(node["node_id"]): node
        for node in nodes
        if isinstance(node, Mapping) and node.get("node_type") == "evidence"
    }
    evidence_parent: dict[str, str] = {}
    for edge in edges:
        if not isinstance(edge, Mapping) or edge.get("relation") != "contains_evidence":
            continue
        source = str(edge.get("source_node_id") or "")
        target = str(edge.get("target_node_id") or "")
        if source not in article_nodes or target not in evidence_nodes:
            raise EvidenceGapError("Invalid Article to Evidence edge")
        evidence_parent[target] = source
    if set(evidence_parent) != set(evidence_nodes):
        raise EvidenceGapError("Every Evidence node must belong to one Article")

    evidence_ids_by_article: dict[str, list[str]] = {
        node_id: [] for node_id in article_nodes
    }
    evidence_records: list[dict[str, Any]] = []
    for source_node_id, node in evidence_nodes.items():
        article_node_id = evidence_parent[source_node_id]
        evidence_id = _merged_evidence_id(claim_id, source_node_id)
        evidence_ids_by_article[article_node_id].append(evidence_id)
        evidence_records.append(
            {
                "evidence_id": evidence_id,
                "source_node_id": source_node_id,
                "claim_id": claim_id,
                "article_node_id": article_node_id,
                "article_id": str(node.get("article_id") or ""),
                "pmid": node.get("pmid"),
                "label": str(node.get("label") or ""),
                "text": str(node.get("text") or ""),
                "source_evidence_id": node.get("evidence_id"),
                "sentence_id": str(node.get("sentence_id") or ""),
                "sentence_index": int(node.get("sentence_index", -1)),
                "sentence_index_within_section": int(
                    node.get("sentence_index_within_section", -1)
                ),
                "section": node.get("section"),
                "section_index": int(node.get("section_index", -1)),
                "character_start": int(node.get("character_start", -1)),
                "character_end": int(node.get("character_end", -1)),
                "source_text_fingerprint": node.get("source_text_fingerprint"),
                "splitter_fingerprint": node.get("splitter_fingerprint"),
            }
        )

    article_records = [
        {
            "article_node_id": node_id,
            "claim_id": claim_id,
            "article_id": str(node.get("article_id") or ""),
            "pmid": node.get("pmid"),
            "rank": int(node.get("final_article_rank", -1)),
            "title": str(node.get("label") or ""),
            "rationale": str(node.get("text") or ""),
            "stance": str(node.get("stance") or ""),
            "confidence": float(node.get("confidence") or 0.0),
            "probabilities": dict(node.get("probabilities") or {}),
            "evidence_ids": evidence_ids_by_article[node_id],
            "provider": node.get("provider"),
            "model": node.get("model"),
            "model_fingerprint": node.get("model_fingerprint"),
            "prompt_version": node.get("prompt_version"),
        }
        for node_id, node in sorted(
            article_nodes.items(),
            key=lambda item: int(item[1].get("final_article_rank", -1)),
        )
    ]
    rank_by_article = {
        row["article_node_id"]: row["rank"] for row in article_records
    }
    evidence_records.sort(
        key=lambda row: (rank_by_article[row["article_node_id"]], row["sentence_index"])
    )
    return article_records, evidence_records


def build_statement_bundle(
    decomposition: Mapping[str, Any],
    statement_result: Mapping[str, Any],
    graphs_by_claim: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    validate_decomposition_bundle(decomposition)
    validate_statement_analysis_bundle(statement_result)
    for key in ("statement_id", "original_statement", "source_language"):
        if decomposition.get(key) != statement_result.get(key):
            raise EvidenceGapError(f"Statement source mismatch: {key}")

    source_claims = decomposition.get("claims", [])
    analysis_claims = statement_result.get("claim_results", [])
    if len(source_claims) != len(analysis_claims):
        raise EvidenceGapError("Statement bundle claim count mismatch")

    claims: list[dict[str, Any]] = []
    articles: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for source_claim, result in zip(source_claims, analysis_claims, strict=True):
        for key in ("claim_id", "source_text", "canonical_claim_en"):
            if source_claim.get(key) != result.get(key):
                raise EvidenceGapError(f"Statement bundle claim mismatch: {key}")

        claim_id = str(result["claim_id"])
        status = str(result["status"])
        record: dict[str, Any] = {
            "claim_id": claim_id,
            "source_text": str(result["source_text"]),
            "canonical_claim_en": str(result["canonical_claim_en"]),
            "analysis_status": status,
            "verdict": result.get("verdict"),
            "article_counts": None,
            "rationale": None,
            "scope": None,
            "boundary": None,
            "article_node_ids": [],
            "error": result.get("error"),
        }
        if status == "completed":
            graph = graphs_by_claim.get(claim_id)
            if not isinstance(graph, Mapping):
                raise EvidenceGapError(f"Missing completed final graph for {claim_id}")
            if (
                graph.get("claim_text") != result.get("canonical_claim_en")
                or graph.get("verdict") != result.get("verdict")
            ):
                raise EvidenceGapError(f"Final graph result mismatch for {claim_id}")
            summary = graph.get("summary")
            boundary = graph.get("boundary")
            if not isinstance(summary, Mapping) or not isinstance(boundary, Mapping):
                raise EvidenceGapError(f"Invalid final graph metadata for {claim_id}")
            claim_articles, claim_evidence = _flatten_graph(claim_id, graph)
            record.update(
                {
                    "article_counts": dict(summary.get("article_counts") or {}),
                    "rationale": summary.get("rationale"),
                    "scope": summary.get("scope"),
                    "boundary": dict(boundary),
                    "article_node_ids": [
                        row["article_node_id"] for row in claim_articles
                    ],
                }
            )
            articles.extend(claim_articles)
            evidence.extend(claim_evidence)
        claims.append(record)

    bundle = {
        "schema_version": STATEMENT_BUNDLE_SCHEMA_VERSION,
        "contract_id": STATEMENT_BUNDLE_CONTRACT_ID,
        "statement": {
            "statement_id": str(decomposition["statement_id"]),
            "original_text": str(decomposition["original_statement"]),
            "source_language": str(decomposition["source_language"]),
            "analysis_status": str(statement_result["analysis_status"]),
        },
        "claims": claims,
        "inference_steps": [
            {
                "inference_step_id": str(step["inference_step_id"]),
                "premise_claim_ids": list(step["premise_claim_ids"]),
                "conclusion_claim_id": str(step["conclusion_claim_id"]),
            }
            for step in decomposition.get("inference_steps", [])
        ],
        "articles": articles,
        "evidence": evidence,
        "summary": {
            "total_claims": len(claims),
            "completed_claims": sum(
                row["analysis_status"] == "completed" for row in claims
            ),
            "failed_claims": sum(
                row["analysis_status"] == "failed" for row in claims
            ),
            "articles": len(articles),
            "evidence": len(evidence),
        },
    }
    validate_statement_bundle(bundle)
    return bundle


def validate_statement_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    if (
        bundle.get("schema_version") != STATEMENT_BUNDLE_SCHEMA_VERSION
        or bundle.get("contract_id") != STATEMENT_BUNDLE_CONTRACT_ID
    ):
        raise EvidenceGapError("Unexpected statement bundle contract")
    statement = bundle.get("statement")
    collections = [
        bundle.get("claims"),
        bundle.get("inference_steps"),
        bundle.get("articles"),
        bundle.get("evidence"),
    ]
    if not isinstance(statement, Mapping) or not all(
        isinstance(value, list) for value in collections
    ):
        raise EvidenceGapError("Invalid statement bundle structure")
    if any(
        not str(statement.get(key) or "").strip()
        for key in (
            "statement_id",
            "original_text",
            "source_language",
            "analysis_status",
        )
    ):
        raise EvidenceGapError("Statement bundle source fields cannot be blank")

    claims, steps, articles, evidence = collections
    claim_by_id: dict[str, Mapping[str, Any]] = {}
    for claim in claims:
        if not isinstance(claim, Mapping):
            raise EvidenceGapError("Statement bundle claim must be an object")
        claim_id = str(claim.get("claim_id") or "")
        status = str(claim.get("analysis_status") or "")
        if (
            not claim_id
            or claim_id in claim_by_id
            or status not in {"completed", "failed"}
        ):
            raise EvidenceGapError("Invalid statement bundle claim")
        if status == "completed":
            if (
                claim.get("verdict") not in _VALID_VERDICTS
                or claim.get("error") is not None
            ):
                raise EvidenceGapError("Invalid completed claim result")
        elif (
            claim.get("verdict") is not None
            or claim.get("article_counts") is not None
            or claim.get("article_node_ids")
            or not str(claim.get("error") or "").strip()
        ):
            raise EvidenceGapError("Invalid failed claim result")
        claim_by_id[claim_id] = claim

    inference_step_ids: set[str] = set()
    inference_step_keys: set[tuple[tuple[str, ...], str]] = set()
    for step in steps:
        if not isinstance(step, Mapping):
            raise EvidenceGapError("Invalid inference step")
        premises = step.get("premise_claim_ids")
        conclusion = str(step.get("conclusion_claim_id") or "").strip()
        if (
            not isinstance(premises, list)
            or not premises
            or any(
                not isinstance(value, str) or not value.strip()
                for value in premises
            )
        ):
            raise EvidenceGapError("Invalid inference step references")
        premise_ids = [value.strip() for value in premises]
        if (
            len(premise_ids) != len(set(premise_ids))
            or not set(premise_ids).issubset(claim_by_id)
            or conclusion not in claim_by_id
            or conclusion in premise_ids
        ):
            raise EvidenceGapError("Invalid inference step references")
        inference_step_id = str(step.get("inference_step_id") or "").strip()
        expected_step_id = runtime_inference_step_id(premise_ids, conclusion)
        step_key = (tuple(sorted(premise_ids)), conclusion)
        if inference_step_id != expected_step_id:
            raise EvidenceGapError("Invalid inference step identity")
        if (
            inference_step_id in inference_step_ids
            or step_key in inference_step_keys
        ):
            raise EvidenceGapError("Duplicate inference step")
        inference_step_ids.add(inference_step_id)
        inference_step_keys.add(step_key)

    article_by_id: dict[str, Mapping[str, Any]] = {}
    for article in articles:
        if not isinstance(article, Mapping):
            raise EvidenceGapError("Invalid Article record")
        node_id = str(article.get("article_node_id") or "")
        claim_id = str(article.get("claim_id") or "")
        if (
            not node_id
            or node_id in article_by_id
            or claim_id not in claim_by_id
            or claim_by_id[claim_id].get("analysis_status") != "completed"
            or article.get("stance") not in _VALID_STANCES
            or not isinstance(article.get("evidence_ids"), list)
        ):
            raise EvidenceGapError("Invalid Article identity")
        article_by_id[node_id] = article

    evidence_by_id: dict[str, Mapping[str, Any]] = {}
    for item in evidence:
        if not isinstance(item, Mapping):
            raise EvidenceGapError("Invalid Evidence record")
        evidence_id = str(item.get("evidence_id") or "")
        article_node_id = str(item.get("article_node_id") or "")
        if (
            not evidence_id
            or evidence_id in evidence_by_id
            or "stance" in item
            or article_node_id not in article_by_id
            or item.get("claim_id") != article_by_id[article_node_id].get("claim_id")
        ):
            raise EvidenceGapError("Invalid Evidence identity")
        evidence_by_id[evidence_id] = item

    for article in articles:
        expected = {
            evidence_id
            for evidence_id, item in evidence_by_id.items()
            if item.get("article_node_id") == article.get("article_node_id")
        }
        if set(article["evidence_ids"]) != expected:
            raise EvidenceGapError("Article Evidence references do not match")
    for claim in claims:
        expected = {
            article_id
            for article_id, article in article_by_id.items()
            if article.get("claim_id") == claim.get("claim_id")
        }
        if set(claim.get("article_node_ids") or []) != expected:
            raise EvidenceGapError("Claim Article references do not match")

    summary = bundle.get("summary")
    expected_summary = {
        "total_claims": len(claims),
        "completed_claims": sum(
            claim.get("analysis_status") == "completed" for claim in claims
        ),
        "failed_claims": sum(
            claim.get("analysis_status") == "failed" for claim in claims
        ),
        "articles": len(articles),
        "evidence": len(evidence),
    }
    if not isinstance(summary, Mapping) or any(
        int(summary.get(key, -1)) != value for key, value in expected_summary.items()
    ):
        raise EvidenceGapError("Statement bundle summary count mismatch")
    return {
        "status": "PASS",
        "statement_id": str(statement["statement_id"]),
        "analysis_status": str(statement["analysis_status"]),
        **expected_summary,
        "empty_claims": not claims,
    }


def run_statement_bundle(
    root: Path,
    *,
    statement_analysis_artifact_dir: Path,
    run_name: str,
    artifact_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    analysis_dir = _resolve(root, statement_analysis_artifact_dir)
    validate_statement_analysis_artifact(analysis_dir)
    request = _read_json_object(analysis_dir / "request.json")
    statement_result_path = analysis_dir / "statement_result.json"
    statement_result = _read_json_object(statement_result_path)
    decomposition_path = (
        _resolve(root, str(request.get("decomposition_artifact_dir") or ""))
        / "decomposition.json"
    )
    decomposition = _read_json_object(decomposition_path)

    graphs: dict[str, dict[str, Any]] = {}
    graph_meta: list[dict[str, str]] = []
    for result in statement_result.get("claim_results", []):
        if result.get("status") != "completed":
            continue
        claim_id = str(result["claim_id"])
        graph_path = _resolve(root, str(result.get("graph_bundle_path") or ""))
        graphs[claim_id] = _read_json_object(graph_path)
        graph_meta.append(
            {
                "claim_id": claim_id,
                "path": relative_path(root, graph_path),
                "sha256": sha256_file(graph_path),
            }
        )

    bundle = build_statement_bundle(decomposition, statement_result, graphs)
    validation = validate_statement_bundle(bundle)
    name = _safe_name(run_name)
    target = (
        artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    ) / name
    with atomic_directory(target, force=force) as staging:
        bundle_path = staging / "statement_bundle.json"
        atomic_write_json(bundle_path, bundle)
        atomic_write_json(
            staging / "run_manifest.json",
            {
                "schema_version": STATEMENT_BUNDLE_SCHEMA_VERSION,
                "contract_id": STATEMENT_BUNDLE_CONTRACT_ID,
                "run_type": "phase075_statement_bundle",
                "run_name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": {
                    "statement_analysis_artifact_dir": relative_path(
                        root, analysis_dir
                    ),
                    "statement_result": {
                        "path": relative_path(root, statement_result_path),
                        "sha256": sha256_file(statement_result_path),
                    },
                    "decomposition": {
                        "path": relative_path(root, decomposition_path),
                        "sha256": sha256_file(decomposition_path),
                    },
                    "final_graphs": graph_meta,
                },
                "output": {
                    "statement_bundle": {
                        "path": relative_path(root, target / bundle_path.name),
                        "sha256": sha256_file(bundle_path),
                    }
                },
                "counts": dict(bundle["summary"]),
                "analysis_status": bundle["statement"]["analysis_status"],
            },
        )
    return {
        **validation,
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "statement_bundle_path": relative_path(
            root, target / "statement_bundle.json"
        ),
    }


def validate_statement_bundle_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    manifest = _read_json_object(artifact_dir / "run_manifest.json")
    if (
        manifest.get("schema_version") != STATEMENT_BUNDLE_SCHEMA_VERSION
        or manifest.get("contract_id") != STATEMENT_BUNDLE_CONTRACT_ID
    ):
        raise EvidenceGapError("Unexpected statement bundle manifest contract")
    root = _find_repo_root(artifact_dir)
    source = manifest.get("source")
    output = manifest.get("output")
    if not isinstance(source, Mapping) or not isinstance(output, Mapping):
        raise EvidenceGapError("Invalid statement bundle manifest")

    analysis_dir = _resolve(
        root, str(source.get("statement_analysis_artifact_dir") or "")
    )
    validate_statement_analysis_artifact(analysis_dir)
    file_meta = [
        (source.get("statement_result"), "statement result"),
        (source.get("decomposition"), "decomposition"),
        (output.get("statement_bundle"), "statement bundle"),
    ]
    loaded: list[dict[str, Any]] = []
    for meta, label in file_meta:
        if not isinstance(meta, Mapping):
            raise EvidenceGapError(f"Missing {label} metadata")
        path = _resolve(root, str(meta.get("path") or ""))
        if not path.is_file() or sha256_file(path) != str(meta.get("sha256") or ""):
            raise EvidenceGapError(f"Statement bundle {label} checksum mismatch")
        loaded.append(_read_json_object(path))
    statement_result, decomposition, actual = loaded

    graphs: dict[str, dict[str, Any]] = {}
    graph_meta = source.get("final_graphs")
    if not isinstance(graph_meta, list):
        raise EvidenceGapError("Statement bundle final_graphs must be an array")
    for meta in graph_meta:
        if not isinstance(meta, Mapping):
            raise EvidenceGapError("Invalid final graph metadata")
        claim_id = str(meta.get("claim_id") or "")
        path = _resolve(root, str(meta.get("path") or ""))
        if (
            not claim_id
            or claim_id in graphs
            or not path.is_file()
            or sha256_file(path) != str(meta.get("sha256") or "")
        ):
            raise EvidenceGapError("Invalid final graph source")
        graphs[claim_id] = _read_json_object(path)

    expected = build_statement_bundle(decomposition, statement_result, graphs)
    if actual != expected:
        raise EvidenceGapError("Statement bundle does not match source artifacts")
    validation = validate_statement_bundle(actual)
    if manifest.get("counts") != actual.get("summary"):
        raise EvidenceGapError("Statement bundle manifest count mismatch")
    if manifest.get("analysis_status") != validation["analysis_status"]:
        raise EvidenceGapError("Statement bundle manifest status mismatch")
    return {
        **validation,
        "run_name": manifest.get("run_name"),
        "checksums": "PASS",
    }
