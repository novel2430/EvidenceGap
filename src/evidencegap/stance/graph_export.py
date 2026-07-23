from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    require_empty_or_force,
    sha256_file,
)
from evidencegap.stance.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    RUN_SCHEMA_VERSION,
    iter_prediction_rows,
    validate_prediction_artifact,
)
from evidencegap.stance.graph_artifacts import (
    write_edge_rows_atomic,
    write_jsonl_atomic,
    write_node_rows_atomic,
    write_summary_rows_atomic,
)
from evidencegap.stance.graph_contracts import (
    GRAPH_CONTRACT_ID,
    GRAPH_SCHEMA_VERSION,
    aggregate_prediction_rows,
    build_graph_bundle,
    build_graph_rows,
    validate_graph_rows,
)

GRAPH_RUN_TYPE = "stance_graph_ready_export"
DEFAULT_GRAPH_ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT / "graph_ready"


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)
    cleaned = cleaned.strip("._")
    if not cleaned:
        raise EvidenceGapError("run_name cannot be empty after sanitization")
    return cleaned


def _load_source_manifest(prediction_path: Path) -> dict[str, Any] | None:
    path = prediction_path.parent / "run_manifest.json"
    if not path.exists():
        return None
    value = load_json(path)
    if not isinstance(value, dict):
        raise EvidenceGapError(f"Expected JSON object in {path}")
    return value


def _render_markdown(report: Mapping[str, Any]) -> str:
    patterns = report["query_directional_pattern_counts"]
    outputs = report["outputs"]
    return "\n".join(
        [
            "# Phase 06.7 Graph-ready Evidence Export",
            "",
            f"- Source predictions: `{report['source_prediction_path']}`",
            f"- Source rows: {report['source_rows']}",
            f"- Query graphs: {report['queries']}",
            f"- Paper summaries: {report['papers']}",
            f"- Graph nodes: {report['nodes']}",
            f"- Graph edges: {report['edges']}",
            f"- Source partial: {str(report['source_partial']).lower()}",
            "",
            "## Query directional evidence patterns",
            "",
            "| Pattern | Queries |",
            "|---|---:|",
            f"| support_only | {patterns.get('support_only', 0)} |",
            f"| refute_only | {patterns.get('refute_only', 0)} |",
            f"| mixed | {patterns.get('mixed', 0)} |",
            f"| none | {patterns.get('none', 0)} |",
            "",
            "## Transparent aggregation",
            "",
            "```text",
            "rank_weight = 1 / Phase 05 evidence_rank",
            "stance_mass(label) = sum(rank_weight * LLM_probability(label))",
            "```",
            "",
            (
                "`mass_leader`, `directional_evidence_pattern`, "
                "`directional_margin`, and `directional_mass_share` are "
                "display-oriented evidence summaries, not a final medical verdict. "
                "LLM probabilities "
                "are self-reported and not calibrated."
            ),
            "",
            "## Outputs",
            "",
            *[
                f"- `{value['path']}` (SHA-256 `{value['sha256']}`)"
                for value in outputs.values()
            ],
            "",
            "## Boundary",
            "",
            (
                "This export stops at evidence stance organization. Study-quality "
                "weighting, cross-article aggregation, and the final Verdict node "
                "belong to the downstream Evidence Graph/pipeline phase."
            ),
            "",
        ]
    )


def export_graph_ready_stance(
    root: Path,
    *,
    prediction_path: Path,
    run_name: str | None = None,
    artifact_root: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    prediction_path = prediction_path.resolve()
    source_validation = validate_prediction_artifact(prediction_path)
    rows = list(iter_prediction_rows(prediction_path))
    query_summaries, paper_summaries, groups = aggregate_prediction_rows(rows)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    papers_by_graph: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for paper_summary in paper_summaries:
        papers_by_graph[str(paper_summary["graph_id"])].append(paper_summary)
    summaries_by_graph = {
        str(summary["graph_id"]): summary for summary in query_summaries
    }
    for graph_id, group in groups.items():
        graph_nodes, graph_edges = build_graph_rows(graph_id, group)
        nodes.extend(graph_nodes)
        edges.extend(graph_edges)
        bundles.append(
            build_graph_bundle(
                summaries_by_graph[graph_id],
                papers_by_graph[graph_id],
                graph_nodes,
                graph_edges,
            )
        )
    graph_validation = validate_graph_rows(
        query_summaries, paper_summaries, nodes, edges
    )

    source_manifest = _load_source_manifest(prediction_path)
    source_partial = bool(source_manifest.get("partial")) if source_manifest else False
    name = _safe_name(run_name or f"{prediction_path.parent.name}_graph_ready")
    base = (
        artifact_root.resolve()
        if artifact_root is not None
        else root / DEFAULT_GRAPH_ARTIFACT_ROOT
    )
    target = base / name
    require_empty_or_force(target, force=force)

    with atomic_directory(target, force=force) as staging:
        output_paths = {
            "query_summaries": staging / "query_summaries.parquet",
            "paper_summaries": staging / "paper_summaries.parquet",
            "nodes": staging / "graph_nodes.parquet",
            "edges": staging / "graph_edges.parquet",
            "bundles": staging / "graph_bundles.jsonl",
        }
        query_count = write_summary_rows_atomic(
            output_paths["query_summaries"], query_summaries
        )
        paper_count = write_summary_rows_atomic(
            output_paths["paper_summaries"], paper_summaries
        )
        node_count = write_node_rows_atomic(output_paths["nodes"], nodes)
        edge_count = write_edge_rows_atomic(output_paths["edges"], edges)
        bundle_count = write_jsonl_atomic(output_paths["bundles"], bundles)

        directional_pattern_counts = Counter(
            str(summary["directional_evidence_pattern"])
            for summary in query_summaries
        )
        mass_leader_counts = Counter(
            str(summary["mass_leader"]) for summary in query_summaries
        )
        output_manifest: dict[str, dict[str, str]] = {}
        for key, staging_path in output_paths.items():
            final_path = target / staging_path.name
            output_manifest[key] = {
                "path": relative_path(root, final_path),
                "sha256": sha256_file(staging_path),
            }
        report = {
            "schema_version": RUN_SCHEMA_VERSION,
            "graph_schema_version": GRAPH_SCHEMA_VERSION,
            "contract_id": GRAPH_CONTRACT_ID,
            "run_name": name,
            "run_type": GRAPH_RUN_TYPE,
            "source_prediction_path": relative_path(root, prediction_path),
            "source_prediction_sha256": source_validation["sha256"],
            "source_prediction_run": source_validation["run_name"],
            "source_model_name": source_validation["model_name"],
            "source_model_fingerprint": source_validation["model_fingerprint"],
            "source_partial": source_partial,
            "source_manifest_path": (
                relative_path(root, prediction_path.parent / "run_manifest.json")
                if source_manifest is not None
                else None
            ),
            "source_coverage": (
                source_manifest.get("coverage") if source_manifest is not None else None
            ),
            "source_rows": len(rows),
            "queries": query_count,
            "papers": paper_count,
            "nodes": node_count,
            "edges": edge_count,
            "bundles": bundle_count,
            "query_directional_pattern_counts": dict(
                sorted(directional_pattern_counts.items())
            ),
            "query_mass_leader_counts": dict(sorted(mass_leader_counts.items())),
            "aggregation": {
                "rank_weight": "1 / evidence_rank",
                "stance_mass": "sum(rank_weight * stance_probability)",
                "directional_margin": (
                    "abs(support_mass - refute_mass) / "
                    "(support_mass + refute_mass)"
                ),
                "directional_mass_share": (
                    "(support_mass + refute_mass) / "
                    "(support_mass + refute_mass + insufficient_mass)"
                ),
                "probability_calibrated": False,
                "is_final_medical_verdict": False,
            },
            "validation": graph_validation,
            "outputs": output_manifest,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "run_manifest.json", report)

    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_root.mkdir(parents=True, exist_ok=True)
    json_report_path = report_root / f"stance_graph_{name}.json"
    markdown_report_path = report_root / f"stance_graph_{name}.md"
    atomic_write_json(json_report_path, report)
    markdown_report_path.write_text(_render_markdown(report), encoding="utf-8")
    return {
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        "manifest_path": relative_path(root, target / "run_manifest.json"),
        "report_path": relative_path(root, json_report_path),
        "markdown_report_path": relative_path(root, markdown_report_path),
        "source_rows": len(rows),
        "queries": query_count,
        "papers": paper_count,
        "nodes": node_count,
        "edges": edge_count,
        "bundles": bundle_count,
        "query_directional_pattern_counts": report[
            "query_directional_pattern_counts"
        ],
        "source_partial": source_partial,
        "validation": graph_validation,
        "api_requests": 0,
    }
