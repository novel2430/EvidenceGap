from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evidencegap_backend.agent.schemas import EvidenceWorkspace
from evidencegap_backend.common import (
    EvidenceGapError,
    atomic_write_json,
    load_json,
    relative_path,
    sha256_file,
)
from evidencegap_backend.config import PipelineConfig
from evidencegap_backend.pipeline.statement_analysis import (
    STATEMENT_ANALYSIS_CONTRACT_ID,
    STATEMENT_ANALYSIS_SCHEMA_VERSION,
    validate_statement_analysis_artifact,
    validate_statement_analysis_bundle,
)
from evidencegap_backend.pipeline.statement_decomposition import (
    validate_decomposition_bundle,
)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def materialize_statement_analysis(
    root: Path,
    *,
    workspace: EvidenceWorkspace,
    decomposition_artifact_dir: Path,
    artifact_dir: Path,
    pipeline_config: PipelineConfig,
    provider: str,
    model: str | None,
    device: str,
    amp: str,
    request_batch_size: int,
    max_tokens: int,
    thinking: bool,
    runtime_resources_present: bool,
) -> dict[str, Any]:
    """Materialize the selected immutable attempts into the standard contract.

    Only the canonical statement-analysis JSON files are overwritten. The
    ``agent_attempts`` subtree remains immutable across evidence cycles.
    """

    root = root.resolve()
    artifact_dir = artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    decomposition = dict(workspace.decomposition)
    decomposition_validation = validate_decomposition_bundle(decomposition)
    decomposition_dir = decomposition_artifact_dir.resolve()
    decomposition_path = decomposition_dir / "decomposition.json"

    claim_results: list[dict[str, Any]] = []
    graph_bundles: dict[str, dict[str, Any]] = {}
    for claim_id in workspace.claim_order:
        claim = workspace.claims[claim_id]
        selected = next(
            (
                attempt
                for attempt in claim.attempts
                if attempt.attempt_id == claim.selected_attempt_id
                and attempt.status == "successful"
            ),
            None,
        )
        base = {
            "claim_id": claim_id,
            "source_text": claim.source_text,
            "source_spans": claim.source_spans,
            "canonical_claim_en": claim.canonical_claim_en,
            "agent_status": claim.status,
        }
        if selected is None:
            claim_results.append(
                {
                    **base,
                    "status": "failed",
                    "phase07_artifact_dir": None,
                    "graph_bundle_path": None,
                    "verdict": None,
                    "error": claim.last_error
                    or claim.terminal_reason
                    or "No successful evidence attempt",
                }
            )
            continue
        if not selected.artifact_dir or not selected.graph_bundle_path:
            raise EvidenceGapError(
                f"Selected attempt artifact is incomplete for {claim_id}"
            )
        graph_path = _resolve(root, selected.graph_bundle_path)
        graph_bundle = load_json(graph_path)
        if not isinstance(graph_bundle, dict):
            raise EvidenceGapError(f"Selected graph is invalid for {claim_id}")
        graph_bundles[claim_id] = graph_bundle
        claim_results.append(
            {
                **base,
                "status": "completed",
                "phase07_artifact_dir": selected.artifact_dir,
                "graph_bundle_path": selected.graph_bundle_path,
                "verdict": selected.verdict,
                "error": None,
                "selected_attempt_id": selected.attempt_id,
            }
        )

    completed = sum(row["status"] == "completed" for row in claim_results)
    failed = len(claim_results) - completed
    analysis_status = (
        "completed"
        if not failed
        else ("failed" if not completed else "partial_failure")
    )
    bundle = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "statement_id": decomposition_validation["statement_id"],
        "original_statement": decomposition["original_statement"],
        "source_language": decomposition["source_language"],
        "analysis_status": analysis_status,
        "analysis_context": pipeline_config.analysis_context(),
        "claim_results": claim_results,
        "summary": {
            "total_claims": len(claim_results),
            "completed_claims": completed,
            "failed_claims": failed,
        },
    }
    validation = validate_statement_analysis_bundle(bundle)
    result_path = artifact_dir / "statement_result.json"
    atomic_write_json(result_path, bundle)
    request = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "run_name": artifact_dir.name,
        "statement_id": decomposition_validation["statement_id"],
        "decomposition_artifact_dir": relative_path(root, decomposition_dir),
        "decomposition_path": relative_path(root, decomposition_path),
        "decomposition_sha256": sha256_file(decomposition_path),
        "evidence_cycle": workspace.evidence_cycle,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(artifact_dir / "request.json", request)
    manifest = {
        "schema_version": STATEMENT_ANALYSIS_SCHEMA_VERSION,
        "contract_id": STATEMENT_ANALYSIS_CONTRACT_ID,
        "run_type": "langgraph_agent_multi_claim_analysis",
        "run_name": artifact_dir.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "statement_id": decomposition_validation["statement_id"],
        "analysis_status": analysis_status,
        "execution": {
            "provider": provider,
            "model": model,
            "device": device,
            "amp": amp,
            "request_batch_size": request_batch_size,
            "max_tokens": max_tokens,
            "thinking": thinking if provider == "deepseek" else None,
            "resource_lifecycle": (
                "engine_resident" if runtime_resources_present else "per_call"
            ),
            "decomposition_handoff": "workspace",
            "orchestration": "langgraph",
            "evidence_cycle": workspace.evidence_cycle,
        },
        "counts": dict(bundle["summary"]),
        "source": {
            "decomposition_artifact_dir": relative_path(root, decomposition_dir),
            "decomposition": {
                "path": relative_path(root, decomposition_path),
                "sha256": sha256_file(decomposition_path),
            },
        },
        "outputs": {
            "statement_result": {
                "path": relative_path(root, result_path),
                "sha256": sha256_file(result_path),
            }
        },
        "seconds": round(time.perf_counter() - started, 6),
    }
    atomic_write_json(artifact_dir / "run_manifest.json", manifest)
    validate_statement_analysis_artifact(artifact_dir)
    return {
        **validation,
        "status": analysis_status.upper(),
        "artifact_status": "PASS",
        "run_name": artifact_dir.name,
        "artifact_dir": relative_path(root, artifact_dir),
        "statement_result_path": relative_path(root, result_path),
        "statement_result": bundle,
        "decomposition": decomposition,
        "claim_graph_bundles": graph_bundles,
    }


def gap_round_record(
    root: Path,
    *,
    agent_dir: Path,
    workspace: EvidenceWorkspace,
    statement_analysis_dir: Path,
    statement_bundle_dir: Path,
    gap_dir: Path,
    gap_decision: Mapping[str, Any],
) -> dict[str, Any]:
    rounds_dir = agent_dir / "gap_rounds"
    rounds_dir.mkdir(parents=True, exist_ok=True)
    gap_summary = dict(workspace.latest_gap_summary or {})
    gap_summary_path = (
        rounds_dir / f"round_{workspace.gap_round:03d}_gap_summary.json"
    )
    atomic_write_json(gap_summary_path, gap_summary)
    record = {
        "schema_version": "1.0.0",
        "evidence_cycle": workspace.evidence_cycle,
        "gap_round": workspace.gap_round,
        "selected_attempts": {
            claim_id: workspace.claims[claim_id].selected_attempt_id
            for claim_id in workspace.claim_order
        },
        "artifacts": {
            "statement_analysis": {
                "path": relative_path(root, statement_analysis_dir),
                "sha256": sha256_file(
                    statement_analysis_dir / "statement_result.json"
                ),
            },
            "statement_bundle": {
                "path": relative_path(root, statement_bundle_dir),
                "sha256": sha256_file(
                    statement_bundle_dir / "statement_bundle.json"
                ),
            },
            "inference_gap_analysis": {
                "path": relative_path(root, gap_dir),
                "sha256": sha256_file(
                    gap_dir / "inference_gap_analysis.json"
                ),
            },
        },
        "gap_summary": gap_summary,
        "gap_summary_artifact": {
            "path": relative_path(root, gap_summary_path),
            "sha256": sha256_file(gap_summary_path),
        },
        "gap_decision": dict(gap_decision),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = rounds_dir / f"round_{workspace.gap_round:03d}.json"
    atomic_write_json(path, record)
    return record
