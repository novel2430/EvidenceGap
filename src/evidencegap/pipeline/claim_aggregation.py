from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
)
from evidencegap.pipeline.article_evidence import validate_article_evidence_artifact
from evidencegap.stance.contracts import canonical_stance_label

CLAIM_AGGREGATION_SCHEMA_VERSION = "1.0.0"
CLAIM_AGGREGATION_CONTRACT_ID = "phase07.claim-aggregation.v1"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/pipeline/claim_aggregation")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid name: {value!r}")
    return cleaned


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing article evidence artifact: {path}") from exc

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceGapError(f"Invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(row, Mapping):
            raise EvidenceGapError(f"Expected JSON object at {path}:{line_number}")
        rows.append(dict(row))
    return rows


def _verdict(support_count: int, refute_count: int) -> str:
    if support_count and refute_count:
        return "mixed"
    if support_count:
        return "supported"
    if refute_count:
        return "refuted"
    return "insufficient"


def _rationale(verdict: str) -> str:
    return {
        "supported": (
            "The retrieved direct evidence supports the claim, with no refuting "
            "articles identified."
        ),
        "refuted": (
            "The retrieved direct evidence refutes the claim, with no supporting "
            "articles identified."
        ),
        "mixed": (
            "The retrieved evidence contains both supporting and refuting articles."
        ),
        "insufficient": (
            "The retrieved articles do not provide direct evidence that supports or "
            "refutes the claim."
        ),
    }[verdict]


def aggregate_article_evidence_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    claim_keys: set[tuple[str, str]] = set()
    article_ids: set[str] = set()
    ranks: set[int] = set()

    for index, source in enumerate(rows):
        row = dict(source)
        article_id = str(row.get("article_id") or "").strip()
        claim_id = str(row.get("claim_id") or "").strip()
        claim_text = str(row.get("claim_text") or "").strip()
        if not article_id or not claim_id or not claim_text:
            raise EvidenceGapError(
                f"Article evidence row {index} is missing article_id, claim_id, or claim_text"
            )
        if article_id in article_ids:
            raise EvidenceGapError(f"Duplicate article_id in aggregation input: {article_id}")
        article_ids.add(article_id)
        claim_keys.add((claim_id, claim_text))

        try:
            rank = int(row["final_article_rank"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EvidenceGapError(
                f"Invalid final_article_rank for article {article_id}"
            ) from exc
        if rank <= 0 or rank in ranks:
            raise EvidenceGapError(
                f"Article rank must be positive and unique: {article_id} rank={rank}"
            )
        ranks.add(rank)

        label = canonical_stance_label(str(row.get("predicted_label") or ""))
        evidence = row.get("selected_evidence")
        if not isinstance(evidence, list):
            raise EvidenceGapError(
                f"selected_evidence must be an array for article {article_id}"
            )
        if label == "insufficient" and evidence:
            raise EvidenceGapError(
                f"insufficient article {article_id} cannot contain selected evidence"
            )
        if label != "insufficient" and not evidence:
            raise EvidenceGapError(
                f"{label} article {article_id} must contain selected evidence"
            )

        normalized.append(
            {
                "article_id": article_id,
                "rank": rank,
                "label": label,
            }
        )

    if not normalized:
        raise EvidenceGapError("Article evidence aggregation input cannot be empty")
    if len(claim_keys) != 1:
        raise EvidenceGapError("Aggregation input contains multiple claims")

    claim_id, claim_text = next(iter(claim_keys))
    grouped = {
        label: [
            row["article_id"]
            for row in sorted(normalized, key=lambda value: value["rank"])
            if row["label"] == label
        ]
        for label in ("support", "refute", "insufficient")
    }
    verdict = _verdict(len(grouped["support"]), len(grouped["refute"]))

    return {
        "schema_version": CLAIM_AGGREGATION_SCHEMA_VERSION,
        "contract_id": CLAIM_AGGREGATION_CONTRACT_ID,
        "claim_id": claim_id,
        "claim_text": claim_text,
        "verdict": verdict,
        "article_counts": {
            "total": len(normalized),
            "support": len(grouped["support"]),
            "refute": len(grouped["refute"]),
            "insufficient": len(grouped["insufficient"]),
        },
        "support_article_ids": grouped["support"],
        "refute_article_ids": grouped["refute"],
        "insufficient_article_ids": grouped["insufficient"],
        "rationale": _rationale(verdict),
        "scope": "retrieved_top_articles",
    }


def _validation_summary(result: Mapping[str, Any]) -> dict[str, Any]:
    counts = result["article_counts"]
    return {
        "verdict": result["verdict"],
        "articles": counts["total"],
        "support_articles": counts["support"],
        "refute_articles": counts["refute"],
        "insufficient_articles": counts["insufficient"],
    }


def run_claim_aggregation(
    root: Path,
    *,
    article_evidence_artifact_dir: Path,
    run_name: str | None = None,
    artifact_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    source_dir = article_evidence_artifact_dir.resolve()
    validate_article_evidence_artifact(source_dir)

    source_path = source_dir / "article_evidence.jsonl"
    source_manifest_path = source_dir / "run_manifest.json"
    rows = _read_jsonl(source_path)
    result = aggregate_article_evidence_rows(rows)
    validation = _validation_summary(result)

    name = _safe_name(run_name or source_dir.name)
    target = (
        artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    ) / name
    with atomic_directory(target, force=force) as staging:
        result_path = staging / "claim_result.json"
        atomic_write_json(result_path, result)
        atomic_write_json(
            staging / "run_manifest.json",
            {
                "schema_version": CLAIM_AGGREGATION_SCHEMA_VERSION,
                "contract_id": CLAIM_AGGREGATION_CONTRACT_ID,
                "run_type": "phase07_claim_aggregation",
                "run_name": name,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "rule": {
                    "support_and_refute": "mixed",
                    "support_only": "supported",
                    "refute_only": "refuted",
                    "neither": "insufficient",
                    "uses_confidence_weighting": False,
                },
                "source": {
                    "article_evidence_artifact_dir": relative_path(root, source_dir),
                    "article_evidence": {
                        "path": relative_path(root, source_path),
                        "sha256": sha256_file(source_path),
                    },
                    "run_manifest": {
                        "path": relative_path(root, source_manifest_path),
                        "sha256": sha256_file(source_manifest_path),
                    },
                },
                "output": {
                    "claim_result": {
                        "path": relative_path(root, target / result_path.name),
                        "sha256": sha256_file(result_path),
                    }
                },
                "validation": validation,
            },
        )

    return {
        "status": "PASS",
        "run_name": name,
        "artifact_dir": relative_path(root, target),
        **validation,
        "claim_result_path": relative_path(root, target / "claim_result.json"),
    }


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    while current.parent != current:
        if (current / "src/evidencegap").exists():
            return current
        current = current.parent
    return start.resolve()


def validate_claim_aggregation_artifact(artifact_dir: Path) -> dict[str, Any]:
    artifact_dir = artifact_dir.resolve()
    try:
        manifest = json.loads(
            (artifact_dir / "run_manifest.json").read_text(encoding="utf-8")
        )
        result = json.loads(
            (artifact_dir / "claim_result.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise EvidenceGapError(
            f"Missing claim aggregation artifact in {artifact_dir}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid claim aggregation JSON in {artifact_dir}") from exc

    if manifest.get("schema_version") != CLAIM_AGGREGATION_SCHEMA_VERSION:
        raise EvidenceGapError("Unexpected claim aggregation manifest schema_version")
    if manifest.get("contract_id") != CLAIM_AGGREGATION_CONTRACT_ID:
        raise EvidenceGapError("Unexpected claim aggregation manifest contract_id")

    root = _find_repo_root(artifact_dir)
    output_meta = manifest["output"]["claim_result"]
    result_path = _resolve(root, str(output_meta["path"]))
    if sha256_file(result_path) != str(output_meta["sha256"]):
        raise EvidenceGapError("Claim result checksum mismatch")

    source_meta = manifest["source"]
    source_path = _resolve(root, str(source_meta["article_evidence"]["path"]))
    source_manifest_path = _resolve(root, str(source_meta["run_manifest"]["path"]))
    if sha256_file(source_path) != str(source_meta["article_evidence"]["sha256"]):
        raise EvidenceGapError("Source article evidence checksum mismatch")
    if sha256_file(source_manifest_path) != str(source_meta["run_manifest"]["sha256"]):
        raise EvidenceGapError("Source article evidence manifest checksum mismatch")

    source_dir = _resolve(root, str(source_meta["article_evidence_artifact_dir"]))
    validate_article_evidence_artifact(source_dir)
    expected = aggregate_article_evidence_rows(_read_jsonl(source_path))
    if result != expected:
        raise EvidenceGapError("Claim result does not match source article evidence")

    return {
        "status": "PASS",
        "run_name": manifest.get("run_name"),
        **_validation_summary(result),
        "checksums": "PASS",
    }
