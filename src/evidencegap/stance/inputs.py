from __future__ import annotations

import csv
import json
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    relative_path,
    sha256_file,
    sha256_text,
)
from evidencegap.sentence_retrieval.artifacts import read_rows_by_query
from evidencegap.sentence_retrieval.evidencebench import load_canonical_queries
from evidencegap.stance.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    RUN_SCHEMA_VERSION,
    validate_input_artifact,
    write_inputs_atomic,
)
from evidencegap.stance.contracts import SCHEMA_VERSION, TASK_ID, StanceInput

DEFAULT_PHASE05_FROZEN_CONFIG = Path(
    "configs/v1/phase05_evidence_sentence_retrieval_frozen.json"
)
DEFAULT_HEALTHFC_MANIFEST = Path("data/processed/v1/manifests/healthfc_eval.jsonl")
DEFAULT_HEALTHFC_RAW = Path("data/raw/v1/healthfc/Datensatz.csv")


def _normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def _text_hash(value: Any, length: int = 16) -> str:
    return sha256_text(_normalize_text(value))[:length]


def _parse_int_label(value: Any, *, field: str) -> int:
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError as exc:
        raise EvidenceGapError(f"Invalid integer label for {field}: {value!r}") from exc
    if not number.is_integer():
        raise EvidenceGapError(f"Non-integer label for {field}: {value!r}")
    return int(number)


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing JSONL file: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvidenceGapError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise EvidenceGapError(f"JSONL record is not an object: {path}:{line_number}")
            yield value


def _safe_name(value: str) -> str:
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid run name: {value!r}")
    return cleaned


def _phase05_defaults(root: Path, split: str) -> tuple[Path, Path]:
    config_path = root / DEFAULT_PHASE05_FROZEN_CONFIG
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing Phase 05 frozen config: {config_path}") from exc
    split_config = config.get(split)
    if not isinstance(split_config, dict) or not split_config.get("ranking_path"):
        raise EvidenceGapError(f"Phase 05 frozen config has no {split} ranking")
    ranking_path = (root / str(split_config["ranking_path"])).resolve()
    canonical_dir = (
        root / "artifacts/v1/evidence_sentence_retrieval/canonical" / f"{split}_full"
    ).resolve()
    return canonical_dir, ranking_path


def _adjacent_context(
    sentences: Sequence[str],
    *,
    sentence_index: int,
    context_window: int,
) -> tuple[str | None, str | None, list[int], list[int]]:
    """Return exact neighboring canonical sentences without altering the target."""

    if context_window < 0:
        raise EvidenceGapError("context_window cannot be negative")
    if sentence_index < 0 or sentence_index >= len(sentences):
        raise EvidenceGapError(f"sentence_index out of range: {sentence_index}")
    if context_window == 0:
        return None, None, [], []
    before_indices = list(
        range(max(0, sentence_index - context_window), sentence_index)
    )
    after_indices = list(
        range(
            sentence_index + 1,
            min(len(sentences), sentence_index + context_window + 1),
        )
    )
    before = "\n".join(sentences[index] for index in before_indices) or None
    after = "\n".join(sentences[index] for index in after_indices) or None
    return before, after, before_indices, after_indices

def prepare_phase05_stance_inputs(
    root: Path,
    *,
    split: str = "dev",
    canonical_dir: Path | None = None,
    ranking_path: Path | None = None,
    top_k: int = 5,
    context_window: int = 1,
    run_name: str | None = None,
    artifact_root: Path | None = None,
    allow_test: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    if split not in {"dev", "test"}:
        raise EvidenceGapError("Phase 05 stance inputs currently support dev or test")
    if split == "test" and not allow_test:
        raise EvidenceGapError(
            "Phase 06 development must not consume the Phase 05 test artifact. "
            "Pass --allow-test only for the final frozen evaluation."
        )
    if top_k <= 0:
        raise EvidenceGapError("top_k must be positive")
    if context_window < 0:
        raise EvidenceGapError("context_window cannot be negative")
    if canonical_dir is None or ranking_path is None:
        default_canonical, default_ranking = _phase05_defaults(root, split)
        canonical_dir = canonical_dir or default_canonical
        ranking_path = ranking_path or default_ranking
    canonical_dir = canonical_dir.resolve()
    ranking_path = ranking_path.resolve()
    queries, canonical_manifest = load_canonical_queries(canonical_dir)
    if canonical_manifest.get("split") != split:
        raise EvidenceGapError("Canonical EvidenceBench split does not match requested split")
    ranking_sha = sha256_file(ranking_path)
    source_manifest_path = ranking_path.parent / "run_manifest.json"
    source_manifest: dict[str, Any] = {}
    if source_manifest_path.exists():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        expected_sha = source_manifest.get("output_sha256")
        if expected_sha is not None and str(expected_sha) != ranking_sha:
            raise EvidenceGapError("Phase 05 ranking checksum differs from its run manifest")
        if source_manifest.get("split") not in {None, split}:
            raise EvidenceGapError("Phase 05 ranking manifest split mismatch")
    rows_by_query = read_rows_by_query(ranking_path)
    query_by_id = {query.query_id: query for query in queries}
    missing = sorted(set(query_by_id) - set(rows_by_query))
    extra = sorted(set(rows_by_query) - set(query_by_id))
    if missing or extra:
        raise EvidenceGapError(
            "Phase 05 ranking/canonical query mismatch: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    source_run_name = str(source_manifest.get("run_name") or "")
    if not source_run_name:
        first_rows = next(iter(rows_by_query.values()), [])
        source_run_name = str(first_rows[0]["run_name"]) if first_rows else "phase05"
    name = _safe_name(
        run_name
        or f"phase05_{source_run_name}_{split}_top{top_k}_ctx{context_window}"
    )
    base = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT / "inputs"
    target = base / name

    label_counts: Counter[str] = Counter()
    expected_rows = 0

    def records() -> Iterator[StanceInput]:
        nonlocal expected_rows
        for query in queries:
            ordered = sorted(
                rows_by_query[query.query_id], key=lambda row: int(row["final_rank"])
            )
            ranks = [int(row["final_rank"]) for row in ordered]
            if ranks != list(range(1, len(ordered) + 1)):
                raise EvidenceGapError(
                    f"Non-contiguous Phase 05 final ranks for {query.query_id}"
                )
            indices = [int(row["sentence_index"]) for row in ordered]
            if len(indices) != len(set(indices)):
                raise EvidenceGapError(
                    f"Duplicate Phase 05 sentence indices for {query.query_id}"
                )
            for row in ordered:
                if str(row.get("query_id")) != query.query_id:
                    raise EvidenceGapError(f"Query mismatch for {query.query_id}")
                if str(row.get("split")) != split:
                    raise EvidenceGapError(f"Split mismatch for {query.query_id}")
                if str(row.get("run_name")) != source_run_name:
                    raise EvidenceGapError(f"Mixed run names for {query.query_id}")
            selected = ordered[: min(top_k, len(ordered))]
            if not selected:
                raise EvidenceGapError(f"No Phase 05 evidence rows for {query.query_id}")
            for row in selected:
                index = int(row["sentence_index"])
                rank = int(row["final_rank"])
                if str(row["paper_id"]) != query.paper_id:
                    raise EvidenceGapError(f"Paper mismatch for {query.query_id}")
                if str(row["pool_fingerprint"]) != query.pool_fingerprint:
                    raise EvidenceGapError(f"Candidate pool mismatch for {query.query_id}")
                if index < 0 or index >= len(query.candidate_sentences):
                    raise EvidenceGapError(f"Invalid sentence index for {query.query_id}: {index}")
                if str(row["sentence_text"]) != query.candidate_sentences[index]:
                    raise EvidenceGapError(
                        f"Phase 05 changed sentence text for {query.query_id}:{index}"
                    )
                if str(row["sentence_type"]) != query.sentence_types[index]:
                    raise EvidenceGapError(
                        f"Phase 05 changed sentence type for {query.query_id}:{index}"
                    )
                score_value = row.get("final_score")
                if score_value is None:
                    score_value = row.get("retrieval_score")
                (
                    context_before,
                    context_after,
                    context_before_indices,
                    context_after_indices,
                ) = _adjacent_context(
                    query.candidate_sentences,
                    sentence_index=index,
                    context_window=context_window,
                )
                expected_rows += 1
                label_counts["unlabeled"] += 1
                yield StanceInput(
                    input_id=f"stance:phase05:{query.query_id}:{index}",
                    dataset="evidencebench_100k",
                    split=split,
                    claim_id=query.query_id,
                    query_id=query.query_id,
                    claim_text=query.hypothesis,
                    paper_id=query.paper_id,
                    sentence_index=index,
                    sentence_type=query.sentence_types[index],
                    evidence_rank=rank,
                    evidence_text=query.candidate_sentences[index],
                    evidence_unit="sentence",
                    context_before=context_before,
                    context_after=context_after,
                    retrieval_model=str(row["retrieval_model"]),
                    retrieval_score=None if score_value is None else float(score_value),
                    cross_encoder_score=(
                        None
                        if row.get("cross_encoder_score") is None
                        else float(row["cross_encoder_score"])
                    ),
                    source_run_name=source_run_name,
                    source_artifact_sha256=ranking_sha,
                    gold_label=None,
                    source_locator={
                        "canonical_dir": relative_path(root, canonical_dir),
                        "ranking_path": relative_path(root, ranking_path),
                        "query_id": query.query_id,
                        "paper_id": query.paper_id,
                        "sentence_index": index,
                        "phase05_final_rank": rank,
                        "context_window": context_window,
                        "context_before_indices": context_before_indices,
                        "context_after_indices": context_after_indices,
                    },
                )

    with atomic_directory(target, force=force) as staging:
        output_path = staging / "stance_inputs.parquet"
        written = write_inputs_atomic(output_path, records())
        if written != expected_rows:
            raise EvidenceGapError("Stance input row count changed during write")
        validation = validate_input_artifact(output_path)
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "stance_schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_name": name,
            "run_type": "phase05_ranked_sentence_stance_inputs",
            "split": split,
            "parameters": {
                "top_k": top_k,
                "context_window": context_window,
                "allow_test": allow_test,
            },
            "canonical_dir": relative_path(root, canonical_dir),
            "canonical_sha256": canonical_manifest.get("canonical_sha256"),
            "source_ranking_path": relative_path(root, ranking_path),
            "source_ranking_sha256": ranking_sha,
            "source_ranking_manifest": (
                relative_path(root, source_manifest_path)
                if source_manifest_path.exists()
                else None
            ),
            "source_run_name": source_run_name,
            "rows": written,
            "queries": len(queries),
            "label_counts": dict(label_counts),
            "output_path": relative_path(root, target / "stance_inputs.parquet"),
            "output_sha256": validation["sha256"],
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "run_manifest.json", manifest)
    return manifest


def prepare_healthfc_stance_inputs(
    root: Path,
    *,
    manifest_path: Path | None = None,
    raw_path: Path | None = None,
    run_name: str = "healthfc_eval",
    artifact_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = (manifest_path or (root / DEFAULT_HEALTHFC_MANIFEST)).resolve()
    raw_path = (raw_path or (root / DEFAULT_HEALTHFC_RAW)).resolve()
    manifest_rows = list(_iter_jsonl(manifest_path))
    if not manifest_rows:
        raise EvidenceGapError(f"Empty HealthFC manifest: {manifest_path}")
    with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    name = _safe_name(run_name)
    base = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT / "inputs"
    target = base / name
    label_map = {
        0: "support",
        1: "insufficient",
        2: "refute",
    }
    label_counts: Counter[str] = Counter()
    manifest_sha = sha256_file(manifest_path)

    def records() -> Iterator[StanceInput]:
        for record in manifest_rows:
            if record.get("dataset") != "healthfc" or record.get("split") != "eval":
                raise EvidenceGapError("Unexpected record in HealthFC evaluation manifest")
            locator = record.get("raw_locator")
            if not isinstance(locator, dict) or locator.get("row_index") is None:
                raise EvidenceGapError("HealthFC manifest record has invalid raw_locator")
            row_index = int(locator["row_index"])
            if row_index < 0 or row_index >= len(raw_rows):
                raise EvidenceGapError(f"HealthFC raw row index out of range: {row_index}")
            raw = raw_rows[row_index]
            claim = _normalize_text(raw.get("en_claim"))
            evidence = _normalize_text(raw.get("en_top_sentences"))
            if not claim or not evidence:
                raise EvidenceGapError(f"Empty HealthFC text at row {row_index}")
            if _text_hash(claim) != str(record.get("claim_text_hash")):
                raise EvidenceGapError(f"HealthFC claim hash mismatch at row {row_index}")
            if _text_hash(evidence) != str(record.get("evidence_text_hash")):
                raise EvidenceGapError(f"HealthFC evidence hash mismatch at row {row_index}")
            label_id = _parse_int_label(record["label"], field="HealthFC manifest")
            raw_label_id = _parse_int_label(
                raw.get("label", ""), field=f"HealthFC raw row {row_index}"
            )
            if raw_label_id != label_id:
                raise EvidenceGapError(
                    f"HealthFC label mismatch at row {row_index}: "
                    f"manifest={label_id}, raw={raw_label_id}"
                )
            try:
                label = label_map[label_id]
            except KeyError as exc:
                raise EvidenceGapError(f"Unknown HealthFC label: {label_id}") from exc
            case_id = str(record["case_id"])
            label_counts[label] += 1
            yield StanceInput(
                input_id=f"stance:{case_id}",
                dataset="healthfc",
                split="eval",
                claim_id=case_id,
                query_id=None,
                claim_text=claim,
                paper_id=None,
                sentence_index=None,
                sentence_type=None,
                evidence_rank=None,
                evidence_text=evidence,
                evidence_unit="bundle",
                source_run_name="healthfc_gold_evidence",
                source_artifact_sha256=manifest_sha,
                gold_label=label,
                source_locator={
                    "manifest_path": relative_path(root, manifest_path),
                    "raw_path": relative_path(root, raw_path),
                    "row_index": row_index,
                    "source_url": record.get("source_url"),
                },
            )

    with atomic_directory(target, force=force) as staging:
        output_path = staging / "stance_inputs.parquet"
        written = write_inputs_atomic(output_path, records())
        validation = validate_input_artifact(output_path)
        if written != len(manifest_rows):
            raise EvidenceGapError("HealthFC stance input count differs from manifest")
        manifest = {
            "schema_version": RUN_SCHEMA_VERSION,
            "stance_schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_name": name,
            "run_type": "healthfc_expert_bundle_stance_inputs",
            "split": "eval",
            "source_manifest_path": relative_path(root, manifest_path),
            "source_manifest_sha256": manifest_sha,
            "source_raw_path": relative_path(root, raw_path),
            "source_raw_sha256": sha256_file(raw_path),
            "rows": written,
            "label_counts": dict(sorted(label_counts.items())),
            "output_path": relative_path(root, target / "stance_inputs.parquet"),
            "output_sha256": validation["sha256"],
            "validation": validation,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "run_manifest.json", manifest)
    return manifest
