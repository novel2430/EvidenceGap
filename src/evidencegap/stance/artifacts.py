from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from evidencegap.common import EvidenceGapError, sha256_file
from evidencegap.stance.contracts import (
    EVIDENCE_TYPES,
    INPUT_RECORD_TYPE,
    PREDICTION_RECORD_TYPE,
    SCHEMA_VERSION,
    STANCE_LABELS,
    TASK_ID,
    StanceInput,
    StancePrediction,
    canonical_stance_label,
)

RUN_SCHEMA_VERSION = "1.0.0"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/stance_verification")
DEFAULT_REPORT_ROOT = Path("reports/v1")


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing pyarrow. Install requirements/v1-phase06.txt"
        ) from exc
    return pa, pq


def _base_fields(pa: Any) -> list[Any]:
    return [
        pa.field("schema_version", pa.string(), nullable=False),
        pa.field("task_id", pa.string(), nullable=False),
        pa.field("record_type", pa.string(), nullable=False),
        pa.field("input_id", pa.string(), nullable=False),
        pa.field("input_fingerprint", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("claim_id", pa.string(), nullable=False),
        pa.field("query_id", pa.string(), nullable=True),
        pa.field("claim_text", pa.string(), nullable=False),
        pa.field("paper_id", pa.string(), nullable=True),
        pa.field("sentence_index", pa.int32(), nullable=True),
        pa.field("sentence_type", pa.string(), nullable=True),
        pa.field("evidence_rank", pa.int32(), nullable=True),
        pa.field("evidence_text", pa.string(), nullable=False),
        pa.field("evidence_unit", pa.string(), nullable=False),
        pa.field("context_before", pa.string(), nullable=True),
        pa.field("context_after", pa.string(), nullable=True),
        pa.field("retrieval_model", pa.string(), nullable=True),
        pa.field("retrieval_score", pa.float32(), nullable=True),
        pa.field("cross_encoder_score", pa.float32(), nullable=True),
        pa.field("source_run_name", pa.string(), nullable=True),
        pa.field("source_artifact_sha256", pa.string(), nullable=True),
        pa.field("gold_label", pa.string(), nullable=True),
        pa.field("source_locator_json", pa.string(), nullable=True),
    ]


def input_schema() -> Any:
    pa, _pq = _pyarrow()
    return pa.schema(_base_fields(pa))


def prediction_schema() -> Any:
    pa, _pq = _pyarrow()
    return pa.schema(
        _base_fields(pa)
        + [
            pa.field("run_name", pa.string(), nullable=False),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("model_fingerprint", pa.string(), nullable=False),
            pa.field("stance_input_artifact_sha256", pa.string(), nullable=False),
            pa.field("predicted_label", pa.string(), nullable=False),
            pa.field("probability_support", pa.float32(), nullable=False),
            pa.field("probability_refute", pa.float32(), nullable=False),
            pa.field("probability_insufficient", pa.float32(), nullable=False),
            pa.field("confidence", pa.float32(), nullable=False),
            pa.field("probability_margin", pa.float32(), nullable=False),
            pa.field("abstained", pa.bool_(), nullable=False),
            pa.field("rationale", pa.string(), nullable=True),
            pa.field("evidence_type", pa.string(), nullable=True),
            pa.field("requires_context", pa.bool_(), nullable=True),
            pa.field("provider", pa.string(), nullable=True),
            pa.field("provider_request_id", pa.string(), nullable=True),
            pa.field("raw_response_sha256", pa.string(), nullable=True),
            pa.field("prompt_version", pa.string(), nullable=True),
        ]
    )


def _write_rows_atomic(path: Path, rows: Iterable[Mapping[str, Any]], schema: Any) -> int:
    pa, pq = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    writer = pq.ParquetWriter(temp, schema, compression="zstd")
    count = 0
    buffer: list[dict[str, Any]] = []
    try:
        for row in rows:
            buffer.append(dict(row))
            if len(buffer) >= 4096:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            count += len(buffer)
    finally:
        writer.close()
    os.replace(temp, path)
    return count


def write_inputs_atomic(path: Path, inputs: Iterable[StanceInput]) -> int:
    return _write_rows_atomic(
        path,
        (stance_input.to_dict() for stance_input in inputs),
        input_schema(),
    )


def write_predictions_atomic(
    path: Path, predictions: Iterable[StancePrediction]
) -> int:
    return _write_rows_atomic(
        path,
        (prediction.to_dict() for prediction in predictions),
        prediction_schema(),
    )


def _iter_rows(path: Path, *, batch_size: int = 4096) -> Iterator[dict[str, Any]]:
    _pa, pq = _pyarrow()
    try:
        parquet = pq.ParquetFile(path)
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing stance parquet: {path}") from exc
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_inputs(path: Path, *, batch_size: int = 4096) -> Iterator[StanceInput]:
    for row in _iter_rows(path, batch_size=batch_size):
        if row.get("schema_version") != SCHEMA_VERSION or row.get("task_id") != TASK_ID:
            raise EvidenceGapError(f"Stance input schema/task mismatch in {path}")
        if row.get("record_type") != INPUT_RECORD_TYPE:
            raise EvidenceGapError(
                f"Expected {INPUT_RECORD_TYPE} in {path}, got {row.get('record_type')!r}"
            )
        yield StanceInput.from_dict(row)


def iter_prediction_rows(
    path: Path, *, batch_size: int = 4096
) -> Iterator[dict[str, Any]]:
    yield from _iter_rows(path, batch_size=batch_size)


def validate_input_artifact(path: Path) -> dict[str, Any]:
    count = 0
    input_ids: set[str] = set()
    datasets: set[str] = set()
    splits: set[str] = set()
    gold_rows = 0
    sentence_rows = 0
    bundle_rows = 0
    for stance_input in iter_inputs(path):
        if stance_input.input_id in input_ids:
            raise EvidenceGapError(f"Duplicate stance input_id: {stance_input.input_id}")
        input_ids.add(stance_input.input_id)
        datasets.add(stance_input.dataset)
        splits.add(stance_input.split)
        gold_rows += stance_input.gold_label is not None
        sentence_rows += stance_input.evidence_unit == "sentence"
        bundle_rows += stance_input.evidence_unit == "bundle"
        count += 1
    if count == 0:
        raise EvidenceGapError(f"Empty stance input artifact: {path}")
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rows": count,
        "unique_input_ids": len(input_ids),
        "datasets": sorted(datasets),
        "splits": sorted(splits),
        "gold_rows": gold_rows,
        "sentence_rows": sentence_rows,
        "bundle_rows": bundle_rows,
        "sha256": sha256_file(path),
    }


def validate_prediction_artifact(
    path: Path,
    *,
    expected_input_sha256: str | None = None,
    expected_run_name: str | None = None,
) -> dict[str, Any]:
    count = 0
    input_ids: set[str] = set()
    prediction_counts = {"support": 0, "refute": 0, "insufficient": 0}
    gold_rows = 0
    confidence_sum = 0.0
    margin_sum = 0.0
    abstained = 0
    input_artifact_hashes: set[str] = set()
    run_names: set[str] = set()
    model_names: set[str] = set()
    model_fingerprints: set[str] = set()
    providers: set[str] = set()
    rationale_rows = 0
    requires_context_rows = 0
    evidence_type_counts = {value: 0 for value in EVIDENCE_TYPES}
    for row in iter_prediction_rows(path):
        if row.get("schema_version") != SCHEMA_VERSION or row.get("task_id") != TASK_ID:
            raise EvidenceGapError(f"Prediction schema/task mismatch in {path}")
        if row.get("record_type") != PREDICTION_RECORD_TYPE:
            raise EvidenceGapError(f"Invalid prediction record_type in {path}")
        stance_input = StanceInput.from_dict(row)
        input_id = stance_input.input_id
        if not input_id or input_id in input_ids:
            raise EvidenceGapError(f"Missing or duplicate prediction input_id: {input_id!r}")
        input_ids.add(input_id)
        run_name = str(row.get("run_name", "")).strip()
        model_name = str(row.get("model_name", "")).strip()
        model_fingerprint = str(row.get("model_fingerprint", "")).strip()
        if not run_name or not model_name or not model_fingerprint:
            raise EvidenceGapError(f"Missing prediction model/run provenance for {input_id}")
        run_names.add(run_name)
        model_names.add(model_name)
        model_fingerprints.add(model_fingerprint)
        if expected_run_name is not None and run_name != expected_run_name:
            raise EvidenceGapError(f"Prediction run_name mismatch for {input_id}")
        label = canonical_stance_label(str(row.get("predicted_label")))
        probabilities = [
            float(row["probability_support"]),
            float(row["probability_refute"]),
            float(row["probability_insufficient"]),
        ]
        if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in probabilities):
            raise EvidenceGapError(f"Invalid probability for {input_id}")
        if abs(sum(probabilities) - 1.0) > 1e-4:
            raise EvidenceGapError(f"Probabilities do not sum to one for {input_id}")
        confidence = float(row["confidence"])
        margin = float(row["probability_margin"])
        if (
            not math.isfinite(confidence)
            or not math.isfinite(margin)
            or margin < 0.0
            or margin > 1.0
        ):
            raise EvidenceGapError(f"Invalid confidence/margin for {input_id}")
        by_label = dict(zip(STANCE_LABELS, probabilities))
        expected_label = max(STANCE_LABELS, key=lambda value: by_label[value])
        if label != expected_label:
            raise EvidenceGapError(f"predicted_label is not probability argmax for {input_id}")
        if abs(confidence - by_label[label]) > 1e-5:
            raise EvidenceGapError(f"confidence does not match prediction for {input_id}")
        ordered_probabilities = sorted(probabilities, reverse=True)
        expected_margin = ordered_probabilities[0] - ordered_probabilities[1]
        if abs(margin - expected_margin) > 1e-5:
            raise EvidenceGapError(f"probability_margin mismatch for {input_id}")
        prediction_counts[label] += 1
        confidence_sum += confidence
        margin_sum += margin
        abstained += bool(row["abstained"])
        gold_rows += row.get("gold_label") is not None
        input_artifact_hash = row.get("stance_input_artifact_sha256")
        if not input_artifact_hash:
            raise EvidenceGapError(f"Missing stance_input_artifact_sha256 for {input_id}")
        input_artifact_hashes.add(str(input_artifact_hash))
        provider = row.get("provider")
        if provider is not None:
            provider_text = str(provider).strip()
            if not provider_text:
                raise EvidenceGapError(f"Blank provider for {input_id}")
            providers.add(provider_text)
            rationale = str(row.get("rationale") or "").strip()
            evidence_type = str(row.get("evidence_type") or "").strip()
            prompt_version = str(row.get("prompt_version") or "").strip()
            raw_response_sha256 = str(row.get("raw_response_sha256") or "").strip()
            if not rationale or not prompt_version or not raw_response_sha256:
                raise EvidenceGapError(f"Missing LLM prediction metadata for {input_id}")
            if evidence_type not in EVIDENCE_TYPES:
                raise EvidenceGapError(
                    f"Invalid evidence_type {evidence_type!r} for {input_id}"
                )
            rationale_rows += 1
            evidence_type_counts[evidence_type] += 1
            requires_context_rows += bool(row.get("requires_context"))
        count += 1
    if count == 0:
        raise EvidenceGapError(f"Empty stance prediction artifact: {path}")
    if len(input_artifact_hashes) != 1:
        raise EvidenceGapError("Prediction artifact mixes multiple stance input artifacts")
    if len(run_names) != 1 or len(model_names) != 1 or len(model_fingerprints) != 1:
        raise EvidenceGapError("Prediction artifact mixes multiple runs or models")
    if expected_input_sha256 is not None and input_artifact_hashes != {expected_input_sha256}:
        raise EvidenceGapError(
            "Prediction stance_input_artifact_sha256 does not match the stance input artifact"
        )
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "task_id": TASK_ID,
        "rows": count,
        "unique_input_ids": len(input_ids),
        "gold_rows": gold_rows,
        "run_name": next(iter(run_names)),
        "model_name": next(iter(model_names)),
        "model_fingerprint": next(iter(model_fingerprints)),
        "stance_input_artifact_sha256": next(iter(input_artifact_hashes)),
        "prediction_counts": prediction_counts,
        "mean_confidence": confidence_sum / count,
        "mean_probability_margin": margin_sum / count,
        "abstained_rows": abstained,
        "providers": sorted(providers),
        "rationale_rows": rationale_rows,
        "requires_context_rows": requires_context_rows,
        "evidence_type_counts": {
            key: value for key, value in evidence_type_counts.items() if value
        },
        "sha256": sha256_file(path),
    }
