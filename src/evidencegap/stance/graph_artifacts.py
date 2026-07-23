from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from evidencegap.common import EvidenceGapError


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing pyarrow. Install requirements/v1-phase06.txt"
        ) from exc
    return pa, pq


def _write_rows_atomic(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    schema: Any,
) -> int:
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


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    count = 0
    with temp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    os.replace(temp, path)
    return count

def _summary_schema() -> Any:
    pa, _pq = _pyarrow()
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("contract_id", pa.string(), nullable=False),
            pa.field("record_type", pa.string(), nullable=False),
            pa.field("graph_id", pa.string(), nullable=False),
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("claim_id", pa.string(), nullable=False),
            pa.field("query_id", pa.string(), nullable=False),
            pa.field("claim_text", pa.string(), nullable=False),
            pa.field("paper_id", pa.string(), nullable=True),
            pa.field("paper_count", pa.int32(), nullable=False),
            pa.field("evidence_count", pa.int32(), nullable=False),
            pa.field("support_count", pa.int32(), nullable=False),
            pa.field("refute_count", pa.int32(), nullable=False),
            pa.field("insufficient_count", pa.int32(), nullable=False),
            pa.field("support_mass", pa.float64(), nullable=False),
            pa.field("refute_mass", pa.float64(), nullable=False),
            pa.field("insufficient_mass", pa.float64(), nullable=False),
            pa.field("mass_leader", pa.string(), nullable=False),
            pa.field("directional_margin", pa.float64(), nullable=False),
            pa.field("directional_mass_share", pa.float64(), nullable=False),
            pa.field("directional_evidence_pattern", pa.string(), nullable=False),
            pa.field("has_conflict", pa.bool_(), nullable=False),
            pa.field("requires_context_count", pa.int32(), nullable=False),
            pa.field("direct_result_count", pa.int32(), nullable=False),
            pa.field("background_count", pa.int32(), nullable=False),
            pa.field("method_count", pa.int32(), nullable=False),
            pa.field("top_support_input_id", pa.string(), nullable=True),
            pa.field("top_refute_input_id", pa.string(), nullable=True),
            pa.field("top_insufficient_input_id", pa.string(), nullable=True),
            pa.field("source_prediction_run", pa.string(), nullable=False),
            pa.field("source_model_name", pa.string(), nullable=False),
            pa.field("source_model_fingerprint", pa.string(), nullable=False),
            pa.field("source_prompt_version", pa.string(), nullable=True),
        ]
    )


def _node_schema() -> Any:
    pa, _pq = _pyarrow()
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("contract_id", pa.string(), nullable=False),
            pa.field("record_type", pa.string(), nullable=False),
            pa.field("graph_id", pa.string(), nullable=False),
            pa.field("node_id", pa.string(), nullable=False),
            pa.field("node_type", pa.string(), nullable=False),
            pa.field("label", pa.string(), nullable=False),
            pa.field("text", pa.string(), nullable=True),
            pa.field("claim_id", pa.string(), nullable=False),
            pa.field("query_id", pa.string(), nullable=False),
            pa.field("paper_id", pa.string(), nullable=True),
            pa.field("input_id", pa.string(), nullable=True),
            pa.field("sentence_index", pa.int32(), nullable=True),
            pa.field("evidence_rank", pa.int32(), nullable=True),
            pa.field("stance_label", pa.string(), nullable=True),
            pa.field("evidence_type", pa.string(), nullable=True),
            pa.field("confidence", pa.float32(), nullable=True),
            pa.field("retrieval_score", pa.float32(), nullable=True),
            pa.field("rank_weight", pa.float64(), nullable=True),
            pa.field("stance_mass", pa.float64(), nullable=True),
            pa.field("requires_context", pa.bool_(), nullable=True),
            pa.field("metadata_json", pa.string(), nullable=False),
        ]
    )


def _edge_schema() -> Any:
    pa, _pq = _pyarrow()
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("contract_id", pa.string(), nullable=False),
            pa.field("record_type", pa.string(), nullable=False),
            pa.field("graph_id", pa.string(), nullable=False),
            pa.field("edge_id", pa.string(), nullable=False),
            pa.field("source_node_id", pa.string(), nullable=False),
            pa.field("target_node_id", pa.string(), nullable=False),
            pa.field("relation", pa.string(), nullable=False),
            pa.field("claim_id", pa.string(), nullable=False),
            pa.field("query_id", pa.string(), nullable=False),
            pa.field("paper_id", pa.string(), nullable=False),
            pa.field("input_id", pa.string(), nullable=True),
            pa.field("sentence_index", pa.int32(), nullable=True),
            pa.field("evidence_rank", pa.int32(), nullable=True),
            pa.field("retrieval_model", pa.string(), nullable=True),
            pa.field("retrieval_score", pa.float32(), nullable=True),
            pa.field("stance_label", pa.string(), nullable=True),
            pa.field("stance_probability", pa.float32(), nullable=True),
            pa.field("rank_weight", pa.float64(), nullable=True),
            pa.field("stance_mass", pa.float64(), nullable=True),
            pa.field("model_name", pa.string(), nullable=False),
            pa.field("model_fingerprint", pa.string(), nullable=False),
            pa.field("source_reference_json", pa.string(), nullable=False),
        ]
    )




def write_summary_rows_atomic(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> int:
    return _write_rows_atomic(path, rows, _summary_schema())


def write_node_rows_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    return _write_rows_atomic(path, rows, _node_schema())


def write_edge_rows_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    return _write_rows_atomic(path, rows, _edge_schema())
