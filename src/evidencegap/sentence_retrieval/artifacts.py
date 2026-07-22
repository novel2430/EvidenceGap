from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from evidencegap.common import EvidenceGapError, atomic_write_json, sha256_file

RUN_SCHEMA_VERSION = "1.0.0"
DEFAULT_ARTIFACT_ROOT = Path("artifacts/v1/evidence_sentence_retrieval")
DEFAULT_REPORT_ROOT = Path("reports/v1")


def _pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing pyarrow. Install requirements/v1-phase05.txt"
        ) from exc
    return pa, pq


def safe_run_name(value: str) -> str:
    value = value.strip()
    if not value:
        raise EvidenceGapError("run_name cannot be empty")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not cleaned:
        raise EvidenceGapError(f"Invalid run_name: {value!r}")
    return cleaned


def ranking_schema() -> Any:
    pa, _pq = _pyarrow()
    return pa.schema(
        [
            pa.field("schema_version", pa.string(), nullable=False),
            pa.field("task_id", pa.string(), nullable=False),
            pa.field("split", pa.string(), nullable=False),
            pa.field("run_name", pa.string(), nullable=False),
            pa.field("query_id", pa.string(), nullable=False),
            pa.field("paper_id", pa.string(), nullable=False),
            pa.field("pool_fingerprint", pa.string(), nullable=False),
            pa.field("sentence_index", pa.int32(), nullable=False),
            pa.field("sentence_type", pa.string(), nullable=False),
            pa.field("sentence_text", pa.string(), nullable=False),
            pa.field("retrieval_model", pa.string(), nullable=False),
            pa.field("retrieval_score", pa.float32(), nullable=False),
            pa.field("retrieval_rank", pa.int32(), nullable=False),
            pa.field("cross_encoder_score", pa.float32(), nullable=True),
            pa.field("final_score", pa.float32(), nullable=True),
            pa.field("final_rank", pa.int32(), nullable=False),
        ]
    )


def write_rows_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    _pa, pq = _pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    schema = ranking_schema()
    writer = pq.ParquetWriter(temp, schema, compression="zstd")
    count = 0
    buffer: list[dict[str, Any]] = []
    try:
        for row in rows:
            buffer.append(dict(row))
            if len(buffer) >= 8192:
                writer.write_table(_pa.Table.from_pylist(buffer, schema=schema))
                count += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(_pa.Table.from_pylist(buffer, schema=schema))
            count += len(buffer)
    finally:
        writer.close()
    os.replace(temp, path)
    return count


def iter_ranking_rows(path: Path, *, batch_size: int = 8192) -> Iterator[dict[str, Any]]:
    _pa, pq = _pyarrow()
    try:
        parquet = pq.ParquetFile(path)
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing ranking parquet: {path}") from exc
    for batch in parquet.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def read_rows_by_query(path: Path) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in iter_ranking_rows(path):
        result.setdefault(str(row["query_id"]), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: int(row["final_rank"]))
    return result


def validate_ranking_rows(
    path: Path,
    *,
    expected_queries: Mapping[str, int] | None = None,
    expected_depths: Mapping[str, int] | None = None,
    expected_run_name: str | None = None,
) -> dict[str, Any]:
    by_query = read_rows_by_query(path)
    duplicate_indices = 0
    rank_gaps = 0
    invalid_scores = 0
    rows_total = 0
    for query_id, rows in by_query.items():
        rows_total += len(rows)
        indices = [int(row["sentence_index"]) for row in rows]
        if len(indices) != len(set(indices)):
            duplicate_indices += 1
        ranks = [int(row["final_rank"]) for row in rows]
        if ranks != list(range(1, len(rows) + 1)):
            rank_gaps += 1
        for row in rows:
            if expected_run_name is not None and row["run_name"] != expected_run_name:
                raise EvidenceGapError(f"Run name mismatch in {path}: {query_id}")
            for key in ("retrieval_score", "final_score", "cross_encoder_score"):
                value = row.get(key)
                if value is not None and not math.isfinite(float(value)):
                    invalid_scores += 1
            index = int(row["sentence_index"])
            if index < 0:
                raise EvidenceGapError(f"Negative sentence index for {query_id}")
            if expected_queries is not None:
                candidate_count = expected_queries.get(query_id)
                if candidate_count is None:
                    raise EvidenceGapError(f"Unexpected query in run: {query_id}")
                if index >= candidate_count:
                    raise EvidenceGapError(
                        f"Sentence index {index} out of range for {query_id}"
                    )
    missing_queries: list[str] = []
    short_queries: list[str] = []
    if expected_queries is not None:
        missing_queries = sorted(set(expected_queries) - set(by_query))
    if expected_depths is not None:
        short_queries = sorted(
            query_id
            for query_id, depth in expected_depths.items()
            if len(by_query.get(query_id, ())) != depth
        )
    if duplicate_indices or rank_gaps or invalid_scores or missing_queries or short_queries:
        raise EvidenceGapError(
            "Invalid sentence ranking artifact: "
            f"duplicate_index_queries={duplicate_indices}, rank_gap_queries={rank_gaps}, "
            f"invalid_scores={invalid_scores}, missing_queries={len(missing_queries)}, "
            f"wrong_depth_queries={len(short_queries)}"
        )
    return {
        "rows": rows_total,
        "queries": len(by_query),
        "duplicate_index_queries": duplicate_indices,
        "rank_gap_queries": rank_gaps,
        "invalid_scores": invalid_scores,
        "missing_queries": len(missing_queries),
        "wrong_depth_queries": len(short_queries),
        "sha256": sha256_file(path),
    }


def combine_shards(
    shard_paths: Sequence[Path],
    output_path: Path,
    *,
    expected_queries: Mapping[str, int],
    run_name: str,
    force: bool,
) -> dict[str, Any]:
    _pa, pq = _pyarrow()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".tmp")
    temp.unlink(missing_ok=True)
    schema = ranking_schema()
    writer = pq.ParquetWriter(temp, schema, compression="zstd")
    try:
        for shard in shard_paths:
            if not shard.exists():
                raise EvidenceGapError(f"Missing completed ranking shard: {shard}")
            parquet = pq.ParquetFile(shard)
            if parquet.schema_arrow != schema:
                raise EvidenceGapError(f"Ranking shard schema mismatch: {shard}")
            for batch in parquet.iter_batches(batch_size=8192):
                writer.write_batch(batch)
    finally:
        writer.close()
    os.replace(temp, output_path)
    return validate_ranking_rows(
        output_path,
        expected_queries=expected_queries,
        expected_run_name=run_name,
    )


def load_metadata(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceGapError(f"Missing metadata file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise EvidenceGapError(f"Invalid metadata JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceGapError(f"Metadata is not an object: {path}")
    return value


def reuse_or_reject_shard(
    *,
    output_path: Path,
    metadata_path: Path,
    expected_signature: Mapping[str, Any],
    force: bool,
) -> dict[str, Any] | None:
    if force:
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return None
    if not output_path.exists() and not metadata_path.exists():
        return None
    if not output_path.exists() or not metadata_path.exists():
        output_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        return None
    metadata = load_metadata(metadata_path)
    comparable = {key: metadata.get(key) for key in expected_signature}
    if comparable != dict(expected_signature):
        raise EvidenceGapError(
            f"Stale Phase 05 shard {output_path}; use --force to rebuild"
        )
    if metadata.get("output_sha256") != sha256_file(output_path):
        raise EvidenceGapError(f"Phase 05 shard checksum mismatch: {output_path}")
    return metadata


def write_shard_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(payload))
