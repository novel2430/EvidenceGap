from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from evidencegap.common import EvidenceGapError, atomic_write_json, relative_path, sha256_file, sha256_text
from evidencegap.dense.encoders import DenseEncoder, encoder_spec, model_fingerprint
from evidencegap.sentence_retrieval.artifacts import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_REPORT_ROOT,
    RUN_SCHEMA_VERSION,
    combine_shards,
    reuse_or_reject_shard,
    safe_run_name,
    validate_ranking_rows,
    write_rows_atomic,
    write_shard_metadata,
)
from evidencegap.sentence_retrieval.bm25 import required_depth
from evidencegap.sentence_retrieval.contracts import EvidenceQuery, SCHEMA_VERSION, TASK_ID
from evidencegap.sentence_retrieval.evaluation import evaluate_sentence_run
from evidencegap.sentence_retrieval.evidencebench import ensure_canonical


def normalize_devices(values: Sequence[str]) -> list[str]:
    devices: list[str] = []
    for raw in values:
        raw = raw.strip()
        if not raw:
            continue
        if raw == "cpu" or raw.startswith("cuda:"):
            device = raw
        elif raw.isdigit():
            device = f"cuda:{raw}"
        else:
            raise EvidenceGapError(f"Invalid device {raw!r}; use 0,1,..., cuda:N, or cpu")
        devices.append(device)
    if not devices:
        raise EvidenceGapError("At least one device is required")
    if len(set(devices)) != len(devices):
        raise EvidenceGapError("Devices must be unique")
    return devices


def _stable_shard(pool_fingerprint: str, num_shards: int) -> int:
    return int(pool_fingerprint[:16], 16) % num_shards


def _combined_model_fingerprint(root: Path, model_key: str) -> str:
    spec = encoder_spec(root, model_key)
    payload = {
        "model_key": model_key,
        "query": model_fingerprint(spec, article=False),
        "candidate": model_fingerprint(spec, article=True),
        "query_format": (
            "raw_hypothesis" if model_key == "medcpt" else "BMR_TASK + Query: hypothesis"
        ),
        "candidate_format": (
            "medcpt_pair(empty_title, exact_sentence)"
            if model_key == "medcpt"
            else "Represent this passage\\npassage: exact_sentence"
        ),
        "pooling": spec.pooling,
        "similarity": spec.similarity,
        "normalize": spec.normalize,
    }
    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _atomic_save_npy(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.unlink(missing_ok=True)
    with temp.open("wb") as handle:
        np.save(handle, matrix, allow_pickle=False)
    os.replace(temp, path)


def _load_or_encode_pool(
    *,
    encoder: DenseEncoder,
    query: EvidenceQuery,
    model_key: str,
    model_fp: str,
    cache_root: Path,
    batch_size: int | None,
    force: bool,
) -> np.ndarray:
    cache_dir = cache_root / "sentence_embeddings" / model_key
    matrix_path = cache_dir / f"{query.pool_fingerprint}.npy"
    metadata_path = cache_dir / f"{query.pool_fingerprint}.json"
    signature = {
        "schema_version": RUN_SCHEMA_VERSION,
        "model_key": model_key,
        "model_fingerprint": model_fp,
        "pool_fingerprint": query.pool_fingerprint,
        "sentence_count": len(query.candidate_sentences),
        "input_format": (
            "medcpt_pair(empty_title, exact_sentence)"
            if model_key == "medcpt"
            else "Represent this passage\\npassage: exact_sentence"
        ),
    }
    if force:
        matrix_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    if matrix_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if {key: metadata.get(key) for key in signature} != signature:
            raise EvidenceGapError(f"Stale sentence embedding cache: {matrix_path}")
        if metadata.get("output_sha256") != sha256_file(matrix_path):
            raise EvidenceGapError(f"Sentence embedding checksum mismatch: {matrix_path}")
        matrix = np.load(matrix_path, allow_pickle=False)
        if matrix.shape != (len(query.candidate_sentences), encoder.spec.dimension):
            raise EvidenceGapError(f"Sentence embedding shape mismatch: {matrix_path}")
        return matrix.astype(np.float32, copy=False)
    if matrix_path.exists() or metadata_path.exists():
        matrix_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    records = [("", sentence) for sentence in query.candidate_sentences]
    matrix = encoder.encode_articles(records, batch_size=batch_size).astype(np.float32, copy=False)
    _atomic_save_npy(matrix_path, matrix)
    atomic_write_json(
        metadata_path,
        {
            **signature,
            "output_sha256": sha256_file(matrix_path),
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
        },
    )
    return matrix


def _load_or_encode_query(
    *,
    encoder: DenseEncoder,
    query: EvidenceQuery,
    model_key: str,
    model_fp: str,
    cache_root: Path,
    batch_size: int | None,
    force: bool,
) -> np.ndarray:
    hypothesis_hash = sha256_text(query.hypothesis)
    cache_dir = cache_root / "query_embeddings" / model_key
    matrix_path = cache_dir / f"{query.query_id.replace(':', '_')}_{hypothesis_hash[:16]}.npy"
    metadata_path = matrix_path.with_suffix(".json")
    signature = {
        "schema_version": RUN_SCHEMA_VERSION,
        "model_key": model_key,
        "model_fingerprint": model_fp,
        "query_id": query.query_id,
        "hypothesis_sha256": hypothesis_hash,
    }
    if force:
        matrix_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    if matrix_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if {key: metadata.get(key) for key in signature} != signature:
            raise EvidenceGapError(f"Stale query embedding cache: {matrix_path}")
        if metadata.get("output_sha256") != sha256_file(matrix_path):
            raise EvidenceGapError(f"Query embedding checksum mismatch: {matrix_path}")
        vector = np.load(matrix_path, allow_pickle=False)
        if vector.shape != (encoder.spec.dimension,):
            raise EvidenceGapError(f"Query embedding shape mismatch: {matrix_path}")
        return vector.astype(np.float32, copy=False)
    if matrix_path.exists() or metadata_path.exists():
        matrix_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    vector = encoder.encode_queries([query.hypothesis], batch_size=batch_size)[0].astype(np.float32, copy=False)
    _atomic_save_npy(matrix_path, vector)
    atomic_write_json(
        metadata_path,
        {
            **signature,
            "output_sha256": sha256_file(matrix_path),
            "shape": list(vector.shape),
            "dtype": str(vector.dtype),
        },
    )
    return vector


def _dense_rows(
    query: EvidenceQuery,
    scores: np.ndarray,
    *,
    split: str,
    run_name: str,
    model_key: str,
    top_k: int,
) -> Iterable[dict[str, Any]]:
    ranking = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    depth = required_depth(query, top_k)
    for rank, index in enumerate(ranking[:depth], start=1):
        score = float(scores[index])
        yield {
            "schema_version": SCHEMA_VERSION,
            "task_id": TASK_ID,
            "split": split,
            "run_name": run_name,
            "query_id": query.query_id,
            "paper_id": query.paper_id,
            "pool_fingerprint": query.pool_fingerprint,
            "sentence_index": index,
            "sentence_type": query.sentence_types[index],
            "sentence_text": query.candidate_sentences[index],
            "retrieval_model": model_key,
            "retrieval_score": score,
            "retrieval_rank": rank,
            "cross_encoder_score": None,
            "final_score": score,
            "final_rank": rank,
        }


def _dense_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(payload["root"])
    model_key = str(payload["model_key"])
    device = str(payload["device"])
    amp = str(payload["amp"])
    batch_size = payload.get("batch_size")
    queries = [EvidenceQuery.from_dict(value) for value in payload["queries"]]
    output_path = Path(payload["output_path"])
    metadata_path = Path(payload["metadata_path"])
    signature = dict(payload["signature"])
    reused = reuse_or_reject_shard(
        output_path=output_path,
        metadata_path=metadata_path,
        expected_signature=signature,
        force=bool(payload["force"]),
    )
    if reused is not None:
        return reused
    if not queries:
        row_count = write_rows_atomic(output_path, ())
        metadata = {
            **signature,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": device,
            "queries": 0,
            "rows": row_count,
            "unique_pools": 0,
            "output_sha256": sha256_file(output_path),
        }
        write_shard_metadata(metadata_path, metadata)
        return metadata
    encoder = DenseEncoder(root, model_key, device=device, amp=amp)
    cache_root = Path(payload["cache_root"])
    model_fp = str(payload["model_fingerprint"])
    pool_matrices: dict[str, np.ndarray] = {}

    def rows() -> Iterable[dict[str, Any]]:
        for query in queries:
            matrix = pool_matrices.get(query.pool_fingerprint)
            if matrix is None:
                matrix = _load_or_encode_pool(
                    encoder=encoder,
                    query=query,
                    model_key=model_key,
                    model_fp=model_fp,
                    cache_root=cache_root,
                    batch_size=batch_size,
                    force=bool(payload["force"]),
                )
                pool_matrices[query.pool_fingerprint] = matrix
            vector = _load_or_encode_query(
                encoder=encoder,
                query=query,
                model_key=model_key,
                model_fp=model_fp,
                cache_root=cache_root,
                batch_size=batch_size,
                force=bool(payload["force"]),
            )
            scores = matrix @ vector
            yield from _dense_rows(
                query,
                scores,
                split=str(payload["split"]),
                run_name=str(payload["run_name"]),
                model_key=model_key,
                top_k=int(payload["top_k"]),
            )

    row_count = write_rows_atomic(output_path, rows())
    metadata = {
        **signature,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "queries": len(queries),
        "rows": row_count,
        "unique_pools": len({query.pool_fingerprint for query in queries}),
        "output_sha256": sha256_file(output_path),
    }
    write_shard_metadata(metadata_path, metadata)
    return metadata


def _dense_worker_group(payloads: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [_dense_worker(payload) for payload in payloads]


def run_dense_sentence_retrieval(
    root: Path,
    *,
    model_key: str,
    split: str,
    devices: Sequence[str],
    max_queries: int | None = None,
    canonical_dir: Path | None = None,
    run_name: str | None = None,
    top_k: int = 20,
    num_shards: int | None = None,
    batch_size: int | None = None,
    amp: str = "fp16",
    artifact_root: Path | None = None,
    report_dir: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if model_key not in {"medcpt", "bmretriever"}:
        raise EvidenceGapError("model_key must be medcpt or bmretriever")
    if top_k <= 0:
        raise EvidenceGapError("top_k must be positive")
    root = root.resolve()
    normalized_devices = normalize_devices(devices)
    if any(device == "cpu" for device in normalized_devices) and amp == "fp16":
        raise EvidenceGapError("fp16 dense inference is not supported on CPU")
    shard_count = num_shards or len(normalized_devices)
    if shard_count <= 0:
        raise EvidenceGapError("num_shards must be positive")
    canonical_path, queries, canonical_manifest = ensure_canonical(
        root,
        split=split,
        max_queries=max_queries,
        canonical_dir=canonical_dir,
    )
    name = safe_run_name(
        run_name or f"{model_key}_{split}_{'full' if max_queries is None else max_queries}"
    )
    base_root = artifact_root.resolve() if artifact_root else root / DEFAULT_ARTIFACT_ROOT
    run_base = base_root / "runs" / name
    shard_dir = run_base / "shards"
    cache_root = base_root
    if force and run_base.exists():
        shutil.rmtree(run_base)
    shard_dir.mkdir(parents=True, exist_ok=True)
    model_fp = _combined_model_fingerprint(root, model_key)
    partitions: list[list[EvidenceQuery]] = [[] for _ in range(shard_count)]
    for query in queries:
        partitions[_stable_shard(query.pool_fingerprint, shard_count)].append(query)

    payloads: list[dict[str, Any]] = []
    shard_paths: list[Path] = []
    for shard_index, partition in enumerate(partitions):
        output_path = shard_dir / f"shard-{shard_index:05d}-of-{shard_count:05d}.parquet"
        metadata_path = output_path.with_suffix(".json")
        shard_paths.append(output_path)
        signature = {
            "schema_version": RUN_SCHEMA_VERSION,
            "task_id": TASK_ID,
            "run_name": name,
            "split": split,
            "model_key": model_key,
            "model_fingerprint": model_fp,
            "canonical_sha256": canonical_manifest["canonical_sha256"],
            "shard_index": shard_index,
            "num_shards": shard_count,
            "top_k": top_k,
            "amp": amp,
            "assignment": "int(pool_fingerprint[:16],16) modulo num_shards",
        }
        payloads.append(
            {
                "root": str(root),
                "model_key": model_key,
                "model_fingerprint": model_fp,
                "device": normalized_devices[shard_index % len(normalized_devices)],
                "amp": amp,
                "batch_size": batch_size,
                "queries": [query.to_dict() for query in partition],
                "output_path": str(output_path),
                "metadata_path": str(metadata_path),
                "cache_root": str(cache_root),
                "signature": signature,
                "force": force,
                "split": split,
                "run_name": name,
                "top_k": top_k,
            }
        )

    groups = [
        [payload for payload in payloads if payload["device"] == device]
        for device in normalized_devices
    ]
    groups = [group for group in groups if group]
    if len(groups) == 1:
        nested_metadata = [_dense_worker_group(groups[0])]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=len(groups)) as pool:
            nested_metadata = pool.map(_dense_worker_group, groups)
    shard_metadata = [item for group in nested_metadata for item in group]
    shard_metadata.sort(key=lambda item: int(item["shard_index"]))

    output_path = run_base / "ranked_sentences.parquet"
    validation = combine_shards(
        shard_paths,
        output_path,
        expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
        run_name=name,
        force=force,
    )
    validation = validate_ranking_rows(
        output_path,
        expected_queries={query.query_id: len(query.candidate_sentences) for query in queries},
        expected_depths={
            query.query_id: required_depth(query, top_k) for query in queries
        },
        expected_run_name=name,
    )
    run_manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "task_id": TASK_ID,
        "run_name": name,
        "split": split,
        "retrieval_model": model_key,
        "model_fingerprint": model_fp,
        "canonical_dir": relative_path(root, canonical_path),
        "canonical_sha256": canonical_manifest["canonical_sha256"],
        "parameters": {
            "top_k": top_k,
            "num_shards": shard_count,
            "devices": normalized_devices,
            "batch_size": batch_size,
            "amp": amp,
            "similarity": "inner_product",
            "normalization": False,
            "sentence_input_format": (
                "medcpt_pair(empty_title, exact_sentence)"
                if model_key == "medcpt"
                else "Represent this passage\\npassage: exact_sentence"
            ),
        },
        "queries": len(queries),
        "rows": validation["rows"],
        "output_path": relative_path(root, output_path),
        "output_sha256": validation["sha256"],
        "shards": shard_metadata,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "score_semantics": "inner_product_higher_is_more_relevant",
    }
    manifest_path = run_base / "run_manifest.json"
    atomic_write_json(manifest_path, run_manifest)
    report_root = report_dir.resolve() if report_dir else root / DEFAULT_REPORT_ROOT
    report_path = report_root / f"evidence_sentence_retrieval_{name}_{split}.json"
    evaluation = evaluate_sentence_run(
        root,
        canonical_dir=canonical_path,
        run_path=output_path,
        report_path=report_path,
    )
    return {
        "run_name": name,
        "run_path": relative_path(root, output_path),
        "manifest_path": relative_path(root, manifest_path),
        "report_path": relative_path(root, report_path),
        "validation": validation,
        "metrics": evaluation["metrics"],
    }
