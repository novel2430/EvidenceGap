from __future__ import annotations

import multiprocessing as mp
import os
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    sha256_file,
)
from evidencegap.dense.article_inputs import DEFAULT_OUTPUT_DIR
from evidencegap.dense.encoders import DenseEncoder, encoder_spec, model_fingerprint

EMBEDDING_SCHEMA_VERSION = "1.0.0"
DEFAULT_DENSE_DIR = Path("artifacts/v1/dense")
DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")


@dataclass(frozen=True)
class ShardInfo:
    path: Path
    start: int
    end: int
    rows: int
    dimension: int
    dtype: str
    sha256: str


def _range(total: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    start = total * shard_index // num_shards
    end = total * (shard_index + 1) // num_shards
    return start, end


def _embedding_root(root: Path, model_key: str) -> Path:
    return root / DEFAULT_DENSE_DIR / model_key / "article_embeddings"


def _query_root(root: Path, model_key: str, split: str) -> Path:
    return root / DEFAULT_DENSE_DIR / model_key / "query_embeddings" / split


def _article_input_manifest(root: Path, input_dir: Path | None) -> tuple[Path, dict[str, Any]]:
    directory = (root / (input_dir or DEFAULT_OUTPUT_DIR)).resolve()
    path = directory / "article_inputs_manifest.json"
    return directory, load_json(path)


def _existing_shard_valid(
    npy_path: Path,
    meta_path: Path,
    *,
    expected: dict[str, Any],
) -> bool:
    if not npy_path.exists() or not meta_path.exists():
        return False
    try:
        meta = load_json(meta_path)
    except EvidenceGapError:
        return False
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    if sha256_file(npy_path) != meta.get("sha256"):
        return False
    matrix = np.load(npy_path, mmap_mode="r")
    return list(matrix.shape) == meta.get("shape") and str(matrix.dtype) == meta.get("dtype")


def encode_article_shard(
    root: Path,
    *,
    model_key: str,
    shard_index: int,
    num_shards: int,
    device: str,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int | None = None,
    amp: str = "fp16",
    force: bool = False,
    expected_model_fingerprint: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if shard_index < 0 or shard_index >= num_shards:
        raise EvidenceGapError("invalid shard_index/num_shards")
    input_directory, input_manifest = _article_input_manifest(root, input_dir)
    input_path = input_directory / "article_inputs.parquet"
    total = int(input_manifest["rows"])
    start, end = _range(total, shard_index, num_shards)
    rows = end - start
    spec = encoder_spec(root, model_key)
    article_model_fingerprint = expected_model_fingerprint or model_fingerprint(spec, article=True)
    output_root = (
        (root / output_dir).resolve()
        if output_dir is not None
        else _embedding_root(root, model_key)
    )
    shard_dir = output_root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    stem = f"shard-{shard_index:05d}-of-{num_shards:05d}"
    npy_path = shard_dir / f"{stem}.npy"
    meta_path = shard_dir / f"{stem}.json"
    expected = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "model_key": model_key,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "start_doc_idx": start,
        "end_doc_idx": end,
        "article_input_sha256": input_manifest["output"]["sha256"],
        "model_fingerprint": article_model_fingerprint,
    }
    if not force and _existing_shard_valid(npy_path, meta_path, expected=expected):
        print(f"  reuse {stem}", flush=True)
        return load_json(meta_path)
    if force:
        npy_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)

    temporary = npy_path.with_name(npy_path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float16,
        shape=(rows, spec.dimension),
    )
    encoder = DenseEncoder(root, model_key, device=device, amp=amp)
    dataset = ds.dataset(input_path, format="parquet")
    scanner = dataset.scanner(
        columns=["doc_idx", "title", "abstract"],
        filter=(ds.field("doc_idx") >= start) & (ds.field("doc_idx") < end),
        batch_size=max(256, (batch_size or spec.default_article_batch_size) * 8),
        use_threads=False,
    )
    written = 0
    expected_doc_idx = start
    try:
        for record_batch in scanner.to_batches():
            doc_indices = record_batch.column("doc_idx").to_pylist()
            titles = record_batch.column("title").to_pylist()
            abstracts = record_batch.column("abstract").to_pylist()
            if doc_indices and int(doc_indices[0]) != expected_doc_idx:
                raise EvidenceGapError(
                    f"article input order drift in {stem}: "
                    f"expected {expected_doc_idx}, got {doc_indices[0]}"
                )
            records = [
                (str(title or ""), str(abstract or ""))
                for title, abstract in zip(titles, abstracts)
            ]
            embeddings = encoder.encode_articles(records, batch_size=batch_size)
            if not np.isfinite(embeddings).all():
                raise EvidenceGapError(f"non-finite embeddings in {stem}")
            next_written = written + len(records)
            matrix[written:next_written] = embeddings.astype(np.float16)
            written = next_written
            expected_doc_idx += len(records)
            if written and written % 10000 < len(records):
                print(
                    f"  {stem} on {device}: {written:,}/{rows:,}",
                    flush=True,
                )
        matrix.flush()
    except Exception:
        del matrix
        temporary.unlink(missing_ok=True)
        raise
    del matrix
    if written != rows:
        temporary.unlink(missing_ok=True)
        raise EvidenceGapError(f"{stem} wrote {written:,}, expected {rows:,}")
    os.replace(temporary, npy_path)
    meta = {
        **expected,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "amp": amp,
        "shape": [rows, spec.dimension],
        "dtype": "float16",
        "sha256": sha256_file(npy_path),
        "bytes": npy_path.stat().st_size,
        "path": relative_path(root, npy_path),
    }
    atomic_write_json(meta_path, meta)
    return meta


def _worker(
    root_text: str,
    kwargs: dict[str, Any],
    device: str,
    shard_indices: list[int],
    result_queue: Any,
) -> None:
    try:
        for shard_index in shard_indices:
            result = encode_article_shard(
                Path(root_text),
                device=device,
                shard_index=shard_index,
                **kwargs,
            )
            result_queue.put(("ok", shard_index, result))
    except Exception as exc:
        result_queue.put(("error", shard_indices, f"{type(exc).__name__}: {exc}"))


def encode_article_embeddings(
    root: Path,
    *,
    model_key: str,
    devices: Sequence[str],
    num_shards: int | None = None,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int | None = None,
    amp: str = "fp16",
    force: bool = False,
) -> dict[str, Any]:
    if not devices:
        raise EvidenceGapError("at least one device is required")
    root = root.resolve()
    num_shards = num_shards or len(devices)
    assignments = {device: [] for device in devices}
    for shard_index in range(num_shards):
        assignments[devices[shard_index % len(devices)]].append(shard_index)

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    spec = encoder_spec(root, model_key)
    shared_model_fingerprint = model_fingerprint(spec, article=True)
    kwargs = {
        "model_key": model_key,
        "num_shards": num_shards,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "batch_size": batch_size,
        "amp": amp,
        "force": force,
        "expected_model_fingerprint": shared_model_fingerprint,
    }
    processes = [
        context.Process(
            target=_worker,
            args=(str(root), kwargs, device, indices, result_queue),
        )
        for device, indices in assignments.items()
        if indices
    ]
    for process in processes:
        process.start()

    expected_results = num_shards
    completed = 0
    errors: list[str] = []
    while completed < expected_results and any(p.is_alive() for p in processes):
        try:
            status, identifier, payload = result_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if status == "ok":
            completed += 1
            print(f"completed article shard {identifier}", flush=True)
        else:
            errors.append(f"shards {identifier}: {payload}")
            break
    if errors:
        for process in processes:
            if process.is_alive():
                process.terminate()
    for process in processes:
        process.join()
        if process.exitcode not in {0, None, -15} and not errors:
            errors.append(f"worker pid={process.pid} exited {process.exitcode}")
    while not result_queue.empty():
        status, identifier, payload = result_queue.get()
        if status == "ok":
            completed += 1
        else:
            errors.append(f"shards {identifier}: {payload}")
    if errors:
        raise EvidenceGapError("article encoding failed: " + "; ".join(errors))

    return finalize_article_embeddings(
        root,
        model_key=model_key,
        num_shards=num_shards,
        input_dir=input_dir,
        output_dir=output_dir,
    )


def finalize_article_embeddings(
    root: Path,
    *,
    model_key: str,
    num_shards: int,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    input_directory, input_manifest = _article_input_manifest(root, input_dir)
    output_root = (
        (root / output_dir).resolve()
        if output_dir is not None
        else _embedding_root(root, model_key)
    )
    spec = encoder_spec(root, model_key)
    shards: list[dict[str, Any]] = []
    cursor = 0
    for shard_index in range(num_shards):
        stem = f"shard-{shard_index:05d}-of-{num_shards:05d}"
        meta_path = output_root / "shards" / f"{stem}.json"
        meta = load_json(meta_path)
        npy_path = output_root / "shards" / f"{stem}.npy"
        if int(meta["start_doc_idx"]) != cursor:
            raise EvidenceGapError(f"non-contiguous embedding ranges at {stem}")
        if int(meta["shape"][1]) != spec.dimension:
            raise EvidenceGapError(f"dimension mismatch at {stem}")
        if sha256_file(npy_path) != meta["sha256"]:
            raise EvidenceGapError(f"fingerprint mismatch at {stem}")
        cursor = int(meta["end_doc_idx"])
        shards.append(meta)
    total = int(input_manifest["rows"])
    if cursor != total:
        raise EvidenceGapError(
            f"embedding rows end at {cursor:,}, expected {total:,}"
        )
    manifest = {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_key": model_key,
        "dimension": spec.dimension,
        "dtype": "float16",
        "rows": total,
        "similarity": spec.similarity,
        "normalized": spec.normalize,
        "pooling": spec.pooling,
        "article_input_manifest": relative_path(
            root, input_directory / "article_inputs_manifest.json"
        ),
        "article_input_sha256": input_manifest["output"]["sha256"],
        "article_model_fingerprint": model_fingerprint(spec, article=True),
        "shards": shards,
    }
    atomic_write_json(output_root / "embedding_manifest.json", manifest)
    return manifest


def encode_query_embeddings(
    root: Path,
    *,
    model_key: str,
    split: str,
    device: str,
    corpus_dir: Path | None = None,
    output_dir: Path | None = None,
    batch_size: int | None = None,
    amp: str = "fp16",
    force: bool = False,
) -> dict[str, Any]:
    if split not in {"dev", "test"}:
        raise EvidenceGapError("split must be dev or test")
    root = root.resolve()
    corpus_dir = (root / (corpus_dir or DEFAULT_CORPUS_DIR)).resolve()
    output_root = (
        (root / output_dir).resolve()
        if output_dir is not None
        else _query_root(root, model_key, split)
    )
    claims_path = corpus_dir / "claims.parquet"
    corpus_manifest_path = corpus_dir / "corpus_manifest.json"
    if not claims_path.exists():
        raise EvidenceGapError(f"Missing claims table: {claims_path}")
    table = pq.read_table(
        claims_path,
        columns=["claim_id", "claim_text", "split"],
        filters=[("split", "=", split)],
    ).sort_by([("claim_id", "ascending")])
    claim_ids = [str(value) for value in table["claim_id"].to_pylist()]
    texts = [str(value) for value in table["claim_text"].to_pylist()]
    spec = encoder_spec(root, model_key)
    encoder = DenseEncoder(root, model_key, device=device, amp=amp)

    with atomic_directory(output_root, force=force) as staging:
        embeddings = encoder.encode_queries(texts, batch_size=batch_size)
        if not np.isfinite(embeddings).all():
            raise EvidenceGapError("non-finite query embeddings")
        npy_path = staging / "embeddings.npy"
        np.save(npy_path, embeddings.astype(np.float32), allow_pickle=False)
        metadata_path = staging / "queries.parquet"
        pq.write_table(
            pa.table(
                {
                    "query_idx": pa.array(range(len(claim_ids)), type=pa.int64()),
                    "claim_id": pa.array(claim_ids, type=pa.string()),
                }
            ),
            metadata_path,
            compression="zstd",
        )
        manifest = {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_key": model_key,
            "split": split,
            "rows": len(claim_ids),
            "dimension": spec.dimension,
            "dtype": "float32",
            "similarity": spec.similarity,
            "normalized": spec.normalize,
            "pooling": spec.pooling,
            "query_model_fingerprint": model_fingerprint(spec, article=False),
            "source_corpus_manifest_sha256": sha256_file(corpus_manifest_path),
            "embeddings": {
                "path": relative_path(root, output_root / "embeddings.npy"),
                "sha256": sha256_file(npy_path),
            },
            "queries": {
                "path": relative_path(root, output_root / "queries.parquet"),
                "sha256": sha256_file(metadata_path),
            },
        }
        atomic_write_json(staging / "query_embedding_manifest.json", manifest)
    return manifest


class ShardedEmbeddingStore:
    def __init__(self, root: Path, manifest_path: Path) -> None:
        self.root = root.resolve()
        self.manifest_path = manifest_path.resolve()
        self.manifest = load_json(self.manifest_path)
        self.rows = int(self.manifest["rows"])
        self.dimension = int(self.manifest["dimension"])
        self.shards: list[tuple[int, int, np.ndarray, Path]] = []
        for meta in self.manifest["shards"]:
            path = self.root / meta["path"]
            matrix = np.load(path, mmap_mode="r")
            self.shards.append(
                (
                    int(meta["start_doc_idx"]),
                    int(meta["end_doc_idx"]),
                    matrix,
                    path,
                )
            )

    def get_rows(self, indices: Sequence[int]) -> np.ndarray:
        result = np.empty((len(indices), self.dimension), dtype=np.float32)
        assignments: dict[int, list[tuple[int, int]]] = {}
        for output_row, value in enumerate(indices):
            index = int(value)
            if index < 0 or index >= self.rows:
                raise EvidenceGapError(f"embedding row out of range: {index}")
            for shard_number, (start, end, _matrix, _path) in enumerate(self.shards):
                if start <= index < end:
                    assignments.setdefault(shard_number, []).append(
                        (output_row, index - start)
                    )
                    break
        for shard_number, pairs in assignments.items():
            start, _end, matrix, _path = self.shards[shard_number]
            source_rows = [relative for _out, relative in pairs]
            values = np.asarray(matrix[source_rows], dtype=np.float32)
            for value, (output_row, _relative) in zip(values, pairs):
                result[output_row] = value
        return result

    def iter_batches(self, batch_size: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start, end, matrix, _path in self.shards:
            for offset in range(0, end - start, batch_size):
                stop = min(end - start, offset + batch_size)
                ids = np.arange(start + offset, start + stop, dtype=np.int64)
                values = np.asarray(matrix[offset:stop], dtype=np.float32)
                yield ids, values


class QueryEmbeddingStore:
    def __init__(self, root: Path, directory: Path) -> None:
        self.root = root.resolve()
        self.directory = directory.resolve()
        self.manifest = load_json(self.directory / "query_embedding_manifest.json")
        self.embeddings = np.load(self.directory / "embeddings.npy", mmap_mode="r")
        table = pq.read_table(self.directory / "queries.parquet").sort_by(
            [("query_idx", "ascending")]
        )
        self.claim_ids = [str(value) for value in table["claim_id"].to_pylist()]
        self.index_by_claim_id = {
            claim_id: index for index, claim_id in enumerate(self.claim_ids)
        }

    def rows_for_claims(self, claim_ids: Sequence[str]) -> np.ndarray:
        try:
            indices = [self.index_by_claim_id[claim_id] for claim_id in claim_ids]
        except KeyError as exc:
            raise EvidenceGapError(f"query embedding missing claim: {exc.args[0]}") from exc
        return np.asarray(self.embeddings[indices], dtype=np.float32)


def validate_article_embeddings(
    root: Path,
    *,
    model_key: str,
    embedding_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    directory = (
        (root / embedding_dir).resolve()
        if embedding_dir is not None
        else _embedding_root(root, model_key)
    )
    errors: list[str] = []
    try:
        store = ShardedEmbeddingStore(root, directory / "embedding_manifest.json")
    except Exception as exc:
        return {"status": "FAIL", "errors": [str(exc)]}
    cursor = 0
    for meta, (start, end, matrix, path) in zip(
        store.manifest["shards"], store.shards
    ):
        if start != cursor:
            errors.append(f"non-contiguous range at {path.name}")
        if matrix.shape != (end - start, store.dimension):
            errors.append(f"shape mismatch at {path.name}")
        if sha256_file(path) != meta.get("sha256"):
            errors.append(f"fingerprint mismatch at {path.name}")
        if len(matrix):
            sample = np.asarray(matrix[[0, len(matrix) - 1]], dtype=np.float32)
            if not np.isfinite(sample).all():
                errors.append(f"non-finite sampled values at {path.name}")
        cursor = end
    if cursor != store.rows:
        errors.append("final embedding row count mismatch")
    input_manifest_path = root / store.manifest["article_input_manifest"]
    if input_manifest_path.exists():
        input_manifest = load_json(input_manifest_path)
        if input_manifest["output"]["sha256"] != store.manifest.get(
            "article_input_sha256"
        ):
            errors.append("article input fingerprint drift")
    else:
        errors.append("missing article input manifest")
    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": store.rows,
        "dimension": store.dimension,
    }


def validate_query_embeddings(
    root: Path,
    *,
    model_key: str,
    split: str,
    query_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    directory = (
        (root / query_dir).resolve()
        if query_dir is not None
        else _query_root(root, model_key, split)
    )
    errors: list[str] = []
    try:
        store = QueryEmbeddingStore(root, directory)
    except Exception as exc:
        return {"status": "FAIL", "errors": [str(exc)]}
    if len(store.claim_ids) != store.embeddings.shape[0]:
        errors.append("query ID/embedding count mismatch")
    if int(store.manifest["dimension"]) != store.embeddings.shape[1]:
        errors.append("query embedding dimension mismatch")
    if str(store.embeddings.dtype) != str(store.manifest.get("dtype")):
        errors.append("query embedding dtype mismatch")
    embeddings_path = directory / "embeddings.npy"
    queries_path = directory / "queries.parquet"
    if sha256_file(embeddings_path) != store.manifest.get("embeddings", {}).get("sha256"):
        errors.append("query embedding fingerprint mismatch")
    if sha256_file(queries_path) != store.manifest.get("queries", {}).get("sha256"):
        errors.append("query metadata fingerprint mismatch")
    if len(store.embeddings):
        sample = np.asarray(store.embeddings[[0, len(store.embeddings) - 1]])
        if not np.isfinite(sample).all():
            errors.append("non-finite sampled query embeddings")
    return {
        "schema_version": EMBEDDING_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": len(store.claim_ids),
        "dimension": int(store.embeddings.shape[1]),
    }
