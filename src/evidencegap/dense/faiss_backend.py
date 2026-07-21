from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from evidencegap.common import (
    EvidenceGapError,
    atomic_directory,
    atomic_write_json,
    load_json,
    relative_path,
    sha256_file,
)
from evidencegap.dense.embeddings import (
    DEFAULT_DENSE_DIR,
    ShardedEmbeddingStore,
)

FAISS_SCHEMA_VERSION = "1.0.0"
DEFAULT_CORPUS_DIR = Path("artifacts/v1/article_corpus")


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing faiss-cpu dependency. Install requirements/v1-phase03.txt"
        ) from exc
    return faiss


def _embedding_dir(root: Path, model_key: str) -> Path:
    return root / DEFAULT_DENSE_DIR / model_key / "article_embeddings"


def _index_dir(root: Path, model_key: str) -> Path:
    return root / DEFAULT_DENSE_DIR / model_key / "faiss_index"


def build_faiss_index(
    root: Path,
    *,
    model_key: str,
    embedding_dir: Path | None = None,
    index_dir: Path | None = None,
    index_type: str = "ivf-sq-fp16",
    nlist: int = 4096,
    nprobe: int = 64,
    train_size: int = 200000,
    seed: int = 20260721,
    add_batch_size: int = 32768,
    threads: int = 16,
    force: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    embedding_dir = (
        (root / embedding_dir).resolve()
        if embedding_dir is not None
        else _embedding_dir(root, model_key)
    )
    index_dir = (
        (root / index_dir).resolve()
        if index_dir is not None
        else _index_dir(root, model_key)
    )
    embedding_manifest_path = embedding_dir / "embedding_manifest.json"
    store = ShardedEmbeddingStore(root, embedding_manifest_path)
    faiss = _faiss()
    faiss.omp_set_num_threads(max(1, threads))
    if index_type not in {"flat", "ivf-flat", "ivf-sq-fp16"}:
        raise EvidenceGapError("index_type must be flat, ivf-flat, or ivf-sq-fp16")
    if nprobe < 1:
        raise EvidenceGapError("nprobe must be positive")

    with atomic_directory(index_dir, force=force) as staging:
        dimension = store.dimension
        if index_type == "flat":
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
            effective_nlist = None
        else:
            if nlist < 1 or nlist > store.rows:
                raise EvidenceGapError("invalid nlist")
            minimum_training = nlist * 39
            if train_size < minimum_training:
                raise EvidenceGapError(
                    f"train_size={train_size:,} is too small for nlist={nlist:,}; "
                    f"use at least {minimum_training:,}"
                )
            quantizer = faiss.IndexFlatIP(dimension)
            if index_type == "ivf-flat":
                index = faiss.IndexIVFFlat(
                    quantizer,
                    dimension,
                    nlist,
                    faiss.METRIC_INNER_PRODUCT,
                )
            else:
                index = faiss.IndexIVFScalarQuantizer(
                    quantizer,
                    dimension,
                    nlist,
                    faiss.ScalarQuantizer.QT_fp16,
                    faiss.METRIC_INNER_PRODUCT,
                )
            rng = np.random.default_rng(seed)
            sample_count = min(store.rows, train_size)
            sample_ids = np.sort(
                rng.choice(store.rows, size=sample_count, replace=False).astype(np.int64)
            )
            training = store.get_rows(sample_ids)
            print(
                f"Training FAISS {index_type} with {sample_count:,} vectors...",
                flush=True,
            )
            index.train(np.ascontiguousarray(training, dtype=np.float32))
            effective_nlist = nlist

        added = 0
        for ids, vectors in store.iter_batches(add_batch_size):
            index.add_with_ids(
                np.ascontiguousarray(vectors, dtype=np.float32),
                np.ascontiguousarray(ids, dtype=np.int64),
            )
            added += len(ids)
            if added % 100000 < len(ids):
                print(f"  FAISS add: {added:,}/{store.rows:,}", flush=True)
        if added != store.rows or int(index.ntotal) != store.rows:
            raise EvidenceGapError(
                f"FAISS row count mismatch: added={added}, ntotal={index.ntotal}, "
                f"expected={store.rows}"
            )
        effective_nprobe = min(nprobe, nlist) if hasattr(index, "nprobe") else None
        if effective_nprobe is not None:
            index.nprobe = effective_nprobe

        index_path = staging / "index.faiss"
        faiss.write_index(index, str(index_path))
        manifest = {
            "schema_version": FAISS_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model_key": model_key,
            "index_type": index_type,
            "metric": "inner_product",
            "normalized": bool(store.manifest.get("normalized", False)),
            "dimension": dimension,
            "rows": store.rows,
            "nlist": effective_nlist,
            "default_nprobe": effective_nprobe,
            "training": {
                "size": min(store.rows, train_size) if effective_nlist else 0,
                "seed": seed,
            },
            "article_embedding_manifest": relative_path(
                root, embedding_manifest_path
            ),
            "article_embedding_manifest_sha256": sha256_file(
                embedding_manifest_path
            ),
            "index": {
                "path": relative_path(root, index_dir / "index.faiss"),
                "sha256": sha256_file(index_path),
                "bytes": index_path.stat().st_size,
            },
        }
        atomic_write_json(staging / "index_manifest.json", manifest)
    return manifest


class DenseFaissBackend:
    def __init__(
        self,
        root: Path,
        index_dir: Path,
        *,
        nprobe: int | None = None,
    ) -> None:
        self.root = root.resolve()
        self.index_dir = index_dir.resolve()
        self.manifest = load_json(self.index_dir / "index_manifest.json")
        faiss = _faiss()
        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        if hasattr(self.index, "nprobe"):
            value = nprobe or self.manifest.get("default_nprobe") or 1
            self.index.nprobe = min(int(value), int(self.manifest["nlist"]))
        self.nprobe = getattr(self.index, "nprobe", None)

    def search(
        self,
        queries: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if queries.ndim != 2:
            raise EvidenceGapError("query embeddings must be a 2-D matrix")
        scores, ids = self.index.search(
            np.ascontiguousarray(queries, dtype=np.float32),
            top_k,
        )
        return scores, ids


def validate_faiss_index(
    root: Path,
    *,
    model_key: str,
    embedding_dir: Path | None = None,
    index_dir: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    embedding_dir = (
        (root / embedding_dir).resolve()
        if embedding_dir is not None
        else _embedding_dir(root, model_key)
    )
    index_dir = (
        (root / index_dir).resolve()
        if index_dir is not None
        else _index_dir(root, model_key)
    )
    errors: list[str] = []
    try:
        manifest = load_json(index_dir / "index_manifest.json")
        embedding_manifest_path = embedding_dir / "embedding_manifest.json"
        embedding_manifest = load_json(embedding_manifest_path)
        backend = DenseFaissBackend(root, index_dir)
    except Exception as exc:
        return {"status": "FAIL", "errors": [str(exc)]}
    if manifest.get("schema_version") != FAISS_SCHEMA_VERSION:
        errors.append("FAISS schema version mismatch")
    if manifest.get("model_key") != model_key:
        errors.append("FAISS model key mismatch")
    if sha256_file(index_dir / "index.faiss") != manifest["index"]["sha256"]:
        errors.append("FAISS index fingerprint mismatch")
    if (
        sha256_file(embedding_manifest_path)
        != manifest["article_embedding_manifest_sha256"]
    ):
        errors.append("article embedding manifest drift")
    if int(backend.index.ntotal) != int(embedding_manifest["rows"]):
        errors.append("FAISS/embedding row count mismatch")
    if int(backend.index.d) != int(embedding_manifest["dimension"]):
        errors.append("FAISS/embedding dimension mismatch")
    return {
        "schema_version": FAISS_SCHEMA_VERSION,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": int(backend.index.ntotal),
        "dimension": int(backend.index.d),
        "index": relative_path(root, index_dir),
    }
