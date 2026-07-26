from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from evidencegap_backend.common import EvidenceGapError, load_json


def _faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise EvidenceGapError(
            "Missing faiss-cpu dependency. Install the backend runtime dependencies"
        ) from exc
    return faiss


class DenseFaissBackend:
    """Load and query an existing Phase 03 FAISS article index."""

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

    @property
    def loaded(self) -> bool:
        return self.index is not None

    def close(self) -> None:
        self.index = None

    def search(
        self,
        queries: np.ndarray,
        *,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise EvidenceGapError("FAISS backend is closed")
        if queries.ndim != 2:
            raise EvidenceGapError("query embeddings must be a 2-D matrix")
        scores, ids = self.index.search(
            np.ascontiguousarray(queries, dtype=np.float32),
            top_k,
        )
        return scores, ids
