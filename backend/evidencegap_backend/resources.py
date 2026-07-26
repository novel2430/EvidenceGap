from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidencegap_backend.common import (
    EvidenceGapError,
    load_json,
    relative_path,
    sha256_file,
)
from evidencegap_backend.config import BackendConfig
from evidencegap_backend.dense.encoders import DenseEncoder, model_fingerprint
from evidencegap_backend.dense.faiss_backend import DenseFaissBackend
from evidencegap_backend.pipeline.sentence_materialization import (
    StanzaSentenceSplitter,
)
from evidencegap_backend.reranking.cross_encoder import CrossEncoderScorer
from evidencegap_backend.retrieval.bm25s_backend import BM25SBackend


def _quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


class ArticleTextStore:
    """Persistent DuckDB-backed article lookup with a bounded in-memory cache."""

    def __init__(
        self,
        *,
        article_input_path: Path,
        corpus_articles_path: Path,
        max_cached_articles: int = 5000,
    ) -> None:
        try:
            import duckdb
            import pyarrow as pa
        except ImportError as exc:
            raise EvidenceGapError(
                "Missing duckdb/pyarrow dependencies for article lookup"
            ) from exc
        if max_cached_articles <= 0:
            raise EvidenceGapError("max_cached_articles must be positive")
        self._pa = pa
        self._connection = duckdb.connect()
        self._connection.execute(
            f"""
            CREATE VIEW runtime_article_inputs AS
            SELECT
                CAST(article_id AS VARCHAR) AS article_id,
                CAST(doc_idx AS BIGINT) AS doc_idx,
                coalesce(CAST(title AS VARCHAR), '') AS title,
                coalesce(CAST(abstract AS VARCHAR), '') AS abstract
            FROM read_parquet('{_quote(article_input_path)}')
            """
        )
        self._connection.execute(
            f"""
            CREATE VIEW runtime_corpus_articles AS
            SELECT
                CAST(article_id AS VARCHAR) AS article_id,
                CASE WHEN pmid IS NULL THEN NULL ELSE CAST(pmid AS VARCHAR) END AS pmid
            FROM read_parquet('{_quote(corpus_articles_path)}')
            """
        )
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_cached_articles = max_cached_articles
        self._cache_hits = 0
        self._cache_misses = 0
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._connection is not None

    def fetch(self, article_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if self._connection is None:
            raise EvidenceGapError("Article text store is closed")
        ordered_ids = [str(value) for value in article_ids]
        if len(ordered_ids) != len(set(ordered_ids)):
            raise EvidenceGapError("Article lookup contains duplicate article IDs")
        with self._lock:
            missing: list[str] = []
            for article_id in ordered_ids:
                if article_id in self._cache:
                    self._cache_hits += 1
                    self._cache.move_to_end(article_id)
                else:
                    self._cache_misses += 1
                    missing.append(article_id)
            if missing:
                table = self._pa.Table.from_pylist(
                    [{"article_id": value} for value in missing]
                )
                self._connection.register("runtime_candidates", table)
                try:
                    rows = self._connection.execute(
                        """
                        SELECT
                            i.article_id,
                            i.doc_idx,
                            a.pmid,
                            i.title,
                            i.abstract
                        FROM runtime_article_inputs i
                        JOIN runtime_candidates c USING (article_id)
                        LEFT JOIN runtime_corpus_articles a USING (article_id)
                        """
                    ).fetch_arrow_table().to_pylist()
                finally:
                    self._connection.unregister("runtime_candidates")
                found = {str(row["article_id"]): dict(row) for row in rows}
                unresolved = [value for value in missing if value not in found]
                if unresolved:
                    raise EvidenceGapError(
                        "Article input join missed "
                        f"{len(unresolved)} candidates; first={unresolved[0]}"
                    )
                for article_id in missing:
                    self._cache[article_id] = found[article_id]
                    self._cache.move_to_end(article_id)
                    while len(self._cache) > self._max_cached_articles:
                        self._cache.popitem(last=False)
            return {article_id: dict(self._cache[article_id]) for article_id in ordered_ids}

    def status(self) -> dict[str, Any]:
        return {
            "loaded": self.loaded,
            "cached_articles": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "max_cached_articles": self._max_cached_articles,
        }

    def close(self) -> None:
        connection = self._connection
        self._connection = None
        self._cache.clear()
        if connection is not None:
            connection.close()


class RuntimeResources:
    """Long-lived model, index, tokenizer, and article-store resources."""

    def __init__(self, config: BackendConfig) -> None:
        self.config = config
        self.bm25: BM25SBackend | None = None
        self.medcpt_encoder: DenseEncoder | None = None
        self.medcpt_index: DenseFaissBackend | None = None
        self.bmretriever_encoder: DenseEncoder | None = None
        self.bmretriever_index: DenseFaissBackend | None = None
        self.cross_encoder: CrossEncoderScorer | None = None
        self.sentence_splitter: StanzaSentenceSplitter | None = None
        self.article_store: ArticleTextStore | None = None
        self.corpus_manifest: Mapping[str, Any] | None = None
        self.article_input_manifest: Mapping[str, Any] | None = None
        self.expected_corpus_articles_sha256 = ""
        self.expected_article_input_sha256 = ""
        self.corpus_manifest_sha256 = ""
        self.article_input_manifest_sha256 = ""
        self.medcpt_metadata: dict[str, Any] = {}
        self.bmretriever_metadata: dict[str, Any] = {}
        self._loaded = False
        self._load_count = 0
        self._claim_queries = 0
        self._analysis_runs = 0
        self._lock = threading.RLock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def article_ids(self) -> Any:
        self._require_loaded()
        assert self.bm25 is not None
        return self.bm25.article_ids

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise EvidenceGapError("Runtime resources are not loaded")

    def load(self, *, validate_paths: bool = True) -> None:
        with self._lock:
            if self._loaded:
                return
            if validate_paths:
                missing = [
                    path for path in self.config.required_resource_paths if not path.exists()
                ]
                if missing:
                    rendered = "\n".join(f"- {path}" for path in missing)
                    raise EvidenceGapError(
                        f"Missing backend runtime resources:\n{rendered}"
                    )
            try:
                self._load_impl()
            except Exception:
                self.close()
                raise
            self._loaded = True
            self._load_count += 1

    def _load_impl(self) -> None:
        cfg = self.config
        corpus_manifest_path = cfg.corpus_dir / "corpus_manifest.json"
        article_input_manifest_path = (
            cfg.article_input_dir / "article_inputs_manifest.json"
        )
        corpus_manifest = load_json(corpus_manifest_path)
        article_input_manifest = load_json(article_input_manifest_path)
        if not isinstance(corpus_manifest, Mapping) or not isinstance(
            article_input_manifest, Mapping
        ):
            raise EvidenceGapError("Article corpus/input manifests must be JSON objects")
        self.corpus_manifest = corpus_manifest
        self.article_input_manifest = article_input_manifest
        self.corpus_manifest_sha256 = sha256_file(corpus_manifest_path)
        self.article_input_manifest_sha256 = sha256_file(article_input_manifest_path)
        self.expected_corpus_articles_sha256 = str(
            corpus_manifest.get("files", {})
            .get("articles.parquet", {})
            .get("sha256", "")
        )
        self.expected_article_input_sha256 = str(
            article_input_manifest.get("output", {}).get("sha256", "")
        )
        if not self.expected_corpus_articles_sha256 or not self.expected_article_input_sha256:
            raise EvidenceGapError("Article corpus/input manifests are incomplete")
        if (
            article_input_manifest.get("source_corpus_manifest_sha256")
            != self.corpus_manifest_sha256
        ):
            raise EvidenceGapError(
                "Phase 03 article inputs do not match the Phase 02 corpus"
            )

        self.bm25 = BM25SBackend(cfg.bm25_index_dir, mmap=True)
        article_ids_metadata = self.bm25.manifest.get("index", {}).get(
            "article_ids", {}
        )
        if article_ids_metadata.get("sha256") != sha256_file(
            cfg.bm25_index_dir / "article_ids.npy"
        ):
            raise EvidenceGapError("BM25 article ID map checksum mismatch")
        if self.bm25.manifest.get("corpus", {}).get(
            "articles_sha256"
        ) != self.expected_corpus_articles_sha256:
            raise EvidenceGapError(
                "BM25 index does not match the Phase 02 article corpus"
            )

        self.medcpt_encoder = DenseEncoder(
            cfg.workspace_root, "medcpt", device=cfg.device, amp=cfg.amp
        )
        self.medcpt_encoder.load_query_model()
        self.medcpt_index = DenseFaissBackend(
            cfg.workspace_root,
            cfg.medcpt_index_dir,
            nprobe=cfg.pipeline.dense_nprobe,
        )
        self.medcpt_metadata = self._validate_dense_runtime(
            "medcpt", self.medcpt_encoder, self.medcpt_index
        )

        self.bmretriever_encoder = DenseEncoder(
            cfg.workspace_root, "bmretriever", device=cfg.device, amp=cfg.amp
        )
        self.bmretriever_encoder.load_query_model()
        self.bmretriever_index = DenseFaissBackend(
            cfg.workspace_root,
            cfg.bmretriever_index_dir,
            nprobe=cfg.pipeline.dense_nprobe,
        )
        self.bmretriever_metadata = self._validate_dense_runtime(
            "bmretriever", self.bmretriever_encoder, self.bmretriever_index
        )

        self.cross_encoder = CrossEncoderScorer(
            cfg.workspace_root,
            model_dir=cfg.cross_encoder_model_dir,
            device=cfg.device,
            amp=cfg.amp,
            max_length=512,
        )
        self.sentence_splitter = StanzaSentenceSplitter(
            model_dir=cfg.stanza_model_dir,
            device=cfg.device,
            package=cfg.stanza_package,
            batch_size=cfg.stanza_batch_size,
            allow_cpu_fallback=cfg.allow_cpu_fallback,
        )
        self.article_store = ArticleTextStore(
            article_input_path=cfg.article_input_dir / "article_inputs.parquet",
            corpus_articles_path=cfg.corpus_dir / "articles.parquet",
            max_cached_articles=cfg.article_cache_size,
        )

    def _validate_dense_runtime(
        self,
        model_key: str,
        encoder: DenseEncoder,
        backend: DenseFaissBackend,
    ) -> dict[str, Any]:
        cfg = self.config
        index_dir = (
            cfg.medcpt_index_dir
            if model_key == "medcpt"
            else cfg.bmretriever_index_dir
        )
        if backend.manifest.get("model_key") != model_key:
            raise EvidenceGapError(
                f"{model_key} FAISS manifest model mismatch: "
                f"{backend.manifest.get('model_key')}"
            )
        expected_index_sha256 = str(
            backend.manifest.get("index", {}).get("sha256", "")
        )
        if not expected_index_sha256 or sha256_file(
            index_dir / "index.faiss"
        ) != expected_index_sha256:
            raise EvidenceGapError(f"{model_key} FAISS index checksum mismatch")
        assert self.bm25 is not None
        article_ids = self.bm25.article_ids
        if int(backend.index.ntotal) != len(article_ids):
            raise EvidenceGapError(
                f"{model_key} FAISS/article ID row mismatch: "
                f"{backend.index.ntotal} != {len(article_ids)}"
            )
        embedding_manifest_value = backend.manifest.get(
            "article_embedding_manifest"
        )
        if not embedding_manifest_value:
            raise EvidenceGapError(
                f"{model_key} FAISS manifest has no article embedding manifest"
            )
        embedding_manifest_path = Path(str(embedding_manifest_value))
        if not embedding_manifest_path.is_absolute():
            embedding_manifest_path = cfg.workspace_root / embedding_manifest_path
        embedding_manifest = load_json(embedding_manifest_path)
        if embedding_manifest.get(
            "article_input_sha256"
        ) != self.expected_article_input_sha256:
            raise EvidenceGapError(
                f"{model_key} FAISS index was built from a different article input"
            )
        return {
            "model_key": model_key,
            "query_model_path": relative_path(
                cfg.workspace_root, encoder.spec.query_model
            ),
            "query_model_fingerprint": model_fingerprint(
                encoder.spec, article=False
            ),
            "index_path": relative_path(cfg.workspace_root, index_dir),
            "index_manifest_sha256": sha256_file(
                index_dir / "index_manifest.json"
            ),
            "article_embedding_manifest_path": relative_path(
                cfg.workspace_root, embedding_manifest_path
            ),
            "article_embedding_manifest_sha256": sha256_file(
                embedding_manifest_path
            ),
            "article_input_sha256": self.expected_article_input_sha256,
            "requested_nprobe": cfg.pipeline.dense_nprobe,
            "actual_nprobe": backend.nprobe,
            "device": cfg.device,
            "amp": cfg.amp,
        }

    def query_dense(
        self, model_key: str, claim_text: str, *, top_k: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self._require_loaded()
        if model_key == "medcpt":
            encoder = self.medcpt_encoder
            backend = self.medcpt_index
            metadata = self.medcpt_metadata
        elif model_key == "bmretriever":
            encoder = self.bmretriever_encoder
            backend = self.bmretriever_index
            metadata = self.bmretriever_metadata
        else:
            raise EvidenceGapError(f"Unsupported dense model: {model_key}")
        assert encoder is not None and backend is not None
        started = time.perf_counter()
        query = encoder.encode_queries([claim_text])
        scores, ids = backend.search(query, top_k=top_k)
        rows: list[dict[str, Any]] = []
        article_ids = self.article_ids
        for rank, (doc_idx_value, score_value) in enumerate(
            zip(ids[0], scores[0], strict=True), start=1
        ):
            doc_idx = int(doc_idx_value)
            if doc_idx < 0:
                continue
            if doc_idx >= len(article_ids):
                raise EvidenceGapError(
                    f"{model_key} FAISS returned out-of-range doc_idx {doc_idx}"
                )
            rows.append(
                {
                    "rank": rank,
                    "doc_idx": doc_idx,
                    "article_id": str(article_ids[doc_idx]),
                    "score": float(score_value),
                }
            )
        self._claim_queries += 1
        result_metadata = dict(metadata)
        result_metadata.update(
            {
                "rows": len(rows),
                "seconds": round(time.perf_counter() - started, 6),
                "resource_lifecycle": "engine_resident",
            }
        )
        return rows, result_metadata

    def fetch_article_texts(
        self, article_ids: Sequence[str]
    ) -> dict[str, dict[str, Any]]:
        self._require_loaded()
        assert self.article_store is not None
        return self.article_store.fetch(article_ids)

    def score_articles(
        self,
        *,
        claim_text: str,
        articles: Sequence[Mapping[str, Any]],
        batch_size: int,
    ) -> dict[str, Any]:
        self._require_loaded()
        assert self.cross_encoder is not None
        result = self.cross_encoder.score(
            claim_text=claim_text,
            articles=articles,
            batch_size=batch_size,
        )
        result["metadata"]["resource_lifecycle"] = "engine_resident"
        return result

    def record_analysis_run(self) -> None:
        self._analysis_runs += 1

    def status(self) -> dict[str, Any]:
        article_store = (
            self.article_store.status() if self.article_store is not None else None
        )
        return {
            "loaded": self._loaded,
            "load_count": self._load_count,
            "analysis_runs": self._analysis_runs,
            "dense_query_calls": self._claim_queries,
            "resource_ids": {
                "bm25": None if self.bm25 is None else id(self.bm25),
                "medcpt_encoder": (
                    None if self.medcpt_encoder is None else id(self.medcpt_encoder)
                ),
                "medcpt_faiss": (
                    None if self.medcpt_index is None else id(self.medcpt_index)
                ),
                "bmretriever_encoder": (
                    None
                    if self.bmretriever_encoder is None
                    else id(self.bmretriever_encoder)
                ),
                "bmretriever_faiss": (
                    None
                    if self.bmretriever_index is None
                    else id(self.bmretriever_index)
                ),
                "cross_encoder": (
                    None if self.cross_encoder is None else id(self.cross_encoder)
                ),
                "stanza": (
                    None
                    if self.sentence_splitter is None
                    else id(self.sentence_splitter)
                ),
                "article_store": (
                    None if self.article_store is None else id(self.article_store)
                ),
            },
            "resources": {
                "bm25": bool(self.bm25 and self.bm25.loaded),
                "medcpt_encoder": bool(
                    self.medcpt_encoder and self.medcpt_encoder.query_model_loaded
                ),
                "medcpt_faiss": bool(
                    self.medcpt_index and self.medcpt_index.loaded
                ),
                "bmretriever_encoder": bool(
                    self.bmretriever_encoder
                    and self.bmretriever_encoder.query_model_loaded
                ),
                "bmretriever_faiss": bool(
                    self.bmretriever_index and self.bmretriever_index.loaded
                ),
                "cross_encoder": bool(
                    self.cross_encoder and self.cross_encoder.loaded
                ),
                "stanza": self.sentence_splitter is not None,
                "article_store": article_store,
            },
        }

    def close(self) -> None:
        with self._lock:
            for resource in (
                self.article_store,
                self.sentence_splitter,
                self.cross_encoder,
                self.bmretriever_index,
                self.bmretriever_encoder,
                self.medcpt_index,
                self.medcpt_encoder,
                self.bm25,
            ):
                close = getattr(resource, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            self.bm25 = None
            self.medcpt_encoder = None
            self.medcpt_index = None
            self.bmretriever_encoder = None
            self.bmretriever_index = None
            self.cross_encoder = None
            self.sentence_splitter = None
            self.article_store = None
            self._loaded = False
