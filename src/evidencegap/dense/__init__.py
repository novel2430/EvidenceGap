"""Dense retrieval public API with lazy imports.

Keeping this module dependency-light lets CLI help and sentence-level contracts load
without importing PyArrow/FAISS until a dense artifact operation is invoked.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "build_dense_article_inputs": (
        "evidencegap.dense.article_inputs",
        "build_dense_article_inputs",
    ),
    "validate_dense_article_inputs": (
        "evidencegap.dense.article_inputs",
        "validate_dense_article_inputs",
    ),
    "ShardedEmbeddingStore": (
        "evidencegap.dense.embeddings",
        "ShardedEmbeddingStore",
    ),
    "encode_article_embeddings": (
        "evidencegap.dense.embeddings",
        "encode_article_embeddings",
    ),
    "encode_query_embeddings": (
        "evidencegap.dense.embeddings",
        "encode_query_embeddings",
    ),
    "validate_article_embeddings": (
        "evidencegap.dense.embeddings",
        "validate_article_embeddings",
    ),
    "validate_query_embeddings": (
        "evidencegap.dense.embeddings",
        "validate_query_embeddings",
    ),
    "DenseFaissBackend": (
        "evidencegap.dense.faiss_backend",
        "DenseFaissBackend",
    ),
    "build_faiss_index": (
        "evidencegap.dense.faiss_backend",
        "build_faiss_index",
    ),
    "validate_faiss_index": (
        "evidencegap.dense.faiss_backend",
        "validate_faiss_index",
    ),
    "compare_retrieval_reports": (
        "evidencegap.dense.evaluation",
        "compare_retrieval_reports",
    ),
    "query_dense_index": (
        "evidencegap.dense.evaluation",
        "query_dense_index",
    ),
    "run_dense_article_retrieval": (
        "evidencegap.dense.evaluation",
        "run_dense_article_retrieval",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
