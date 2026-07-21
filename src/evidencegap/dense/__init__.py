from evidencegap.dense.article_inputs import (
    build_dense_article_inputs,
    validate_dense_article_inputs,
)
from evidencegap.dense.embeddings import (
    ShardedEmbeddingStore,
    encode_article_embeddings,
    encode_query_embeddings,
    validate_article_embeddings,
    validate_query_embeddings,
)
from evidencegap.dense.faiss_backend import (
    DenseFaissBackend,
    build_faiss_index,
    validate_faiss_index,
)
from evidencegap.dense.evaluation import (
    compare_retrieval_reports,
    query_dense_index,
    run_dense_article_retrieval,
)

__all__ = [
    "DenseFaissBackend",
    "ShardedEmbeddingStore",
    "build_dense_article_inputs",
    "build_faiss_index",
    "compare_retrieval_reports",
    "encode_article_embeddings",
    "encode_query_embeddings",
    "query_dense_index",
    "run_dense_article_retrieval",
    "validate_article_embeddings",
    "validate_dense_article_inputs",
    "validate_faiss_index",
    "validate_query_embeddings",
]
