from evidencegap.pipeline.contracts import (
    RUNTIME_SENTENCE_CONTRACT_ID,
    RUNTIME_SENTENCE_SCHEMA_VERSION,
    ArticleSection,
    RuntimeArticle,
    RuntimeSentence,
)
from evidencegap.pipeline.sentence_materialization import (
    StanzaSentenceSplitter,
    canonicalize_article_text,
    check_stanza_runtime,
    download_stanza_sentence_model,
    load_runtime_articles,
    materialize_article_sentences,
    materialize_runtime_sentences,
    validate_runtime_sentence_artifact,
    validate_runtime_sentence_rows,
)

__all__ = [
    "RUNTIME_SENTENCE_CONTRACT_ID",
    "RUNTIME_SENTENCE_SCHEMA_VERSION",
    "ArticleSection",
    "RuntimeArticle",
    "RuntimeSentence",
    "StanzaSentenceSplitter",
    "canonicalize_article_text",
    "check_stanza_runtime",
    "download_stanza_sentence_model",
    "load_runtime_articles",
    "materialize_article_sentences",
    "materialize_runtime_sentences",
    "validate_runtime_sentence_artifact",
    "validate_runtime_sentence_rows",
]

from evidencegap.pipeline.retrieval_adapters import (
    RUNTIME_RETRIEVAL_CONTRACT_ID,
    RUNTIME_RETRIEVAL_SCHEMA_VERSION,
    fuse_article_rankings,
    fuse_sentence_rankings,
    retrieve_runtime_articles,
    retrieve_runtime_evidence,
    run_retrieval_adapters,
    runtime_claim_id,
    validate_retrieval_adapter_artifact,
    validate_runtime_evidence_rows,
)

__all__.extend([
    "RUNTIME_RETRIEVAL_CONTRACT_ID",
    "RUNTIME_RETRIEVAL_SCHEMA_VERSION",
    "fuse_article_rankings",
    "fuse_sentence_rankings",
    "retrieve_runtime_articles",
    "retrieve_runtime_evidence",
    "run_retrieval_adapters",
    "runtime_claim_id",
    "validate_retrieval_adapter_artifact",
    "validate_runtime_evidence_rows",
])

from evidencegap.pipeline.article_evidence import (
    ARTICLE_EVIDENCE_CONTRACT_ID,
    ARTICLE_EVIDENCE_SCHEMA_VERSION,
    ARTICLE_EVIDENCE_PROMPT_VERSION,
    load_article_prompt_inputs,
    run_article_evidence_extractor,
    validate_article_evidence_artifact,
    validate_article_evidence_rows,
)

__all__.extend([
    "ARTICLE_EVIDENCE_CONTRACT_ID",
    "ARTICLE_EVIDENCE_SCHEMA_VERSION",
    "ARTICLE_EVIDENCE_PROMPT_VERSION",
    "load_article_prompt_inputs",
    "run_article_evidence_extractor",
    "validate_article_evidence_artifact",
    "validate_article_evidence_rows",
])

from evidencegap.pipeline.claim_aggregation import (
    CLAIM_AGGREGATION_CONTRACT_ID,
    CLAIM_AGGREGATION_SCHEMA_VERSION,
    aggregate_article_evidence_rows,
    run_claim_aggregation,
    validate_claim_aggregation_artifact,
)

__all__.extend([
    "CLAIM_AGGREGATION_CONTRACT_ID",
    "CLAIM_AGGREGATION_SCHEMA_VERSION",
    "aggregate_article_evidence_rows",
    "run_claim_aggregation",
    "validate_claim_aggregation_artifact",
])
