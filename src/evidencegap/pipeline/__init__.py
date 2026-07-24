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
from evidencegap.pipeline.final_graph import (
    FINAL_GRAPH_CONTRACT_ID,
    FINAL_GRAPH_SCHEMA_VERSION,
    build_final_graph_bundle,
    run_final_graph,
    validate_final_graph_artifact,
)

__all__.extend([
    "FINAL_GRAPH_CONTRACT_ID",
    "FINAL_GRAPH_SCHEMA_VERSION",
    "build_final_graph_bundle",
    "run_final_graph",
    "validate_final_graph_artifact",
])
from evidencegap.pipeline.analysis import (
    ANALYSIS_CONTRACT_ID,
    ANALYSIS_SCHEMA_VERSION,
    run_analysis,
    validate_analysis_artifact,
)

__all__.extend([
    "ANALYSIS_CONTRACT_ID",
    "ANALYSIS_SCHEMA_VERSION",
    "run_analysis",
    "validate_analysis_artifact",
])
from evidencegap.pipeline.statement_decomposition import (
    STATEMENT_DECOMPOSITION_CONTRACT_ID,
    STATEMENT_DECOMPOSITION_PROMPT_VERSION,
    STATEMENT_DECOMPOSITION_SCHEMA_VERSION,
    run_statement_decomposition,
    validate_decomposition_bundle,
    validate_statement_decomposition_artifact,
)

__all__.extend([
    "STATEMENT_DECOMPOSITION_CONTRACT_ID",
    "STATEMENT_DECOMPOSITION_PROMPT_VERSION",
    "STATEMENT_DECOMPOSITION_SCHEMA_VERSION",
    "run_statement_decomposition",
    "validate_decomposition_bundle",
    "validate_statement_decomposition_artifact",
])

from evidencegap.pipeline.statement_analysis import (
    STATEMENT_ANALYSIS_CONTRACT_ID,
    STATEMENT_ANALYSIS_SCHEMA_VERSION,
    run_statement_analysis,
    validate_statement_analysis_artifact,
    validate_statement_analysis_bundle,
)

__all__.extend([
    "STATEMENT_ANALYSIS_CONTRACT_ID",
    "STATEMENT_ANALYSIS_SCHEMA_VERSION",
    "run_statement_analysis",
    "validate_statement_analysis_artifact",
    "validate_statement_analysis_bundle",
])

from evidencegap.pipeline.statement_bundle import (
    STATEMENT_BUNDLE_CONTRACT_ID,
    STATEMENT_BUNDLE_SCHEMA_VERSION,
    build_statement_bundle,
    run_statement_bundle,
    validate_statement_bundle,
    validate_statement_bundle_artifact,
)

__all__.extend([
    "STATEMENT_BUNDLE_CONTRACT_ID",
    "STATEMENT_BUNDLE_SCHEMA_VERSION",
    "build_statement_bundle",
    "run_statement_bundle",
    "validate_statement_bundle",
    "validate_statement_bundle_artifact",
])

from evidencegap.pipeline.statement_run import (
    STATEMENT_RUN_CONTRACT_ID,
    STATEMENT_RUN_SCHEMA_VERSION,
    run_statement_pipeline,
    validate_statement_pipeline_artifact,
)

__all__.extend([
    "STATEMENT_RUN_CONTRACT_ID",
    "STATEMENT_RUN_SCHEMA_VERSION",
    "run_statement_pipeline",
    "validate_statement_pipeline_artifact",
])
