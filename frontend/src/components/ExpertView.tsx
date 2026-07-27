import { Wrench } from 'lucide-react'
import type {
  LocalizationStatusResponse,
  PresentationBundle,
  RunStage,
  RunStatusResponse,
} from '../contracts'
import { useArticleContextQuery } from '../hooks/useArticleContextQuery'
import { UI_TEXT } from '../uiText'
import {
  getSelectionArticleNodeId,
  type GraphSelection,
  type PresentationIndexes,
} from '../utils/presentation'

interface ExpertViewProps {
  run: RunStatusResponse
  presentation: PresentationBundle
  selectedLocalization: LocalizationStatusResponse | null
  selection: GraphSelection | null
  indexes: PresentationIndexes
}

const TECHNICAL_STAGE_LABELS: Record<RunStage, string> =
  UI_TEXT.expert.stages

function displayNumber(value: unknown, maximumFractionDigits = 6): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits })
    : UI_TEXT.common.dash
}

function stageSeconds(
  executionSummary: Record<string, unknown> | null,
  stage: RunStage,
): string {
  const stages = executionSummary?.stages
  if (!stages || typeof stages !== 'object') return UI_TEXT.common.dash
  const stageValue = (stages as Record<string, unknown>)[stage]
  if (!stageValue || typeof stageValue !== 'object') {
    return UI_TEXT.common.dash
  }
  return displayNumber(
    (stageValue as Record<string, unknown>).seconds,
    3,
  )
}

function uniqueValues(values: Array<string | null>): string {
  const available = [...new Set(values.filter(
    (value): value is string => Boolean(value?.trim()),
  ))]
  return available.length > 0
    ? available.join(', ')
    : UI_TEXT.common.dash
}

export function ExpertView({
  run,
  presentation,
  selectedLocalization,
  selection,
  indexes,
}: ExpertViewProps) {
  const articleNodeId = getSelectionArticleNodeId(selection, indexes)
  const article = articleNodeId
    ? indexes.articlesById.get(articleNodeId) ?? null
    : null
  const articleContextQuery = useArticleContextQuery(
    run.run_id,
    articleNodeId,
  )
  const articleEvidence = article
    ? indexes.evidenceByArticleId.get(article.article_node_id) ?? []
    : []
  const analysisContext = presentation.analysis_context
  const totalSeconds = run.execution_summary?.total_seconds

  return (
    <details className="expert-view">
      <summary>
        <span><Wrench size={14} /> {UI_TEXT.expert.title}</span>
        <small>{UI_TEXT.expert.subtitle}</small>
      </summary>

      <section className="expert-section">
        <h3>{UI_TEXT.expert.runDetails}</h3>
        <dl className="technical-grid">
          <div><dt>{UI_TEXT.expert.fields.runId}</dt><dd>{run.run_id}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.outputLanguage}</dt><dd>{presentation.output_language || UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.totalSeconds}</dt><dd>{displayNumber(totalSeconds, 3)}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.schemaVersion}</dt><dd>{presentation.schema_version || UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.contractId}</dt><dd>{presentation.contract_id || UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.localizationId}</dt><dd>{selectedLocalization?.localization_id ?? UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.localizationTarget}</dt><dd>{selectedLocalization?.language ?? UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.localizationStatus}</dt><dd>{selectedLocalization?.status ?? UI_TEXT.common.dash}</dd></div>
        </dl>

        <h4>{UI_TEXT.expert.pipelineTimings}</h4>
        <table className="technical-table">
          <thead><tr><th>{UI_TEXT.expert.columns.stage}</th><th>{UI_TEXT.expert.columns.seconds}</th></tr></thead>
          <tbody>
            {(Object.keys(TECHNICAL_STAGE_LABELS) as RunStage[]).map((stage) => (
              <tr key={stage}>
                <td>{TECHNICAL_STAGE_LABELS[stage]}</td>
                <td>{stageSeconds(run.execution_summary, stage)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h4>{UI_TEXT.expert.analysisContext}</h4>
        <dl className="technical-grid">
          <div><dt>{UI_TEXT.expert.fields.scope}</dt><dd>{analysisContext.scope ?? UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.aggregationMethod}</dt><dd>{analysisContext.aggregation_method ?? UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.retrievalMethods}</dt><dd>{analysisContext.retrieval_methods?.join(', ') || UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.fusionMethod}</dt><dd>{analysisContext.fusion_method ?? UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.reranker}</dt><dd>{analysisContext.reranker ?? UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.sourceDepth}</dt><dd>{displayNumber(analysisContext.source_depth, 0)}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.denseNprobe}</dt><dd>{displayNumber(analysisContext.dense_nprobe, 0)}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.rrfK}</dt><dd>{displayNumber(analysisContext.rrf_k, 0)}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.rerankDepth}</dt><dd>{displayNumber(analysisContext.rerank_depth, 0)}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.articleTopK}</dt><dd>{displayNumber(analysisContext.article_top_k, 0)}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.maxEvidence}</dt><dd>{displayNumber(analysisContext.max_evidence_sentences_per_article, 0)}</dd></div>
        </dl>

        <h4>{UI_TEXT.expert.modelMetadata}</h4>
        <dl className="technical-grid">
          <div><dt>{UI_TEXT.expert.fields.providers}</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.provider))}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.models}</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.model))}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.modelFingerprints}</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.model_fingerprint))}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.promptVersions}</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.prompt_version))}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.statementSha}</dt><dd>{presentation.source_statement_bundle_sha256 || UI_TEXT.common.dash}</dd></div>
          <div><dt>{UI_TEXT.expert.fields.inferenceSha}</dt><dd>{presentation.source_inference_gap_analysis_sha256 || UI_TEXT.common.dash}</dd></div>
        </dl>
      </section>

      {article && (
        <section className="expert-section">
          <h3>{UI_TEXT.expert.articleDetails}</h3>
          <dl className="technical-grid">
            <div><dt>{UI_TEXT.expert.fields.articleNodeId}</dt><dd>{article.article_node_id}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.articleId}</dt><dd>{article.article_id || UI_TEXT.common.dash}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.pmid}</dt><dd>{article.pmid ?? UI_TEXT.common.dash}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.provider}</dt><dd>{article.provider ?? UI_TEXT.common.dash}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.model}</dt><dd>{article.model ?? UI_TEXT.common.dash}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.modelFingerprint}</dt><dd>{article.model_fingerprint ?? UI_TEXT.common.dash}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.promptVersion}</dt><dd>{article.prompt_version ?? UI_TEXT.common.dash}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.stanceConfidence}</dt><dd>{displayNumber(article.confidence)}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.evidenceCount}</dt><dd>{articleEvidence.length}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.canonicalFingerprint}</dt><dd>{articleContextQuery.data?.source_text_fingerprint ?? UI_TEXT.common.dash}</dd></div>
          </dl>

          <h4>{UI_TEXT.expert.retrievalTrace}</h4>
          <table className="technical-table">
            <thead><tr><th>{UI_TEXT.expert.columns.method}</th><th>{UI_TEXT.expert.columns.rank}</th><th>{UI_TEXT.expert.columns.score}</th></tr></thead>
            <tbody>
              <tr><td>{UI_TEXT.expert.methods.bm25}</td><td>{displayNumber(article.retrieval_trace.bm25.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.bm25.score)}</td></tr>
              <tr><td>{UI_TEXT.expert.methods.medcpt}</td><td>{displayNumber(article.retrieval_trace.medcpt.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.medcpt.score)}</td></tr>
              <tr><td>{UI_TEXT.expert.methods.bmretriever}</td><td>{displayNumber(article.retrieval_trace.bmretriever.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.bmretriever.score)}</td></tr>
              <tr><td>{UI_TEXT.expert.methods.rrf}</td><td>{displayNumber(article.retrieval_trace.fusion.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.fusion.rrf_score)}</td></tr>
              <tr><td>{UI_TEXT.expert.methods.crossEncoder}</td><td>{UI_TEXT.common.dash}</td><td>{displayNumber(article.retrieval_trace.cross_encoder.score)}</td></tr>
              <tr><td>{UI_TEXT.expert.methods.finalRank}</td><td>{displayNumber(article.retrieval_trace.final_article_rank, 0)}</td><td>{UI_TEXT.common.dash}</td></tr>
            </tbody>
          </table>
          <p className="technical-note">
            {UI_TEXT.expert.scoreNote}
          </p>

          <h4>{UI_TEXT.expert.evidenceProvenance}</h4>
          <dl className="technical-grid">
            <div><dt>{UI_TEXT.expert.fields.evidenceIds}</dt><dd>{articleEvidence.length > 0 ? articleEvidence.map((evidence) => evidence.evidence_id).join(', ') : UI_TEXT.common.dash}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.sourceFingerprints}</dt><dd>{uniqueValues(articleEvidence.map((evidence) => evidence.source_text_fingerprint))}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.splitterFingerprints}</dt><dd>{uniqueValues(articleEvidence.map((evidence) => evidence.splitter_fingerprint))}</dd></div>
            <div><dt>{UI_TEXT.expert.fields.fingerprintVerified}</dt><dd>{articleContextQuery.data ? articleContextQuery.data.fingerprint_verified ? UI_TEXT.common.yes : UI_TEXT.common.no : UI_TEXT.common.dash}</dd></div>
          </dl>
        </section>
      )}
    </details>
  )
}
