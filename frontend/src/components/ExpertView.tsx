import { Wrench } from 'lucide-react'
import type {
  LocalizationStatusResponse,
  PresentationBundle,
  RunStage,
  RunStatusResponse,
} from '../contracts'
import { useArticleContextQuery } from '../hooks/useArticleContextQuery'
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

const TECHNICAL_STAGE_LABELS: Record<RunStage, string> = {
  statement_decomposition: 'Claim decomposition',
  claim_analysis: 'Claim analysis',
  statement_bundle: 'Statement bundle',
  inference_gap_analysis: 'Inference gap analysis',
  output_generation: 'Output generation',
}

function displayNumber(value: unknown, maximumFractionDigits = 6): string {
  return typeof value === 'number' && Number.isFinite(value)
    ? value.toLocaleString(undefined, { maximumFractionDigits })
    : '—'
}

function stageSeconds(
  executionSummary: Record<string, unknown> | null,
  stage: RunStage,
): string {
  const stages = executionSummary?.stages
  if (!stages || typeof stages !== 'object') return '—'
  const stageValue = (stages as Record<string, unknown>)[stage]
  if (!stageValue || typeof stageValue !== 'object') return '—'
  return displayNumber(
    (stageValue as Record<string, unknown>).seconds,
    3,
  )
}

function uniqueValues(values: Array<string | null>): string {
  const available = [...new Set(values.filter(
    (value): value is string => Boolean(value?.trim()),
  ))]
  return available.length > 0 ? available.join(', ') : '—'
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
        <span><Wrench size={14} /> Expert View</span>
        <small>Technical metadata</small>
      </summary>

      <section className="expert-section">
        <h3>Run-level Technical Details</h3>
        <dl className="technical-grid">
          <div><dt>Run ID</dt><dd>{run.run_id}</dd></div>
          <div><dt>Output language</dt><dd>{presentation.output_language || '—'}</dd></div>
          <div><dt>Total execution time (seconds)</dt><dd>{displayNumber(totalSeconds, 3)}</dd></div>
          <div><dt>Schema version</dt><dd>{presentation.schema_version || '—'}</dd></div>
          <div><dt>Contract ID</dt><dd>{presentation.contract_id || '—'}</dd></div>
          <div><dt>Localization ID</dt><dd>{selectedLocalization?.localization_id ?? '—'}</dd></div>
          <div><dt>Localization target</dt><dd>{selectedLocalization?.language ?? '—'}</dd></div>
          <div><dt>Localization status</dt><dd>{selectedLocalization?.status ?? '—'}</dd></div>
        </dl>

        <h4>Pipeline stage timings</h4>
        <table className="technical-table">
          <thead><tr><th>Stage</th><th>Seconds</th></tr></thead>
          <tbody>
            {(Object.keys(TECHNICAL_STAGE_LABELS) as RunStage[]).map((stage) => (
              <tr key={stage}>
                <td>{TECHNICAL_STAGE_LABELS[stage]}</td>
                <td>{stageSeconds(run.execution_summary, stage)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        <h4>Analysis context</h4>
        <dl className="technical-grid">
          <div><dt>Scope</dt><dd>{analysisContext.scope ?? '—'}</dd></div>
          <div><dt>Aggregation method</dt><dd>{analysisContext.aggregation_method ?? '—'}</dd></div>
          <div><dt>Retrieval methods</dt><dd>{analysisContext.retrieval_methods?.join(', ') || '—'}</dd></div>
          <div><dt>Fusion method</dt><dd>{analysisContext.fusion_method ?? '—'}</dd></div>
          <div><dt>Reranker</dt><dd>{analysisContext.reranker ?? '—'}</dd></div>
          <div><dt>Source depth</dt><dd>{displayNumber(analysisContext.source_depth, 0)}</dd></div>
          <div><dt>Dense nprobe</dt><dd>{displayNumber(analysisContext.dense_nprobe, 0)}</dd></div>
          <div><dt>RRF k</dt><dd>{displayNumber(analysisContext.rrf_k, 0)}</dd></div>
          <div><dt>Rerank depth</dt><dd>{displayNumber(analysisContext.rerank_depth, 0)}</dd></div>
          <div><dt>Article top k</dt><dd>{displayNumber(analysisContext.article_top_k, 0)}</dd></div>
          <div><dt>Max evidence sentences / article</dt><dd>{displayNumber(analysisContext.max_evidence_sentences_per_article, 0)}</dd></div>
        </dl>

        <h4>Model and artifact metadata</h4>
        <dl className="technical-grid">
          <div><dt>Providers</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.provider))}</dd></div>
          <div><dt>Models</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.model))}</dd></div>
          <div><dt>Model fingerprints</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.model_fingerprint))}</dd></div>
          <div><dt>Prompt versions</dt><dd>{uniqueValues(presentation.articles.map((articleItem) => articleItem.prompt_version))}</dd></div>
          <div><dt>Statement bundle SHA-256</dt><dd>{presentation.source_statement_bundle_sha256 || '—'}</dd></div>
          <div><dt>Inference gap bundle SHA-256</dt><dd>{presentation.source_inference_gap_analysis_sha256 || '—'}</dd></div>
        </dl>
      </section>

      {article && (
        <section className="expert-section">
          <h3>Article-level Technical Details</h3>
          <dl className="technical-grid">
            <div><dt>Article node ID</dt><dd>{article.article_node_id}</dd></div>
            <div><dt>Article ID</dt><dd>{article.article_id || '—'}</dd></div>
            <div><dt>PMID</dt><dd>{article.pmid ?? '—'}</dd></div>
            <div><dt>Provider</dt><dd>{article.provider ?? '—'}</dd></div>
            <div><dt>Model</dt><dd>{article.model ?? '—'}</dd></div>
            <div><dt>Model fingerprint</dt><dd>{article.model_fingerprint ?? '—'}</dd></div>
            <div><dt>Prompt version</dt><dd>{article.prompt_version ?? '—'}</dd></div>
            <div><dt>Stance confidence</dt><dd>{displayNumber(article.confidence)}</dd></div>
            <div><dt>Evidence count</dt><dd>{articleEvidence.length}</dd></div>
            <div><dt>Canonical text fingerprint</dt><dd>{articleContextQuery.data?.source_text_fingerprint ?? '—'}</dd></div>
          </dl>

          <h4>Retrieval and reranking trace</h4>
          <table className="technical-table">
            <thead><tr><th>Method</th><th>Rank</th><th>Score</th></tr></thead>
            <tbody>
              <tr><td>BM25 retrieval</td><td>{displayNumber(article.retrieval_trace.bm25.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.bm25.score)}</td></tr>
              <tr><td>MedCPT retrieval</td><td>{displayNumber(article.retrieval_trace.medcpt.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.medcpt.score)}</td></tr>
              <tr><td>BMRetriever retrieval</td><td>{displayNumber(article.retrieval_trace.bmretriever.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.bmretriever.score)}</td></tr>
              <tr><td>RRF fusion</td><td>{displayNumber(article.retrieval_trace.fusion.rank, 0)}</td><td>{displayNumber(article.retrieval_trace.fusion.rrf_score)}</td></tr>
              <tr><td>Cross-Encoder reranking</td><td>—</td><td>{displayNumber(article.retrieval_trace.cross_encoder.score)}</td></tr>
              <tr><td>Final article rank</td><td>{displayNumber(article.retrieval_trace.final_article_rank, 0)}</td><td>—</td></tr>
            </tbody>
          </table>
          <p className="technical-note">
            Retrieval and reranking scores are backend-provided method-specific values and are not combined or interpreted as stance confidence.
          </p>

          <h4>Evidence provenance</h4>
          <dl className="technical-grid">
            <div><dt>Evidence IDs</dt><dd>{articleEvidence.length > 0 ? articleEvidence.map((evidence) => evidence.evidence_id).join(', ') : '—'}</dd></div>
            <div><dt>Source text fingerprints</dt><dd>{uniqueValues(articleEvidence.map((evidence) => evidence.source_text_fingerprint))}</dd></div>
            <div><dt>Splitter fingerprints</dt><dd>{uniqueValues(articleEvidence.map((evidence) => evidence.splitter_fingerprint))}</dd></div>
            <div><dt>Fingerprint verified</dt><dd>{articleContextQuery.data ? articleContextQuery.data.fingerprint_verified ? 'Yes' : 'No' : '—'}</dd></div>
          </dl>
        </section>
      )}
    </details>
  )
}
