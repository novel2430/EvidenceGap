import { useMemo } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  ExternalLink,
  FileText,
  Fingerprint,
  RefreshCw,
} from 'lucide-react'
import type {
  PresentationArticle,
  PresentationEvidence,
  ArticleEvidenceSpanResponse,
} from '../contracts'
import { useArticleContextQuery } from '../hooks/useArticleContextQuery'
import { getApiErrorMessage } from '../utils/format'
import {
  APPLICABILITY_DIMENSIONS,
  formatApplicabilityDimension,
  formatEnumLabel,
  getApplicabilityStatusClassName,
  getApplicabilityStatusLabel,
} from '../utils/presentationLabels'
import type {
  GraphSelection,
  PresentationIndexes,
} from '../utils/presentation'
import { ArticleCanonicalText } from './ArticleCanonicalText'

interface ArticleInspectorProps {
  runId: string
  selection: Extract<GraphSelection, { kind: 'article' | 'evidence' }>
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}

function formatConfidence(confidence: number): string {
  if (!Number.isFinite(confidence)) return '—'
  return new Intl.NumberFormat(undefined, {
    style: 'percent',
    maximumFractionDigits: 1,
  }).format(confidence)
}

function shortEvidenceId(evidenceId: string): string {
  if (evidenceId.length <= 18) return evidenceId
  return `${evidenceId.slice(0, 8)}…${evidenceId.slice(-7)}`
}

function sectionName(
  evidence: PresentationEvidence,
  contextSpan?: ArticleEvidenceSpanResponse,
): string {
  return (
    evidence.section?.trim() ||
    contextSpan?.section?.trim() ||
    'Section unavailable'
  )
}

function ArticleMetadata({
  article,
}: {
  article: PresentationArticle
}) {
  return (
    <section className={`article-overview-card article-overview-card--${article.stance} is-selected`}>
      <div className="article-overview-heading">
        <span>Rank {Number.isFinite(article.rank) ? article.rank : '—'}</span>
        <span className={`article-stance article-stance--${article.stance}`}>
          {article.stance}
        </span>
      </div>
      <h3>{article.display_title || article.title || '—'}</h3>
      <dl className="article-metadata-grid">
        <div><dt>PMID</dt><dd>{article.pmid ?? '—'}</dd></div>
        <div><dt>Stance confidence</dt><dd>{formatConfidence(article.confidence)}</dd></div>
        <div><dt>Evidence sentences</dt><dd>{article.evidence_ids.length}</dd></div>
      </dl>
      <div className="article-rationale">
        <strong>Rationale</strong>
        <p>{article.display_rationale || article.rationale || '—'}</p>
      </div>
    </section>
  )
}

function ArticleApplicabilitySection({
  article,
}: {
  article: PresentationArticle
}) {
  const applicability = article.applicability
  const issues = article.applicability_issues

  return (
    <section className="article-applicability-section">
      <div className="inspector-section-heading">
        <h3>Article Applicability</h3>
        {applicability && <span>8 dimensions</span>}
      </div>
      <p className="applicability-description">
        How directly this article matches the exact claim.
      </p>

      {!applicability ? (
        <p className="applicability-unavailable">
          Article applicability unavailable for this Run.
        </p>
      ) : (
        <>
          <dl className="applicability-matrix">
            {APPLICABILITY_DIMENSIONS.map((dimension) => {
              const status = applicability[dimension]
              return (
                <div key={dimension}>
                  <dt>{formatApplicabilityDimension(dimension)}</dt>
                  <dd>
                    <span className={`applicability-status applicability-status--${getApplicabilityStatusClassName(status)}`}>
                      {getApplicabilityStatusLabel(status)}
                    </span>
                  </dd>
                </div>
              )
            })}
          </dl>

          <section className="applicability-issues">
            <h4>Applicability Issues</h4>
            {issues === undefined ? (
              <p className="applicability-issues-empty">
                Applicability issues unavailable for this Run.
              </p>
            ) : issues.length === 0 ? (
              <p className="applicability-issues-empty">
                No explicit applicability issues reported.
              </p>
            ) : (
              <div className="applicability-issue-list">
                {issues.map((issue, issueIndex) => (
                  <article
                    className="applicability-issue"
                    key={`${issue.dimension}-${issue.code}-${issueIndex}`}
                  >
                    <div>
                      <strong>
                        {formatApplicabilityDimension(issue.dimension)}
                      </strong>
                      <span>{formatEnumLabel(issue.code).toUpperCase()}</span>
                    </div>
                    <p>{issue.reason}</p>
                  </article>
                ))}
              </div>
            )}
          </section>
        </>
      )}
    </section>
  )
}

export function ArticleInspector({
  runId,
  selection,
  indexes,
  onSelect,
}: ArticleInspectorProps) {
  const selectedEvidence =
    selection.kind === 'evidence'
      ? indexes.evidenceById.get(selection.evidenceId) ?? null
      : null
  const articleNodeId =
    selection.kind === 'article'
      ? selection.articleNodeId
      : selectedEvidence?.article_node_id ?? null
  const article = articleNodeId
    ? indexes.articlesById.get(articleNodeId) ?? null
    : null
  const claim = article
    ? indexes.claimsById.get(article.claim_id) ?? null
    : null
  const evidenceItems = article
    ? indexes.evidenceByArticleId.get(article.article_node_id) ?? []
    : []
  const contextQuery = useArticleContextQuery(runId, articleNodeId)
  const contextSpansByEvidenceId = useMemo(
    () => new Map(
      contextQuery.data?.evidence_spans.map((span) => [
        span.evidence_id,
        span,
      ]) ?? [],
    ),
    [contextQuery.data],
  )

  if (!article || !claim) return null

  const activeEvidenceId =
    selection.kind === 'evidence' ? selection.evidenceId : null
  const pubmedId = article.pmid ?? contextQuery.data?.pmid ?? null
  const activeContextSpan = selectedEvidence
    ? contextSpansByEvidenceId.get(selectedEvidence.evidence_id)
    : undefined

  return (
    <section className="selection-card article-inspector">
      <nav className="inspector-breadcrumb" aria-label="Inspector breadcrumb">
        <button
          type="button"
          onClick={() => onSelect({ kind: 'claim', claimId: claim.claim_id })}
        >
          Claim
        </button>
        <span>›</span>
        {selection.kind === 'evidence' ? (
          <button
            type="button"
            onClick={() => onSelect({
              kind: 'article',
              articleNodeId: article.article_node_id,
            })}
          >
            Article
          </button>
        ) : (
          <strong>Article</strong>
        )}
        {selection.kind === 'evidence' && (
          <>
            <span>›</span>
            <strong>Evidence</strong>
          </>
        )}
      </nav>

      <ArticleMetadata article={article} />
      <ArticleApplicabilitySection article={article} />

      {pubmedId && (
        <a
          className="pubmed-link"
          href={`https://pubmed.ncbi.nlm.nih.gov/${encodeURIComponent(pubmedId)}/`}
          target="_blank"
          rel="noreferrer noopener"
        >
          <ExternalLink size={13} />
          Open in PubMed
        </a>
      )}

      {selectedEvidence && (
        <section className="active-evidence-summary">
          <span className="eyebrow">Evidence selection</span>
          <strong>{selectedEvidence.display_text || selectedEvidence.text}</strong>
          <dl>
            <div><dt>Section</dt><dd>{sectionName(selectedEvidence, activeContextSpan)}</dd></div>
            <div><dt>Sentence index</dt><dd>{selectedEvidence.sentence_index}</dd></div>
            <div><dt>Evidence ID</dt><dd title={selectedEvidence.evidence_id}>{shortEvidenceId(selectedEvidence.evidence_id)}</dd></div>
          </dl>
          {contextQuery.isSuccess && !activeContextSpan && (
            <p className="span-unavailable">
              No canonical-text span is available for this evidence sentence.
            </p>
          )}
        </section>
      )}

      <section className="article-evidence-list-section">
        <div className="inspector-section-heading">
          <h3>Evidence Sentences</h3>
          <span>{evidenceItems.length}</span>
        </div>
        <div className="article-evidence-list">
          {evidenceItems.map((evidence) => {
            const isSelected = activeEvidenceId === evidence.evidence_id
            const isUnlocated =
              contextQuery.isSuccess &&
              !contextSpansByEvidenceId.has(evidence.evidence_id)
            const contextSpan = contextSpansByEvidenceId.get(evidence.evidence_id)
            return (
              <button
                className={`article-evidence-item${isSelected ? ' is-selected' : ''}`}
                type="button"
                key={evidence.evidence_id}
                onClick={() => onSelect({
                  kind: 'evidence',
                  evidenceId: evidence.evidence_id,
                })}
              >
                <span className="evidence-item-heading">
                  <span><FileText size={11} /> {sectionName(evidence, contextSpan)}</span>
                  <span>Sentence {evidence.sentence_index}</span>
                </span>
                <strong>{evidence.display_text || evidence.text}</strong>
                <small title={evidence.evidence_id}>
                  {shortEvidenceId(evidence.evidence_id)}
                  {isUnlocated ? ' · Span unavailable' : ''}
                </small>
              </button>
            )
          })}
          {evidenceItems.length === 0 && (
            <p className="reference-empty">No evidence sentences were linked to this Article.</p>
          )}
        </div>
      </section>

      <section className="article-context-section">
        <div className="inspector-section-heading">
          <h3>Canonical Article Text</h3>
          {contextQuery.isSuccess && (
            <span>{contextQuery.data.sections.length} sections</span>
          )}
        </div>

        {contextQuery.isPending && (
          <div className="article-context-loading" role="status">
            <span className="context-skeleton" />
            <span className="context-skeleton context-skeleton--short" />
            <span className="context-skeleton" />
            <small>Loading Article Context…</small>
          </div>
        )}

        {contextQuery.isError && (
          <div className="article-context-error" role="alert">
            <AlertTriangle size={17} />
            <strong>Article Context could not be loaded</strong>
            <p>{getApiErrorMessage(contextQuery.error)}</p>
            <button
              className="secondary-button"
              type="button"
              onClick={() => void contextQuery.refetch()}
            >
              <RefreshCw size={13} /> Retry
            </button>
          </div>
        )}

        {contextQuery.isSuccess && (
          <>
            <div className={`fingerprint-status fingerprint-status--${typeof contextQuery.data.fingerprint_verified !== 'boolean' ? 'unavailable' : contextQuery.data.fingerprint_verified ? 'verified' : 'unverified'}`}>
              {contextQuery.data.fingerprint_verified === true ? (
                <CheckCircle2 size={14} />
              ) : (
                <Fingerprint size={14} />
              )}
              <div>
                <strong>
                  {typeof contextQuery.data.fingerprint_verified !== 'boolean'
                    ? 'Verification unavailable'
                    : contextQuery.data.fingerprint_verified
                    ? 'Fingerprint verified'
                    : 'Fingerprint not verified'}
                </strong>
                <p>
                  {typeof contextQuery.data.fingerprint_verified !== 'boolean'
                    ? 'The backend did not provide fingerprint verification status.'
                    : contextQuery.data.fingerprint_verified
                    ? 'The displayed canonical text matches the evidence artifact used by the analysis.'
                    : 'Cross-version fingerprint verification was unavailable; this does not by itself mean the article text is incorrect.'}
                </p>
              </div>
            </div>
            <ArticleCanonicalText
              context={contextQuery.data}
              activeEvidenceId={activeEvidenceId}
              onSelectEvidence={(evidenceId) =>
                onSelect({ kind: 'evidence', evidenceId })}
            />
          </>
        )}
      </section>
    </section>
  )
}
