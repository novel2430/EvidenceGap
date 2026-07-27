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
import { UI_TEXT } from '../uiText'
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
  if (!Number.isFinite(confidence)) return UI_TEXT.common.dash
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
    UI_TEXT.common.sectionUnavailable
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
        <span>
          {UI_TEXT.articles.rank(
            Number.isFinite(article.rank)
              ? article.rank
              : UI_TEXT.common.dash,
          )}
        </span>
        <span className={`article-stance article-stance--${article.stance}`}>
          {UI_TEXT.articles.groups[article.stance]}
        </span>
      </div>
      <h3>
        {article.display_title || article.title || UI_TEXT.common.dash}
      </h3>
      <dl className="article-metadata-grid">
        <div><dt>{UI_TEXT.articles.pmid}</dt><dd>{article.pmid ?? UI_TEXT.common.dash}</dd></div>
        <div><dt>{UI_TEXT.articles.stanceConfidence}</dt><dd>{formatConfidence(article.confidence)}</dd></div>
        <div><dt>{UI_TEXT.articles.evidenceSentences}</dt><dd>{article.evidence_ids.length}</dd></div>
      </dl>
      <div className="article-rationale">
        <strong>{UI_TEXT.articles.rationale}</strong>
        <p>
          {article.display_rationale ||
            article.rationale ||
            UI_TEXT.common.dash}
        </p>
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
        <h3>{UI_TEXT.articles.applicability.title}</h3>
        {applicability && (
          <span>{UI_TEXT.articles.applicability.dimensions}</span>
        )}
      </div>
      <p className="applicability-description">
        {UI_TEXT.articles.applicability.description}
      </p>

      {!applicability ? (
        <p className="applicability-unavailable">
          {UI_TEXT.articles.applicability.unavailable}
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
            <h4>{UI_TEXT.articles.applicability.issuesTitle}</h4>
            {issues === undefined ? (
              <p className="applicability-issues-empty">
                {UI_TEXT.articles.applicability.issuesUnavailable}
              </p>
            ) : issues.length === 0 ? (
              <p className="applicability-issues-empty">
                {UI_TEXT.articles.applicability.noIssues}
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
      <nav
        className="inspector-breadcrumb"
        aria-label={UI_TEXT.articles.breadcrumbLabel}
      >
        <button
          type="button"
          onClick={() => onSelect({ kind: 'claim', claimId: claim.claim_id })}
        >
          {UI_TEXT.common.claim}
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
            {UI_TEXT.common.article}
          </button>
        ) : (
          <strong>{UI_TEXT.common.article}</strong>
        )}
        {selection.kind === 'evidence' && (
          <>
            <span>›</span>
            <strong>{UI_TEXT.common.evidence}</strong>
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
          {UI_TEXT.articles.openPubMed}
        </a>
      )}

      {selectedEvidence && (
        <section className="active-evidence-summary">
          <span className="eyebrow">{UI_TEXT.articles.evidenceSelection}</span>
          <strong>{selectedEvidence.display_text || selectedEvidence.text}</strong>
          <dl>
            <div><dt>{UI_TEXT.articles.section}</dt><dd>{sectionName(selectedEvidence, activeContextSpan)}</dd></div>
            <div><dt>{UI_TEXT.articles.sentenceIndex}</dt><dd>{selectedEvidence.sentence_index}</dd></div>
            <div><dt>{UI_TEXT.articles.evidenceId}</dt><dd title={selectedEvidence.evidence_id}>{shortEvidenceId(selectedEvidence.evidence_id)}</dd></div>
          </dl>
          {contextQuery.isSuccess && !activeContextSpan && (
            <p className="span-unavailable">
              {UI_TEXT.articles.noCanonicalSpan}
            </p>
          )}
        </section>
      )}

      <section className="article-evidence-list-section">
        <div className="inspector-section-heading">
          <h3>{UI_TEXT.articles.evidenceSentences}</h3>
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
                  <span>{UI_TEXT.articles.sentence(evidence.sentence_index)}</span>
                </span>
                <strong>{evidence.display_text || evidence.text}</strong>
                <small title={evidence.evidence_id}>
                  {isUnlocated
                    ? UI_TEXT.articles.unlocatedEvidence(
                        shortEvidenceId(evidence.evidence_id),
                      )
                    : shortEvidenceId(evidence.evidence_id)}
                </small>
              </button>
            )
          })}
          {evidenceItems.length === 0 && (
            <p className="reference-empty">{UI_TEXT.articles.noEvidence}</p>
          )}
        </div>
      </section>

      <section className="article-context-section">
        <div className="inspector-section-heading">
          <h3>{UI_TEXT.articles.canonicalText}</h3>
          {contextQuery.isSuccess && (
            <span>
              {UI_TEXT.articles.sectionCount(
                contextQuery.data.sections.length,
              )}
            </span>
          )}
        </div>

        {contextQuery.isPending && (
          <div className="article-context-loading" role="status">
            <span className="context-skeleton" />
            <span className="context-skeleton context-skeleton--short" />
            <span className="context-skeleton" />
            <small>{UI_TEXT.articles.loadingContext}</small>
          </div>
        )}

        {contextQuery.isError && (
          <div className="article-context-error" role="alert">
            <AlertTriangle size={17} />
            <strong>{UI_TEXT.articles.contextLoadFailed}</strong>
            <p>{getApiErrorMessage(contextQuery.error)}</p>
            <button
              className="secondary-button"
              type="button"
              onClick={() => void contextQuery.refetch()}
            >
              <RefreshCw size={13} /> {UI_TEXT.common.retry}
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
                    ? UI_TEXT.articles.fingerprint.unavailable
                    : contextQuery.data.fingerprint_verified
                    ? UI_TEXT.articles.fingerprint.verified
                    : UI_TEXT.articles.fingerprint.unverified}
                </strong>
                <p>
                  {typeof contextQuery.data.fingerprint_verified !== 'boolean'
                    ? UI_TEXT.articles.fingerprint.unavailableDescription
                    : contextQuery.data.fingerprint_verified
                    ? UI_TEXT.articles.fingerprint.verifiedDescription
                    : UI_TEXT.articles.fingerprint.unverifiedDescription}
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
