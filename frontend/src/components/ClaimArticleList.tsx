import { useState } from 'react'
import { ChevronDown, ChevronRight, FileText } from 'lucide-react'
import type {
  ArticleStance,
  PresentationArticle,
  PresentationClaim,
} from '../contracts'
import {
  getSelectionArticleNodeId,
  type GraphSelection,
  type PresentationIndexes,
} from '../utils/presentation'

interface ClaimArticleListProps {
  claim: PresentationClaim
  indexes: PresentationIndexes
  selection: GraphSelection | null
  onSelect: (selection: GraphSelection) => void
}

const ARTICLE_GROUPS: Array<{
  stance: ArticleStance
  label: string
}> = [
  { stance: 'support', label: 'Supporting' },
  { stance: 'refute', label: 'Refuting' },
  { stance: 'insufficient', label: 'Insufficient' },
]

function formatConfidence(confidence: number): string {
  if (!Number.isFinite(confidence)) return '—'
  return new Intl.NumberFormat(undefined, {
    style: 'percent',
    maximumFractionDigits: 1,
  }).format(confidence)
}

function ArticleListCard({
  article,
  isSelected,
  onSelect,
}: {
  article: PresentationArticle
  isSelected: boolean
  onSelect: () => void
}) {
  const title = article.display_title || article.title || '—'
  const rationale = article.display_rationale || article.rationale || '—'

  return (
    <button
      className={`article-list-card article-list-card--${article.stance}${isSelected ? ' is-selected' : ''}`}
      type="button"
      onClick={onSelect}
    >
      <span className="article-list-card-heading">
        <span>Rank {Number.isFinite(article.rank) ? article.rank : '—'}</span>
        <span className={`article-stance article-stance--${article.stance}`}>
          {article.stance}
        </span>
      </span>
      <strong>{title}</strong>
      <span className="article-card-metadata">
        PMID {article.pmid ?? '—'} · Confidence {formatConfidence(article.confidence)}
      </span>
      <span className="article-card-rationale">{rationale}</span>
      <span className="article-evidence-count">
        <FileText size={11} />
        {article.evidence_ids.length} evidence sentence{article.evidence_ids.length === 1 ? '' : 's'}
      </span>
    </button>
  )
}

export function ClaimArticleList({
  claim,
  indexes,
  selection,
  onSelect,
}: ClaimArticleListProps) {
  const [expandedGroups, setExpandedGroups] = useState<
    Record<ArticleStance, boolean>
  >({
    support: true,
    refute: true,
    insufficient: true,
  })
  const articles = indexes.articlesByClaimId.get(claim.claim_id) ?? []
  const activeArticleNodeId = getSelectionArticleNodeId(selection, indexes)

  function toggleGroup(stance: ArticleStance) {
    setExpandedGroups((current) => ({
      ...current,
      [stance]: !current[stance],
    }))
  }

  return (
    <section className="claim-articles">
      <div className="inspector-section-heading">
        <h3>Ranked Articles</h3>
        <span>{articles.length}</span>
      </div>
      <p className="article-list-boundary">Retrieved Top Articles for this Claim</p>
      {ARTICLE_GROUPS.map(({ stance, label }) => {
        const groupedArticles = articles.filter(
          (article) => article.stance === stance,
        )
        if (groupedArticles.length === 0) return null
        const isExpanded = expandedGroups[stance]

        return (
          <div className="article-group" key={stance}>
            <button
              className="article-group-toggle"
              type="button"
              onClick={() => toggleGroup(stance)}
              aria-expanded={isExpanded}
            >
              {isExpanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              <span>{label}</span>
              <strong>{groupedArticles.length}</strong>
            </button>
            {isExpanded && (
              <div className="article-group-list">
                {groupedArticles.map((article) => (
                  <ArticleListCard
                    article={article}
                    isSelected={activeArticleNodeId === article.article_node_id}
                    key={article.article_node_id}
                    onSelect={() => onSelect({
                      kind: 'article',
                      articleNodeId: article.article_node_id,
                    })}
                  />
                ))}
              </div>
            )}
          </div>
        )
      })}
      {articles.length === 0 && (
        <p className="reference-empty">No ranked articles were provided for this Claim.</p>
      )}
    </section>
  )
}
