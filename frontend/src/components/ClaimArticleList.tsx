import { useState } from 'react'
import { ChevronDown, ChevronRight, FileText } from 'lucide-react'
import type {
  ArticleStance,
  PresentationArticle,
  PresentationClaim,
} from '../contracts'
import { UI_TEXT } from '../uiText'
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
  { stance: 'support', label: UI_TEXT.articles.groups.support },
  { stance: 'refute', label: UI_TEXT.articles.groups.refute },
  { stance: 'insufficient', label: UI_TEXT.articles.groups.insufficient },
]

function formatConfidence(confidence: number): string {
  if (!Number.isFinite(confidence)) return UI_TEXT.common.dash
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
  const title =
    article.display_title || article.title || UI_TEXT.common.dash
  const rationale =
    article.display_rationale || article.rationale || UI_TEXT.common.dash

  return (
    <button
      className={`article-list-card article-list-card--${article.stance}${isSelected ? ' is-selected' : ''}`}
      type="button"
      onClick={onSelect}
    >
      <span className="article-list-card-heading">
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
      </span>
      <strong>{title}</strong>
      <span className="article-card-metadata">
        {UI_TEXT.articles.metadata(
          article.pmid ?? UI_TEXT.common.dash,
          formatConfidence(article.confidence),
        )}
      </span>
      <span className="article-card-rationale">{rationale}</span>
      <span className="article-evidence-count">
        <FileText size={11} />
        {UI_TEXT.articles.evidenceSentenceCount(article.evidence_ids.length)}
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
        <h3>{UI_TEXT.articles.rankedTitle}</h3>
        <span>{articles.length}</span>
      </div>
      <p className="article-list-boundary">{UI_TEXT.articles.boundary}</p>
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
        <p className="reference-empty">{UI_TEXT.articles.empty}</p>
      )}
    </section>
  )
}
