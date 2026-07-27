import type { PresentationArticle, PresentationClaim } from '../contracts'
import { UI_TEXT } from '../uiText'
import { countArticleStances } from '../utils/presentation'

interface EvidenceBalanceProps {
  claim: PresentationClaim
  articles: PresentationArticle[]
}

export function EvidenceBalance({ claim, articles }: EvidenceBalanceProps) {
  const counts = countArticleStances(articles)
  const percentage = (count: number) =>
    counts.total === 0 ? 0 : (count / counts.total) * 100

  return (
    <section className={`evidence-balance${claim.evidence_state === 'CONFLICTED' ? ' evidence-balance--conflicted' : ''}`}>
      <div className="inspector-section-heading">
        <h3>{UI_TEXT.evidenceBalance.title}</h3>
        <span>{UI_TEXT.evidenceBalance.articleCount(counts.total)}</span>
      </div>
      <p className="balance-boundary">
        {UI_TEXT.evidenceBalance.boundary}
      </p>
      <div
        className="balance-bar"
        role="img"
        aria-label={UI_TEXT.evidenceBalance.aria(
          counts.support,
          counts.refute,
          counts.insufficient,
        )}
      >
        <span
          className="balance-bar-segment balance-bar-segment--support"
          style={{ width: `${percentage(counts.support)}%` }}
        />
        <span
          className="balance-bar-segment balance-bar-segment--refute"
          style={{ width: `${percentage(counts.refute)}%` }}
        />
        <span
          className="balance-bar-segment balance-bar-segment--insufficient"
          style={{ width: `${percentage(counts.insufficient)}%` }}
        />
      </div>
      <dl className="balance-legend">
        <div><dt><span className="balance-key balance-key--support" />{UI_TEXT.evidenceBalance.supporting}</dt><dd>{counts.support}</dd></div>
        <div><dt><span className="balance-key balance-key--refute" />{UI_TEXT.evidenceBalance.refuting}</dt><dd>{counts.refute}</dd></div>
        <div><dt><span className="balance-key balance-key--insufficient" />{UI_TEXT.evidenceBalance.insufficient}</dt><dd>{counts.insufficient}</dd></div>
      </dl>
    </section>
  )
}
