import type { PresentationArticle, PresentationClaim } from '../contracts'
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
        <h3>Evidence Balance</h3>
        <span>{counts.total} articles</span>
      </div>
      <p className="balance-boundary">Retrieved Top Articles distribution</p>
      <div
        className="balance-bar"
        role="img"
        aria-label={`${counts.support} supporting, ${counts.refute} refuting, and ${counts.insufficient} insufficient articles`}
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
        <div><dt><span className="balance-key balance-key--support" />Supporting</dt><dd>{counts.support}</dd></div>
        <div><dt><span className="balance-key balance-key--refute" />Refuting</dt><dd>{counts.refute}</dd></div>
        <div><dt><span className="balance-key balance-key--insufficient" />Insufficient</dt><dd>{counts.insufficient}</dd></div>
      </dl>
    </section>
  )
}
