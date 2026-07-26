import { Network } from 'lucide-react'
import type { PresentationBundle } from '../contracts'

interface ClaimGraphProps {
  presentation: PresentationBundle | null
}

export function ClaimGraph({ presentation }: ClaimGraphProps) {
  return (
    <section className="graph-panel panel">
      <div className="graph-grid" />
      <div className="graph-empty-state" role="status">
        <div className="empty-state-icon"><Network size={24} /></div>
        <span className="eyebrow">Claim graph</span>
        <h2>{presentation ? 'Graph view unavailable' : 'No analysis selected'}</h2>
        <p>
          {presentation
            ? 'Claim and inference graph is not displayed in this build.'
            : 'Submit a biomedical statement or open a recent analysis.'}
        </p>
      </div>
    </section>
  )
}
