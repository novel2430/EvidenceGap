import { AlertTriangle, GitBranch, LockKeyhole, Scale, Target } from 'lucide-react'
import type { DemoCase, GapType } from '../types'

const gapIcon: Record<GapType, typeof AlertTriangle> = {
  PRIVATE_DATA_GAP: LockKeyhole,
  DECISION_GAP: Scale,
  SCOPE_GAP: Target,
  CAUSAL_GAP: GitBranch,
  CONFLICT_GAP: AlertTriangle,
}

export function GapReport({ currentCase }: { currentCase: DemoCase }) {
  const gaps = currentCase.gaps.slice(0, 4)

  return (
    <section className="gap-report panel">
      <div className="panel-kicker">Gap Report</div>
      {gaps.length === 0 ? (
        <div className="closed-gap-state">
          <span>Evidence chain closed</span>
          <p>No critical Evidence Gap blocks the target recommendation.</p>
        </div>
      ) : (
        <div className="gap-grid">
          {gaps.map((gap) => {
            const Icon = gapIcon[gap.type]
            return (
              <article className={`gap-card priority-${gap.priority.toLowerCase()}`} key={gap.id}>
                <div className="gap-card-head">
                  <Icon size={18} />
                  <span>{gap.priority}</span>
                </div>
                <strong>{gap.type.replaceAll('_', ' ')}</strong>
                <p>{gap.reason}</p>
                <div className="need-line">Need: {gap.requiredEvidence}</div>
                <div className="blocks-line">Blocks: {gap.blocks.join(', ')}</div>
              </article>
            )
          })}
        </div>
      )}
    </section>
  )
}
