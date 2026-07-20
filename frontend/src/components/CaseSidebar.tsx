import { AlertTriangle, BookOpen, ShieldCheck } from 'lucide-react'
import type { DemoCase } from '../types'

export function CaseSidebar({
  cases,
  currentId,
  onSelect,
}: {
  cases: DemoCase[]
  currentId: string
  onSelect: (id: string) => void
}) {
  return (
    <aside className="sidebar panel">
      <div className="case-list">
        {cases.map((item) => {
          const Icon = item.status === 'SUPPORTED' ? ShieldCheck : AlertTriangle
          return (
            <button
              type="button"
              className={item.id === currentId ? 'case-card active' : 'case-card'}
              onClick={() => onSelect(item.id)}
              key={item.id}
            >
              <div className="case-card-top">
                <span>{item.label}</span>
                <Icon size={16} />
              </div>
              <strong>{item.title}</strong>
              <div className="case-stats">
                <span>{item.claims.length} Claims</span>
                <span>{item.gaps.length} Gaps</span>
              </div>
            </button>
          )
        })}
      </div>

      <div className="original-claim">
        <div className="section-title">
          <BookOpen size={16} />
          Original Claim
        </div>
        <p>{cases.find((item) => item.id === currentId)?.originalConclusion}</p>
      </div>
    </aside>
  )
}
