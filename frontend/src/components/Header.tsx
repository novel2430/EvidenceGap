import { GitBranch, Sparkles } from 'lucide-react'
import type { DemoCase, ViewMode } from '../types'

export function Header({
  currentCase,
  viewMode,
  targetActive,
  onViewModeChange,
  onToggleTarget,
}: {
  currentCase: DemoCase
  viewMode: ViewMode
  targetActive: boolean
  onViewModeChange: (mode: ViewMode) => void
  onToggleTarget: () => void
}) {
  return (
    <header className="app-header">
      <div className="brand-block">
        <div className="brand-mark">
          <GitBranch size={20} />
        </div>
        <div className="brand-title">EvidenceGap</div>
      </div>

      <div className="case-headline">
        <strong>{currentCase.title}</strong>
        <em className={`status-pill status-${currentCase.status.toLowerCase()}`}>{currentCase.status}</em>
      </div>

      <div className="header-actions">
        <div className="mode-switch">
          {(['all', 'gaps', 'conflicts'] as ViewMode[]).map((mode) => (
            <button
              type="button"
              className={viewMode === mode ? 'active' : ''}
              onClick={() => onViewModeChange(mode)}
              key={mode}
            >
              {mode}
            </button>
          ))}
        </div>
        <button
          type="button"
          className={targetActive ? 'target-toggle active' : 'target-toggle'}
          onClick={onToggleTarget}
        >
          <Sparkles size={15} />
          Target Path
        </button>
      </div>
    </header>
  )
}
