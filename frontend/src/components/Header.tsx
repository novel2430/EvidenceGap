import { GitBranch, History, Server } from 'lucide-react'
import type { RunStatusResponse } from '../contracts'

interface HeaderProps {
  run: RunStatusResponse | null
  onOpenHistory: () => void
}

export function Header({ run, onOpenHistory }: HeaderProps) {
  const integrationLabel = run ? `Run ${run.status}` : 'Ready for analysis'

  return (
    <header className="app-header">
      <div className="brand-block">
        <div className="brand-mark">
          <GitBranch size={20} />
        </div>
        <div>
          <div className="brand-title">EvidenceGap</div>
          <div className="brand-subtitle">Statement evidence analysis</div>
        </div>
      </div>

      <div className="header-actions">
        <div className="integration-state" role="status">
          <Server size={15} />
          {integrationLabel}
        </div>
        <button
          className="history-button"
          type="button"
          onClick={onOpenHistory}
          aria-label="Open Recent Runs"
          title="Recent Runs"
        >
          <History size={17} />
        </button>
      </div>
    </header>
  )
}
