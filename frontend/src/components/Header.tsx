import { GitBranch, History, Plus, Server } from 'lucide-react'
import type { RunStatusResponse } from '../contracts'
import { UI_TEXT } from '../uiText'

interface HeaderProps {
  run: RunStatusResponse | null
  onOpenNewAnalysis: () => void
  onOpenHistory: () => void
}

export function Header({
  run,
  onOpenNewAnalysis,
  onOpenHistory,
}: HeaderProps) {
  const integrationLabel =
    run?.status === 'running' || run?.status === 'queued'
      ? UI_TEXT.header.status.running
      : run?.status === 'failed'
        ? UI_TEXT.header.status.failed
        : run
          ? UI_TEXT.header.status.loaded
          : UI_TEXT.header.status.ready

  return (
    <header className="app-header">
      <div className="brand-block">
        <div className="brand-mark">
          <GitBranch size={20} />
        </div>
        <div>
          <div className="brand-title">{UI_TEXT.brand.name}</div>
          <div className="brand-subtitle">{UI_TEXT.brand.subtitle}</div>
        </div>
      </div>

      <div className="header-actions">
        <button
          className="new-analysis-button"
          type="button"
          onClick={onOpenNewAnalysis}
        >
          <Plus size={16} />
          {UI_TEXT.header.newAnalysis}
        </button>
        <div
          className={`integration-state integration-state--${run?.status ?? 'idle'}`}
          role="status"
        >
          <Server size={15} />
          {integrationLabel}
        </div>
        <button
          className="history-button"
          type="button"
          onClick={onOpenHistory}
          aria-label={UI_TEXT.header.openRecentRuns}
          title={UI_TEXT.header.recentRuns}
        >
          <History size={17} />
        </button>
      </div>
    </header>
  )
}
