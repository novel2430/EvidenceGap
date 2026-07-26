import { History, RefreshCw } from 'lucide-react'
import { useNavigate } from 'react-router'
import type { RunListItemResponse } from '../contracts'
import { formatCreatedAt, formatDuration } from '../utils/format'

interface RunSidebarProps {
  runs: RunListItemResponse[]
  selectedRunId: string | null
  isLoading: boolean
  errorMessage: string | null
  onRetry: () => void
  onRunSelected?: () => void
}

export function RunSidebar({
  runs,
  selectedRunId,
  isLoading,
  errorMessage,
  onRetry,
  onRunSelected,
}: RunSidebarProps) {
  const navigate = useNavigate()

  function openRun(runId: string) {
    navigate(`/runs/${encodeURIComponent(runId)}`)
    onRunSelected?.()
  }

  return (
    <section className="sidebar panel">
      <div className="panel-heading">
        <div><span className="eyebrow">Workspace</span><h2>Recent Runs</h2></div>
        <History size={18} />
      </div>

      {isLoading && <div className="sidebar-message">Loading recent analyses…</div>}

      {errorMessage && (
        <div className="sidebar-error" role="alert">
          <strong>Could not load Recent Runs</strong>
          <p>{errorMessage}</p>
          <button type="button" onClick={onRetry}><RefreshCw size={14} /> Retry</button>
        </div>
      )}

      {!isLoading && !errorMessage && runs.length === 0 && (
        <div className="empty-card empty-card--compact">
          <History size={20} />
          <strong>No analyses yet</strong>
          <p>Your completed and in-progress runs will appear here.</p>
        </div>
      )}

      <div className="run-list">
        {runs.map((run) => {
          const gaps = run.summary
            ? run.summary.gaps.SCOPE_GAP + run.summary.gaps.CAUSAL_GAP
            : null
          return (
            <button
              className={`run-list-item${selectedRunId === run.run_id ? ' is-selected' : ''}`}
              type="button"
              key={run.run_id}
              onClick={() => openRun(run.run_id)}
            >
              <div className="run-item-topline">
                <span className={`status-dot status-dot--${run.status}`} />
                <span className={`run-status run-status--${run.status}`}>{run.status}</span>
                <time>{formatCreatedAt(run.created_at)}</time>
              </div>
              <p>{run.statement_preview}</p>
              <dl className="run-item-metrics">
                <div><dt>Claims</dt><dd>{run.summary?.total_claims ?? '—'}</dd></div>
                <div><dt>Gaps</dt><dd>{gaps ?? '—'}</dd></div>
                <div><dt>Time</dt><dd>{formatDuration(run.total_seconds)}</dd></div>
              </dl>
            </button>
          )
        })}
      </div>
    </section>
  )
}
