import { History, RefreshCw } from 'lucide-react'
import { useNavigate } from 'react-router'
import type { RunListItemResponse } from '../contracts'
import { UI_TEXT } from '../uiText'
import { formatCreatedAt, formatDuration } from '../utils/format'

interface RunSidebarProps {
  runs: RunListItemResponse[]
  selectedRunId: string | null
  isLoading: boolean
  errorMessage: string | null
  onRetry: () => void
  onRunSelected?: (runId: string) => void
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
    onRunSelected?.(runId)
  }

  return (
    <section className="sidebar panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">{UI_TEXT.history.eyebrow}</span>
          <h2>{UI_TEXT.history.title}</h2>
        </div>
        <History size={18} />
      </div>

      {isLoading && (
        <div className="sidebar-message">{UI_TEXT.history.loading}</div>
      )}

      {errorMessage && (
        <div className="sidebar-error" role="alert">
          <strong>{UI_TEXT.history.loadFailed}</strong>
          <p>{errorMessage}</p>
          <button type="button" onClick={onRetry}>
            <RefreshCw size={14} /> {UI_TEXT.common.retry}
          </button>
        </div>
      )}

      {!isLoading && !errorMessage && runs.length === 0 && (
        <div className="empty-card empty-card--compact">
          <History size={20} />
          <strong>{UI_TEXT.history.emptyTitle}</strong>
          <p>{UI_TEXT.history.emptyDescription}</p>
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
                <span className={`run-status run-status--${run.status}`}>
                  {UI_TEXT.statusLabels[run.status]}
                </span>
                <time>{formatCreatedAt(run.created_at)}</time>
              </div>
              <p>{run.statement_preview}</p>
              <dl className="run-item-metrics">
                <div><dt>{UI_TEXT.history.metrics.claims}</dt><dd>{run.summary?.total_claims ?? UI_TEXT.common.dash}</dd></div>
                <div><dt>{UI_TEXT.history.metrics.gaps}</dt><dd>{gaps ?? UI_TEXT.common.dash}</dd></div>
                <div><dt>{UI_TEXT.history.metrics.time}</dt><dd>{formatDuration(run.total_seconds)}</dd></div>
              </dl>
            </button>
          )
        })}
      </div>
    </section>
  )
}
