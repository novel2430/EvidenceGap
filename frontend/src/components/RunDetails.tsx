import { Activity, CalendarDays, Clock, FileText, Languages } from 'lucide-react'
import type { RunStatusResponse } from '../contracts'
import { formatCreatedAt, formatDuration } from '../utils/format'
import { ExportActions } from './ExportActions'
import { RunErrorPanel } from './RunErrorPanel'
import { RunProgress } from './RunProgress'

interface RunDetailsProps {
  run: RunStatusResponse | null
}

function executionSeconds(run: RunStatusResponse): number | null {
  const value = run.execution_summary?.total_seconds
  return typeof value === 'number' ? value : null
}

export function RunDetails({ run }: RunDetailsProps) {
  if (!run) {
    return (
      <aside className="run-details panel">
        <div className="panel-heading">
          <div><span className="eyebrow">Run details</span><h2>No run selected</h2></div>
          <Activity size={18} />
        </div>
        <div className="empty-card">
          <FileText size={22} />
          <strong>Analysis details will appear here</strong>
          <p>Create a run or select one from Recent Runs.</p>
        </div>
      </aside>
    )
  }

  return (
    <aside className="run-details panel">
      <div className="panel-heading">
        <div><span className="eyebrow">Run details</span><h2>Analysis status</h2></div>
        <span className={`status-badge status-badge--${run.status}`}>{run.status}</span>
      </div>

      <dl className="run-metadata">
        <div><dt><CalendarDays size={14} /> Created</dt><dd>{formatCreatedAt(run.created_at)}</dd></div>
        <div><dt><Languages size={14} /> Output</dt><dd>{run.language}</dd></div>
        <div><dt><Clock size={14} /> Execution</dt><dd>{formatDuration(executionSeconds(run))}</dd></div>
      </dl>

      <RunProgress run={run} />
      {run.status === 'failed' && <RunErrorPanel run={run} />}

      {run.status === 'succeeded' && run.result && (
        <>
          <section className="statement-section">
            <h3>Original Statement</h3>
            <p>{run.result.statement.original_text}</p>
          </section>
          <ExportActions runId={run.run_id} />
        </>
      )}

      <div className="run-id" title={run.run_id}>Run ID: {run.run_id}</div>
    </aside>
  )
}
