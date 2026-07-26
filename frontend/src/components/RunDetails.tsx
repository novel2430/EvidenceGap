import { Activity, CalendarDays, Clock, FileText, Languages } from 'lucide-react'
import type { RunStatusResponse } from '../contracts'
import { formatCreatedAt, formatDuration } from '../utils/format'
import type {
  GraphSelection,
  PresentationIndexes,
} from '../utils/presentation'
import { getSelectionClaimId } from '../utils/presentation'
import { ClaimList } from './ClaimList'
import { ExportActions } from './ExportActions'
import { RunErrorPanel } from './RunErrorPanel'
import { RunProgress } from './RunProgress'
import { SelectionInspector } from './SelectionInspector'
import { StatementClaimHighlighter } from './StatementClaimHighlighter'

interface RunDetailsProps {
  run: RunStatusResponse | null
  indexes: PresentationIndexes
  selection: GraphSelection | null
  onSelectionChange: (selection: GraphSelection | null) => void
}

function executionSeconds(run: RunStatusResponse): number | null {
  const value = run.execution_summary?.total_seconds
  return typeof value === 'number' ? value : null
}

export function RunDetails({
  run,
  indexes,
  selection,
  onSelectionChange,
}: RunDetailsProps) {
  const activeClaimId = getSelectionClaimId(selection, indexes)

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
            <StatementClaimHighlighter
              originalText={run.result.statement.original_text}
              claims={run.result.claims}
              selection={selection}
              activeClaimId={activeClaimId}
              onSelect={onSelectionChange}
            />
          </section>
          <ClaimList
            claims={run.result.claims}
            selection={selection}
            activeClaimId={activeClaimId}
            onSelect={onSelectionChange}
          />
          <SelectionInspector
            runId={run.run_id}
            selection={selection}
            indexes={indexes}
            onSelect={onSelectionChange}
          />
          <ExportActions runId={run.run_id} />
          <section className="method-boundary">
            <h3>Methodological Boundary</h3>
            <p>Only summarizes the retrieved Top Articles.</p>
            <p>Not a systematic review.</p>
            <p>Not a clinical recommendation.</p>
            <p>Not final medical truth.</p>
          </section>
        </>
      )}

      <div className="run-id" title={run.run_id}>Run ID: {run.run_id}</div>
    </aside>
  )
}
