import { Activity, CalendarDays, Clock, FileText, Languages } from 'lucide-react'
import type {
  LocalizationStatusResponse,
  PresentationBundle,
  RunStatusResponse,
} from '../contracts'
import { formatCreatedAt, formatDuration } from '../utils/format'
import type {
  GraphSelection,
  PresentationIndexes,
} from '../utils/presentation'
import { getSelectionClaimId } from '../utils/presentation'
import { ClaimList } from './ClaimList'
import { ExpertView } from './ExpertView'
import { ExportActions } from './ExportActions'
import { LocalizationPanel } from './LocalizationPanel'
import { RunErrorPanel } from './RunErrorPanel'
import { RunProgress } from './RunProgress'
import { SelectionInspector } from './SelectionInspector'
import { StatementClaimHighlighter } from './StatementClaimHighlighter'

interface RunDetailsProps {
  run: RunStatusResponse | null
  sourcePresentation: PresentationBundle | null
  activePresentation: PresentationBundle | null
  localizationList: LocalizationStatusResponse[]
  selectedLocalizationId: string | null
  activeLocalizationId: string | null
  selectedLocalization: LocalizationStatusResponse | null
  isLocalizationListLoading: boolean
  localizationListErrorMessage: string | null
  selectedLocalizationLoadErrorMessage: string | null
  localizationSelectionNotice: string | null
  onRetryLocalizationList: () => void
  onRetrySelectedLocalization: () => void
  onSelectLocalization: (localizationId: string | null) => void
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
  sourcePresentation,
  activePresentation,
  localizationList,
  selectedLocalizationId,
  activeLocalizationId,
  selectedLocalization,
  isLocalizationListLoading,
  localizationListErrorMessage,
  selectedLocalizationLoadErrorMessage,
  localizationSelectionNotice,
  onRetryLocalizationList,
  onRetrySelectedLocalization,
  onSelectLocalization,
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
        <div><dt><Languages size={14} /> Output</dt><dd>{activePresentation?.output_language ?? run.language}</dd></div>
        <div><dt><Clock size={14} /> Execution</dt><dd>{formatDuration(executionSeconds(run))}</dd></div>
      </dl>

      <RunProgress run={run} />
      {run.status === 'failed' && <RunErrorPanel run={run} />}

      {run.status === 'succeeded' && sourcePresentation && activePresentation && (
        <>
          <LocalizationPanel
            runId={run.run_id}
            originalLanguage={sourcePresentation.output_language}
            localizations={localizationList}
            selectedLocalizationId={selectedLocalizationId}
            activeLocalizationId={activeLocalizationId}
            selectedLocalization={selectedLocalization}
            isListLoading={isLocalizationListLoading}
            listErrorMessage={localizationListErrorMessage}
            selectedLoadErrorMessage={selectedLocalizationLoadErrorMessage}
            selectionNotice={localizationSelectionNotice}
            onRetryList={onRetryLocalizationList}
            onRetrySelected={onRetrySelectedLocalization}
            onSelectLocalization={onSelectLocalization}
          />
          <section className="statement-section">
            <h3>Original Statement</h3>
            <StatementClaimHighlighter
              originalText={sourcePresentation.statement.original_text}
              claims={sourcePresentation.claims}
              selection={selection}
              activeClaimId={activeClaimId}
              onSelect={onSelectionChange}
            />
          </section>
          <ClaimList
            claims={activePresentation.claims}
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
          <ExpertView
            run={run}
            presentation={activePresentation}
            selectedLocalization={selectedLocalization}
            selection={selection}
            indexes={indexes}
          />
          <ExportActions
            runId={run.run_id}
            isLocalizedView={Boolean(activeLocalizationId)}
          />
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
