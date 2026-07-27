import {
  useEffect,
  useRef,
  type KeyboardEvent,
} from 'react'
import { CalendarDays, Clock, FileText, Languages } from 'lucide-react'
import type {
  LocalizationStatusResponse,
  PresentationBundle,
  RunStatusResponse,
} from '../contracts'
import { formatCreatedAt, formatDuration } from '../utils/format'
import { UI_TEXT } from '../uiText'
import type {
  GraphSelection,
  PresentationIndexes,
} from '../utils/presentation'
import { getSelectionClaimId } from '../utils/presentation'
import { AnalysisSummary } from './AnalysisSummary'
import { ClaimList } from './ClaimList'
import { ExpertView } from './ExpertView'
import { ExportActions } from './ExportActions'
import { LocalizationPanel } from './LocalizationPanel'
import { RunErrorPanel } from './RunErrorPanel'
import { RunProgress } from './RunProgress'
import { SelectionInspector } from './SelectionInspector'
import { StatementClaimHighlighter } from './StatementClaimHighlighter'

export type DetailsTab = 'overview' | 'inspect' | 'run'

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
  activeTab: DetailsTab
  onTabChange: (tab: DetailsTab) => void
}

const DETAILS_TABS: Array<{ id: DetailsTab; label: string }> = [
  { id: 'overview', label: UI_TEXT.details.tabs.overview },
  { id: 'inspect', label: UI_TEXT.details.tabs.inspect },
  { id: 'run', label: UI_TEXT.details.tabs.run },
]

function getSelectionKey(selection: GraphSelection | null): string {
  if (!selection) return 'none'
  if (selection.kind === 'claim') return `claim:${selection.claimId}`
  if (selection.kind === 'inference_step') {
    return `inference:${selection.inferenceStepId}`
  }
  if (selection.kind === 'gap') {
    return `gap:${selection.inferenceStepId}:${selection.gapIndex}`
  }
  if (selection.kind === 'article') return `article:${selection.articleNodeId}`
  return `evidence:${selection.evidenceId}`
}

function executionSeconds(run: RunStatusResponse): number | null {
  const value = run.execution_summary?.total_seconds
  return typeof value === 'number' ? value : null
}

function EmptyDetails({
  title,
  message,
}: {
  title: string
  message: string
}) {
  return (
    <div className="empty-card details-empty-state">
      <FileText size={22} />
      <strong>{title}</strong>
      <p>{message}</p>
    </div>
  )
}

function OverviewTab({
  run,
  sourcePresentation,
  activePresentation,
  indexes,
  selection,
  onSelectionChange,
}: Pick<
  RunDetailsProps,
  | 'run'
  | 'sourcePresentation'
  | 'activePresentation'
  | 'indexes'
  | 'selection'
  | 'onSelectionChange'
>) {
  const activeClaimId = getSelectionClaimId(selection, indexes)

  if (!run) {
    return (
      <EmptyDetails
        title={UI_TEXT.details.noAnalysisTitle}
        message={UI_TEXT.details.noAnalysisDescription}
      />
    )
  }

  return (
    <div className="details-tab-stack">
      <AnalysisSummary
        summary={activePresentation?.summary ?? null}
        claims={activePresentation?.claims}
        embedded
      />

      {run.status === 'succeeded' &&
        sourcePresentation &&
        activePresentation && (
          <>
            <ClaimList
              claims={activePresentation.claims}
              selection={selection}
              activeClaimId={activeClaimId}
              onSelect={onSelectionChange}
            />
            <section className="statement-section">
              <h3>{UI_TEXT.details.originalStatement}</h3>
              <StatementClaimHighlighter
                originalText={sourcePresentation.statement.original_text}
                claims={sourcePresentation.claims}
                selection={selection}
                activeClaimId={activeClaimId}
                onSelect={onSelectionChange}
              />
            </section>
            <details className="method-boundary">
              <summary>{UI_TEXT.details.methodologicalBoundary}</summary>
              <div>
                {UI_TEXT.details.methodologicalBoundaryItems.map((item) => (
                  <p key={item}>{item}</p>
                ))}
                <small>
                  {UI_TEXT.details.sourceDepth(
                    activePresentation.analysis_context.source_depth,
                    activePresentation.analysis_context.article_top_k,
                  )}
                </small>
              </div>
            </details>
          </>
        )}
    </div>
  )
}

function InspectTab({
  run,
  activePresentation,
  indexes,
  selection,
  onSelectionChange,
}: Pick<
  RunDetailsProps,
  'run' | 'activePresentation' | 'indexes' | 'selection' | 'onSelectionChange'
>) {
  if (!run || run.status !== 'succeeded' || !activePresentation) {
    return (
      <EmptyDetails
        title={UI_TEXT.details.nothingToInspectTitle}
        message={UI_TEXT.details.nothingToInspectDescription}
      />
    )
  }

  return (
    <SelectionInspector
      runId={run.run_id}
      selection={selection}
      indexes={indexes}
      onSelect={onSelectionChange}
    />
  )
}

function RunTab(props: Omit<
  RunDetailsProps,
  'activeTab' | 'onTabChange' | 'sourcePresentation'
> & {
  sourcePresentation: PresentationBundle | null
}) {
  const {
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
  } = props

  if (!run) {
    return (
      <EmptyDetails
        title={UI_TEXT.details.noRunTitle}
        message={UI_TEXT.details.noRunDescription}
      />
    )
  }

  return (
    <div className="details-tab-stack">
      <RunProgress run={run} />
      {run.status === 'failed' && <RunErrorPanel run={run} />}

      <section className="run-execution-section">
        <h3>{UI_TEXT.details.execution}</h3>
        <dl className="run-metadata">
          <div><dt><CalendarDays size={14} /> {UI_TEXT.details.created}</dt><dd>{formatCreatedAt(run.created_at)}</dd></div>
          <div><dt><Languages size={14} /> {UI_TEXT.details.output}</dt><dd>{activePresentation?.output_language ?? run.language}</dd></div>
          <div><dt><Clock size={14} /> {UI_TEXT.details.duration}</dt><dd>{formatDuration(executionSeconds(run))}</dd></div>
        </dl>
      </section>

      {run.status === 'succeeded' &&
        sourcePresentation &&
        activePresentation && (
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
          </>
        )}

      <div className="run-id" title={run.run_id}>
        {UI_TEXT.details.runId(run.run_id)}
      </div>
    </div>
  )
}

export function RunDetails({
  activeTab,
  onTabChange,
  ...props
}: RunDetailsProps) {
  const contentRef = useRef<HTMLDivElement>(null)
  const previousSelectionKey = useRef('none')
  const selectionKey = getSelectionKey(props.selection)

  useEffect(() => {
    if (
      activeTab !== 'inspect' ||
      selectionKey === 'none' ||
      selectionKey === previousSelectionKey.current
    ) {
      previousSelectionKey.current = selectionKey
      return
    }
    contentRef.current?.scrollTo({ top: 0, behavior: 'auto' })
    previousSelectionKey.current = selectionKey
  }, [activeTab, selectionKey])

  function handleTabKeyDown(
    event: KeyboardEvent<HTMLButtonElement>,
    tabIndex: number,
  ) {
    const direction =
      event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0
    let nextIndex = tabIndex
    if (direction !== 0) {
      nextIndex = (tabIndex + direction + DETAILS_TABS.length) %
        DETAILS_TABS.length
    } else if (event.key === 'Home') {
      nextIndex = 0
    } else if (event.key === 'End') {
      nextIndex = DETAILS_TABS.length - 1
    } else {
      return
    }
    event.preventDefault()
    onTabChange(DETAILS_TABS[nextIndex].id)
    const buttons = event.currentTarget.parentElement
      ?.querySelectorAll<HTMLButtonElement>('[role="tab"]')
    buttons?.[nextIndex]?.focus()
  }

  return (
    <aside className="run-details panel">
      <div className="details-panel-heading">
        <div>
          <span className="eyebrow">{UI_TEXT.details.eyebrow}</span>
          <h2>{UI_TEXT.details.title}</h2>
        </div>
      </div>

      <div
        className="details-tabs"
        role="tablist"
        aria-label={UI_TEXT.details.tabListLabel}
      >
        {DETAILS_TABS.map((tab, tabIndex) => (
          <button
            id={`details-tab-${tab.id}`}
            className={activeTab === tab.id ? 'is-active' : ''}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`details-panel-${tab.id}`}
            tabIndex={activeTab === tab.id ? 0 : -1}
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            onKeyDown={(event) => handleTabKeyDown(event, tabIndex)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <div
        ref={contentRef}
        id={`details-panel-${activeTab}`}
        className="details-tab-content"
        role="tabpanel"
        aria-labelledby={`details-tab-${activeTab}`}
      >
        <div
          className="details-tab-transition"
          key={activeTab === 'inspect'
            ? `${activeTab}:${selectionKey}`
            : activeTab}
        >
          {activeTab === 'overview' && <OverviewTab {...props} />}
          {activeTab === 'inspect' && <InspectTab {...props} />}
          {activeTab === 'run' && <RunTab {...props} />}
        </div>
      </div>
    </aside>
  )
}
