import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { AlertCircle, LoaderCircle, RefreshCw } from 'lucide-react'
import {
  Navigate,
  Route,
  Routes,
  useParams,
  useSearchParams,
} from 'react-router'
import { Group, Panel, Separator } from 'react-resizable-panels'
import './App.css'
import { evidenceGapApi, EvidenceGapApiError } from './api'
import { ClaimGraph } from './components/ClaimGraph'
import { Header } from './components/Header'
import { NewAnalysisDrawer } from './components/NewAnalysisDrawer'
import {
  RunDetails,
  type DetailsTab,
} from './components/RunDetails'
import { RunHistoryDrawer } from './components/RunHistoryDrawer'
import { useToast } from './hooks/useToast'
import { useRunQuery } from './hooks/useRunQuery'
import {
  useLocalizationList,
  useLocalizationQuery,
} from './hooks/useLocalizations'
import { getApiErrorMessage, getToastErrorMessage } from './utils/format'
import { RUN_STAGE_LABELS } from './utils/run'
import { UI_TEXT } from './uiText'
import {
  buildPresentationIndexes,
  getSelectionClaimId,
  isSelectionValid,
  type GraphSelection,
} from './utils/presentation'

function isSameSelection(
  current: GraphSelection | null,
  next: GraphSelection | null,
) {
  if (current?.kind !== next?.kind) return false
  if (!current || !next) return current === next
  if (current.kind === 'claim' && next.kind === 'claim') {
    return current.claimId === next.claimId
  }
  if (
    current.kind === 'inference_step' &&
    next.kind === 'inference_step'
  ) {
    return current.inferenceStepId === next.inferenceStepId
  }
  if (current.kind === 'gap' && next.kind === 'gap') {
    return current.inferenceStepId === next.inferenceStepId &&
      current.gapIndex === next.gapIndex
  }
  if (current.kind === 'article' && next.kind === 'article') {
    return current.articleNodeId === next.articleNodeId
  }
  return current.kind === 'evidence' &&
    next.kind === 'evidence' &&
    current.evidenceId === next.evidenceId
}

function Workspace() {
  const { runId } = useParams<{ runId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const [isNewAnalysisOpen, setIsNewAnalysisOpen] = useState(() => !runId)
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [selection, setSelection] = useState<GraphSelection | null>(null)
  const [activeDetailsTab, setActiveDetailsTab] =
    useState<DetailsTab>(runId ? 'run' : 'overview')
  const observedRunStatuses = useRef(new Map<string, string>())
  const observedLocalizationStatuses = useRef(new Map<string, string>())
  const {
    isOperationTracked,
    notifyOnce,
    untrackOperation,
  } = useToast()
  const closeNewAnalysis = useCallback(
    () => setIsNewAnalysisOpen(false),
    [],
  )
  const closeHistory = useCallback(() => setIsHistoryOpen(false), [])
  const handleSelectionChange = useCallback(
    (nextSelection: GraphSelection | null) => {
      setSelection((current) =>
        isSameSelection(current, nextSelection) ? current : nextSelection)
      if (nextSelection) setActiveDetailsTab('inspect')
    },
    [],
  )
  const queryClient = useQueryClient()
  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: ({ signal }) => evidenceGapApi.listRuns({ limit: 20, signal }),
  })
  const runQuery = useRunQuery(runId)
  const run = runQuery.data ?? null
  const runStatus = run?.status
  const sourcePresentation =
    run?.status === 'succeeded' ? run.result : null
  const selectedLocalizationId =
    searchParams.get('localization')?.trim() || null
  const localizationListQuery = useLocalizationList(
    runId,
    Boolean(sourcePresentation),
  )
  const selectedLocalizationQuery = useLocalizationQuery(
    runId,
    selectedLocalizationId,
    Boolean(sourcePresentation),
  )
  const selectedLocalizationData = selectedLocalizationQuery.data
  const selectedLocalization =
    selectedLocalizationData &&
    selectedLocalizationData.source_run_id === runId
      ? selectedLocalizationData
      : null
  const activeLocalization =
    selectedLocalization?.status === 'succeeded' &&
    selectedLocalization.result
      ? selectedLocalization
      : null
  const activePresentation =
    activeLocalization?.result ?? sourcePresentation
  const sourcePresentationIndexes = useMemo(
    () => buildPresentationIndexes(sourcePresentation),
    [sourcePresentation],
  )
  const presentationIndexes = useMemo(
    () => buildPresentationIndexes(activePresentation),
    [activePresentation],
  )
  const handleSelectLocalization = useCallback(
    (localizationId: string | null) => {
      if (
        runId &&
        selectedLocalizationId &&
        localizationId !== selectedLocalizationId
      ) {
        untrackOperation(
          `localization:${runId}:${selectedLocalizationId}`,
        )
      }
      const nextSearchParams = new URLSearchParams(searchParams)
      if (localizationId) {
        nextSearchParams.set('localization', localizationId)
      } else {
        nextSearchParams.delete('localization')
      }
      setSearchParams(nextSearchParams)
    },
    [
      runId,
      searchParams,
      selectedLocalizationId,
      setSearchParams,
      untrackOperation,
    ],
  )

  useEffect(() => {
    if (!runId || !runStatus) return
    void queryClient.invalidateQueries({ queryKey: ['runs'] })
  }, [queryClient, runId, runStatus])

  useEffect(() => {
    if (!run) return
    const operationKey = `run:${run.run_id}`
    const previousStatus = observedRunStatuses.current.get(run.run_id)
    observedRunStatuses.current.set(run.run_id, run.status)
    if (!isOperationTracked(operationKey)) return

    if (run.status === 'succeeded' && previousStatus !== 'succeeded') {
      const claimCount = run.result?.claims.length
      notifyOnce(`${operationKey}:succeeded`, {
        type: 'success',
        title: UI_TEXT.toast.analysisCompleted,
        description: typeof claimCount === 'number'
          ? UI_TEXT.toast.analyzedClaims(claimCount)
          : UI_TEXT.toast.resultsReady,
      })
    } else if (run.status === 'failed' && previousStatus !== 'failed') {
      const stage = run.progress?.stage
      notifyOnce(`${operationKey}:failed`, {
        type: 'error',
        title: UI_TEXT.toast.analysisFailed,
        description: stage
          ? UI_TEXT.toast.stoppedDuring(RUN_STAGE_LABELS[stage])
          : UI_TEXT.toast.stoppedBeforeComplete,
      })
    }
  }, [isOperationTracked, notifyOnce, run])

  useEffect(() => {
    setSelection(null)
  }, [runId])

  useEffect(() => {
    if (!runId) setIsNewAnalysisOpen(true)
  }, [runId])

  useEffect(() => {
    setActiveDetailsTab(runStatus === 'succeeded' ? 'overview' : 'run')
  }, [runId, runStatus])

  useEffect(() => {
    if (!selection || isSelectionValid(selection, presentationIndexes)) return

    if (selection.kind === 'gap') {
      const inferenceStep = presentationIndexes.inferenceStepsById.get(
        selection.inferenceStepId,
      )
      setSelection(
        inferenceStep
          ? {
              kind: 'inference_step',
              inferenceStepId: inferenceStep.inference_step_id,
            }
          : null,
      )
      return
    }

    if (selection.kind === 'article' || selection.kind === 'evidence') {
      const sourceClaimId = getSelectionClaimId(
        selection,
        sourcePresentationIndexes,
      )
      setSelection(
        sourceClaimId && presentationIndexes.claimsById.has(sourceClaimId)
          ? { kind: 'claim', claimId: sourceClaimId }
          : null,
      )
      return
    }

    setSelection(null)
  }, [
    presentationIndexes,
    selection,
    sourcePresentationIndexes,
  ])

  useEffect(() => {
    const localizationStatus = selectedLocalization?.status
    if (
      !runId ||
      (localizationStatus !== 'succeeded' && localizationStatus !== 'failed')
    ) {
      return
    }
    void queryClient.invalidateQueries({
      queryKey: ['localizations', runId],
    })
  }, [queryClient, runId, selectedLocalization?.status])

  useEffect(() => {
    if (!selectedLocalization) return
    const operationKey =
      `localization:${selectedLocalization.source_run_id}:${selectedLocalization.localization_id}`
    const previousStatus = observedLocalizationStatuses.current.get(
      operationKey,
    )
    observedLocalizationStatuses.current.set(
      operationKey,
      selectedLocalization.status,
    )
    if (!isOperationTracked(operationKey)) return

    if (
      selectedLocalization.status === 'succeeded' &&
      previousStatus !== 'succeeded'
    ) {
      notifyOnce(`${operationKey}:succeeded`, {
        type: 'success',
        title: UI_TEXT.toast.localizationReady,
        description: UI_TEXT.toast.localizationDisplayed(
          selectedLocalization.language,
        ),
      })
    } else if (
      selectedLocalization.status === 'failed' &&
      previousStatus !== 'failed'
    ) {
      notifyOnce(`${operationKey}:failed`, {
        type: 'error',
        title: UI_TEXT.toast.localizationFailed,
        description: getToastErrorMessage(
          selectedLocalization.error?.message ??
          UI_TEXT.toast.localizationCouldNotGenerate(
            selectedLocalization.language,
          ),
        ),
      })
    }
  }, [
    isOperationTracked,
    notifyOnce,
    selectedLocalization,
  ])

  useEffect(() => {
    if (!isNewAnalysisOpen && !isHistoryOpen) return
    const previousOverflow = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      document.body.style.overflow = previousOverflow
    }
  }, [isHistoryOpen, isNewAnalysisOpen])

  const runLoadError = runQuery.isError ? getApiErrorMessage(runQuery.error) : null
  const isNotFound = runQuery.error instanceof EvidenceGapApiError && runQuery.error.status === 404
  const localizationSelectionNotice =
    selectedLocalizationQuery.isSuccess &&
    selectedLocalizationQuery.data.source_run_id !== runId
      ? UI_TEXT.localization.invalidSelection
      : selectedLocalization?.status === 'succeeded' &&
          !selectedLocalization.result
        ? UI_TEXT.localization.missingResult
        : null

  return (
    <main className="workbench-shell">
      <Header
        run={run}
        onOpenNewAnalysis={() => {
          setIsHistoryOpen(false)
          setIsNewAnalysisOpen(true)
        }}
        onOpenHistory={() => {
          setIsNewAnalysisOpen(false)
          setIsHistoryOpen(true)
        }}
      />

      <div className="workbench-body">
        <Group className="workbench-split-group" orientation="horizontal">
          <Panel id="graph-workspace" defaultSize="70%" minSize="560px">
            <div className="center-result">
              {runQuery.isLoading && (
                <section className="load-state panel" role="status">
                  <LoaderCircle className="spin" size={24} />
                  <h2>{UI_TEXT.workspace.loadingRun}</h2>
                  <p>{UI_TEXT.workspace.loadingRunDescription}</p>
                </section>
              )}

              {runLoadError && (
                <section className="load-state load-state--error panel" role="alert">
                  <AlertCircle size={24} />
                  <h2>
                    {isNotFound
                      ? UI_TEXT.workspace.runNotFound
                      : UI_TEXT.workspace.runLoadFailed}
                  </h2>
                  <p>{runLoadError}</p>
                  <button className="secondary-button" type="button" onClick={() => void runQuery.refetch()}>
                    <RefreshCw size={15} /> {UI_TEXT.common.retry}
                  </button>
                </section>
              )}

              {!runQuery.isLoading && !runLoadError && (
                <ClaimGraph
                  key={runId ?? 'no-run'}
                  presentation={activePresentation}
                  indexes={presentationIndexes}
                  selection={selection}
                  onSelectionChange={handleSelectionChange}
                />
              )}
            </div>
          </Panel>

          <Separator
            className="resize-handle resize-handle--vertical"
            aria-label={UI_TEXT.workspace.resizePanels}
          />

          <Panel id="run-details" defaultSize="30%" minSize="340px" maxSize="46%">
            <div className="split-panel-content">
              <RunDetails
                run={run}
                sourcePresentation={sourcePresentation}
                activePresentation={activePresentation}
                localizationList={localizationListQuery.data?.localizations ?? []}
                selectedLocalizationId={selectedLocalizationId}
                activeLocalizationId={activeLocalization?.localization_id ?? null}
                selectedLocalization={selectedLocalization}
                isLocalizationListLoading={localizationListQuery.isLoading}
                localizationListErrorMessage={
                  localizationListQuery.isError
                    ? getApiErrorMessage(localizationListQuery.error)
                    : null
                }
                selectedLocalizationLoadErrorMessage={
                  selectedLocalizationQuery.isError
                    ? getApiErrorMessage(selectedLocalizationQuery.error)
                    : null
                }
                localizationSelectionNotice={localizationSelectionNotice}
                onRetryLocalizationList={() =>
                  void localizationListQuery.refetch()}
                onRetrySelectedLocalization={() =>
                  void selectedLocalizationQuery.refetch()}
                onSelectLocalization={handleSelectLocalization}
                indexes={presentationIndexes}
                selection={selection}
                onSelectionChange={handleSelectionChange}
                activeTab={activeDetailsTab}
                onTabChange={setActiveDetailsTab}
              />
            </div>
          </Panel>
        </Group>
      </div>

      <NewAnalysisDrawer
        isOpen={isNewAnalysisOpen}
        onClose={closeNewAnalysis}
      />
      <RunHistoryDrawer
        isOpen={isHistoryOpen}
        runs={runsQuery.data?.runs ?? []}
        selectedRunId={runId ?? null}
        isLoading={runsQuery.isLoading}
        errorMessage={runsQuery.isError ? getApiErrorMessage(runsQuery.error) : null}
        onRetry={() => void runsQuery.refetch()}
        onClose={closeHistory}
        onRunSelected={(nextRunId) => {
          if (runId && nextRunId !== runId) {
            untrackOperation(`run:${runId}`)
          }
          closeHistory()
        }}
      />
    </main>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<Workspace />} />
      <Route path="/runs/:runId" element={<Workspace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
