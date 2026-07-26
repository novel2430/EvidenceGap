import { useCallback, useEffect, useMemo, useState } from 'react'
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
import { AnalysisForm } from './components/AnalysisForm'
import { AnalysisSummary } from './components/AnalysisSummary'
import { ClaimGraph } from './components/ClaimGraph'
import { Header } from './components/Header'
import { RunDetails } from './components/RunDetails'
import { RunHistoryDrawer } from './components/RunHistoryDrawer'
import { useRunQuery } from './hooks/useRunQuery'
import {
  useLocalizationList,
  useLocalizationQuery,
} from './hooks/useLocalizations'
import { getApiErrorMessage } from './utils/format'
import {
  buildPresentationIndexes,
  getSelectionClaimId,
  isSelectionValid,
  type GraphSelection,
} from './utils/presentation'

function Workspace() {
  const { runId } = useParams<{ runId: string }>()
  const [searchParams, setSearchParams] = useSearchParams()
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [selection, setSelection] = useState<GraphSelection | null>(null)
  const closeHistory = useCallback(() => setIsHistoryOpen(false), [])
  const handleSelectionChange = useCallback(
    (nextSelection: GraphSelection | null) => setSelection(nextSelection),
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
      const nextSearchParams = new URLSearchParams(searchParams)
      if (localizationId) {
        nextSearchParams.set('localization', localizationId)
      } else {
        nextSearchParams.delete('localization')
      }
      setSearchParams(nextSearchParams)
    },
    [searchParams, setSearchParams],
  )

  useEffect(() => {
    if (!runId || !runStatus) return
    void queryClient.invalidateQueries({ queryKey: ['runs'] })
  }, [queryClient, runId, runStatus])

  useEffect(() => {
    setSelection(null)
  }, [runId])

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

  const runLoadError = runQuery.isError ? getApiErrorMessage(runQuery.error) : null
  const isNotFound = runQuery.error instanceof EvidenceGapApiError && runQuery.error.status === 404
  const localizationSelectionNotice =
    selectedLocalizationQuery.isSuccess &&
    selectedLocalizationQuery.data.source_run_id !== runId
      ? 'The requested localization does not belong to this Run.'
      : selectedLocalization?.status === 'succeeded' &&
          !selectedLocalization.result
        ? 'The requested localization completed without a presentation result.'
        : null

  return (
    <main className="workbench-shell">
      <Header run={run} onOpenHistory={() => setIsHistoryOpen(true)} />

      <div className="workbench-body">
        <Group className="workbench-split-group" orientation="vertical">
          <Panel id="workspace" defaultSize="74%" minSize="410px">
            <div className="split-panel-content">
              <Group className="workbench-split-group" orientation="horizontal">
                <Panel id="new-analysis" defaultSize="25%" minSize="280px" maxSize="36%">
                  <div className="split-panel-content">
                    <AnalysisForm />
                  </div>
                </Panel>

                <Separator className="resize-handle resize-handle--vertical" aria-label="Resize new analysis and graph workspace" />

                <Panel id="graph-workspace" defaultSize="48%" minSize="500px">
                  <div className="center-result">
                    {runQuery.isLoading && (
                      <section className="load-state panel" role="status">
                        <LoaderCircle className="spin" size={24} />
                        <h2>Loading analysis run</h2>
                        <p>Retrieving the current backend state…</p>
                      </section>
                    )}

                    {runLoadError && (
                      <section className="load-state load-state--error panel" role="alert">
                        <AlertCircle size={24} />
                        <h2>{isNotFound ? 'Run not found' : 'Run could not be loaded'}</h2>
                        <p>{runLoadError}</p>
                        <button className="secondary-button" type="button" onClick={() => void runQuery.refetch()}>
                          <RefreshCw size={15} /> Retry
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

                <Separator className="resize-handle resize-handle--vertical" aria-label="Resize analysis workspace and run details" />

                <Panel id="run-details" defaultSize="27%" minSize="300px" maxSize="42%">
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
                    />
                  </div>
                </Panel>
              </Group>
            </div>
          </Panel>

          <Separator className="resize-handle resize-handle--horizontal" aria-label="Resize workspace and analysis summary" />

          <Panel id="analysis-summary" defaultSize="26%" minSize="175px" maxSize="44%">
            <div className="split-panel-content">
              <AnalysisSummary
                summary={activePresentation?.summary ?? null}
                analysisContext={activePresentation?.analysis_context ?? null}
              />
            </div>
          </Panel>
        </Group>
      </div>

      <RunHistoryDrawer
        isOpen={isHistoryOpen}
        runs={runsQuery.data?.runs ?? []}
        selectedRunId={runId ?? null}
        isLoading={runsQuery.isLoading}
        errorMessage={runsQuery.isError ? getApiErrorMessage(runsQuery.error) : null}
        onRetry={() => void runsQuery.refetch()}
        onClose={closeHistory}
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
