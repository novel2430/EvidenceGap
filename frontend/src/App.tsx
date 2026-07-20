import { useMemo, useState } from 'react'
import { Group, Panel, Separator } from 'react-resizable-panels'
import './App.css'
import { CaseSidebar } from './components/CaseSidebar'
import { ClaimGraph } from './components/ClaimGraph'
import { ConclusionCompare } from './components/ConclusionCompare'
import { EvidenceInspector } from './components/EvidenceInspector'
import { GapReport } from './components/GapReport'
import { Header } from './components/Header'
import { cases } from './data/cases'
import type { InspectorSelection, ViewMode } from './types'

function App() {
  const [caseId, setCaseId] = useState('V0-B-GAP-001')
  const [selection, setSelection] = useState<InspectorSelection | null>({ kind: 'claim', id: 'B-C5' })
  const [viewMode, setViewMode] = useState<ViewMode>('all')
  const [targetActive, setTargetActive] = useState(true)

  const currentCase = useMemo(() => cases.find((item) => item.id === caseId) ?? cases[0], [caseId])

  function handleSelectCase(id: string) {
    const nextCase = cases.find((item) => item.id === id)
    if (!nextCase) return
    setCaseId(id)
    setSelection({ kind: 'claim', id: nextCase.claims.find((claim) => claim.isTarget)?.id ?? nextCase.claims[0].id })
    setViewMode(id === 'V0-C-OVERREACH-001' ? 'conflicts' : 'all')
    setTargetActive(true)
  }

  return (
    <main className="workbench-shell">
      <Header
        currentCase={currentCase}
        viewMode={viewMode}
        targetActive={targetActive}
        onViewModeChange={setViewMode}
        onToggleTarget={() => setTargetActive((value) => !value)}
      />

      <div className="workbench-body">
        <Group className="workbench-split-group" orientation="vertical">
          <Panel id="workspace" defaultSize="72%" minSize="360px">
            <div className="split-panel-content">
              <Group className="workbench-split-group" orientation="horizontal">
                <Panel id="case-sidebar" defaultSize="18%" minSize="220px" maxSize="34%">
                  <div className="split-panel-content">
                    <CaseSidebar cases={cases} currentId={currentCase.id} onSelect={handleSelectCase} />
                  </div>
                </Panel>

                <Separator
                  className="resize-handle resize-handle--vertical"
                  aria-label="調整案例列表與圖表寬度"
                />

                <Panel id="claim-graph" defaultSize="57%" minSize="480px">
                  <div className="split-panel-content">
                    <ClaimGraph
                      currentCase={currentCase}
                      selection={selection}
                      viewMode={viewMode}
                      targetActive={targetActive}
                      onSelect={setSelection}
                    />
                  </div>
                </Panel>

                <Separator
                  className="resize-handle resize-handle--vertical"
                  aria-label="調整圖表與證據檢視器寬度"
                />

                <Panel id="evidence-inspector" defaultSize="25%" minSize="280px" maxSize="42%">
                  <div className="split-panel-content">
                    <EvidenceInspector currentCase={currentCase} selection={selection} />
                  </div>
                </Panel>
              </Group>
            </div>
          </Panel>

          <Separator
            className="resize-handle resize-handle--horizontal"
            aria-label="調整主工作區與底部報告高度"
          />

          <Panel id="report-area" defaultSize="28%" minSize="150px" maxSize="48%">
            <div className="split-panel-content">
              <Group className="workbench-split-group" orientation="horizontal">
                <Panel id="gap-report" defaultSize="43%" minSize="340px">
                  <div className="split-panel-content">
                    <GapReport currentCase={currentCase} />
                  </div>
                </Panel>

                <Separator
                  className="resize-handle resize-handle--vertical"
                  aria-label="調整缺口報告與結論比較寬度"
                />

                <Panel id="conclusion-compare" defaultSize="57%" minSize="420px">
                  <div className="split-panel-content">
                    <ConclusionCompare currentCase={currentCase} />
                  </div>
                </Panel>
              </Group>
            </div>
          </Panel>
        </Group>
      </div>
    </main>
  )
}

export default App
