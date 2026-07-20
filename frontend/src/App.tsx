import { useMemo, useState } from 'react'
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

      <div className="workspace-grid">
        <CaseSidebar cases={cases} currentId={currentCase.id} onSelect={handleSelectCase} />
        <ClaimGraph
          currentCase={currentCase}
          selection={selection}
          viewMode={viewMode}
          targetActive={targetActive}
          onSelect={setSelection}
        />
        <EvidenceInspector currentCase={currentCase} selection={selection} />
      </div>

      <div className="bottom-grid">
        <GapReport currentCase={currentCase} />
        <ConclusionCompare currentCase={currentCase} />
      </div>
    </main>
  )
}

export default App
