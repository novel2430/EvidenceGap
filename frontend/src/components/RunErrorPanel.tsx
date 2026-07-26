import { AlertTriangle, ArrowLeft, RotateCcw } from 'lucide-react'
import { useNavigate } from 'react-router'
import type { RunStatusResponse } from '../contracts'
import { RUN_STAGE_LABELS } from '../utils/run'

interface RunErrorPanelProps {
  run: RunStatusResponse
}

export function RunErrorPanel({ run }: RunErrorPanelProps) {
  const navigate = useNavigate()
  const failedStage = run.progress ? RUN_STAGE_LABELS[run.progress.stage] : '—'

  return (
    <section className="run-error-panel" role="alert">
      <div className="error-panel-icon"><AlertTriangle size={22} /></div>
      <span className="eyebrow">Analysis failed</span>
      <h2>The run could not be completed</h2>
      <dl className="error-details">
        <div><dt>Failed stage</dt><dd>{failedStage}</dd></div>
        <div><dt>Error code</dt><dd>{run.error?.code ?? '—'}</dd></div>
        <div><dt>Error message</dt><dd>{run.error?.message ?? 'The backend did not provide an error message.'}</dd></div>
      </dl>
      <div className="button-row">
        <button className="secondary-button" type="button" onClick={() => navigate('/')}>
          <RotateCcw size={15} /> Try Again
        </button>
        <button className="text-button" type="button" onClick={() => navigate('/')}>
          <ArrowLeft size={15} /> Back to New Analysis
        </button>
      </div>
    </section>
  )
}
