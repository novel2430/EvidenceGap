import { AlertTriangle, ArrowLeft, RotateCcw } from 'lucide-react'
import { useNavigate } from 'react-router'
import type { RunStatusResponse } from '../contracts'
import { UI_TEXT } from '../uiText'
import { RUN_STAGE_LABELS } from '../utils/run'

interface RunErrorPanelProps {
  run: RunStatusResponse
}

export function RunErrorPanel({ run }: RunErrorPanelProps) {
  const navigate = useNavigate()
  const failedStage = run.progress
    ? RUN_STAGE_LABELS[run.progress.stage]
    : UI_TEXT.common.dash

  return (
    <section className="run-error-panel" role="alert">
      <div className="error-panel-icon"><AlertTriangle size={22} /></div>
      <span className="eyebrow">{UI_TEXT.runError.eyebrow}</span>
      <h2>{UI_TEXT.runError.title}</h2>
      <dl className="error-details">
        <div><dt>{UI_TEXT.runError.failedStage}</dt><dd>{failedStage}</dd></div>
        <div><dt>{UI_TEXT.runError.code}</dt><dd>{run.error?.code ?? UI_TEXT.common.dash}</dd></div>
        <div><dt>{UI_TEXT.runError.message}</dt><dd>{run.error?.message ?? UI_TEXT.runError.missingMessage}</dd></div>
      </dl>
      <div className="button-row">
        <button className="secondary-button" type="button" onClick={() => navigate('/')}>
          <RotateCcw size={15} /> {UI_TEXT.runError.tryAgain}
        </button>
        <button className="text-button" type="button" onClick={() => navigate('/')}>
          <ArrowLeft size={15} /> {UI_TEXT.runError.back}
        </button>
      </div>
    </section>
  )
}
