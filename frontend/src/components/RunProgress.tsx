import { Check, Circle, LoaderCircle, X } from 'lucide-react'
import type { RunStatusResponse } from '../contracts'
import { RUN_STAGES, RUN_STAGE_LABELS } from '../utils/run'

interface RunProgressProps {
  run: RunStatusResponse
}

export function RunProgress({ run }: RunProgressProps) {
  const currentStageIndex = run.progress?.stage_index ?? 0

  return (
    <section className="run-progress" aria-label="Analysis progress">
      <div className="progress-heading">
        <div>
          <span className="eyebrow">Run progress</span>
          <h2>
            {run.status === 'queued'
              ? 'Waiting to start'
              : run.status === 'running'
                ? 'Analysis in progress'
                : run.status === 'succeeded'
                  ? 'Analysis completed'
                  : 'Analysis stopped'}
          </h2>
        </div>
        <span className={`status-badge status-badge--${run.status}`}>{run.status}</span>
      </div>

      <ol className="stage-list">
        {RUN_STAGES.map((stage, index) => {
          const stagePosition = index + 1
          const isFailed = run.status === 'failed' && currentStageIndex === stagePosition
          const isCurrent = run.status === 'running' && currentStageIndex === stagePosition
          const isCompleted = run.status === 'succeeded' || currentStageIndex > stagePosition
          const state = isFailed ? 'failed' : isCurrent ? 'current' : isCompleted ? 'completed' : 'pending'
          const Icon = isFailed ? X : isCurrent ? LoaderCircle : isCompleted ? Check : Circle

          return (
            <li className={`stage-item stage-item--${state}`} key={stage}>
              <span className="stage-icon"><Icon size={15} /></span>
              <span className="stage-copy">
                <strong>{RUN_STAGE_LABELS[stage]}</strong>
                <small>{state}</small>
              </span>
            </li>
          )
        })}
      </ol>

      {run.progress && (
        <div className="progress-message" role="status">
          {run.progress.stage === 'claim_analysis' &&
            run.progress.completed_units !== null &&
            run.progress.total_units !== null && (
              <strong>Claim analysis: {run.progress.completed_units} / {run.progress.total_units}</strong>
            )}
          <span>{run.progress.message}</span>
        </div>
      )}
    </section>
  )
}
