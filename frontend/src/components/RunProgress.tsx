import {
  Check,
  CheckCircle2,
  ChevronDown,
  Circle,
  LoaderCircle,
  X,
} from 'lucide-react'
import type {
  RunStage,
  RunStatusResponse,
} from '../contracts'
import { formatDuration } from '../utils/format'
import { RUN_STAGES, RUN_STAGE_LABELS } from '../utils/run'
import { UI_TEXT } from '../uiText'

interface RunProgressProps {
  run: RunStatusResponse
}

function executionSeconds(run: RunStatusResponse): number | null {
  const value = run.execution_summary?.total_seconds
  return typeof value === 'number' ? value : null
}

function stageSeconds(
  run: RunStatusResponse,
  stage: RunStage,
): number | null {
  const stages = run.execution_summary?.stages
  if (!stages || typeof stages !== 'object') return null
  const stageValue = (stages as Record<string, unknown>)[stage]
  if (!stageValue || typeof stageValue !== 'object') return null
  const seconds = (stageValue as Record<string, unknown>).seconds
  return typeof seconds === 'number' ? seconds : null
}

function StageList({ run }: RunProgressProps) {
  const currentStageIndex = run.progress?.stage_index ?? 0

  return (
    <ol className="stage-list">
      {RUN_STAGES.map((stage, index) => {
        const stagePosition = index + 1
        const isFailed =
          run.status === 'failed' && currentStageIndex === stagePosition
        const isCurrent =
          run.status === 'running' && currentStageIndex === stagePosition
        const isCompleted =
          run.status === 'succeeded' || currentStageIndex > stagePosition
        const state = isFailed
          ? 'failed'
          : isCurrent
            ? 'current'
            : isCompleted
              ? 'completed'
              : 'pending'
        const Icon = isFailed
          ? X
          : isCurrent
            ? LoaderCircle
            : isCompleted
              ? Check
              : Circle
        const timing = stageSeconds(run, stage)

        return (
          <li className={`stage-item stage-item--${state}`} key={stage}>
            <span className="stage-icon"><Icon size={15} /></span>
            <span className="stage-copy">
              <strong>{RUN_STAGE_LABELS[stage]}</strong>
              <small>
                {timing !== null
                  ? UI_TEXT.runProgress.stageStateTiming(
                      UI_TEXT.runProgress.stageStates[state],
                      formatDuration(timing),
                    )
                  : UI_TEXT.runProgress.stageStates[state]}
              </small>
            </span>
          </li>
        )
      })}
    </ol>
  )
}

export function RunProgress({ run }: RunProgressProps) {
  if (run.status === 'succeeded') {
    return (
      <details
        className="run-progress run-progress--compact"
        aria-label={UI_TEXT.runProgress.label}
      >
        <summary>
          <span className="completed-run-summary">
            <CheckCircle2 size={16} />
            <strong>{UI_TEXT.runProgress.completed}</strong>
            <span>{UI_TEXT.runProgress.stageCount(RUN_STAGES.length)}</span>
            <span>
              {UI_TEXT.runProgress.duration(
                formatDuration(executionSeconds(run)),
              )}
            </span>
          </span>
          <span className="progress-expand-label">
            {UI_TEXT.runProgress.stageDetails} <ChevronDown size={13} />
          </span>
        </summary>
        <div className="completed-stage-details">
          <StageList run={run} />
        </div>
      </details>
    )
  }

  const completedStages = run.status === 'running'
    ? Math.max(0, (run.progress?.stage_index ?? 1) - 1)
    : 0

  return (
    <section
      className={`run-progress run-progress--${run.status}`}
      aria-label={UI_TEXT.runProgress.label}
    >
      <div className="progress-heading">
        <div>
          <span className="eyebrow">{UI_TEXT.runProgress.eyebrow}</span>
          <h2>
            {run.status === 'queued'
              ? UI_TEXT.runProgress.waiting
              : run.status === 'running'
                ? UI_TEXT.runProgress.running
                : UI_TEXT.runProgress.stopped}
          </h2>
        </div>
        <span className={`status-badge status-badge--${run.status}`}>
          {UI_TEXT.statusLabels[run.status]}
        </span>
      </div>

      {(run.status === 'queued' || run.status === 'running') && (
        <div
          className="stage-progress-track"
          role="progressbar"
          aria-label={UI_TEXT.runProgress.completedStagesLabel}
          aria-valuemin={0}
          aria-valuemax={RUN_STAGES.length}
          aria-valuenow={completedStages}
        >
          <span
            style={{
              width: `${(completedStages / RUN_STAGES.length) * 100}%`,
            }}
          />
        </div>
      )}

      <StageList run={run} />

      {run.progress && (
        <div className="progress-message" role="status">
          {run.progress.stage === 'claim_analysis' &&
            run.progress.completed_units !== null &&
            run.progress.total_units !== null && (
              <strong>
                {UI_TEXT.runProgress.claimProgress(
                  run.progress.completed_units,
                  run.progress.total_units,
                )}
              </strong>
            )}
          <span>{run.progress.message}</span>
        </div>
      )}
    </section>
  )
}
