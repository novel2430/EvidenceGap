import type { KeyboardEvent, MouseEvent } from 'react'
import type { NodeProps } from '@xyflow/react'
import { Handle, Position } from '@xyflow/react'
import { GitMerge, Route, TriangleAlert } from 'lucide-react'
import type { InferenceGraphNode } from '../graph/types'
import { UI_TEXT } from '../uiText'
import {
  getGapTypeLabel,
  getInferenceIntegrityLabel,
  getIntegrityClassName,
} from '../utils/presentationLabels'

export function InferenceNode({ data }: NodeProps<InferenceGraphNode>) {
  const {
    inferenceStep,
    stepNumber,
    visual,
    selectedGapIndex,
    onSelect,
  } = data

  function selectInference() {
    onSelect?.({
      kind: 'inference_step',
      inferenceStepId: inferenceStep.inference_step_id,
    })
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      selectInference()
    }
  }

  function selectGap(event: MouseEvent<HTMLButtonElement>, gapIndex: number) {
    event.stopPropagation()
    onSelect?.({
      kind: 'gap',
      inferenceStepId: inferenceStep.inference_step_id,
      gapIndex,
    })
  }

  return (
    <div
      className={[
        'inference-node',
        visual.selected ? 'is-selected' : '',
        visual.spotlight ? 'is-spotlight' : '',
        visual.dimmed ? 'is-dimmed' : '',
      ].filter(Boolean).join(' ')}
      role="button"
      tabIndex={0}
      onClick={selectInference}
      onKeyDown={handleKeyDown}
      aria-label={UI_TEXT.graph.inferenceAria(stepNumber)}
    >
      <Handle type="target" position={Position.Left} />
      <div className="graph-node-heading">
        <span>
          <GitMerge size={13} /> {UI_TEXT.graph.inferenceStep(stepNumber)}
        </span>
      </div>
      <p>
        {UI_TEXT.graph.inferenceRelation(
          inferenceStep.premise_claim_ids.length,
        )}
      </p>
      <div className="inference-integrity-row">
        <span className={`inference-integrity-badge inference-integrity-badge--${getIntegrityClassName(inferenceStep.inference_integrity)}`}>
          <Route size={11} />
          {getInferenceIntegrityLabel(inferenceStep.inference_integrity, true)}
        </span>
      </div>
      <div className="gap-badges">
        {inferenceStep.gaps.map((gap, gapIndex) => (
          <button
            className={`nodrag nopan gap-badge gap-badge--${gap.gap_type.toLowerCase()}${selectedGapIndex === gapIndex ? ' is-selected' : ''}`}
            type="button"
            key={`${gap.gap_type}-${gapIndex}`}
            onClick={(event) => selectGap(event, gapIndex)}
          >
            <TriangleAlert size={11} />
            {getGapTypeLabel(gap.gap_type)}
          </button>
        ))}
      </div>
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
