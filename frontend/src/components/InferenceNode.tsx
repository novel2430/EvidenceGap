import { GitBranch, Scale, Sparkles } from 'lucide-react'
import { Handle, Position } from '@xyflow/react'
import type { InferenceStep } from '../types'

const inferenceIcon = {
  DECISION: Scale,
  DOMAIN_RULE_APPLICATION: GitBranch,
  CAUSAL: GitBranch,
  GENERALIZATION: Sparkles,
}

export function InferenceNode({
  data,
  selected,
}: {
  data: { step: InferenceStep; faded?: boolean; spotlight?: boolean }
  selected: boolean
}) {
  const step = data.step
  const Icon = inferenceIcon[step.inferenceType]

  return (
    <div
      className={[
        'inference-node',
        selected ? 'selected-node' : '',
        data.faded ? 'faded-node' : '',
        data.spotlight ? 'spotlight-node' : '',
      ].join(' ')}
    >
      <Handle id="in" type="target" position={Position.Left} />
      <Handle id="out" type="source" position={Position.Right} />
      <div className="inference-icon">
        <Icon size={18} />
      </div>
      <div className="inference-content">
        <div className="inference-label">{step.inferenceType.replaceAll('_', ' ')}</div>
        <div className="inference-meta">
          {step.requiredAssumptions.length} assumptions
          {step.expertJudgment ? ' · expert' : ''}
        </div>
      </div>
    </div>
  )
}
