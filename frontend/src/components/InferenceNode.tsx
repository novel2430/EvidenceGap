import { GitBranch, Scale, Sparkles } from 'lucide-react'
import type { Node, NodeProps } from '@xyflow/react'
import type { GraphPort } from '../graph/claimGraphLayout'
import type { InferenceStep } from '../types'
import { GraphHandles } from './GraphHandles'

const inferenceIcon = {
  DECISION: Scale,
  DOMAIN_RULE_APPLICATION: GitBranch,
  CAUSAL: GitBranch,
  GENERALIZATION: Sparkles,
}

export type InferenceNodeType = Node<{
  step: InferenceStep
  ports: GraphPort[]
  faded?: boolean
  spotlight?: boolean
}, 'inference'>

export function InferenceNode({ data, selected }: NodeProps<InferenceNodeType>) {
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
      <GraphHandles ports={data.ports} />
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
