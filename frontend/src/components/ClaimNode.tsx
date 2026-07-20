import {
  AlertTriangle,
  Ban,
  CircleHelp,
  Database,
  FileSearch,
  ShieldCheck,
  Sparkles,
  Target,
} from 'lucide-react'
import type { Node, NodeProps } from '@xyflow/react'
import type { GraphPort } from '../graph/claimGraphLayout'
import type { Claim } from '../types'
import { GraphHandles } from './GraphHandles'

const statusIcon = {
  SUPPORTED: ShieldCheck,
  PARTIAL: CircleHelp,
  UNKNOWN: CircleHelp,
  CONTRADICTED: Ban,
  CONFLICTED: AlertTriangle,
  BLOCKED: AlertTriangle,
}

const typeIcon = {
  OBSERVED_FACT: FileSearch,
  COMPUTED_METRIC: Database,
  DOMAIN_RULE: ShieldCheck,
  ASSUMPTION: CircleHelp,
  INFERRED_CLAIM: Sparkles,
  RECOMMENDATION: Target,
}

export type ClaimNodeType = Node<{
  claim: Claim
  ports: GraphPort[]
  faded?: boolean
  spotlight?: boolean
}, 'claim'>

export function ClaimNode({ data, selected }: NodeProps<ClaimNodeType>) {
  const claim = data.claim
  const StatusIcon = statusIcon[claim.status]
  const TypeIcon = typeIcon[claim.type]

  return (
    <div
      className={[
        'claim-node',
        `status-${claim.status.toLowerCase()}`,
        `type-${claim.type.toLowerCase()}`,
        claim.isTarget ? 'target-node' : '',
        selected ? 'selected-node' : '',
        data.faded ? 'faded-node' : '',
        data.spotlight ? 'spotlight-node' : '',
      ].join(' ')}
    >
      <GraphHandles ports={data.ports} />
      <div className="node-topline">
        <span className="type-badge">
          <TypeIcon size={13} />
          {claim.type.replace('_', ' ')}
        </span>
        <span className="node-id">{claim.id}</span>
      </div>
      <div className="node-title">{claim.shortText}</div>
      <div className="node-footer">
        <div className="node-metrics">
          <span>{claim.evidenceIds.length} Evidence</span>
          <span>{claim.gapIds.length} Gap</span>
        </div>
        <div className="node-status">
          <StatusIcon size={14} />
          {claim.status}
        </div>
      </div>
    </div>
  )
}
