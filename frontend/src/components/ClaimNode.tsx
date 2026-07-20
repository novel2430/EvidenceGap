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
import { Handle, Position } from '@xyflow/react'
import type { Claim } from '../types'

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

export function ClaimNode({ data, selected }: { data: { claim: Claim; faded?: boolean; spotlight?: boolean }; selected: boolean }) {
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
      <Handle id="in" type="target" position={Position.Left} />
      <Handle id="out" type="source" position={Position.Right} />
      <div className="node-topline">
        <span className="type-badge">
          <TypeIcon size={13} />
          {claim.type.replace('_', ' ')}
        </span>
        <span className="node-id">{claim.id}</span>
      </div>
      <div className="node-title">{claim.shortText}</div>
      <div className="node-metrics">
        <span>{claim.evidenceIds.length} Evidence</span>
        <span>{claim.gapIds.length} Gap</span>
      </div>
      <div className="node-status">
        <StatusIcon size={15} />
        {claim.status}
      </div>
    </div>
  )
}
