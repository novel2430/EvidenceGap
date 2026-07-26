import type { KeyboardEvent } from 'react'
import type { NodeProps } from '@xyflow/react'
import { Handle, Position } from '@xyflow/react'
import {
  CheckCircle2,
  CircleHelp,
  Scale,
  TriangleAlert,
  XCircle,
  type LucideIcon,
} from 'lucide-react'
import type { EvidenceState } from '../contracts'
import type { ClaimGraphNode } from '../graph/types'

const EVIDENCE_ICONS: Record<EvidenceState, LucideIcon> = {
  SUPPORTED: CheckCircle2,
  REFUTED: XCircle,
  CONFLICTED: Scale,
  INSUFFICIENT: CircleHelp,
  ERROR: TriangleAlert,
}

export function ClaimNode({ data }: NodeProps<ClaimGraphNode>) {
  const { claim, claimNumber, articleCounts, visual, onSelect } = data
  const EvidenceIcon = EVIDENCE_ICONS[claim.evidence_state]

  function selectClaim() {
    onSelect?.({ kind: 'claim', claimId: claim.claim_id })
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault()
      selectClaim()
    }
  }

  return (
    <div
      className={[
        'claim-node',
        `claim-node--${claim.evidence_state.toLowerCase()}`,
        visual.selected ? 'is-selected' : '',
        visual.spotlight ? 'is-spotlight' : '',
        visual.dimmed ? 'is-dimmed' : '',
        visual.terminal ? 'is-terminal' : '',
      ].filter(Boolean).join(' ')}
      role="button"
      tabIndex={0}
      onClick={selectClaim}
      onKeyDown={handleKeyDown}
      aria-label={`Claim ${claimNumber}: ${claim.display_text}`}
    >
      <Handle type="target" position={Position.Left} />
      <div className="graph-node-heading">
        <span>Claim {claimNumber}</span>
        <span className="role-badge">{claim.argument_role}</span>
      </div>
      <p>{claim.display_text}</p>
      <div className="claim-node-footer">
        <span className="evidence-state-badge">
          <EvidenceIcon size={13} />
          {claim.evidence_state}
        </span>
        <span className="article-counts" title="Retrieved Top Articles">
          S {articleCounts.support} · R {articleCounts.refute} · I {articleCounts.insufficient}
        </span>
      </div>
      {visual.terminal && <span className="terminal-badge">Terminal claim</span>}
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
