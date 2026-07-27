import type { KeyboardEvent } from 'react'
import type { NodeProps } from '@xyflow/react'
import { Handle, Position } from '@xyflow/react'
import {
  CheckCircle2,
  CircleHelp,
  Route,
  Scale,
  TriangleAlert,
  XCircle,
  type LucideIcon,
} from 'lucide-react'
import type { EvidenceState } from '../contracts'
import type { ClaimGraphNode } from '../graph/types'
import { UI_TEXT } from '../uiText'
import {
  getInferenceIntegrityLabel,
  getIntegrityClassName,
} from '../utils/presentationLabels'

const EVIDENCE_ICONS: Record<EvidenceState, LucideIcon> = {
  SUPPORTED: CheckCircle2,
  REFUTED: XCircle,
  CONFLICTED: Scale,
  INSUFFICIENT: CircleHelp,
  ERROR: TriangleAlert,
}

export function ClaimNode({ data }: NodeProps<ClaimGraphNode>) {
  const { claim, claimNumber, articleCounts, visual, onSelect } = data
  const evidenceStatus = claim.audit?.evidence_status ?? claim.evidence_state
  const inferenceIntegrity = claim.audit?.inference_integrity
  const EvidenceIcon = EVIDENCE_ICONS[evidenceStatus]

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
      aria-label={UI_TEXT.graph.claimAria(claimNumber, claim.display_text)}
    >
      <Handle type="target" position={Position.Left} />
      <div className="graph-node-heading">
        <span>{UI_TEXT.graph.claim(claimNumber)}</span>
        <span className="role-badge">{claim.argument_role}</span>
      </div>
      <p>{claim.display_text}</p>
      <div className="claim-node-footer">
        <div className="claim-axis-badges">
          <span className={`evidence-state-badge evidence-state-badge--${evidenceStatus.toLowerCase()}`}>
            <EvidenceIcon size={12} />
            {evidenceStatus}
          </span>
          <span className={`inference-integrity-badge inference-integrity-badge--${getIntegrityClassName(inferenceIntegrity)}`}>
            <Route size={11} />
            {getInferenceIntegrityLabel(inferenceIntegrity, true)}
          </span>
        </div>
        <span
          className="article-counts"
          title={UI_TEXT.graph.retrievedTopArticles}
        >
          {UI_TEXT.graph.articleCounts(
            articleCounts.support,
            articleCounts.refute,
            articleCounts.insufficient,
          )}
        </span>
      </div>
      {visual.terminal && (
        <span className="terminal-badge">{UI_TEXT.graph.terminalClaim}</span>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  )
}
