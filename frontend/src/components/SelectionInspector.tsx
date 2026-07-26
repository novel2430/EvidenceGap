import { GitMerge, MousePointer2, TriangleAlert } from 'lucide-react'
import type {
  PresentationClaim,
  PresentationInferenceStep,
} from '../contracts'
import type {
  GraphSelection,
  PresentationIndexes,
} from '../utils/presentation'
import { EvidenceBalance } from './EvidenceBalance'

interface SelectionInspectorProps {
  selection: GraphSelection | null
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}

interface ClaimReferenceListProps {
  claimIds: string[]
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
  emptyLabel?: string
}

function ClaimReferenceList({
  claimIds,
  indexes,
  onSelect,
  emptyLabel = 'None',
}: ClaimReferenceListProps) {
  if (claimIds.length === 0) return <p className="reference-empty">{emptyLabel}</p>

  return (
    <div className="claim-reference-list">
      {claimIds.map((claimId) => {
        const claim = indexes.claimsById.get(claimId)
        return (
          <button
            type="button"
            key={claimId}
            onClick={() => onSelect({ kind: 'claim', claimId })}
          >
            {claim?.display_text ?? claimId}
          </button>
        )
      })}
    </div>
  )
}

function ImpactDetails({
  inferenceStep,
  indexes,
  onSelect,
}: {
  inferenceStep: PresentationInferenceStep
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}) {
  const impact = inferenceStep.impact
  return (
    <div className="impact-details">
      <h4>Direct conclusion</h4>
      <ClaimReferenceList
        claimIds={[impact.direct_conclusion_claim_id]}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>Downstream claims</h4>
      <ClaimReferenceList
        claimIds={impact.downstream_claim_ids}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>Terminal claims</h4>
      <ClaimReferenceList
        claimIds={impact.terminal_claim_ids}
        indexes={indexes}
        onSelect={onSelect}
      />
      <dl className="impact-summary">
        <div><dt>Downstream inference steps</dt><dd>{impact.downstream_inference_step_ids.length}</dd></div>
        <div><dt>Affects terminal conclusion</dt><dd>{impact.affects_terminal_conclusion ? 'Yes' : 'No'}</dd></div>
      </dl>
    </div>
  )
}

function ClaimDetails({
  claim,
  indexes,
}: {
  claim: PresentationClaim
  indexes: PresentationIndexes
}) {
  return (
    <section className="selection-card">
      <div className="selection-card-heading">
        <span className="eyebrow">Claim selection</span>
        <span className={`claim-list-state claim-list-state--${claim.evidence_state.toLowerCase()}`}>
          {claim.evidence_state}
        </span>
      </div>
      <h3>{claim.display_text}</h3>
      <dl className="claim-detail-fields">
        <div><dt>Original quote</dt><dd>{claim.source_text}</dd></div>
        <div><dt>Canonical English Claim</dt><dd>{claim.canonical_claim_en}</dd></div>
        <div><dt>Display text</dt><dd>{claim.display_text}</dd></div>
        <div><dt>Argument Role</dt><dd>{claim.argument_role}</dd></div>
        <div><dt>Evidence State</dt><dd>{claim.evidence_state}</dd></div>
      </dl>
      <EvidenceBalance
        claim={claim}
        articles={indexes.articlesByClaimId.get(claim.claim_id) ?? []}
      />
    </section>
  )
}

function InferenceDetails({
  inferenceStep,
  indexes,
  onSelect,
}: {
  inferenceStep: PresentationInferenceStep
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}) {
  return (
    <section className="selection-card">
      <div className="selection-card-heading">
        <span className="eyebrow">Inference selection</span>
        <GitMerge size={17} />
      </div>
      <h3>Inference Step</h3>
      <h4>Premise claims</h4>
      <ClaimReferenceList
        claimIds={inferenceStep.premise_claim_ids}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>Conclusion claim</h4>
      <ClaimReferenceList
        claimIds={[inferenceStep.conclusion_claim_id]}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>Detected gaps</h4>
      {inferenceStep.gaps.length === 0 ? (
        <p className="reference-empty">No Scope or Causal Gap detected.</p>
      ) : (
        <div className="inspector-gap-list">
          {inferenceStep.gaps.map((gap, gapIndex) => (
            <button
              type="button"
              key={`${gap.gap_type}-${gapIndex}`}
              onClick={() => onSelect({
                kind: 'gap',
                inferenceStepId: inferenceStep.inference_step_id,
                gapIndex,
              })}
            >
              <TriangleAlert size={13} />
              <span>
                <strong>{gap.gap_type === 'SCOPE_GAP' ? 'Scope Gap' : 'Causal Gap'}</strong>
                <small>{gap.display_reason}</small>
              </span>
            </button>
          ))}
        </div>
      )}
      <ImpactDetails
        inferenceStep={inferenceStep}
        indexes={indexes}
        onSelect={onSelect}
      />
    </section>
  )
}

export function SelectionInspector({
  selection,
  indexes,
  onSelect,
}: SelectionInspectorProps) {
  if (!selection) {
    return (
      <div className="selection-hint">
        <MousePointer2 size={18} />
        <div>
          <strong>Select an analysis element</strong>
          <p>Choose highlighted statement text, a Claim, an Inference Step, or a Gap.</p>
        </div>
      </div>
    )
  }

  if (selection.kind === 'claim') {
    const claim = indexes.claimsById.get(selection.claimId)
    return claim ? <ClaimDetails claim={claim} indexes={indexes} /> : null
  }

  const inferenceStep = indexes.inferenceStepsById.get(selection.inferenceStepId)
  if (!inferenceStep) return null

  if (selection.kind === 'inference_step') {
    return (
      <InferenceDetails
        inferenceStep={inferenceStep}
        indexes={indexes}
        onSelect={onSelect}
      />
    )
  }

  const gap = inferenceStep.gaps[selection.gapIndex]
  if (!gap) return null
  return (
    <section className="selection-card selection-card--gap">
      <div className="selection-card-heading">
        <span className="eyebrow">Gap selection</span>
        <TriangleAlert size={17} />
      </div>
      <h3>{gap.gap_type === 'SCOPE_GAP' ? 'Scope Gap' : 'Causal Gap'}</h3>
      <p className="gap-reason">{gap.display_reason}</p>
      <ImpactDetails
        inferenceStep={inferenceStep}
        indexes={indexes}
        onSelect={onSelect}
      />
    </section>
  )
}
