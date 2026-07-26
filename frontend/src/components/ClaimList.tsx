import type { PresentationClaim } from '../contracts'
import type { GraphSelection } from '../utils/presentation'

interface ClaimListProps {
  claims: PresentationClaim[]
  selection: GraphSelection | null
  onSelect: (selection: GraphSelection) => void
}

export function ClaimList({ claims, selection, onSelect }: ClaimListProps) {
  return (
    <section className="claim-list-section">
      <div className="inspector-section-heading">
        <h3>Claims</h3>
        <span>{claims.length}</span>
      </div>
      <div className="claim-list">
        {claims.map((claim, index) => {
          const isSelected =
            selection?.kind === 'claim' && selection.claimId === claim.claim_id
          return (
            <button
              className={`claim-list-item${isSelected ? ' is-selected' : ''}`}
              type="button"
              key={claim.claim_id}
              onClick={() => onSelect({ kind: 'claim', claimId: claim.claim_id })}
            >
              <span className="claim-list-number">{index + 1}</span>
              <span className="claim-list-copy">
                <strong>{claim.display_text}</strong>
                <span>
                  <span className="claim-list-role">{claim.argument_role}</span>
                  <span className={`claim-list-state claim-list-state--${claim.evidence_state.toLowerCase()}`}>
                    {claim.evidence_state}
                  </span>
                </span>
              </span>
            </button>
          )
        })}
      </div>
    </section>
  )
}
