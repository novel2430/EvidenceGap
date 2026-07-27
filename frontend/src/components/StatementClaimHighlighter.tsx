import { useMemo } from 'react'
import type { PresentationClaim } from '../contracts'
import type { GraphSelection } from '../utils/presentation'
import { segmentTextByOffsets } from '../utils/textSegments'
import { UI_TEXT } from '../uiText'

interface StatementClaimHighlighterProps {
  originalText: string
  claims: PresentationClaim[]
  selection: GraphSelection | null
  activeClaimId?: string | null
  onSelect: (selection: GraphSelection) => void
}

export function StatementClaimHighlighter({
  originalText,
  claims,
  selection,
  activeClaimId,
  onSelect,
}: StatementClaimHighlighterProps) {
  const selectedClaimId =
    activeClaimId ??
    (selection?.kind === 'claim' ? selection.claimId : null)
  const segmentedText = useMemo(
    () => segmentTextByOffsets(
      originalText,
      claims.flatMap((claim) =>
        claim.source_spans.map((span) => ({
          id: claim.claim_id,
          characterStart: span.character_start,
          characterEnd: span.character_end,
        })),
      ),
    ),
    [claims, originalText],
  )

  function selectOverlappingClaim(claimIds: string[]) {
    const selectedClaimIndex = selectedClaimId
      ? claimIds.indexOf(selectedClaimId)
      : -1
    const nextClaimId = claimIds[(selectedClaimIndex + 1) % claimIds.length]
    onSelect({ kind: 'claim', claimId: nextClaimId })
  }

  return (
    <p className="highlighted-statement">
      {segmentedText.segments.map((segment) => {
        const text = segmentedText.codePoints.slice(segment.start, segment.end).join('')
        if (segment.rangeIds.length === 0) {
          return <span key={`${segment.start}-${segment.end}`}>{text}</span>
        }

        const isSelected = Boolean(
          selectedClaimId && segment.rangeIds.includes(selectedClaimId),
        )
        return (
          <button
            className={`statement-claim-span${isSelected ? ' is-selected' : ''}${segment.rangeIds.length > 1 ? ' is-overlapping' : ''}`}
            type="button"
            key={`${segment.start}-${segment.end}`}
            onClick={() => selectOverlappingClaim(segment.rangeIds)}
            aria-label={UI_TEXT.claims.selectSpan(
              segment.rangeIds.length > 1,
              text,
            )}
          >
            {text}
          </button>
        )
      })}
    </p>
  )
}
