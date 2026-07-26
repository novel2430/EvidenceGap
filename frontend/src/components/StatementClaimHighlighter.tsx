import { useMemo } from 'react'
import type { PresentationClaim } from '../contracts'
import type { GraphSelection } from '../utils/presentation'

interface StatementSegment {
  start: number
  end: number
  claimIds: string[]
}

interface StatementClaimHighlighterProps {
  originalText: string
  claims: PresentationClaim[]
  selection: GraphSelection | null
  onSelect: (selection: GraphSelection) => void
}

function segmentStatement(
  textLength: number,
  claims: PresentationClaim[],
): StatementSegment[] {
  const spansByClaim = new Map<string, Array<{ start: number; end: number }>>()
  const boundaries = new Set([0, textLength])

  for (const claim of claims) {
    const validSpans: Array<{ start: number; end: number }> = []
    for (const span of claim.source_spans) {
      const start = Math.max(0, Math.min(textLength, span.character_start))
      const end = Math.max(start, Math.min(textLength, span.character_end))
      if (start === end) continue
      validSpans.push({ start, end })
      boundaries.add(start)
      boundaries.add(end)
    }
    spansByClaim.set(claim.claim_id, validSpans)
  }

  const orderedBoundaries = [...boundaries].sort((a, b) => a - b)
  const segments: StatementSegment[] = []

  for (let index = 0; index < orderedBoundaries.length - 1; index += 1) {
    const start = orderedBoundaries[index]
    const end = orderedBoundaries[index + 1]
    if (start === end) continue

    const claimIds = claims
      .filter((claim) =>
        spansByClaim.get(claim.claim_id)?.some(
          (span) => span.start <= start && span.end >= end,
        ),
      )
      .map((claim) => claim.claim_id)

    const previous = segments.at(-1)
    if (
      previous &&
      previous.end === start &&
      previous.claimIds.length === claimIds.length &&
      previous.claimIds.every((claimId, claimIndex) => claimId === claimIds[claimIndex])
    ) {
      previous.end = end
    } else {
      segments.push({ start, end, claimIds })
    }
  }

  return segments
}

export function StatementClaimHighlighter({
  originalText,
  claims,
  selection,
  onSelect,
}: StatementClaimHighlighterProps) {
  const codePoints = useMemo(() => Array.from(originalText), [originalText])
  const segments = useMemo(
    () => segmentStatement(codePoints.length, claims),
    [claims, codePoints.length],
  )

  function selectOverlappingClaim(claimIds: string[]) {
    const selectedClaimIndex = selection?.kind === 'claim'
      ? claimIds.indexOf(selection.claimId)
      : -1
    const nextClaimId = claimIds[(selectedClaimIndex + 1) % claimIds.length]
    onSelect({ kind: 'claim', claimId: nextClaimId })
  }

  return (
    <p className="highlighted-statement">
      {segments.map((segment) => {
        const text = codePoints.slice(segment.start, segment.end).join('')
        if (segment.claimIds.length === 0) {
          return <span key={`${segment.start}-${segment.end}`}>{text}</span>
        }

        const isSelected = selection?.kind === 'claim' &&
          segment.claimIds.includes(selection.claimId)
        return (
          <button
            className={`statement-claim-span${isSelected ? ' is-selected' : ''}${segment.claimIds.length > 1 ? ' is-overlapping' : ''}`}
            type="button"
            key={`${segment.start}-${segment.end}`}
            onClick={() => selectOverlappingClaim(segment.claimIds)}
            aria-label={`Select ${segment.claimIds.length === 1 ? 'claim' : 'overlapping claim'}: ${text}`}
          >
            {text}
          </button>
        )
      })}
    </p>
  )
}
