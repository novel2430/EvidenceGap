export interface OffsetRange {
  id: string
  characterStart: number
  characterEnd: number
}

export interface OffsetSegment {
  start: number
  end: number
  rangeIds: string[]
}

export function segmentTextByOffsets(
  text: string,
  ranges: OffsetRange[],
  extraBoundaries: number[] = [],
): { codePoints: string[]; segments: OffsetSegment[] } {
  const codePoints = Array.from(text)
  const textLength = codePoints.length
  const validRanges: OffsetRange[] = []
  const boundaries = new Set([0, textLength])

  for (const range of ranges) {
    const characterStart = Math.max(
      0,
      Math.min(textLength, range.characterStart),
    )
    const characterEnd = Math.max(
      characterStart,
      Math.min(textLength, range.characterEnd),
    )
    if (characterStart === characterEnd) continue
    validRanges.push({ ...range, characterStart, characterEnd })
    boundaries.add(characterStart)
    boundaries.add(characterEnd)
  }

  for (const boundary of extraBoundaries) {
    boundaries.add(Math.max(0, Math.min(textLength, boundary)))
  }

  const orderedBoundaries = [...boundaries].sort((left, right) => left - right)
  const segments: OffsetSegment[] = []

  for (let index = 0; index < orderedBoundaries.length - 1; index += 1) {
    const start = orderedBoundaries[index]
    const end = orderedBoundaries[index + 1]
    if (start === end) continue

    const rangeIds = validRanges
      .filter(
        (range) =>
          range.characterStart <= start && range.characterEnd >= end,
      )
      .map((range) => range.id)

    const previous = segments.at(-1)
    if (
      previous &&
      previous.end === start &&
      previous.rangeIds.length === rangeIds.length &&
      previous.rangeIds.every((rangeId, rangeIndex) => rangeId === rangeIds[rangeIndex]) &&
      !extraBoundaries.includes(start)
    ) {
      previous.end = end
    } else {
      segments.push({ start, end, rangeIds })
    }
  }

  return { codePoints, segments }
}
