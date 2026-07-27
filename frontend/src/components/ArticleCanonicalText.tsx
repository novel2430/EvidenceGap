import { useEffect, useMemo, useRef, type RefObject } from 'react'
import type { ArticleContextResponse } from '../contracts'
import { useReducedMotion } from '../hooks/useReducedMotion'
import { segmentTextByOffsets } from '../utils/textSegments'
import { UI_TEXT } from '../uiText'

interface ArticleCanonicalTextProps {
  context: ArticleContextResponse
  activeEvidenceId: string | null
  onSelectEvidence: (evidenceId: string) => void
}

function sectionLabel(section: string): string {
  return section.trim() || UI_TEXT.common.sectionUnavailable
}

export function ArticleCanonicalText({
  context,
  activeEvidenceId,
  onSelectEvidence,
}: ArticleCanonicalTextProps) {
  const activeSpanRef = useRef<HTMLButtonElement>(null)
  const lastScrolledEvidenceId = useRef<string | null>(null)
  const reducedMotion = useReducedMotion()
  const segmentedText = useMemo(
    () => segmentTextByOffsets(
      context.canonical_text,
      context.evidence_spans.map((span) => ({
        id: span.evidence_id,
        characterStart: span.character_start,
        characterEnd: span.character_end,
      })),
      context.sections.flatMap((section) => [
        section.character_start,
        section.character_end,
      ]),
    ),
    [context],
  )
  const sectionsByStart = useMemo(() => {
    const result = new Map<number, typeof context.sections>()
    for (const section of [...context.sections].sort(
      (left, right) => left.character_start - right.character_start,
    )) {
      const sections = result.get(section.character_start) ?? []
      sections.push(section)
      result.set(section.character_start, sections)
    }
    return result
  }, [context.sections])
  const activeFirstStart = useMemo(
    () => context.evidence_spans.find(
      (span) => span.evidence_id === activeEvidenceId,
    )?.character_start ?? null,
    [activeEvidenceId, context.evidence_spans],
  )

  useEffect(() => {
    if (!activeEvidenceId) {
      lastScrolledEvidenceId.current = null
      return
    }

    if (
      !activeSpanRef.current ||
      lastScrolledEvidenceId.current === activeEvidenceId
    ) {
      return
    }

    const animationFrame = window.requestAnimationFrame(() => {
      activeSpanRef.current?.scrollIntoView({
        behavior: reducedMotion ? 'auto' : 'smooth',
        block: 'center',
        inline: 'nearest',
        container: 'nearest',
      } as ScrollIntoViewOptions & { container: 'nearest' })
      lastScrolledEvidenceId.current = activeEvidenceId
    })
    return () => window.cancelAnimationFrame(animationFrame)
  }, [activeEvidenceId, context.article_node_id, reducedMotion])

  function selectOverlappingEvidence(rangeIds: string[]) {
    const activeIndex = activeEvidenceId
      ? rangeIds.indexOf(activeEvidenceId)
      : -1
    onSelectEvidence(rangeIds[(activeIndex + 1) % rangeIds.length])
  }

  return (
    <div
      className="canonical-text-scroll"
      aria-label={UI_TEXT.articles.canonicalTextLabel}
    >
      <div className="canonical-text">
        {segmentedText.segments.map((segment) => {
          const text = segmentedText.codePoints
            .slice(segment.start, segment.end)
            .join('')
          const sectionHeadings = sectionsByStart.get(segment.start) ?? []
          const isActive = Boolean(
            activeEvidenceId && segment.rangeIds.includes(activeEvidenceId),
          )
          const isFirstActive = isActive && segment.start === activeFirstStart

          return (
            <span className="canonical-segment" key={`${segment.start}-${segment.end}`}>
              {sectionHeadings.map((section) => (
                <strong
                  className="canonical-section-heading"
                  key={`${section.section_index}-${section.character_start}`}
                >
                  {sectionLabel(section.section)}
                </strong>
              ))}
              {segment.rangeIds.length === 0 ? (
                text
              ) : (
                <button
                  ref={
                    isFirstActive
                      ? activeSpanRef as RefObject<HTMLButtonElement>
                      : undefined
                  }
                  className={`canonical-evidence-span${isActive ? ' is-active' : ''}${segment.rangeIds.length > 1 ? ' is-overlapping' : ''}`}
                  type="button"
                  onClick={() => selectOverlappingEvidence(segment.rangeIds)}
                >
                  {text}
                </button>
              )}
            </span>
          )
        })}
      </div>
    </div>
  )
}
