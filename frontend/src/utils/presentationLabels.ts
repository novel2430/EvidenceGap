import type {
  ArticleApplicability,
  EvidenceStatus,
  GapType,
  InferenceIntegrity,
  InferenceStepIntegrity,
} from '../contracts'
import { UI_TEXT } from '../uiText'

export const APPLICABILITY_DIMENSIONS: ReadonlyArray<
  keyof ArticleApplicability
> = [
  'population_or_species',
  'intervention_or_exposure',
  'comparator',
  'outcome',
  'direction',
  'timeframe',
  'causal_strength',
  'prevention_treatment_scope',
]

export function formatEnumLabel(value: string | null | undefined): string {
  if (!value?.trim()) return UI_TEXT.common.unavailable
  return value
    .toLowerCase()
    .split('_')
    .filter(Boolean)
    .map((part) => `${part.charAt(0).toUpperCase()}${part.slice(1)}`)
    .join(' ')
}

export function getEvidenceStatusLabel(
  status: EvidenceStatus | null | undefined,
): string {
  return status ?? UI_TEXT.common.unavailable
}

export function getInferenceIntegrityLabel(
  integrity: InferenceIntegrity | InferenceStepIntegrity | null | undefined,
  compact = false,
): string {
  if (!integrity) return UI_TEXT.common.unavailable
  if (integrity === 'NOT_APPLICABLE') {
    return compact
      ? UI_TEXT.presentationLabels.notApplicableCompact
      : UI_TEXT.presentationLabels.notApplicable
  }
  return integrity
}

export function getGapTypeLabel(gapType: GapType): string {
  return gapType === 'SCOPE_GAP'
    ? UI_TEXT.presentationLabels.scopeGap
    : UI_TEXT.presentationLabels.causalGap
}

export function getIntegrityClassName(
  integrity: InferenceIntegrity | InferenceStepIntegrity | null | undefined,
): string {
  return integrity?.toLowerCase().replaceAll('_', '-') ?? 'unavailable'
}

export function formatApplicabilityDimension(
  dimension: string | null | undefined,
): string {
  if (
    dimension &&
    Object.hasOwn(UI_TEXT.applicability.dimensions, dimension)
  ) {
    return UI_TEXT.applicability.dimensions[
      dimension as keyof ArticleApplicability
    ]
  }
  return formatEnumLabel(dimension)
}

export function getApplicabilityStatusLabel(
  status: string | null | undefined,
): string {
  switch (status) {
    case 'MATCH':
      return UI_TEXT.applicability.statuses.MATCH
    case 'MISMATCH':
      return UI_TEXT.applicability.statuses.MISMATCH
    case 'NOT_REPORTED':
      return UI_TEXT.applicability.statuses.NOT_REPORTED
    case 'NOT_APPLICABLE':
      return UI_TEXT.applicability.statuses.NOT_APPLICABLE
    default:
      return status?.trim() || UI_TEXT.common.unavailable
  }
}

export function getApplicabilityStatusClassName(
  status: string | null | undefined,
): string {
  switch (status) {
    case 'MATCH':
      return 'match'
    case 'MISMATCH':
      return 'mismatch'
    case 'NOT_REPORTED':
      return 'not-reported'
    case 'NOT_APPLICABLE':
      return 'not-applicable'
    default:
      return 'unavailable'
  }
}
