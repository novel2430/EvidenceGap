import type {
  ArticleApplicability,
  EvidenceStatus,
  GapType,
  InferenceIntegrity,
  InferenceStepIntegrity,
} from '../contracts'

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

const APPLICABILITY_DIMENSION_LABELS: Record<
  keyof ArticleApplicability,
  string
> = {
  population_or_species: 'Population or species',
  intervention_or_exposure: 'Intervention or exposure',
  comparator: 'Comparator',
  outcome: 'Outcome',
  direction: 'Direction',
  timeframe: 'Timeframe',
  causal_strength: 'Causal strength',
  prevention_treatment_scope: 'Prevention / treatment scope',
}

export function formatEnumLabel(value: string | null | undefined): string {
  if (!value?.trim()) return 'Unavailable'
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
  return status ?? 'Unavailable'
}

export function getInferenceIntegrityLabel(
  integrity: InferenceIntegrity | InferenceStepIntegrity | null | undefined,
  compact = false,
): string {
  if (!integrity) return 'Unavailable'
  if (integrity === 'NOT_APPLICABLE') {
    return compact ? 'N/A' : 'Not applicable'
  }
  return integrity
}

export function getGapTypeLabel(gapType: GapType): string {
  return gapType === 'SCOPE_GAP' ? 'Scope Gap' : 'Causal Gap'
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
    Object.hasOwn(APPLICABILITY_DIMENSION_LABELS, dimension)
  ) {
    return APPLICABILITY_DIMENSION_LABELS[
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
      return 'Match'
    case 'MISMATCH':
      return 'Mismatch'
    case 'NOT_REPORTED':
      return 'Not reported'
    case 'NOT_APPLICABLE':
      return 'Not applicable'
    default:
      return status?.trim() || 'Unavailable'
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
