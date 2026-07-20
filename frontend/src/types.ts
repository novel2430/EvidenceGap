export type ClaimType =
  | 'OBSERVED_FACT'
  | 'COMPUTED_METRIC'
  | 'DOMAIN_RULE'
  | 'ASSUMPTION'
  | 'INFERRED_CLAIM'
  | 'RECOMMENDATION'

export type ClaimStatus =
  | 'SUPPORTED'
  | 'PARTIAL'
  | 'UNKNOWN'
  | 'CONTRADICTED'
  | 'CONFLICTED'
  | 'BLOCKED'

export type InferenceType =
  | 'DECISION'
  | 'DOMAIN_RULE_APPLICATION'
  | 'CAUSAL'
  | 'GENERALIZATION'

export type GapType =
  | 'PRIVATE_DATA_GAP'
  | 'DECISION_GAP'
  | 'SCOPE_GAP'
  | 'CAUSAL_GAP'
  | 'CONFLICT_GAP'

export type ViewMode = 'all' | 'gaps' | 'conflicts'

export interface EvidenceItem {
  id: string
  sourceId: string
  sourceType: string
  text: string
  role: 'SUPPORTING' | 'REFUTING' | 'CONTEXT' | 'SCOPE_LIMITATION'
  timeScope: string
  geography: string
  population: string
}

export interface Gap {
  id: string
  type: GapType
  target: string
  blocks: string[]
  reason: string
  requiredEvidence: string
  privateDataRequired: boolean
  priority: 'Critical' | 'High' | 'Medium'
}

export interface Claim {
  id: string
  text: string
  shortText: string
  type: ClaimType
  status: ClaimStatus
  reasonCodes: string[]
  evidenceIds: string[]
  gapIds: string[]
  downstream: string[]
  isTarget?: boolean
  position: {
    x: number
    y: number
  }
}

export interface InferenceStep {
  id: string
  premises: string[]
  conclusion: string
  inferenceType: InferenceType
  ruleDescription: string
  requiredAssumptions: string[]
  expertJudgment: boolean
  position: {
    x: number
    y: number
  }
}

export interface DemoCase {
  id: string
  title: string
  label: string
  caseType: string
  purpose: string
  status: ClaimStatus
  originalConclusion: string
  safeConclusion: string
  safeLimitations: string[]
  claims: Claim[]
  inferenceSteps: InferenceStep[]
  evidence: EvidenceItem[]
  gaps: Gap[]
  targetPath: string[]
  conflictPath: string[]
}

export type InspectorSelection =
  | { kind: 'claim'; id: string }
  | { kind: 'inference'; id: string }
