export type RunStatus = 'queued' | 'running' | 'succeeded' | 'failed'

export type RunStage =
  | 'statement_decomposition'
  | 'claim_analysis'
  | 'statement_bundle'
  | 'inference_gap_analysis'
  | 'output_generation'

export type ClaimAnalysisStatus = 'completed' | 'failed'
export type ClaimVerdict = 'supported' | 'refuted' | 'mixed' | 'insufficient'
export type EvidenceState =
  | 'SUPPORTED'
  | 'REFUTED'
  | 'CONFLICTED'
  | 'INSUFFICIENT'
  | 'ERROR'
export type ArgumentRole = 'PREMISE' | 'INTERMEDIATE' | 'CONCLUSION' | 'STANDALONE'
export type GapType = 'SCOPE_GAP' | 'CAUSAL_GAP'
export type ArticleStance = 'support' | 'refute' | 'insufficient'

export interface RunCreateRequest {
  statement: string
  language?: string | null
}

export interface RunAcceptedResponse {
  run_id: string
  status: 'queued'
  created_at: string
}

export interface RunErrorResponse {
  code: string
  message: string
}

export interface RunProgressResponse {
  stage: RunStage
  stage_index: number
  total_stages: number
  message: string
  completed_units: number | null
  total_units: number | null
  updated_at: string
}

export interface RunStatusResponse {
  run_id: string
  status: RunStatus
  language: string
  created_at: string
  started_at: string | null
  finished_at: string | null
  progress: RunProgressResponse | null
  execution_summary: Record<string, unknown> | null
  error: RunErrorResponse | null
  result: PresentationBundle | null
}

export interface RunListItemResponse {
  run_id: string
  statement_preview: string
  language: string
  status: RunStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  total_seconds: number | null
  summary: PresentationSummary | null
  error: RunErrorResponse | null
}

export interface RunListResponse {
  runs: RunListItemResponse[]
  next_cursor: string | null
}

export interface AnalysisContext {
  schema_version: '1.0.0'
  scope: 'retrieved_top_articles'
  is_systematic_review: false
  is_clinical_recommendation: false
  is_final_medical_truth: false
  aggregation_method: 'deterministic_article_count'
  uses_confidence_weighting: false
  retrieval_methods: ['BM25', 'MedCPT', 'BMRetriever']
  fusion_method: 'reciprocal_rank_fusion'
  reranker: 'MedCPT Cross-Encoder'
  source_depth: number
  dense_nprobe: number
  rrf_k: number
  rerank_depth: number
  article_top_k: number
  max_evidence_sentences_per_article: number
}

export interface PresentationStatement {
  statement_id: string
  original_text: string
  source_language: string
  analysis_status: string
  display_text: string
}

export interface SourceSpan {
  character_start: number
  character_end: number
}

export interface PresentationClaim {
  claim_id: string
  source_text: string
  source_spans: SourceSpan[]
  canonical_claim_en: string
  analysis_status: ClaimAnalysisStatus
  verdict: ClaimVerdict | null
  article_counts: Record<string, number> | null
  rationale: string | null
  scope: string | null
  boundary: Record<string, unknown> | null
  article_node_ids: string[]
  error: string | null
  evidence_state: EvidenceState
  argument_role: ArgumentRole
  premise_inference_step_ids: string[]
  conclusion_inference_step_ids: string[]
  display_text: string
  display_rationale: string | null
}

export interface InferenceImpact {
  direct_conclusion_claim_id: string
  downstream_claim_ids: string[]
  downstream_inference_step_ids: string[]
  terminal_claim_ids: string[]
  affects_terminal_conclusion: boolean
  cycle_detected: boolean
}

export interface PresentationGap {
  gap_type: GapType
  detection_method: 'llm'
  reason_en: string
  display_reason: string
}

export interface PresentationInferenceStep {
  inference_step_id: string
  premise_claim_ids: string[]
  conclusion_claim_id: string
  impact: InferenceImpact
  gaps: PresentationGap[]
}

export interface RetrievalRankScore {
  rank: number | null
  score: number | null
}

export interface RetrievalTrace {
  bm25: RetrievalRankScore
  medcpt: RetrievalRankScore
  bmretriever: RetrievalRankScore
  fusion: {
    rank: number
    rrf_score: number
  }
  cross_encoder: {
    score: number
  }
  final_article_rank: number
}

export interface PresentationArticle {
  article_node_id: string
  claim_id: string
  article_id: string
  pmid: string | null
  rank: number
  retrieval_trace: RetrievalTrace
  title: string
  rationale: string
  stance: ArticleStance
  confidence: number
  probabilities: Record<ArticleStance, number>
  evidence_ids: string[]
  provider: string | null
  model: string | null
  model_fingerprint: string | null
  prompt_version: string | null
  display_title: string
  display_rationale: string
}

export interface PresentationEvidence {
  evidence_id: string
  source_node_id: string
  claim_id: string
  article_node_id: string
  article_id: string
  pmid: string | null
  label: string
  text: string
  source_evidence_id: string | null
  sentence_id: string
  sentence_index: number
  sentence_index_within_section: number
  section: string | null
  section_index: number
  character_start: number
  character_end: number
  source_text_fingerprint: string | null
  splitter_fingerprint: string | null
  display_text: string
}

export interface PresentationSummary {
  total_claims: number
  evidence_states: Record<EvidenceState, number>
  argument_roles: Record<ArgumentRole, number>
  total_inference_steps: number
  gaps: Record<GapType, number>
  articles: number
  evidence: number
}

export interface PresentationBundle {
  schema_version: '1.1.0'
  contract_id: 'phase077.presentation-bundle.v1'
  output_language: string
  localized: boolean
  source_statement_bundle_sha256: string
  source_inference_gap_analysis_sha256: string
  analysis_context: AnalysisContext
  statement: PresentationStatement
  claims: PresentationClaim[]
  inference_steps: PresentationInferenceStep[]
  articles: PresentationArticle[]
  evidence: PresentationEvidence[]
  summary: PresentationSummary
}

export interface ArticleSectionSpanResponse {
  sentence_type: string
  section: string
  section_index: number
  character_start: number
  character_end: number
}

export interface ArticleEvidenceSpanResponse {
  evidence_id: string
  claim_id: string
  section: string | null
  section_index: number
  sentence_index: number
  character_start: number
  character_end: number
  text: string
}

export interface ArticleContextResponse {
  article_node_id: string
  article_id: string
  claim_id: string
  pmid: string | null
  title: string | null
  canonical_text: string
  source_text_fingerprint: string
  fingerprint_verified: boolean
  sections: ArticleSectionSpanResponse[]
  evidence_spans: ArticleEvidenceSpanResponse[]
}

export type InspectorSelection =
  | { kind: 'claim'; claimId: string }
  | { kind: 'inference_step'; inferenceStepId: string }
  | { kind: 'article'; articleNodeId: string }
  | { kind: 'evidence'; evidenceId: string }

export interface LocalizationCreateRequest {
  language: string
}

export interface LocalizationAcceptedResponse {
  localization_id: string
  source_run_id: string
  language: string
  status: 'queued'
  created_at: string
}

export interface LocalizationStatusResponse {
  localization_id: string
  source_run_id: string
  language: string
  status: RunStatus
  created_at: string
  started_at: string | null
  finished_at: string | null
  error: RunErrorResponse | null
  result: PresentationBundle | null
}

export interface LocalizationListResponse {
  localizations: LocalizationStatusResponse[]
}

export interface HealthResponse {
  status: 'ok'
  engine_loaded: boolean
  worker_alive: boolean
  active_run_id: string | null
  queued_runs: number
  load_count: number
  analysis_runs: number
}
