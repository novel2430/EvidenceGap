import type {
  PresentationArticle,
  PresentationBundle,
  PresentationClaim,
  PresentationEvidence,
  PresentationInferenceStep,
} from '../contracts'

export type GraphFocusMode = 'all' | 'gaps' | 'conflicts'

export type GraphSelection =
  | { kind: 'claim'; claimId: string }
  | { kind: 'inference_step'; inferenceStepId: string }
  | { kind: 'gap'; inferenceStepId: string; gapIndex: number }
  | { kind: 'article'; articleNodeId: string }
  | { kind: 'evidence'; evidenceId: string }

export interface ArticleStanceCounts {
  support: number
  refute: number
  insufficient: number
  total: number
}

export interface PresentationIndexes {
  claimsById: Map<string, PresentationClaim>
  inferenceStepsById: Map<string, PresentationInferenceStep>
  articlesById: Map<string, PresentationArticle>
  articlesByClaimId: Map<string, PresentationArticle[]>
  evidenceById: Map<string, PresentationEvidence>
  evidenceByArticleId: Map<string, PresentationEvidence[]>
  inferenceStepsByClaimId: Map<string, PresentationInferenceStep[]>
}

export function buildPresentationIndexes(
  presentation: PresentationBundle | null,
): PresentationIndexes {
  const claimsById = new Map<string, PresentationClaim>()
  const inferenceStepsById = new Map<string, PresentationInferenceStep>()
  const articlesById = new Map<string, PresentationArticle>()
  const articlesByClaimId = new Map<string, PresentationArticle[]>()
  const evidenceById = new Map<string, PresentationEvidence>()
  const evidenceByArticleId = new Map<string, PresentationEvidence[]>()
  const inferenceStepsByClaimId = new Map<string, PresentationInferenceStep[]>()

  if (!presentation) {
    return {
      claimsById,
      inferenceStepsById,
      articlesById,
      articlesByClaimId,
      evidenceById,
      evidenceByArticleId,
      inferenceStepsByClaimId,
    }
  }

  for (const claim of presentation.claims) {
    claimsById.set(claim.claim_id, claim)
  }

  for (const evidence of presentation.evidence) {
    evidenceById.set(evidence.evidence_id, evidence)
  }

  for (const article of presentation.articles) {
    articlesById.set(article.article_node_id, article)
    const articles = articlesByClaimId.get(article.claim_id) ?? []
    articles.push(article)
    articlesByClaimId.set(article.claim_id, articles)
    evidenceByArticleId.set(
      article.article_node_id,
      article.evidence_ids
        .map((evidenceId) => evidenceById.get(evidenceId))
        .filter((evidence): evidence is PresentationEvidence => Boolean(evidence)),
    )
  }

  for (const articles of articlesByClaimId.values()) {
    articles.sort((left, right) => left.rank - right.rank)
  }

  for (const inferenceStep of presentation.inference_steps) {
    inferenceStepsById.set(inferenceStep.inference_step_id, inferenceStep)
    const relatedClaimIds = new Set([
      ...inferenceStep.premise_claim_ids,
      inferenceStep.conclusion_claim_id,
    ])

    for (const claimId of relatedClaimIds) {
      const inferenceSteps = inferenceStepsByClaimId.get(claimId) ?? []
      inferenceSteps.push(inferenceStep)
      inferenceStepsByClaimId.set(claimId, inferenceSteps)
    }
  }

  return {
    claimsById,
    inferenceStepsById,
    articlesById,
    articlesByClaimId,
    evidenceById,
    evidenceByArticleId,
    inferenceStepsByClaimId,
  }
}

export function countArticleStances(
  articles: PresentationArticle[],
): ArticleStanceCounts {
  const counts: ArticleStanceCounts = {
    support: 0,
    refute: 0,
    insufficient: 0,
    total: articles.length,
  }

  for (const article of articles) {
    counts[article.stance] += 1
  }

  return counts
}

export function isSelectionValid(
  selection: GraphSelection,
  indexes: PresentationIndexes,
): boolean {
  if (selection.kind === 'claim') {
    return indexes.claimsById.has(selection.claimId)
  }

  if (selection.kind === 'article') {
    return indexes.articlesById.has(selection.articleNodeId)
  }

  if (selection.kind === 'evidence') {
    return indexes.evidenceById.has(selection.evidenceId)
  }

  const inferenceStep = indexes.inferenceStepsById.get(selection.inferenceStepId)
  if (!inferenceStep) return false
  return selection.kind === 'inference_step' ||
    Boolean(inferenceStep.gaps[selection.gapIndex])
}

export function getSelectionClaimId(
  selection: GraphSelection | null,
  indexes: PresentationIndexes,
): string | null {
  if (!selection) return null
  if (selection.kind === 'claim') return selection.claimId
  if (selection.kind === 'article') {
    return indexes.articlesById.get(selection.articleNodeId)?.claim_id ?? null
  }
  if (selection.kind === 'evidence') {
    return indexes.evidenceById.get(selection.evidenceId)?.claim_id ?? null
  }
  return null
}

export function getSelectionArticleNodeId(
  selection: GraphSelection | null,
  indexes: PresentationIndexes,
): string | null {
  if (!selection) return null
  if (selection.kind === 'article') return selection.articleNodeId
  if (selection.kind === 'evidence') {
    return indexes.evidenceById.get(selection.evidenceId)?.article_node_id ?? null
  }
  return null
}
