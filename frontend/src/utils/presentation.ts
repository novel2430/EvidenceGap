import type {
  PresentationArticle,
  PresentationBundle,
  PresentationClaim,
  PresentationInferenceStep,
} from '../contracts'

export type GraphFocusMode = 'all' | 'gaps' | 'conflicts'

export type GraphSelection =
  | { kind: 'claim'; claimId: string }
  | { kind: 'inference_step'; inferenceStepId: string }
  | { kind: 'gap'; inferenceStepId: string; gapIndex: number }

export interface ArticleStanceCounts {
  support: number
  refute: number
  insufficient: number
  total: number
}

export interface PresentationIndexes {
  claimsById: Map<string, PresentationClaim>
  inferenceStepsById: Map<string, PresentationInferenceStep>
  articlesByClaimId: Map<string, PresentationArticle[]>
  inferenceStepsByClaimId: Map<string, PresentationInferenceStep[]>
}

export function buildPresentationIndexes(
  presentation: PresentationBundle | null,
): PresentationIndexes {
  const claimsById = new Map<string, PresentationClaim>()
  const inferenceStepsById = new Map<string, PresentationInferenceStep>()
  const articlesByClaimId = new Map<string, PresentationArticle[]>()
  const inferenceStepsByClaimId = new Map<string, PresentationInferenceStep[]>()

  if (!presentation) {
    return {
      claimsById,
      inferenceStepsById,
      articlesByClaimId,
      inferenceStepsByClaimId,
    }
  }

  for (const claim of presentation.claims) {
    claimsById.set(claim.claim_id, claim)
  }

  for (const article of presentation.articles) {
    const articles = articlesByClaimId.get(article.claim_id) ?? []
    articles.push(article)
    articlesByClaimId.set(article.claim_id, articles)
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
    articlesByClaimId,
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

  const inferenceStep = indexes.inferenceStepsById.get(selection.inferenceStepId)
  if (!inferenceStep) return false
  return selection.kind === 'inference_step' ||
    Boolean(inferenceStep.gaps[selection.gapIndex])
}
