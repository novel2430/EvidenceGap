import { useQuery } from '@tanstack/react-query'
import { evidenceGapApi } from '../api'

export function useArticleContextQuery(
  runId: string | undefined,
  articleNodeId: string | null,
) {
  return useQuery({
    queryKey: ['article-context', runId, articleNodeId],
    queryFn: ({ signal }) =>
      evidenceGapApi.getArticleContext(runId!, articleNodeId!, { signal }),
    enabled: Boolean(runId && articleNodeId),
    staleTime: 5 * 60 * 1000,
  })
}
