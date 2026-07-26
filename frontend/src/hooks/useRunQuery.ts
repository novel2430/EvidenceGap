import { useQuery } from '@tanstack/react-query'
import { evidenceGapApi } from '../api'

export function useRunQuery(runId: string | undefined) {
  return useQuery({
    queryKey: ['run', runId],
    queryFn: ({ signal }) => evidenceGapApi.getRun(runId!, { signal }),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'queued' || status === 'running' ? 1500 : false
    },
  })
}
