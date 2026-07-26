import { useQuery } from '@tanstack/react-query'
import { evidenceGapApi } from '../api'

export function useLocalizationList(
  runId: string | undefined,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ['localizations', runId],
    queryFn: ({ signal }) =>
      evidenceGapApi.listLocalizations(runId!, { signal }),
    enabled: Boolean(runId && enabled),
  })
}

export function useLocalizationQuery(
  runId: string | undefined,
  localizationId: string | null,
  enabled: boolean,
) {
  return useQuery({
    queryKey: ['localization', runId, localizationId],
    queryFn: ({ signal }) =>
      evidenceGapApi.getLocalization(runId!, localizationId!, { signal }),
    enabled: Boolean(runId && localizationId && enabled),
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === 'queued' || status === 'running' ? 1500 : false
    },
  })
}
