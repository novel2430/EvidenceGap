import type {
  ArticleContextResponse,
  HealthResponse,
  LocalizationAcceptedResponse,
  LocalizationCreateRequest,
  LocalizationListResponse,
  LocalizationStatusResponse,
  PresentationBundle,
  RunAcceptedResponse,
  RunCreateRequest,
  RunListResponse,
  RunStatusResponse,
} from './contracts'
import { UI_TEXT } from './uiText'

export const DEFAULT_API_BASE_URL =
  'https://www.next.zju.edu.cn/novel-evidencegap/back/'

export interface EvidenceGapApiOptions {
  baseUrl?: string
  fetchFn?: typeof fetch
}

export interface ApiRequestOptions {
  signal?: AbortSignal
}

export interface ListRunsOptions extends ApiRequestOptions {
  limit?: number
  cursor?: string
}

export class EvidenceGapApiError extends Error {
  readonly status: number
  readonly url: string
  readonly detail: unknown

  constructor(message: string, status: number, url: string, detail: unknown) {
    super(message)
    this.name = 'EvidenceGapApiError'
    this.status = status
    this.url = url
    this.detail = detail
  }
}

function normalizeBaseUrl(value: string): string {
  const trimmed = value.trim()
  if (!trimmed) {
    throw new Error(UI_TEXT.errors.apiBaseBlank)
  }

  const fallbackOrigin =
    typeof window === 'undefined' ? DEFAULT_API_BASE_URL : window.location.origin
  const url = new URL(trimmed, fallbackOrigin)
  url.search = ''
  url.hash = ''

  if (!url.pathname.endsWith('/')) {
    url.pathname += '/'
  }

  return url.toString()
}

export function getConfiguredApiBaseUrl(): string {
  return normalizeBaseUrl(
    import.meta.env?.VITE_API_BASE_URL || DEFAULT_API_BASE_URL,
  )
}

function formatErrorDetail(detail: unknown, fallback: string): string {
  if (typeof detail === 'string' && detail.trim()) {
    return detail
  }

  if (detail !== undefined && detail !== null) {
    try {
      return JSON.stringify(detail)
    } catch {
      // Use the HTTP status text when the payload cannot be serialized.
    }
  }

  return fallback || UI_TEXT.errors.apiRequestFailed
}

async function buildApiError(response: Response): Promise<EvidenceGapApiError> {
  let detail: unknown

  try {
    const body = await response.text()
    try {
      const payload = JSON.parse(body) as { detail?: unknown }
      detail = payload.detail ?? body
    } catch {
      detail = body || undefined
    }
  } catch {
    detail = undefined
  }

  return new EvidenceGapApiError(
    formatErrorDetail(detail, response.statusText),
    response.status,
    response.url,
    detail,
  )
}

export function createEvidenceGapApi(options: EvidenceGapApiOptions = {}) {
  const baseUrl = normalizeBaseUrl(options.baseUrl ?? getConfiguredApiBaseUrl())
  const fetchFn = options.fetchFn ?? globalThis.fetch.bind(globalThis)

  function resolveUrl(path: string): string {
    return new URL(path.replace(/^\/+/, ''), baseUrl).toString()
  }

  async function requestJson<T>(
    path: string,
    init: RequestInit = {},
  ): Promise<T> {
    const response = await fetchFn(resolveUrl(path), {
      ...init,
      headers: {
        Accept: 'application/json',
        ...init.headers,
      },
    })

    if (!response.ok) {
      throw await buildApiError(response)
    }

    return (await response.json()) as T
  }

  async function postJson<TResponse, TRequest>(
    path: string,
    body: TRequest,
    requestOptions?: ApiRequestOptions,
  ): Promise<TResponse> {
    return requestJson<TResponse>(path, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
      signal: requestOptions?.signal,
    })
  }

  async function requestText(
    path: string,
    requestOptions?: ApiRequestOptions,
  ): Promise<string> {
    const response = await fetchFn(resolveUrl(path), {
      headers: {
        Accept: 'text/markdown, text/plain',
      },
      signal: requestOptions?.signal,
    })

    if (!response.ok) {
      throw await buildApiError(response)
    }

    return response.text()
  }

  async function requestBlob(
    path: string,
    accept: string,
    requestOptions?: ApiRequestOptions,
  ): Promise<Blob> {
    const response = await fetchFn(resolveUrl(path), {
      headers: { Accept: accept },
      signal: requestOptions?.signal,
    })

    if (!response.ok) {
      throw await buildApiError(response)
    }

    return response.blob()
  }

  return {
    baseUrl,

    getHealth(requestOptions?: ApiRequestOptions): Promise<HealthResponse> {
      return requestJson<HealthResponse>('/health', {
        signal: requestOptions?.signal,
      })
    },

    createRun(
      request: RunCreateRequest,
      requestOptions?: ApiRequestOptions,
    ): Promise<RunAcceptedResponse> {
      return postJson<RunAcceptedResponse, RunCreateRequest>(
        '/api/v1/runs',
        request,
        requestOptions,
      )
    },

    listRuns(options: ListRunsOptions = {}): Promise<RunListResponse> {
      const params = new URLSearchParams()
      if (options.limit !== undefined) {
        params.set('limit', String(options.limit))
      }
      if (options.cursor) {
        params.set('cursor', options.cursor)
      }

      const query = params.size > 0 ? `?${params.toString()}` : ''
      return requestJson<RunListResponse>(`/api/v1/runs${query}`, {
        signal: options.signal,
      })
    },

    getRun(
      runId: string,
      requestOptions?: ApiRequestOptions,
    ): Promise<RunStatusResponse> {
      return requestJson<RunStatusResponse>(
        `/api/v1/runs/${encodeURIComponent(runId)}`,
        { signal: requestOptions?.signal },
      )
    },

    getArticleContext(
      runId: string,
      articleNodeId: string,
      requestOptions?: ApiRequestOptions,
    ): Promise<ArticleContextResponse> {
      return requestJson<ArticleContextResponse>(
        `/api/v1/runs/${encodeURIComponent(runId)}/articles/${encodeURIComponent(articleNodeId)}`,
        { signal: requestOptions?.signal },
      )
    },

    getResultExport(
      runId: string,
      requestOptions?: ApiRequestOptions,
    ): Promise<PresentationBundle> {
      return requestJson<PresentationBundle>(
        `/api/v1/runs/${encodeURIComponent(runId)}/exports/result.json`,
        { signal: requestOptions?.signal },
      )
    },

    downloadResultExport(
      runId: string,
      requestOptions?: ApiRequestOptions,
    ): Promise<Blob> {
      return requestBlob(
        `/api/v1/runs/${encodeURIComponent(runId)}/exports/result.json`,
        'application/json',
        requestOptions,
      )
    },

    getMarkdownExport(
      runId: string,
      requestOptions?: ApiRequestOptions,
    ): Promise<string> {
      return requestText(
        `/api/v1/runs/${encodeURIComponent(runId)}/exports/report.md`,
        requestOptions,
      )
    },

    createLocalization(
      runId: string,
      request: LocalizationCreateRequest,
      requestOptions?: ApiRequestOptions,
    ): Promise<LocalizationAcceptedResponse> {
      return postJson<LocalizationAcceptedResponse, LocalizationCreateRequest>(
        `/api/v1/runs/${encodeURIComponent(runId)}/localizations`,
        request,
        requestOptions,
      )
    },

    listLocalizations(
      runId: string,
      requestOptions?: ApiRequestOptions,
    ): Promise<LocalizationListResponse> {
      return requestJson<LocalizationListResponse>(
        `/api/v1/runs/${encodeURIComponent(runId)}/localizations`,
        { signal: requestOptions?.signal },
      )
    },

    getLocalization(
      runId: string,
      localizationId: string,
      requestOptions?: ApiRequestOptions,
    ): Promise<LocalizationStatusResponse> {
      return requestJson<LocalizationStatusResponse>(
        `/api/v1/runs/${encodeURIComponent(runId)}/localizations/${encodeURIComponent(localizationId)}`,
        { signal: requestOptions?.signal },
      )
    },
  }
}

export type EvidenceGapApi = ReturnType<typeof createEvidenceGapApi>

export const evidenceGapApi = createEvidenceGapApi()
