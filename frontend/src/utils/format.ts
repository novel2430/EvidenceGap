import { EvidenceGapApiError } from '../api'

const createdAtFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

export function formatCreatedAt(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : createdAtFormatter.format(date)
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined || !Number.isFinite(seconds)) {
    return '—'
  }

  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`
  }

  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.round(seconds % 60)
  return `${minutes}m ${String(remainingSeconds).padStart(2, '0')}s`
}

export function getApiErrorMessage(error: unknown): string {
  if (error instanceof EvidenceGapApiError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'An unexpected error occurred.'
}
