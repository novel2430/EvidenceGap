import { EvidenceGapApiError } from '../api'
import { UI_TEXT } from '../uiText'

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
    return UI_TEXT.common.dash
  }

  if (seconds < 60) {
    return UI_TEXT.common.duration.seconds(seconds)
  }

  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = Math.round(seconds % 60)
  return UI_TEXT.common.duration.minutes(minutes, remainingSeconds)
}

export function getApiErrorMessage(error: unknown): string {
  if (typeof error === 'string' && error.trim()) {
    return error.trim()
  }
  if (error instanceof EvidenceGapApiError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return UI_TEXT.errors.unexpected
}

export function getToastErrorMessage(error: unknown): string {
  const message = getApiErrorMessage(error).replace(/\s+/g, ' ').trim()
  if (!message || message === '[object Object]') {
    return UI_TEXT.errors.operationFailed
  }
  return message.length > 180 ? `${message.slice(0, 177)}…` : message
}
