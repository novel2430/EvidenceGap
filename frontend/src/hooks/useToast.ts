import {
  createContext,
  useContext,
} from 'react'
import { UI_TEXT } from '../uiText'

export type ToastType = 'success' | 'info' | 'warning' | 'error' | 'loading'

interface ToastAction {
  label: string
  onClick: () => void | Promise<void>
}

export interface ToastInput {
  id?: string
  type: ToastType
  title: string
  description?: string
  duration?: number | null
  action?: ToastAction
}

export interface ToastContextValue {
  showToast: (toast: ToastInput) => string
  notifyOnce: (eventKey: string, toast: ToastInput) => string | null
  dismissToast: (toastId: string) => void
  trackOperation: (operationKey: string) => void
  untrackOperation: (operationKey: string) => void
  isOperationTracked: (operationKey: string) => boolean
}

export const ToastContext = createContext<ToastContextValue | null>(null)

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) {
    throw new Error(UI_TEXT.errors.toastProviderMissing)
  }
  return context
}
