import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Info,
  LoaderCircle,
  X,
  XCircle,
} from 'lucide-react'
import {
  ToastContext,
  type ToastContextValue,
  type ToastInput,
  type ToastType,
} from '../hooks/useToast'
import { UI_TEXT } from '../uiText'

interface ToastItem extends ToastInput {
  id: string
}

const DEFAULT_DURATIONS: Record<ToastType, number | null> = {
  success: 3500,
  info: 3500,
  warning: 5500,
  error: 7000,
  loading: null,
}

const TOAST_ICONS = {
  success: CheckCircle2,
  info: Info,
  warning: AlertTriangle,
  error: XCircle,
  loading: LoaderCircle,
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const timers = useRef(new Map<string, number>())
  const notifiedEvents = useRef(new Set<string>())
  const trackedOperations = useRef(new Set<string>())
  const nextToastId = useRef(0)

  const dismissToast = useCallback((toastId: string) => {
    const timer = timers.current.get(toastId)
    if (timer !== undefined) window.clearTimeout(timer)
    timers.current.delete(toastId)
    setToasts((current) => current.filter((toast) => toast.id !== toastId))
  }, [])

  const showToast = useCallback((input: ToastInput) => {
    const id = input.id ?? `toast-${++nextToastId.current}`
    const toast = { ...input, id }
    setToasts((current) => {
      const existingIndex = current.findIndex((item) => item.id === id)
      if (existingIndex === -1) return [...current.slice(-3), toast]
      return current.map((item) => item.id === id ? toast : item)
    })

    const existingTimer = timers.current.get(id)
    if (existingTimer !== undefined) window.clearTimeout(existingTimer)
    timers.current.delete(id)
    const duration = input.duration === undefined
      ? DEFAULT_DURATIONS[input.type]
      : input.duration
    if (duration !== null) {
      timers.current.set(id, window.setTimeout(() => {
        timers.current.delete(id)
        setToasts((current) => current.filter((item) => item.id !== id))
      }, duration))
    }
    return id
  }, [])

  const notifyOnce = useCallback((
    eventKey: string,
    toast: ToastInput,
  ) => {
    if (notifiedEvents.current.has(eventKey)) return null
    notifiedEvents.current.add(eventKey)
    return showToast(toast)
  }, [showToast])

  const trackOperation = useCallback((operationKey: string) => {
    const operationType = operationKey.split(':', 1)[0]
    if (operationType === 'run' || operationType === 'localization') {
      for (const trackedKey of trackedOperations.current) {
        if (
          trackedKey.startsWith(`${operationType}:`) ||
          (
            operationType === 'run' &&
            trackedKey.startsWith('localization:')
          )
        ) {
          trackedOperations.current.delete(trackedKey)
        }
      }
    }
    trackedOperations.current.add(operationKey)
  }, [])

  const untrackOperation = useCallback((operationKey: string) => {
    trackedOperations.current.delete(operationKey)
  }, [])

  const isOperationTracked = useCallback(
    (operationKey: string) => trackedOperations.current.has(operationKey),
    [],
  )

  useEffect(() => () => {
    for (const timer of timers.current.values()) window.clearTimeout(timer)
    timers.current.clear()
  }, [])

  const value = useMemo<ToastContextValue>(() => ({
    showToast,
    notifyOnce,
    dismissToast,
    trackOperation,
    untrackOperation,
    isOperationTracked,
  }), [
    dismissToast,
    isOperationTracked,
    notifyOnce,
    showToast,
    trackOperation,
    untrackOperation,
  ])

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        className="toast-viewport"
        aria-label={UI_TEXT.toast.viewportLabel}
        aria-live="polite"
        aria-relevant="additions text"
      >
        {toasts.map((toast) => {
          const Icon = TOAST_ICONS[toast.type]
          return (
            <div
              className={`app-toast app-toast--${toast.type}`}
              role={toast.type === 'error' ? 'alert' : 'status'}
              key={toast.id}
            >
              <span className="toast-icon" aria-hidden="true">
                <Icon
                  className={toast.type === 'loading' ? 'spin' : undefined}
                  size={18}
                />
              </span>
              <div className="toast-copy">
                <strong>{toast.title}</strong>
                {toast.description && <p>{toast.description}</p>}
                {toast.action && (
                  <button
                    type="button"
                    onClick={() => {
                      void Promise.resolve()
                        .then(toast.action!.onClick)
                        .then(() => dismissToast(toast.id))
                        .catch(() => {
                          showToast({
                            id: toast.id,
                            type: 'error',
                            title: UI_TEXT.toast.actionFailed,
                            description: UI_TEXT.toast.actionFailedDescription,
                          })
                        })
                    }}
                  >
                    {toast.action.label}
                  </button>
                )}
              </div>
              <button
                className="toast-close"
                type="button"
                aria-label={UI_TEXT.toast.dismiss(toast.title)}
                onClick={() => dismissToast(toast.id)}
              >
                <X size={15} />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}
