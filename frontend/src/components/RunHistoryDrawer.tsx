import { useEffect, useRef } from 'react'
import { X } from 'lucide-react'
import type { RunListItemResponse } from '../contracts'
import { RunSidebar } from './RunSidebar'

interface RunHistoryDrawerProps {
  isOpen: boolean
  runs: RunListItemResponse[]
  selectedRunId: string | null
  isLoading: boolean
  errorMessage: string | null
  onRetry: () => void
  onClose: () => void
}

export function RunHistoryDrawer({
  isOpen,
  runs,
  selectedRunId,
  isLoading,
  errorMessage,
  onRetry,
  onClose,
}: RunHistoryDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null)

  useEffect(() => {
    if (!isOpen) return

    closeButtonRef.current?.focus()
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose()
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  if (!isOpen) return null

  return (
    <div className="history-drawer-layer">
      <button
        className="history-drawer-backdrop"
        type="button"
        aria-label="Close Recent Runs"
        onClick={onClose}
      />
      <aside
        className="history-drawer"
        role="dialog"
        aria-modal="true"
        aria-label="Recent Runs"
      >
        <button
          ref={closeButtonRef}
          className="drawer-close-button"
          type="button"
          aria-label="Close Recent Runs"
          onClick={onClose}
        >
          <X size={18} />
        </button>
        <RunSidebar
          runs={runs}
          selectedRunId={selectedRunId}
          isLoading={isLoading}
          errorMessage={errorMessage}
          onRetry={onRetry}
          onRunSelected={onClose}
        />
      </aside>
    </div>
  )
}
