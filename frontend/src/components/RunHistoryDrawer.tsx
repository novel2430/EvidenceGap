import { useRef } from 'react'
import { X } from 'lucide-react'
import type { RunListItemResponse } from '../contracts'
import { useModalDrawer } from '../hooks/useModalDrawer'
import { UI_TEXT } from '../uiText'
import { RunSidebar } from './RunSidebar'

interface RunHistoryDrawerProps {
  isOpen: boolean
  runs: RunListItemResponse[]
  selectedRunId: string | null
  isLoading: boolean
  errorMessage: string | null
  onRetry: () => void
  onClose: () => void
  onRunSelected: (runId: string) => void
}

export function RunHistoryDrawer({
  isOpen,
  runs,
  selectedRunId,
  isLoading,
  errorMessage,
  onRetry,
  onClose,
  onRunSelected,
}: RunHistoryDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLElement>(null)
  useModalDrawer(isOpen, panelRef, closeButtonRef, onClose)

  return (
    <div
      className={`history-drawer-layer${isOpen ? ' is-open' : ''}`}
      aria-hidden={!isOpen}
    >
      <button
        className="history-drawer-backdrop"
        type="button"
        aria-label={UI_TEXT.drawers.closeRecentRuns}
        tabIndex={-1}
        onClick={onClose}
      />
      <aside
        ref={panelRef}
        className="history-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={UI_TEXT.drawers.recentRuns}
        tabIndex={-1}
        inert={!isOpen}
      >
        <button
          ref={closeButtonRef}
          className="drawer-close-button"
          type="button"
          aria-label={UI_TEXT.drawers.closeRecentRuns}
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
          onRunSelected={onRunSelected}
        />
      </aside>
    </div>
  )
}
