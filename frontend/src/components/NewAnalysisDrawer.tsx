import { useRef } from 'react'
import { X } from 'lucide-react'
import { useModalDrawer } from '../hooks/useModalDrawer'
import { UI_TEXT } from '../uiText'
import { AnalysisForm } from './AnalysisForm'

interface NewAnalysisDrawerProps {
  isOpen: boolean
  onClose: () => void
}

export function NewAnalysisDrawer({
  isOpen,
  onClose,
}: NewAnalysisDrawerProps) {
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const panelRef = useRef<HTMLElement>(null)
  useModalDrawer(isOpen, panelRef, closeButtonRef, onClose)

  return (
    <div
      className={`new-analysis-drawer-layer${isOpen ? ' is-open' : ''}`}
      aria-hidden={!isOpen}
    >
      <button
        className="new-analysis-drawer-backdrop"
        type="button"
        aria-label={UI_TEXT.drawers.closeNewAnalysis}
        tabIndex={-1}
        onClick={onClose}
      />
      <aside
        ref={panelRef}
        className="new-analysis-drawer"
        role="dialog"
        aria-modal="true"
        aria-label={UI_TEXT.drawers.newAnalysis}
        tabIndex={-1}
        inert={!isOpen}
      >
        <button
          ref={closeButtonRef}
          className="drawer-close-button"
          type="button"
          aria-label={UI_TEXT.drawers.closeNewAnalysis}
          onClick={onClose}
        >
          <X size={18} />
        </button>
        <AnalysisForm onRunCreated={onClose} />
      </aside>
    </div>
  )
}
