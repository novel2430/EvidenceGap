import { useEffect, useRef, useState } from 'react'
import { Clipboard, Download } from 'lucide-react'
import { evidenceGapApi } from '../api'
import { useToast } from '../hooks/useToast'
import { UI_TEXT } from '../uiText'
import { getApiErrorMessage, getToastErrorMessage } from '../utils/format'

interface ExportActionsProps {
  runId: string
  isLocalizedView?: boolean
}

function saveBlob(blob: Blob, filename: string) {
  const objectUrl = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = objectUrl
  link.download = filename
  document.body.append(link)
  link.click()
  link.remove()
  URL.revokeObjectURL(objectUrl)
}

export function ExportActions({
  runId,
  isLocalizedView = false,
}: ExportActionsProps) {
  const [pendingAction, setPendingAction] = useState<'json' | 'markdown' | null>(null)
  const [message, setMessage] = useState<string | null>(null)
  const resetTimer = useRef<number | null>(null)
  const { showToast } = useToast()

  useEffect(() => () => {
    if (resetTimer.current !== null) window.clearTimeout(resetTimer.current)
  }, [])

  function showTemporaryMessage(nextMessage: string) {
    setMessage(nextMessage)
    if (resetTimer.current !== null) window.clearTimeout(resetTimer.current)
    resetTimer.current = window.setTimeout(() => setMessage(null), 2500)
  }

  async function downloadJson() {
    const toastId = `export:${runId}:json`
    setPendingAction('json')
    setMessage(null)
    showToast({
      id: toastId,
      type: 'loading',
        title: UI_TEXT.toast.preparingJson,
    })
    try {
      const blob = await evidenceGapApi.downloadResultExport(runId)
      saveBlob(blob, `evidencegap-${runId}.json`)
      showToast({
        id: toastId,
        type: 'success',
        title: UI_TEXT.toast.exportPrepared,
        description: UI_TEXT.toast.jsonStarted,
      })
    } catch (error) {
      const errorMessage = getApiErrorMessage(error)
      setMessage(UI_TEXT.export.jsonFailure(errorMessage))
      showToast({
        id: toastId,
        type: 'error',
        title: UI_TEXT.toast.jsonFailed,
        description: getToastErrorMessage(error),
      })
    } finally {
      setPendingAction(null)
    }
  }

  async function downloadMarkdown() {
    const toastId = `export:${runId}:markdown`
    setPendingAction('markdown')
    setMessage(null)
    showToast({
      id: toastId,
      type: 'loading',
        title: UI_TEXT.toast.preparingMarkdown,
    })
    try {
      const report = await evidenceGapApi.getMarkdownExport(runId)
      saveBlob(new Blob([report], { type: 'text/markdown;charset=utf-8' }), `evidencegap-${runId}.md`)
      showToast({
        id: toastId,
        type: 'success',
        title: UI_TEXT.toast.exportPrepared,
        description: UI_TEXT.toast.markdownStarted,
      })
    } catch (error) {
      const errorMessage = getApiErrorMessage(error)
      setMessage(UI_TEXT.export.markdownFailure(errorMessage))
      showToast({
        id: toastId,
        type: 'error',
        title: UI_TEXT.toast.markdownFailed,
        description: getToastErrorMessage(error),
      })
    } finally {
      setPendingAction(null)
    }
  }

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href)
      showTemporaryMessage(UI_TEXT.export.linkCopied)
      showToast({
        id: `copy-run-link:${runId}`,
        type: 'success',
        title: UI_TEXT.toast.copied,
        description: UI_TEXT.toast.copiedDescription,
      })
    } catch (error) {
      const errorMessage = getApiErrorMessage(error)
      setMessage(UI_TEXT.export.copyFailure(errorMessage))
      showToast({
        id: `copy-run-link:${runId}`,
        type: 'error',
        title: UI_TEXT.toast.copyFailed,
        description: getToastErrorMessage(error),
      })
    }
  }

  return (
    <section className="export-actions">
      <h3>{UI_TEXT.export.title}</h3>
      {isLocalizedView && (
        <p className="export-boundary">
          {UI_TEXT.export.originalBoundary}
        </p>
      )}
      <div className="export-buttons">
        <button type="button" onClick={downloadJson} disabled={pendingAction !== null}>
          <Download size={14} />{' '}
          {pendingAction === 'json'
            ? UI_TEXT.export.downloading
            : UI_TEXT.export.json}
        </button>
        <button type="button" onClick={downloadMarkdown} disabled={pendingAction !== null}>
          <Download size={14} />{' '}
          {pendingAction === 'markdown'
            ? UI_TEXT.export.downloading
            : UI_TEXT.export.markdown}
        </button>
        <button type="button" onClick={copyLink}>
          <Clipboard size={14} /> {UI_TEXT.export.copyLink}
        </button>
      </div>
      {message && <p className="action-message" role="status">{message}</p>}
    </section>
  )
}
