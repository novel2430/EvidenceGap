import { useEffect, useRef, useState } from 'react'
import { Clipboard, Download } from 'lucide-react'
import { evidenceGapApi } from '../api'
import { getApiErrorMessage } from '../utils/format'

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

  useEffect(() => () => {
    if (resetTimer.current !== null) window.clearTimeout(resetTimer.current)
  }, [])

  function showTemporaryMessage(nextMessage: string) {
    setMessage(nextMessage)
    if (resetTimer.current !== null) window.clearTimeout(resetTimer.current)
    resetTimer.current = window.setTimeout(() => setMessage(null), 2500)
  }

  async function downloadJson() {
    setPendingAction('json')
    setMessage(null)
    try {
      const blob = await evidenceGapApi.downloadResultExport(runId)
      saveBlob(blob, `evidencegap-${runId}.json`)
    } catch (error) {
      setMessage(`JSON download failed: ${getApiErrorMessage(error)}`)
    } finally {
      setPendingAction(null)
    }
  }

  async function downloadMarkdown() {
    setPendingAction('markdown')
    setMessage(null)
    try {
      const report = await evidenceGapApi.getMarkdownExport(runId)
      saveBlob(new Blob([report], { type: 'text/markdown;charset=utf-8' }), `evidencegap-${runId}.md`)
    } catch (error) {
      setMessage(`Markdown download failed: ${getApiErrorMessage(error)}`)
    } finally {
      setPendingAction(null)
    }
  }

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(window.location.href)
      showTemporaryMessage('Link copied')
    } catch (error) {
      setMessage(`Could not copy the link: ${getApiErrorMessage(error)}`)
    }
  }

  return (
    <section className="export-actions">
      <h3>Export &amp; share</h3>
      {isLocalizedView && (
        <p className="export-boundary">
          Downloads export the Original Analysis. The backend does not provide localized export endpoints.
        </p>
      )}
      <div className="export-buttons">
        <button type="button" onClick={downloadJson} disabled={pendingAction !== null}>
          <Download size={14} /> {pendingAction === 'json' ? 'Downloading…' : 'Download JSON'}
        </button>
        <button type="button" onClick={downloadMarkdown} disabled={pendingAction !== null}>
          <Download size={14} /> {pendingAction === 'markdown' ? 'Downloading…' : 'Download Markdown'}
        </button>
        <button type="button" onClick={copyLink}>
          <Clipboard size={14} /> Copy Run Link
        </button>
      </div>
      {message && <p className="action-message" role="status">{message}</p>}
    </section>
  )
}
