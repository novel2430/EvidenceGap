import { PanelRight, Search } from 'lucide-react'
import type {
  ArticleContextResponse,
  InspectorSelection,
  PresentationBundle,
} from '../contracts'
import { UI_TEXT } from '../uiText'

interface EvidenceInspectorProps {
  presentation: PresentationBundle | null
  selection: InspectorSelection | null
  articleContext: ArticleContextResponse | null
}

export function EvidenceInspector({
  presentation,
  selection,
  articleContext,
}: EvidenceInspectorProps) {
  const selectionLabel = selection
    ? UI_TEXT.inspector.selected(selection.kind)
    : UI_TEXT.inspector.nothingSelected
  const detailText = articleContext
    ? UI_TEXT.inspector.articleLoaded(articleContext.article_id)
    : presentation
      ? UI_TEXT.inspector.selectItem
      : UI_TEXT.inspector.selectAfterRun

  return (
    <aside className="inspector panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">{UI_TEXT.inspector.legacyEyebrow}</span>
          <h2>{UI_TEXT.inspector.legacyTitle}</h2>
        </div>
        <PanelRight size={18} />
      </div>

      <div className="empty-card">
        <Search size={22} />
        <strong>{selectionLabel}</strong>
        <p>{detailText}</p>
      </div>
    </aside>
  )
}
