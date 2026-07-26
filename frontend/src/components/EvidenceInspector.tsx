import { PanelRight, Search } from 'lucide-react'
import type {
  ArticleContextResponse,
  InspectorSelection,
  PresentationBundle,
} from '../contracts'

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
  const selectionLabel = selection ? `Selected ${selection.kind}` : 'Nothing selected'
  const detailText = articleContext
    ? `Article context loaded for ${articleContext.article_id}.`
    : presentation
      ? 'Select a claim, inference step, article, or evidence item.'
      : 'Select an item after a completed run is loaded.'

  return (
    <aside className="inspector panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">Inspector</span>
          <h2>Evidence details</h2>
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
