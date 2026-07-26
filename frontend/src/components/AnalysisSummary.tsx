import { Route, ShieldCheck } from 'lucide-react'
import type { AnalysisContext, PresentationSummary } from '../contracts'

interface AnalysisSummaryProps {
  summary: PresentationSummary | null
  analysisContext: AnalysisContext | null
}

export function AnalysisSummary({ summary, analysisContext }: AnalysisSummaryProps) {
  if (!summary) {
    return (
      <section className="analysis-summary panel">
        <div className="summary-heading">
          <div><span className="eyebrow">Run output</span><h2>Analysis summary</h2></div>
          <span className="summary-state">Awaiting completed analysis</span>
        </div>
        <div className="summary-empty">Claim Evidence and Inference Integrity will appear after a successful run.</div>
      </section>
    )
  }

  const evidence = summary.evidence_states
  const gaps = summary.gaps

  return (
    <section className="analysis-summary panel">
      <div className="summary-heading">
        <div><span className="eyebrow">Run output</span><h2>Two-axis analysis summary</h2></div>
        <span className="summary-state">{summary.total_claims} claims</span>
      </div>

      <div className="summary-grid">
        <article className="summary-card summary-card--evidence">
          <div className="summary-card-icon"><ShieldCheck size={18} /></div>
          <div>
            <strong>Claim Evidence</strong>
            <p>{evidence.SUPPORTED} Supported · {evidence.REFUTED} Refuted · {evidence.CONFLICTED} Conflicted · {evidence.INSUFFICIENT} Insufficient{evidence.ERROR ? ` · ${evidence.ERROR} Error` : ''}</p>
          </div>
        </article>
        <article className="summary-card summary-card--integrity">
          <div className="summary-card-icon"><Route size={18} /></div>
          <div>
            <strong>Inference Integrity</strong>
            <p>{gaps.SCOPE_GAP} Scope {gaps.SCOPE_GAP === 1 ? 'Gap' : 'Gaps'} · {gaps.CAUSAL_GAP} Causal {gaps.CAUSAL_GAP === 1 ? 'Gap' : 'Gaps'}</p>
          </div>
        </article>
        <article className="summary-card summary-card--boundary">
          <div className="summary-card-icon"><ShieldCheck size={18} /></div>
          <div>
            <strong>Methodological Boundary</strong>
            <p>Only summarizes the retrieved Top Articles. Not a systematic review. Not a clinical recommendation. Not final medical truth.</p>
            {analysisContext && <small>Source depth: {analysisContext.source_depth}; article top K: {analysisContext.article_top_k}.</small>}
          </div>
        </article>
      </div>
    </section>
  )
}
