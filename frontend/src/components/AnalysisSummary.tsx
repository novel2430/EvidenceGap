import { Route, ShieldCheck } from 'lucide-react'
import type { AnalysisContext, PresentationSummary } from '../contracts'
import {
  getEvidenceStatusLabel,
  getInferenceIntegrityLabel,
  getIntegrityClassName,
} from '../utils/presentationLabels'

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
  const claimIntegrity = summary.claim_inference_integrity
  const stepIntegrity = summary.inference_step_integrity
  const terminalConclusions = summary.terminal_conclusions

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
            <p>{evidence.SUPPORTED ?? 0} Supported · {evidence.REFUTED ?? 0} Refuted · {evidence.CONFLICTED ?? 0} Conflicted · {evidence.INSUFFICIENT ?? 0} Insufficient{evidence.ERROR ? ` · ${evidence.ERROR} Error` : ''}</p>
          </div>
        </article>
        <article className="summary-card summary-card--integrity">
          <div className="summary-card-icon"><Route size={18} /></div>
          <div className="summary-card-content">
            <strong>Terminal Conclusions</strong>
            {terminalConclusions === undefined ? (
              <p>Two-axis terminal status unavailable for this Run.</p>
            ) : terminalConclusions.length === 0 ? (
              <p>No terminal conclusions.</p>
            ) : (
              <div className="terminal-conclusion-list">
                {terminalConclusions.map((terminal) => (
                  <div key={terminal.claim_id}>
                    <span title={terminal.claim_id}>{terminal.claim_id}</span>
                    <span className={`summary-axis-badge summary-axis-badge--evidence-${terminal.evidence_status.toLowerCase()}`}>
                      {getEvidenceStatusLabel(terminal.evidence_status)}
                    </span>
                    <span className={`summary-axis-badge summary-axis-badge--integrity-${getIntegrityClassName(terminal.inference_integrity)}`}>
                      {getInferenceIntegrityLabel(terminal.inference_integrity, true)}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <div className="summary-integrity-stats">
              <span>Claims with intact inference: {claimIntegrity ? claimIntegrity.INTACT : 'Unavailable'} · gapped: {claimIntegrity ? claimIntegrity.GAPPED : 'Unavailable'}</span>
              <span>Inference steps intact: {stepIntegrity ? stepIntegrity.INTACT : 'Unavailable'} · gapped: {stepIntegrity ? stepIntegrity.GAPPED : 'Unavailable'}</span>
            </div>
            <small>Detected gaps: {gaps.SCOPE_GAP ?? 0} scope · {gaps.CAUSAL_GAP ?? 0} causal</small>
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
