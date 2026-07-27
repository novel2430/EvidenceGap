import { Route, ShieldCheck } from 'lucide-react'
import type {
  PresentationClaim,
  PresentationSummary,
} from '../contracts'
import { UI_TEXT } from '../uiText'
import {
  getEvidenceStatusLabel,
  getInferenceIntegrityLabel,
  getIntegrityClassName,
} from '../utils/presentationLabels'

interface AnalysisSummaryProps {
  summary: PresentationSummary | null
  claims?: PresentationClaim[]
  embedded?: boolean
}

export function AnalysisSummary({
  summary,
  claims = [],
  embedded = false,
}: AnalysisSummaryProps) {
  const className = embedded
    ? 'analysis-summary analysis-summary--embedded'
    : 'analysis-summary panel'

  if (!summary) {
    return (
      <section className={className}>
        <div className="summary-heading">
          <div>
            <span className="eyebrow">{UI_TEXT.summary.eyebrow}</span>
            <h2>{UI_TEXT.summary.pending}</h2>
          </div>
        </div>
        <p className="summary-empty">
          {UI_TEXT.summary.pendingDescription}
        </p>
      </section>
    )
  }

  const evidence = summary.evidence_states
  const gaps = summary.gaps
  const claimIntegrity = summary.claim_inference_integrity
  const stepIntegrity = summary.inference_step_integrity
  const terminalConclusions = summary.terminal_conclusions
  const totalGaps = (gaps.SCOPE_GAP ?? 0) + (gaps.CAUSAL_GAP ?? 0)
  const claimsById = new Map(claims.map((claim) => [claim.claim_id, claim]))

  return (
    <section className={className}>
      <div className="result-headline">
        <span className="eyebrow">{UI_TEXT.summary.eyebrow}</span>
        <h2>{UI_TEXT.summary.complete}</h2>
        <p>
          {UI_TEXT.summary.headline(
            summary.total_claims,
            evidence.SUPPORTED ?? 0,
            evidence.INSUFFICIENT ?? 0,
            totalGaps,
          )}
        </p>
      </div>

      <div
        className="summary-metrics"
        aria-label={UI_TEXT.summary.metricsLabel}
      >
        <div>
          <strong>{summary.total_claims}</strong>
          <span>{UI_TEXT.summary.metrics.claims}</span>
        </div>
        <div className="summary-metric--supported">
          <strong>{evidence.SUPPORTED ?? 0}</strong>
          <span>{UI_TEXT.summary.metrics.supported}</span>
        </div>
        <div className="summary-metric--insufficient">
          <strong>{evidence.INSUFFICIENT ?? 0}</strong>
          <span>{UI_TEXT.summary.metrics.insufficient}</span>
        </div>
        <div className="summary-metric--gapped">
          <strong>{totalGaps}</strong>
          <span>{UI_TEXT.summary.metrics.detectedGaps}</span>
        </div>
      </div>

      <div className="summary-secondary-metrics">
        <span>{UI_TEXT.summary.secondary.refuted(evidence.REFUTED ?? 0)}</span>
        <span>{UI_TEXT.summary.secondary.conflicted(evidence.CONFLICTED ?? 0)}</span>
        <span>{UI_TEXT.summary.secondary.errors(evidence.ERROR ?? 0)}</span>
      </div>

      <section className="terminal-conclusions-section">
        <div className="summary-section-heading">
          <div>
            <Route size={16} />
            <h3>{UI_TEXT.summary.terminalConclusions}</h3>
          </div>
          <span>{terminalConclusions?.length ?? UI_TEXT.common.dash}</span>
        </div>

        {terminalConclusions === undefined ? (
          <p className="reference-empty">
            {UI_TEXT.summary.terminalUnavailable}
          </p>
        ) : terminalConclusions.length === 0 ? (
          <p className="reference-empty">
            {UI_TEXT.summary.noTerminalConclusions}
          </p>
        ) : (
          <div className="terminal-conclusion-list">
            {terminalConclusions.map((terminal) => {
              const claim = claimsById.get(terminal.claim_id)
              return (
                <article key={terminal.claim_id}>
                  <p>{claim?.display_text ?? terminal.claim_id}</p>
                  <div>
                    <span className={`summary-axis-badge summary-axis-badge--evidence-${terminal.evidence_status.toLowerCase()}`}>
                      <ShieldCheck size={12} />
                      {getEvidenceStatusLabel(terminal.evidence_status)}
                    </span>
                    <span className={`summary-axis-badge summary-axis-badge--integrity-${getIntegrityClassName(terminal.inference_integrity)}`}>
                      <Route size={12} />
                      {getInferenceIntegrityLabel(
                        terminal.inference_integrity,
                        true,
                      )}
                    </span>
                  </div>
                </article>
              )
            })}
          </div>
        )}
      </section>

      <dl className="summary-integrity-stats">
        <div>
          <dt>{UI_TEXT.summary.integrity.claimsIntact}</dt>
          <dd>{claimIntegrity?.INTACT ?? UI_TEXT.common.unavailable}</dd>
        </div>
        <div>
          <dt>{UI_TEXT.summary.integrity.claimsGapped}</dt>
          <dd>{claimIntegrity?.GAPPED ?? UI_TEXT.common.unavailable}</dd>
        </div>
        <div>
          <dt>{UI_TEXT.summary.integrity.stepsIntact}</dt>
          <dd>{stepIntegrity?.INTACT ?? UI_TEXT.common.unavailable}</dd>
        </div>
        <div>
          <dt>{UI_TEXT.summary.integrity.stepsGapped}</dt>
          <dd>{stepIntegrity?.GAPPED ?? UI_TEXT.common.unavailable}</dd>
        </div>
      </dl>
    </section>
  )
}
