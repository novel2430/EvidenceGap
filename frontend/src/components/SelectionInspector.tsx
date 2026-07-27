import { GitMerge, MousePointer2, Route, TriangleAlert } from 'lucide-react'
import type {
  PresentationClaim,
  PresentationGap,
  PresentationInferenceStep,
} from '../contracts'
import type {
  GraphSelection,
  PresentationIndexes,
} from '../utils/presentation'
import { UI_TEXT } from '../uiText'
import { ArticleInspector } from './ArticleInspector'
import { ClaimArticleList } from './ClaimArticleList'
import { EvidenceBalance } from './EvidenceBalance'
import {
  formatEnumLabel,
  getEvidenceStatusLabel,
  getGapTypeLabel,
  getInferenceIntegrityLabel,
  getIntegrityClassName,
} from '../utils/presentationLabels'

interface SelectionInspectorProps {
  runId: string
  selection: GraphSelection | null
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}

interface ClaimReferenceListProps {
  claimIds: string[]
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
  emptyLabel?: string
}

function ClaimReferenceList({
  claimIds,
  indexes,
  onSelect,
  emptyLabel = UI_TEXT.inspector.none,
}: ClaimReferenceListProps) {
  if (claimIds.length === 0) return <p className="reference-empty">{emptyLabel}</p>

  return (
    <div className="claim-reference-list">
      {claimIds.map((claimId) => {
        const claim = indexes.claimsById.get(claimId)
        return (
          <button
            type="button"
            key={claimId}
            onClick={() => onSelect({ kind: 'claim', claimId })}
          >
            {claim?.display_text ?? claimId}
          </button>
        )
      })}
    </div>
  )
}

function InferenceStepReferenceList({
  inferenceStepIds,
  indexes,
  onSelect,
}: {
  inferenceStepIds: string[]
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}) {
  if (inferenceStepIds.length === 0) return null

  return (
    <div className="inference-reference-list">
      {inferenceStepIds.map((inferenceStepId) => (
        <button
          type="button"
          key={inferenceStepId}
          onClick={() => onSelect({
            kind: 'inference_step',
            inferenceStepId,
          })}
          disabled={!indexes.inferenceStepsById.has(inferenceStepId)}
        >
          <GitMerge size={12} />
          {inferenceStepId}
        </button>
      ))}
    </div>
  )
}

function StructuredGapCard({
  gap,
  onSelect,
}: {
  gap: PresentationGap
  onSelect?: () => void
}) {
  const reason =
    gap.display_reason?.trim() ||
    gap.reason_en?.trim() ||
    UI_TEXT.common.unavailable
  const closureRequirement =
    gap.display_closure_requirement?.trim() ||
    gap.closure_requirement_en?.trim() ||
    null
  const hasBasisPair = Boolean(
    gap.supported_basis?.trim() || gap.unsupported_extension?.trim(),
  )
  const headingContent = (
    <>
      <TriangleAlert size={14} />
      <span>
        <strong>{getGapTypeLabel(gap.gap_type)}</strong>
        {gap.subtype && <small>{formatEnumLabel(gap.subtype)}</small>}
      </span>
    </>
  )

  return (
    <article className={`structured-gap-card structured-gap-card--${gap.gap_type.toLowerCase()}`}>
      {onSelect ? (
        <button
          className="structured-gap-heading"
          type="button"
          onClick={onSelect}
        >
          {headingContent}
        </button>
      ) : (
        <div className="structured-gap-heading">{headingContent}</div>
      )}

      {gap.affected_dimensions && gap.affected_dimensions.length > 0 && (
        <div className="gap-detail-block">
          <h5>{UI_TEXT.inspector.affectedDimensions}</h5>
          <div className="gap-dimension-chips">
            {gap.affected_dimensions.map((dimension) => (
              <span key={dimension}>{formatEnumLabel(dimension)}</span>
            ))}
          </div>
        </div>
      )}

      {hasBasisPair && (
        <div className="gap-basis-pair">
          <div>
            <h5>{UI_TEXT.inspector.supportedByPremises}</h5>
            <p>
              {gap.supported_basis?.trim() || UI_TEXT.common.unavailable}
            </p>
          </div>
          <div>
            <h5>{UI_TEXT.inspector.unsupportedExtension}</h5>
            <p>
              {gap.unsupported_extension?.trim() ||
                UI_TEXT.common.unavailable}
            </p>
          </div>
        </div>
      )}

      <div className="gap-detail-block">
        <h5>{UI_TEXT.inspector.whyGap}</h5>
        <p>{reason}</p>
      </div>

      {closureRequirement && (
        <div className="gap-closure-requirement">
          <h5>{UI_TEXT.inspector.closureRequirement}</h5>
          <p>{closureRequirement}</p>
        </div>
      )}
    </article>
  )
}

function ImpactDetails({
  inferenceStep,
  indexes,
  onSelect,
}: {
  inferenceStep: PresentationInferenceStep
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}) {
  const impact = inferenceStep.impact
  return (
    <div className="impact-details">
      <h4>{UI_TEXT.inspector.directConclusion}</h4>
      <ClaimReferenceList
        claimIds={[impact.direct_conclusion_claim_id]}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>{UI_TEXT.inspector.downstreamClaims}</h4>
      <ClaimReferenceList
        claimIds={impact.downstream_claim_ids}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>{UI_TEXT.inspector.terminalClaims}</h4>
      <ClaimReferenceList
        claimIds={impact.terminal_claim_ids}
        indexes={indexes}
        onSelect={onSelect}
      />
      <dl className="impact-summary">
        <div><dt>{UI_TEXT.inspector.downstreamSteps}</dt><dd>{impact.downstream_inference_step_ids.length}</dd></div>
        <div><dt>{UI_TEXT.inspector.affectsTerminal}</dt><dd>{impact.affects_terminal_conclusion ? UI_TEXT.common.yes : UI_TEXT.common.no}</dd></div>
      </dl>
    </div>
  )
}

function ClaimDetails({
  claim,
  indexes,
  selection,
  onSelect,
}: {
  claim: PresentationClaim
  indexes: PresentationIndexes
  selection: GraphSelection
  onSelect: (selection: GraphSelection) => void
}) {
  return (
    <section className="selection-card">
      <div className="selection-card-heading">
        <span className="eyebrow">{UI_TEXT.inspector.claimSelection}</span>
        <span className={`claim-list-state claim-list-state--${claim.evidence_state.toLowerCase()}`}>
          {claim.evidence_state}
        </span>
      </div>
      <h3>{claim.display_text}</h3>
      <dl className="claim-detail-fields">
        <div><dt>{UI_TEXT.inspector.fields.originalQuote}</dt><dd>{claim.source_text}</dd></div>
        <div><dt>{UI_TEXT.inspector.fields.canonicalClaim}</dt><dd>{claim.canonical_claim_en}</dd></div>
        <div><dt>{UI_TEXT.inspector.fields.displayText}</dt><dd>{claim.display_text}</dd></div>
        <div><dt>{UI_TEXT.inspector.fields.argumentRole}</dt><dd>{claim.argument_role}</dd></div>
        <div><dt>{UI_TEXT.inspector.fields.evidenceState}</dt><dd>{claim.evidence_state}</dd></div>
      </dl>
      <section className="argument-audit">
        <h4>{UI_TEXT.inspector.argumentAudit}</h4>
        {claim.audit ? (
          <>
            <dl>
              <div>
                <dt>{UI_TEXT.inspector.fields.evidenceStatus}</dt>
                <dd>{getEvidenceStatusLabel(claim.audit.evidence_status)}</dd>
              </div>
              <div>
                <dt>{UI_TEXT.inspector.fields.inferenceIntegrity}</dt>
                <dd>
                  <span className={`inference-integrity-badge inference-integrity-badge--${getIntegrityClassName(claim.audit.inference_integrity)}`}>
                    <Route size={11} />
                    {getInferenceIntegrityLabel(claim.audit.inference_integrity)}
                  </span>
                </dd>
              </div>
            </dl>
            {claim.audit.affecting_inference_step_ids.length > 0 && (
              <>
                <h5>{UI_TEXT.inspector.affectingSteps}</h5>
                <InferenceStepReferenceList
                  inferenceStepIds={claim.audit.affecting_inference_step_ids}
                  indexes={indexes}
                  onSelect={onSelect}
                />
              </>
            )}
          </>
        ) : (
          <p className="reference-empty">
            {UI_TEXT.inspector.auditUnavailable}
          </p>
        )}
      </section>
      <EvidenceBalance
        claim={claim}
        articles={indexes.articlesByClaimId.get(claim.claim_id) ?? []}
      />
      <ClaimArticleList
        key={claim.claim_id}
        claim={claim}
        indexes={indexes}
        selection={selection}
        onSelect={onSelect}
      />
    </section>
  )
}

function InferenceDetails({
  inferenceStep,
  indexes,
  onSelect,
}: {
  inferenceStep: PresentationInferenceStep
  indexes: PresentationIndexes
  onSelect: (selection: GraphSelection) => void
}) {
  return (
    <section className="selection-card">
      <div className="selection-card-heading">
        <span className="eyebrow">
          {UI_TEXT.inspector.inferenceSelection}
        </span>
        <GitMerge size={17} />
      </div>
      <div className="inference-detail-title">
        <h3>{UI_TEXT.inspector.inferenceStep}</h3>
        <span className={`inference-integrity-badge inference-integrity-badge--${getIntegrityClassName(inferenceStep.inference_integrity)}`}>
          <Route size={11} />
          {getInferenceIntegrityLabel(inferenceStep.inference_integrity, true)}
        </span>
      </div>
      <h4>{UI_TEXT.inspector.premiseClaims}</h4>
      <ClaimReferenceList
        claimIds={inferenceStep.premise_claim_ids}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>{UI_TEXT.inspector.conclusionClaim}</h4>
      <ClaimReferenceList
        claimIds={[inferenceStep.conclusion_claim_id]}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>{UI_TEXT.inspector.detectedGaps}</h4>
      {inferenceStep.gaps.length === 0 ? (
        <p className="reference-empty">{UI_TEXT.inspector.noGaps}</p>
      ) : (
        <div className="structured-gap-list">
          {inferenceStep.gaps.map((gap, gapIndex) => (
            <StructuredGapCard
              key={`${gap.gap_type}-${gapIndex}`}
              gap={gap}
              onSelect={() => onSelect({
                kind: 'gap',
                inferenceStepId: inferenceStep.inference_step_id,
                gapIndex,
              })}
            />
          ))}
        </div>
      )}
      <ImpactDetails
        inferenceStep={inferenceStep}
        indexes={indexes}
        onSelect={onSelect}
      />
    </section>
  )
}

export function SelectionInspector({
  runId,
  selection,
  indexes,
  onSelect,
}: SelectionInspectorProps) {
  if (!selection) {
    return (
      <div className="selection-hint">
        <MousePointer2 size={18} />
        <div>
          <strong>{UI_TEXT.inspector.emptyTitle}</strong>
          <p>{UI_TEXT.inspector.emptyDescription}</p>
        </div>
      </div>
    )
  }

  if (selection.kind === 'claim') {
    const claim = indexes.claimsById.get(selection.claimId)
    return claim ? (
      <ClaimDetails
        claim={claim}
        indexes={indexes}
        selection={selection}
        onSelect={onSelect}
      />
    ) : null
  }

  if (selection.kind === 'article' || selection.kind === 'evidence') {
    return (
      <ArticleInspector
        runId={runId}
        selection={selection}
        indexes={indexes}
        onSelect={onSelect}
      />
    )
  }

  const inferenceStep = indexes.inferenceStepsById.get(selection.inferenceStepId)
  if (!inferenceStep) return null

  if (selection.kind === 'inference_step') {
    return (
      <InferenceDetails
        inferenceStep={inferenceStep}
        indexes={indexes}
        onSelect={onSelect}
      />
    )
  }

  const gap = inferenceStep.gaps[selection.gapIndex]
  if (!gap) return null
  return (
    <section className="selection-card selection-card--gap">
      <div className="selection-card-heading">
        <span className="eyebrow">{UI_TEXT.inspector.gapSelection}</span>
        <TriangleAlert size={17} />
      </div>
      <div className="inference-detail-title">
        <h3>{getGapTypeLabel(gap.gap_type)}</h3>
        <span className={`inference-integrity-badge inference-integrity-badge--${getIntegrityClassName(inferenceStep.inference_integrity)}`}>
          <Route size={11} />
          {getInferenceIntegrityLabel(inferenceStep.inference_integrity, true)}
        </span>
      </div>
      <h4>{UI_TEXT.inspector.premiseClaims}</h4>
      <ClaimReferenceList
        claimIds={inferenceStep.premise_claim_ids}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>{UI_TEXT.inspector.conclusionClaim}</h4>
      <ClaimReferenceList
        claimIds={[inferenceStep.conclusion_claim_id]}
        indexes={indexes}
        onSelect={onSelect}
      />
      <h4>{UI_TEXT.inspector.structuredGap}</h4>
      <StructuredGapCard gap={gap} />
      <ImpactDetails
        inferenceStep={inferenceStep}
        indexes={indexes}
        onSelect={onSelect}
      />
    </section>
  )
}
