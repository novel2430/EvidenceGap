import { AlertTriangle, Database, FileSearch, GitBranch, LockKeyhole, Microscope, ShieldCheck } from 'lucide-react'
import type { DemoCase, InspectorSelection } from '../types'

export function EvidenceInspector({
  currentCase,
  selection,
}: {
  currentCase: DemoCase
  selection: InspectorSelection | null
}) {
  const claim =
    selection?.kind === 'claim' ? currentCase.claims.find((item) => item.id === selection.id) : currentCase.claims.find((item) => item.isTarget)
  const step = selection?.kind === 'inference' ? currentCase.inferenceSteps.find((item) => item.id === selection.id) : null

  if (step) {
    const premises = step.premises.map((id) => currentCase.claims.find((item) => item.id === id)?.shortText ?? id)
    const conclusion = currentCase.claims.find((item) => item.id === step.conclusion)

    return (
      <aside className="inspector panel">
        <div className="panel-kicker">Inference Inspector</div>
        <h2>{step.inferenceType.replaceAll('_', ' ')}</h2>
        <div className="inspector-card reasoning-card">
          <div className="section-title">
            <GitBranch size={16} />
            Reasoning Rule
          </div>
          <p>{step.ruleDescription}</p>
        </div>
        <div className="inspector-group">
          <span>Premises</span>
          {premises.map((item) => (
            <div className="compact-line" key={item}>{item}</div>
          ))}
        </div>
        <div className="inspector-group">
          <span>Conclusion</span>
          <div className="compact-line strong">{conclusion?.shortText}</div>
        </div>
        <div className="inspector-group">
          <span>Required Assumptions</span>
          {step.requiredAssumptions.length === 0 ? (
            <div className="compact-line supported-text">No extra assumption</div>
          ) : (
            step.requiredAssumptions.map((item) => <div className="compact-line warning-text" key={item}>{item}</div>)
          )}
        </div>
        <div className="expert-chip">{step.expertJudgment ? 'Requires expert judgment' : 'Rule-bound comparison'}</div>
      </aside>
    )
  }

  if (!claim) {
    return null
  }

  const evidence = currentCase.evidence.filter((item) => claim.evidenceIds.includes(item.id))
  const gaps = currentCase.gaps.filter((gap) => claim.gapIds.includes(gap.id) || gap.target === claim.id)

  return (
    <aside className="inspector panel">
      <div className="panel-kicker">Evidence Inspector</div>
      <h2>{claim.shortText}</h2>
      <div className="claim-fulltext">{claim.text}</div>

      <div className="inspector-tags">
        <span>{claim.type.replace('_', ' ')}</span>
        <span className={`status-pill status-${claim.status.toLowerCase()}`}>{claim.status}</span>
      </div>

      <div className="inspector-group">
        <span>Reason Codes</span>
        {claim.reasonCodes.length === 0 ? (
          <div className="compact-line supported-text">
            <ShieldCheck size={14} />
            Directly supported
          </div>
        ) : (
          claim.reasonCodes.map((code) => (
            <div className="compact-line warning-text" key={code}>
              <AlertTriangle size={14} />
              {code}
            </div>
          ))
        )}
      </div>

      <div className="inspector-group">
        <span>Evidence</span>
        {evidence.length === 0 ? (
          <div className="empty-evidence">
            <Database size={16} />
            No direct evidence attached
          </div>
        ) : (
          evidence.map((item) => (
            <div className={`evidence-card role-${item.role.toLowerCase()}`} key={item.id}>
              <div className="evidence-source">
                <FileSearch size={14} />
                {item.sourceId}
                <em>{item.role}</em>
              </div>
              <blockquote>{item.text}</blockquote>
              <div className="scope-tags">
                <span>{item.timeScope}</span>
                <span>{item.geography}</span>
                <span>{item.population}</span>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="inspector-group">
        <span>Current Gaps</span>
        {gaps.length === 0 ? (
          <div className="compact-line supported-text">
            <Microscope size={14} />
            No blocking gap on this claim
          </div>
        ) : (
          gaps.map((gap) => (
            <div className="gap-mini" key={gap.id}>
              {gap.privateDataRequired ? <LockKeyhole size={15} /> : <AlertTriangle size={15} />}
              <div>
                <strong>{gap.type.replaceAll('_', ' ')}</strong>
                <p>{gap.reason}</p>
              </div>
            </div>
          ))
        )}
      </div>

      <div className="inspector-group">
        <span>Downstream Impact</span>
        {claim.downstream.length === 0 ? (
          <div className="compact-line">Final recommendation node</div>
        ) : (
          claim.downstream.map((id) => <div className="compact-line" key={id}>{id}</div>)
        )}
      </div>
    </aside>
  )
}
