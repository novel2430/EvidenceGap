import { ArrowRight, Ban, ShieldCheck } from 'lucide-react'
import type { DemoCase } from '../types'

export function ConclusionCompare({ currentCase }: { currentCase: DemoCase }) {
  return (
    <section className="conclusion-compare panel">
      <div className="panel-kicker">Conclusion Downgrade</div>
      <div className="compare-columns">
        <article className="conclusion-card original">
          <div className="section-title">
            <Ban size={16} />
            Original Conclusion
          </div>
          <p>{currentCase.originalConclusion}</p>
        </article>
        <div className="compare-arrow">
          <ArrowRight size={22} />
        </div>
        <article className="conclusion-card safe">
          <div className="section-title">
            <ShieldCheck size={16} />
            Safe Conclusion
          </div>
          <p>{currentCase.safeConclusion}</p>
          <div className="safe-limits">
            {currentCase.safeLimitations.map((item) => (
              <span key={item}>{item}</span>
            ))}
          </div>
        </article>
      </div>
    </section>
  )
}
