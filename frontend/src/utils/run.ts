import type { RunStage } from '../contracts'

export const RUN_STAGE_LABELS: Record<RunStage, string> = {
  statement_decomposition: 'Claim 拆解',
  claim_analysis: '證據分析',
  statement_bundle: 'Bundle 建立',
  inference_gap_analysis: 'Gap 分析',
  output_generation: '輸出生成',
}

export const RUN_STAGES = Object.keys(RUN_STAGE_LABELS) as RunStage[]
