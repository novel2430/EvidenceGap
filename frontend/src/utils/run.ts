import type { RunStage } from '../contracts'
import { UI_TEXT } from '../uiText'

export const RUN_STAGE_LABELS: Record<RunStage, string> =
  UI_TEXT.stageLabels

export const RUN_STAGES = Object.keys(RUN_STAGE_LABELS) as RunStage[]
