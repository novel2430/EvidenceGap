import type { Edge, Node } from '@xyflow/react'
import type {
  PresentationClaim,
  PresentationInferenceStep,
} from '../contracts'
import type {
  ArticleStanceCounts,
  GraphSelection,
} from '../utils/presentation'

export interface GraphNodeVisualState {
  selected: boolean
  spotlight: boolean
  dimmed: boolean
  terminal: boolean
}

export interface ClaimNodeData extends Record<string, unknown> {
  claim: PresentationClaim
  claimNumber: number
  articleCounts: ArticleStanceCounts
  visual: GraphNodeVisualState
  onSelect?: (selection: GraphSelection) => void
}

export interface InferenceNodeData extends Record<string, unknown> {
  inferenceStep: PresentationInferenceStep
  stepNumber: number
  visual: GraphNodeVisualState
  selectedGapIndex: number | null
  onSelect?: (selection: GraphSelection) => void
}

export type ClaimGraphNode = Node<ClaimNodeData, 'claim'>
export type InferenceGraphNode = Node<InferenceNodeData, 'inference'>
export type ArgumentGraphNode = ClaimGraphNode | InferenceGraphNode
export type ArgumentGraphEdge = Edge

export const CLAIM_NODE_WIDTH = 250
export const CLAIM_NODE_HEIGHT = 156
export const INFERENCE_NODE_WIDTH = 190
export const INFERENCE_NODE_HEIGHT = 126
