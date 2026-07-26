import { MarkerType, Position } from '@xyflow/react'
import ELK from 'elkjs/lib/elk.bundled.js'
import type { PresentationBundle } from '../contracts'
import {
  countArticleStances,
  type PresentationIndexes,
} from '../utils/presentation'
import type {
  ArgumentGraphEdge,
  ArgumentGraphNode,
  ClaimGraphNode,
  InferenceGraphNode,
} from './types'
import {
  CLAIM_NODE_HEIGHT,
  CLAIM_NODE_WIDTH,
  INFERENCE_NODE_HEIGHT,
  INFERENCE_NODE_WIDTH,
} from './types'

const elk = new ELK()

export const claimGraphNodeId = (claimId: string) => `claim:${claimId}`
export const inferenceGraphNodeId = (inferenceStepId: string) =>
  `inference:${inferenceStepId}`

export async function layoutPresentationGraph(
  presentation: PresentationBundle,
  indexes: PresentationIndexes,
): Promise<{ nodes: ArgumentGraphNode[]; edges: ArgumentGraphEdge[] }> {
  const claimNodes: ClaimGraphNode[] = presentation.claims.map((claim, index) => ({
    id: claimGraphNodeId(claim.claim_id),
    type: 'claim',
    position: { x: 0, y: 0 },
    sourcePosition: Position.Right,
    targetPosition: Position.Left,
    width: CLAIM_NODE_WIDTH,
    height: CLAIM_NODE_HEIGHT,
    data: {
      claim,
      claimNumber: index + 1,
      articleCounts: countArticleStances(
        indexes.articlesByClaimId.get(claim.claim_id) ?? [],
      ),
      visual: {
        selected: false,
        spotlight: false,
        dimmed: false,
        terminal: false,
      },
    },
  }))

  const inferenceNodes: InferenceGraphNode[] = presentation.inference_steps.map(
    (inferenceStep, index) => ({
      id: inferenceGraphNodeId(inferenceStep.inference_step_id),
      type: 'inference',
      position: { x: 0, y: 0 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
      width: INFERENCE_NODE_WIDTH,
      height: INFERENCE_NODE_HEIGHT,
      data: {
        inferenceStep,
        stepNumber: index + 1,
        visual: {
          selected: false,
          spotlight: false,
          dimmed: false,
          terminal: false,
        },
        selectedGapIndex: null,
      },
    }),
  )

  const edges: ArgumentGraphEdge[] = []
  for (const inferenceStep of presentation.inference_steps) {
    for (const premiseClaimId of inferenceStep.premise_claim_ids) {
      edges.push({
        id: `premise:${premiseClaimId}:${inferenceStep.inference_step_id}`,
        source: claimGraphNodeId(premiseClaimId),
        target: inferenceGraphNodeId(inferenceStep.inference_step_id),
        type: 'smoothstep',
        markerEnd: { type: MarkerType.ArrowClosed },
      })
    }

    edges.push({
      id: `conclusion:${inferenceStep.inference_step_id}:${inferenceStep.conclusion_claim_id}`,
      source: inferenceGraphNodeId(inferenceStep.inference_step_id),
      target: claimGraphNodeId(inferenceStep.conclusion_claim_id),
      type: 'smoothstep',
      markerEnd: { type: MarkerType.ArrowClosed },
    })
  }

  const allNodes: ArgumentGraphNode[] = [...claimNodes, ...inferenceNodes]
  const layout = await elk.layout({
    id: 'argument-graph',
    layoutOptions: {
      'elk.algorithm': 'layered',
      'elk.direction': 'RIGHT',
      'elk.edgeRouting': 'ORTHOGONAL',
      'elk.spacing.nodeNode': '54',
      'elk.layered.spacing.nodeNodeBetweenLayers': '100',
      'elk.layered.spacing.edgeNodeBetweenLayers': '34',
      'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      'elk.layered.crossingMinimization.strategy': 'LAYER_SWEEP',
      'elk.padding': '[top=32,left=32,bottom=32,right=32]',
    },
    children: allNodes.map((node) => ({
      id: node.id,
      width: node.width,
      height: node.height,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  })

  const positions = new Map(
    layout.children?.map((child) => [
      child.id,
      { x: child.x ?? 0, y: child.y ?? 0 },
    ]),
  )

  return {
    nodes: allNodes.map((node) => ({
      ...node,
      position: positions.get(node.id) ?? node.position,
    })) as ArgumentGraphNode[],
    edges,
  }
}
