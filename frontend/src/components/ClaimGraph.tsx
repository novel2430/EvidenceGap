import { useEffect, useMemo, useState } from 'react'
import {
  Background,
  Controls,
  ReactFlow,
  type NodeTypes,
  type ReactFlowInstance,
} from '@xyflow/react'
import { Network } from 'lucide-react'
import '@xyflow/react/dist/style.css'
import type { PresentationBundle } from '../contracts'
import {
  claimGraphNodeId,
  inferenceGraphNodeId,
  layoutPresentationGraph,
} from '../graph/layout'
import type {
  ArgumentGraphEdge,
  ArgumentGraphNode,
} from '../graph/types'
import type {
  GraphFocusMode,
  GraphSelection,
  PresentationIndexes,
} from '../utils/presentation'
import { getSelectionClaimId } from '../utils/presentation'
import { useReducedMotion } from '../hooks/useReducedMotion'
import { UI_TEXT } from '../uiText'
import { ClaimNode } from './ClaimNode'
import { InferenceNode } from './InferenceNode'

const nodeTypes: NodeTypes = {
  claim: ClaimNode,
  inference: InferenceNode,
}

interface ClaimGraphProps {
  presentation: PresentationBundle | null
  indexes: PresentationIndexes
  selection: GraphSelection | null
  onSelectionChange: (selection: GraphSelection | null) => void
}

function addGapImpactNodes(
  nodeIds: Set<string>,
  indexes: PresentationIndexes,
  inferenceStepId: string,
) {
  const inferenceStep = indexes.inferenceStepsById.get(inferenceStepId)
  if (!inferenceStep) return

  nodeIds.add(inferenceGraphNodeId(inferenceStep.inference_step_id))
  nodeIds.add(claimGraphNodeId(inferenceStep.impact.direct_conclusion_claim_id))
  for (const claimId of inferenceStep.impact.downstream_claim_ids) {
    nodeIds.add(claimGraphNodeId(claimId))
  }
  for (const downstreamInferenceStepId of inferenceStep.impact.downstream_inference_step_ids) {
    nodeIds.add(inferenceGraphNodeId(downstreamInferenceStepId))
  }
  for (const claimId of inferenceStep.impact.terminal_claim_ids) {
    nodeIds.add(claimGraphNodeId(claimId))
  }
}

export function ClaimGraph({
  presentation,
  indexes,
  selection,
  onSelectionChange,
}: ClaimGraphProps) {
  const [focusMode, setFocusMode] = useState<GraphFocusMode>('all')
  const [baseNodes, setBaseNodes] = useState<ArgumentGraphNode[]>([])
  const [baseEdges, setBaseEdges] = useState<ArgumentGraphEdge[]>([])
  const [isLayoutPending, setIsLayoutPending] = useState(false)
  const [layoutError, setLayoutError] = useState<string | null>(null)
  const reducedMotion = useReducedMotion()
  const [flowInstance, setFlowInstance] = useState<
    ReactFlowInstance<ArgumentGraphNode, ArgumentGraphEdge> | null
  >(null)

  useEffect(() => {
    let cancelled = false
    if (!presentation) {
      setBaseNodes([])
      setBaseEdges([])
      setIsLayoutPending(false)
      setLayoutError(null)
      return
    }

    setBaseNodes([])
    setBaseEdges([])
    setIsLayoutPending(true)
    setLayoutError(null)
    void layoutPresentationGraph(presentation, indexes)
      .then((layout) => {
        if (cancelled) return
        setBaseNodes(layout.nodes)
        setBaseEdges(layout.edges)
        setIsLayoutPending(false)
      })
      .catch((error: unknown) => {
        if (cancelled) return
        setLayoutError(
          error instanceof Error
            ? error.message
            : UI_TEXT.graph.layoutErrorFallback,
        )
        setIsLayoutPending(false)
      })

    return () => {
      cancelled = true
    }
  }, [indexes, presentation])

  useEffect(() => {
    if (!flowInstance || baseNodes.length === 0) return
    void flowInstance.fitView({
      padding: 0.16,
      duration: reducedMotion ? 0 : 180,
    })
  }, [baseNodes, flowInstance, reducedMotion])

  useEffect(() => {
    if (!flowInstance || !selection) return
    const selectionClaimId = getSelectionClaimId(selection, indexes)
    const nodeId = selectionClaimId
      ? claimGraphNodeId(selectionClaimId)
      : selection.kind === 'inference_step' || selection.kind === 'gap'
        ? inferenceGraphNodeId(selection.inferenceStepId)
        : null
    if (!nodeId) return
    void flowInstance.fitView({
      nodes: [{ id: nodeId }],
      padding: 0.7,
      maxZoom: 1.15,
      duration: reducedMotion ? 0 : 180,
    })
  }, [flowInstance, indexes, reducedMotion, selection])

  const visualState = useMemo(() => {
    const modeNodeIds = new Set<string>()
    const modeEdgeIds = new Set<string>()
    const selectionNodeIds = new Set<string>()
    const selectionEdgeIds = new Set<string>()
    const terminalNodeIds = new Set<string>()

    if (presentation && focusMode === 'gaps') {
      for (const inferenceStep of presentation.inference_steps) {
        if (inferenceStep.gaps.length === 0) continue
        addGapImpactNodes(modeNodeIds, indexes, inferenceStep.inference_step_id)
        for (const claimId of inferenceStep.impact.terminal_claim_ids) {
          terminalNodeIds.add(claimGraphNodeId(claimId))
        }
      }
      for (const edge of baseEdges) {
        if (modeNodeIds.has(edge.source) && modeNodeIds.has(edge.target)) {
          modeEdgeIds.add(edge.id)
        }
      }
    }

    if (presentation && focusMode === 'conflicts') {
      for (const claim of presentation.claims) {
        if (claim.evidence_state !== 'CONFLICTED') continue
        const claimNodeId = claimGraphNodeId(claim.claim_id)
        modeNodeIds.add(claimNodeId)
        for (const inferenceStep of indexes.inferenceStepsByClaimId.get(claim.claim_id) ?? []) {
          modeNodeIds.add(inferenceGraphNodeId(inferenceStep.inference_step_id))
        }
        for (const edge of baseEdges) {
          if (edge.source === claimNodeId || edge.target === claimNodeId) {
            modeEdgeIds.add(edge.id)
          }
        }
      }
    }

    const selectionClaimId = getSelectionClaimId(selection, indexes)
    if (presentation && selectionClaimId) {
      const claimNodeId = claimGraphNodeId(selectionClaimId)
      selectionNodeIds.add(claimNodeId)
      for (const inferenceStep of indexes.inferenceStepsByClaimId.get(selectionClaimId) ?? []) {
        selectionNodeIds.add(inferenceGraphNodeId(inferenceStep.inference_step_id))
      }
      for (const edge of baseEdges) {
        if (edge.source === claimNodeId || edge.target === claimNodeId) {
          selectionEdgeIds.add(edge.id)
        }
      }
    }

    if (presentation && selection?.kind === 'inference_step') {
      const inferenceStep = indexes.inferenceStepsById.get(selection.inferenceStepId)
      if (inferenceStep) {
        const inferenceNodeId = inferenceGraphNodeId(inferenceStep.inference_step_id)
        selectionNodeIds.add(inferenceNodeId)
        for (const claimId of inferenceStep.premise_claim_ids) {
          selectionNodeIds.add(claimGraphNodeId(claimId))
        }
        selectionNodeIds.add(claimGraphNodeId(inferenceStep.conclusion_claim_id))
        for (const edge of baseEdges) {
          if (edge.source === inferenceNodeId || edge.target === inferenceNodeId) {
            selectionEdgeIds.add(edge.id)
          }
        }
      }
    }

    if (presentation && selection?.kind === 'gap') {
      addGapImpactNodes(selectionNodeIds, indexes, selection.inferenceStepId)
      const inferenceStep = indexes.inferenceStepsById.get(selection.inferenceStepId)
      if (inferenceStep) {
        for (const claimId of inferenceStep.impact.terminal_claim_ids) {
          terminalNodeIds.add(claimGraphNodeId(claimId))
        }
      }
      for (const edge of baseEdges) {
        if (
          selectionNodeIds.has(edge.source) &&
          selectionNodeIds.has(edge.target)
        ) {
          selectionEdgeIds.add(edge.id)
        }
      }
    }

    return {
      modeNodeIds,
      modeEdgeIds,
      selectionNodeIds,
      selectionEdgeIds,
      terminalNodeIds,
    }
  }, [baseEdges, focusMode, indexes, presentation, selection])

  const nodes = useMemo(
    () => baseNodes.map((node) => {
      const isExactSelection =
        (selection?.kind === 'claim' &&
          node.id === claimGraphNodeId(selection.claimId)) ||
        ((selection?.kind === 'inference_step' || selection?.kind === 'gap') &&
          node.id === inferenceGraphNodeId(selection.inferenceStepId))
      const inModeFocus = focusMode === 'all' || visualState.modeNodeIds.has(node.id)
      const inSelectionFocus = !selection || visualState.selectionNodeIds.has(node.id)
      const visual = {
        selected: isExactSelection,
        spotlight:
          visualState.selectionNodeIds.has(node.id) ||
          (focusMode !== 'all' && visualState.modeNodeIds.has(node.id)),
        dimmed: !inModeFocus || !inSelectionFocus,
        terminal: visualState.terminalNodeIds.has(node.id),
      }

      if (node.type === 'claim') {
        return {
          ...node,
          data: {
            ...node.data,
            visual,
            onSelect: onSelectionChange,
          },
        }
      }

      return {
        ...node,
        data: {
          ...node.data,
          visual,
          selectedGapIndex:
            selection?.kind === 'gap' &&
            selection.inferenceStepId === node.data.inferenceStep.inference_step_id
              ? selection.gapIndex
              : null,
          onSelect: onSelectionChange,
        },
      }
    }) as ArgumentGraphNode[],
    [baseNodes, focusMode, onSelectionChange, selection, visualState],
  )

  const edges = useMemo(
    () => baseEdges.map((edge) => {
      const inModeFocus = focusMode === 'all' || visualState.modeEdgeIds.has(edge.id)
      const inSelectionFocus = !selection || visualState.selectionEdgeIds.has(edge.id)
      const isHighlighted =
        visualState.selectionEdgeIds.has(edge.id) ||
        (focusMode !== 'all' && visualState.modeEdgeIds.has(edge.id))
      return {
        ...edge,
        className: [
          'argument-edge',
          isHighlighted ? 'is-highlighted' : '',
          !inModeFocus || !inSelectionFocus ? 'is-dimmed' : '',
        ].filter(Boolean).join(' '),
        style: {
          stroke: isHighlighted ? '#1677ff' : '#8a9bb0',
          strokeWidth: isHighlighted ? 2.5 : 1.5,
        },
        zIndex: isHighlighted ? 2 : 0,
      }
    }),
    [baseEdges, focusMode, selection, visualState],
  )

  if (!presentation) {
    return (
      <section className="graph-panel panel">
        <div className="graph-grid" />
        <div className="graph-empty-state" role="status">
          <div className="empty-state-icon"><Network size={24} /></div>
          <span className="eyebrow">{UI_TEXT.graph.eyebrow}</span>
          <h2>{UI_TEXT.graph.noSelection}</h2>
          <p>{UI_TEXT.graph.noSelectionDescription}</p>
        </div>
      </section>
    )
  }

  return (
    <section className="graph-panel panel">
      <div
        className="graph-focus-toolbar"
        aria-label={UI_TEXT.graph.focusModeLabel}
      >
        {(['all', 'gaps', 'conflicts'] as GraphFocusMode[]).map((mode) => (
          <button
            className={focusMode === mode ? 'is-active' : ''}
            type="button"
            key={mode}
            onClick={() => setFocusMode(mode)}
            aria-pressed={focusMode === mode}
          >
            {UI_TEXT.graph.focusModes[mode]}
          </button>
        ))}
      </div>

      {isLayoutPending ? (
        <div className="graph-loading" role="status">
          {UI_TEXT.graph.layingOut}
        </div>
      ) : layoutError ? (
        <div className="graph-empty-state" role="alert">
          <div className="empty-state-icon"><Network size={24} /></div>
          <h2>{UI_TEXT.graph.unavailable}</h2>
          <p>{layoutError}</p>
        </div>
      ) : presentation.claims.length === 0 ? (
        <div className="graph-empty-state" role="status">
          <div className="empty-state-icon"><Network size={24} /></div>
          <h2>{UI_TEXT.graph.noClaims}</h2>
          <p>{UI_TEXT.graph.noClaimsDescription}</p>
        </div>
      ) : (
        <ReactFlow<ArgumentGraphNode, ArgumentGraphEdge>
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          onInit={setFlowInstance}
          onPaneClick={() => onSelectionChange(null)}
          onNodeClick={(event) => event.stopPropagation()}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          minZoom={0.2}
          maxZoom={1.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={24} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      )}
    </section>
  )
}
