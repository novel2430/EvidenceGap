import { useEffect, useMemo, useState } from 'react'
import {
  ReactFlow,
  type Node,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import {
  layoutClaimGraph,
  type ClaimGraphLayout,
} from '../graph/claimGraphLayout'
import type { DemoCase, InspectorSelection, ViewMode } from '../types'
import { ClaimNode } from './ClaimNode'
import { InferenceNode } from './InferenceNode'
import { RoutedEdge, type RoutedEdgeType } from './RoutedEdge'

const nodeTypes = {
  claim: ClaimNode,
  inference: InferenceNode,
}

const edgeTypes = {
  routed: RoutedEdge,
}

function edgeTone(status: string) {
  if (status === 'SUPPORTED') return 'edge-supported'
  if (status === 'CONFLICTED') return 'edge-conflict'
  if (status === 'BLOCKED' || status === 'UNKNOWN') return 'edge-blocked'
  return 'edge-partial'
}

function edgeColor(status: string) {
  if (status === 'SUPPORTED') return '#00a876'
  if (status === 'CONFLICTED') return '#8b5cf6'
  if (status === 'BLOCKED' || status === 'UNKNOWN') return '#f0522a'
  return '#5d96b8'
}

function linkedIds(
  currentCase: DemoCase,
  selection: InspectorSelection | null,
  mode: ViewMode,
  targetActive: boolean,
) {
  const ids = new Set<string>()

  if (targetActive) {
    currentCase.targetPath.forEach((id) => ids.add(id))
  }

  if (mode === 'gaps') {
    currentCase.gaps.forEach((gap) => {
      ids.add(gap.target)
      gap.blocks.forEach((id) => ids.add(id))
    })
    currentCase.claims
      .filter((claim) => claim.gapIds.length > 0 || claim.status === 'BLOCKED')
      .forEach((claim) => ids.add(claim.id))
  }

  if (mode === 'conflicts') {
    currentCase.claims
      .filter(
        (claim) => claim.status === 'CONFLICTED'
          || claim.reasonCodes.includes('CONFLICTING_EVIDENCE'),
      )
      .forEach((claim) => {
        ids.add(claim.id)
        claim.downstream.forEach((id) => ids.add(id))
      })
  }

  if (selection?.kind === 'claim') {
    const claim = currentCase.claims.find((item) => item.id === selection.id)
    ids.add(selection.id)
    claim?.downstream.forEach((id) => ids.add(id))
    currentCase.inferenceSteps
      .filter((step) => step.premises.includes(selection.id) || step.conclusion === selection.id)
      .forEach((step) => {
        ids.add(step.id)
        ids.add(step.conclusion)
        step.premises.forEach((id) => ids.add(id))
      })
  }

  if (selection?.kind === 'inference') {
    const step = currentCase.inferenceSteps.find((item) => item.id === selection.id)
    ids.add(selection.id)
    if (step) {
      ids.add(step.conclusion)
      step.premises.forEach((id) => ids.add(id))
    }
  }

  return ids
}

function edgeStatusById(currentCase: DemoCase) {
  const statusById = new Map<string, string>()

  currentCase.inferenceSteps.forEach((step) => {
    const conclusion = currentCase.claims.find((claim) => claim.id === step.conclusion)
    const status = conclusion?.status ?? 'PARTIAL'

    step.premises.forEach((premise) => {
      statusById.set(`${premise}-${step.id}`, status)
    })
    statusById.set(`${step.id}-${step.conclusion}`, status)
  })

  return statusById
}

export function ClaimGraph({
  currentCase,
  selection,
  viewMode,
  targetActive,
  onSelect,
}: {
  currentCase: DemoCase
  selection: InspectorSelection | null
  viewMode: ViewMode
  targetActive: boolean
  onSelect: (selection: InspectorSelection) => void
}) {
  const [layout, setLayout] = useState<ClaimGraphLayout | null>(null)

  useEffect(() => {
    let cancelled = false
    setLayout(null)

    void layoutClaimGraph(currentCase).then((nextLayout) => {
      if (!cancelled) setLayout(nextLayout)
    })

    return () => {
      cancelled = true
    }
  }, [currentCase])

  const focusIds = useMemo(
    () => linkedIds(currentCase, selection, viewMode, targetActive),
    [currentCase, selection, viewMode, targetActive],
  )
  const hasFocus = focusIds.size > 0

  const nodes = useMemo<Node[]>(() => {
    if (!layout) return []

    const claimById = new Map(currentCase.claims.map((claim) => [claim.id, claim]))
    const inferenceById = new Map(currentCase.inferenceSteps.map((step) => [step.id, step]))

    const result: Node[] = []

    layout.nodes.forEach((layoutNode) => {
      const claim = claimById.get(layoutNode.id)
      if (claim) {
        result.push({
          id: claim.id,
          type: 'claim',
          position: layoutNode.position,
          data: {
            claim,
            ports: layoutNode.ports,
            faded: hasFocus && !focusIds.has(claim.id),
            spotlight: focusIds.has(claim.id),
          },
        })
        return
      }

      const step = inferenceById.get(layoutNode.id)
      if (!step) return

      result.push({
        id: step.id,
        type: 'inference',
        position: layoutNode.position,
        data: {
          step,
          ports: layoutNode.ports,
          faded: hasFocus && !focusIds.has(step.id),
          spotlight: focusIds.has(step.id),
        },
      })
    })

    return result
  }, [currentCase, focusIds, hasFocus, layout])

  const edges = useMemo<RoutedEdgeType[]>(() => {
    if (!layout) return []

    const statusById = edgeStatusById(currentCase)

    return layout.edges.map((layoutEdge) => {
      const status = statusById.get(layoutEdge.id) ?? 'PARTIAL'
      const active = focusIds.has(layoutEdge.source) && focusIds.has(layoutEdge.target)

      return {
        id: layoutEdge.id,
        source: layoutEdge.source,
        sourceHandle: layoutEdge.sourceHandle,
        target: layoutEdge.target,
        targetHandle: layoutEdge.targetHandle,
        type: 'routed',
        data: { path: layoutEdge.path },
        style: { stroke: edgeColor(status) },
        className: [
          edgeTone(status),
          active ? 'edge-active' : '',
          hasFocus && !active ? 'edge-faded' : '',
        ].join(' '),
        animated: active && (targetActive || viewMode !== 'all'),
      }
    })
  }, [currentCase, focusIds, hasFocus, layout, targetActive, viewMode])

  const handleNodeClick: NodeMouseHandler = (_event, node) => {
    onSelect({ kind: node.type === 'inference' ? 'inference' : 'claim', id: node.id })
  }

  return (
    <section className="graph-panel panel">
      {layout ? (
        <ReactFlow
          key={`${currentCase.id}-${layout.usedFallback ? 'fallback' : 'elk'}`}
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodeClick={handleNodeClick}
          fitView
          fitViewOptions={{ padding: 0.12 }}
          minZoom={0.24}
          maxZoom={1.5}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          proOptions={{ hideAttribution: true }}
        />
      ) : (
        <div className="graph-loading" role="status">
          LAYOUTING GRAPH…
        </div>
      )}
    </section>
  )
}
