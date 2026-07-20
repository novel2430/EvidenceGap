import { useMemo } from 'react'
import {
  ReactFlow,
  type Edge,
  type Node,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import type { DemoCase, InspectorSelection, ViewMode } from '../types'
import { ClaimNode } from './ClaimNode'
import { InferenceNode } from './InferenceNode'

const nodeTypes = {
  claim: ClaimNode,
  inference: InferenceNode,
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

function linkedIds(currentCase: DemoCase, selection: InspectorSelection | null, mode: ViewMode, targetActive: boolean) {
  const ids = new Set<string>()

  if (targetActive) {
    currentCase.targetPath.forEach((id) => ids.add(id))
  }

  if (mode === 'gaps') {
    currentCase.gaps.forEach((gap) => {
      ids.add(gap.target)
      gap.blocks.forEach((id) => ids.add(id))
    })
    currentCase.claims.filter((claim) => claim.gapIds.length > 0 || claim.status === 'BLOCKED').forEach((claim) => ids.add(claim.id))
  }

  if (mode === 'conflicts') {
    currentCase.claims
      .filter((claim) => claim.status === 'CONFLICTED' || claim.reasonCodes.includes('CONFLICTING_EVIDENCE'))
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
  const focusIds = useMemo(
    () => linkedIds(currentCase, selection, viewMode, targetActive),
    [currentCase, selection, viewMode, targetActive],
  )
  const hasFocus = focusIds.size > 0

  const nodes = useMemo<Node[]>(() => {
    const claimNodes: Node[] = currentCase.claims.map((claim) => ({
      id: claim.id,
      type: 'claim',
      position: claim.position,
      data: {
        claim,
        faded: hasFocus && !focusIds.has(claim.id),
        spotlight: focusIds.has(claim.id),
      },
    }))

    const inferenceNodes: Node[] = currentCase.inferenceSteps.map((step) => ({
      id: step.id,
      type: 'inference',
      position: step.position,
      data: {
        step,
        faded: hasFocus && !focusIds.has(step.id),
        spotlight: focusIds.has(step.id),
      },
    }))

    return [...claimNodes, ...inferenceNodes]
  }, [currentCase, focusIds, hasFocus])

  const edges = useMemo<Edge[]>(() => {
    const result: Edge[] = []
    currentCase.inferenceSteps.forEach((step) => {
      const conclusion = currentCase.claims.find((claim) => claim.id === step.conclusion)
      const edgeClass = edgeTone(conclusion?.status ?? 'PARTIAL')
      const color = edgeColor(conclusion?.status ?? 'PARTIAL')
      step.premises.forEach((premise) => {
        const id = `${premise}-${step.id}`
        const active = focusIds.has(premise) && focusIds.has(step.id)
        result.push({
          id,
          source: premise,
          sourceHandle: 'out',
          target: step.id,
          targetHandle: 'in',
          type: 'straight',
          style: { stroke: color },
          className: [edgeClass, active ? 'edge-active' : '', hasFocus && !active ? 'edge-faded' : ''].join(' '),
          animated: active && (targetActive || viewMode !== 'all'),
        })
      })

      const id = `${step.id}-${step.conclusion}`
      const active = focusIds.has(step.id) && focusIds.has(step.conclusion)
      result.push({
        id,
        source: step.id,
        sourceHandle: 'out',
        target: step.conclusion,
        targetHandle: 'in',
        type: 'straight',
        style: { stroke: color },
        className: [edgeClass, active ? 'edge-active' : '', hasFocus && !active ? 'edge-faded' : ''].join(' '),
        animated: active && (targetActive || viewMode !== 'all'),
      })
    })
    return result
  }, [currentCase, focusIds, hasFocus, targetActive, viewMode])

  const handleNodeClick: NodeMouseHandler = (_event, node) => {
    onSelect({ kind: node.type === 'inference' ? 'inference' : 'claim', id: node.id })
  }

  return (
    <section className="graph-panel panel">
      <ReactFlow
        key={currentCase.id}
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={handleNodeClick}
        fitView
        fitViewOptions={{ padding: 0.16 }}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        proOptions={{ hideAttribution: true }}
      />
    </section>
  )
}
