import ELK, {
  type ElkExtendedEdge,
  type ElkNode,
  type ElkPoint,
} from 'elkjs/lib/elk.bundled.js'
import type { DemoCase } from '../types'

const elk = new ELK()

const PORT_SIZE = 10
const CLAIM_WIDTH = 238
const CLAIM_HEIGHT = 132
const TARGET_WIDTH = 270
const TARGET_HEIGHT = 150
const INFERENCE_WIDTH = 190
const INFERENCE_HEIGHT = 70

interface LogicalEdge {
  id: string
  source: string
  target: string
}

interface PortDefinition {
  id: string
  nodeId: string
  side: 'left' | 'right'
  edgeId: string
}

export interface GraphPort {
  id: string
  side: 'left' | 'right'
  offset: number
}

export interface GraphLayoutNode {
  id: string
  position: {
    x: number
    y: number
  }
  ports: GraphPort[]
}

export interface GraphLayoutEdge {
  id: string
  source: string
  target: string
  sourceHandle: string
  targetHandle: string
  path: string
}

export interface ClaimGraphLayout {
  nodes: GraphLayoutNode[]
  edges: GraphLayoutEdge[]
  usedFallback: boolean
}

function nodeSize(currentCase: DemoCase, nodeId: string) {
  const claim = currentCase.claims.find((item) => item.id === nodeId)
  if (claim) {
    return claim.isTarget
      ? { width: TARGET_WIDTH, height: TARGET_HEIGHT }
      : { width: CLAIM_WIDTH, height: CLAIM_HEIGHT }
  }

  return { width: INFERENCE_WIDTH, height: INFERENCE_HEIGHT }
}

function originalY(currentCase: DemoCase, nodeId: string) {
  const claim = currentCase.claims.find((item) => item.id === nodeId)
  if (claim) return claim.position.y

  return currentCase.inferenceSteps.find((item) => item.id === nodeId)?.position.y ?? 0
}

function logicalEdges(currentCase: DemoCase) {
  const edges: LogicalEdge[] = []

  currentCase.inferenceSteps.forEach((step) => {
    step.premises.forEach((premise) => {
      edges.push({
        id: `${premise}-${step.id}`,
        source: premise,
        target: step.id,
      })
    })

    edges.push({
      id: `${step.id}-${step.conclusion}`,
      source: step.id,
      target: step.conclusion,
    })
  })

  return edges
}

function sourcePortId(edgeId: string) {
  return `${edgeId}::source`
}

function targetPortId(edgeId: string) {
  return `${edgeId}::target`
}

function portDefinitions(currentCase: DemoCase, edges: LogicalEdge[]) {
  const definitions = new Map<string, PortDefinition[]>()
  const nodeIds = [
    ...currentCase.claims.map((claim) => claim.id),
    ...currentCase.inferenceSteps.map((step) => step.id),
  ]

  nodeIds.forEach((nodeId) => {
    const incoming = edges
      .filter((edge) => edge.target === nodeId)
      .sort((left, right) => originalY(currentCase, left.source) - originalY(currentCase, right.source) || left.id.localeCompare(right.id))
    const outgoing = edges
      .filter((edge) => edge.source === nodeId)
      .sort((left, right) => originalY(currentCase, left.target) - originalY(currentCase, right.target) || left.id.localeCompare(right.id))

    definitions.set(nodeId, [
      ...incoming.map((edge) => ({
        id: targetPortId(edge.id),
        nodeId,
        side: 'left' as const,
        edgeId: edge.id,
      })),
      ...outgoing.map((edge) => ({
        id: sourcePortId(edge.id),
        nodeId,
        side: 'right' as const,
        edgeId: edge.id,
      })),
    ])
  })

  return definitions
}

function defaultPortsForNode(height: number, definitions: PortDefinition[]) {
  const ports: GraphPort[] = []

  ;(['left', 'right'] as const).forEach((side) => {
    const sidePorts = definitions.filter((port) => port.side === side)
    sidePorts.forEach((port, index) => {
      ports.push({
        id: port.id,
        side,
        offset: (height * (index + 1)) / (sidePorts.length + 1),
      })
    })
  })

  return ports
}

function roundedOrthogonalPath(points: ElkPoint[], radius = 14) {
  if (points.length === 0) return ''
  if (points.length === 1) return `M ${points[0].x} ${points[0].y}`

  let path = `M ${points[0].x} ${points[0].y}`

  for (let index = 1; index < points.length - 1; index += 1) {
    const previous = points[index - 1]
    const current = points[index]
    const next = points[index + 1]
    const incomingLength = Math.hypot(current.x - previous.x, current.y - previous.y)
    const outgoingLength = Math.hypot(next.x - current.x, next.y - current.y)

    if (incomingLength === 0 || outgoingLength === 0) continue

    const cornerRadius = Math.min(radius, incomingLength / 2, outgoingLength / 2)
    const before = {
      x: current.x + ((previous.x - current.x) / incomingLength) * cornerRadius,
      y: current.y + ((previous.y - current.y) / incomingLength) * cornerRadius,
    }
    const after = {
      x: current.x + ((next.x - current.x) / outgoingLength) * cornerRadius,
      y: current.y + ((next.y - current.y) / outgoingLength) * cornerRadius,
    }

    path += ` L ${before.x} ${before.y} Q ${current.x} ${current.y} ${after.x} ${after.y}`
  }

  const last = points[points.length - 1]
  return `${path} L ${last.x} ${last.y}`
}

function pathFromSections(
  sections: ElkExtendedEdge['sections'],
  sourcePoint: ElkPoint,
  targetPoint: ElkPoint,
) {
  if (!sections || sections.length === 0) return ''

  return sections
    .map((section, index) => roundedOrthogonalPath([
      index === 0 ? sourcePoint : section.startPoint,
      ...(section.bendPoints ?? []),
      index === sections.length - 1 ? targetPoint : section.endPoint,
    ]))
    .join(' ')
}

function fallbackPath(
  sourcePosition: { x: number; y: number },
  sourceSize: { width: number; height: number },
  sourcePort: GraphPort,
  targetPosition: { x: number; y: number },
  targetPort: GraphPort,
) {
  const sourceX = sourcePosition.x + sourceSize.width
  const sourceY = sourcePosition.y + sourcePort.offset
  const targetX = targetPosition.x
  const targetY = targetPosition.y + targetPort.offset
  const controlDistance = Math.max(60, (targetX - sourceX) * 0.42)

  return `M ${sourceX} ${sourceY} C ${sourceX + controlDistance} ${sourceY}, ${targetX - controlDistance} ${targetY}, ${targetX} ${targetY}`
}

function createFallbackLayout(
  currentCase: DemoCase,
  edges: LogicalEdge[],
  definitions: Map<string, PortDefinition[]>,
): ClaimGraphLayout {
  const nodes: GraphLayoutNode[] = [
    ...currentCase.claims.map((claim) => {
      const size = nodeSize(currentCase, claim.id)
      return {
        id: claim.id,
        position: claim.position,
        ports: defaultPortsForNode(size.height, definitions.get(claim.id) ?? []),
      }
    }),
    ...currentCase.inferenceSteps.map((step) => {
      const size = nodeSize(currentCase, step.id)
      return {
        id: step.id,
        position: step.position,
        ports: defaultPortsForNode(size.height, definitions.get(step.id) ?? []),
      }
    }),
  ]

  const nodeById = new Map(nodes.map((node) => [node.id, node]))
  const routedEdges = edges.map((edge) => {
    const sourceNode = nodeById.get(edge.source)
    const targetNode = nodeById.get(edge.target)
    const sourceHandle = sourcePortId(edge.id)
    const targetHandle = targetPortId(edge.id)
    const sourcePort = sourceNode?.ports.find((port) => port.id === sourceHandle)
    const targetPort = targetNode?.ports.find((port) => port.id === targetHandle)

    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      sourceHandle,
      targetHandle,
      path: sourceNode && targetNode && sourcePort && targetPort
        ? fallbackPath(sourceNode.position, nodeSize(currentCase, edge.source), sourcePort, targetNode.position, targetPort)
        : '',
    }
  })

  return { nodes, edges: routedEdges, usedFallback: true }
}

export async function layoutClaimGraph(currentCase: DemoCase): Promise<ClaimGraphLayout> {
  const edges = logicalEdges(currentCase)
  const definitions = portDefinitions(currentCase, edges)

  try {
    const graph: ElkNode = {
      id: `claim-graph-${currentCase.id}`,
      layoutOptions: {
        'elk.algorithm': 'layered',
        'elk.direction': 'RIGHT',
        'elk.edgeRouting': 'ORTHOGONAL',
        'elk.padding': '[top=44,left=44,bottom=44,right=44]',
        'elk.spacing.nodeNode': '84',
        'elk.spacing.edgeNode': '32',
        'elk.spacing.edgeEdge': '18',
        'elk.spacing.portPort': '20',
        'elk.layered.spacing.nodeNodeBetweenLayers': '126',
        'elk.layered.spacing.edgeNodeBetweenLayers': '36',
        'elk.layered.spacing.edgeEdgeBetweenLayers': '22',
        'elk.layered.mergeEdges': 'false',
        'elk.layered.nodePlacement.strategy': 'NETWORK_SIMPLEX',
      },
      children: [
        ...currentCase.claims.map((claim) => {
          const size = nodeSize(currentCase, claim.id)
          return {
            id: claim.id,
            width: size.width,
            height: size.height,
            layoutOptions: {
              'elk.portConstraints': 'FIXED_SIDE',
            },
            ports: (definitions.get(claim.id) ?? []).map((port) => ({
              id: port.id,
              width: PORT_SIZE,
              height: PORT_SIZE,
              layoutOptions: {
                'elk.port.side': port.side === 'left' ? 'WEST' : 'EAST',
              },
            })),
          }
        }),
        ...currentCase.inferenceSteps.map((step) => {
          const size = nodeSize(currentCase, step.id)
          return {
            id: step.id,
            width: size.width,
            height: size.height,
            layoutOptions: {
              'elk.portConstraints': 'FIXED_SIDE',
            },
            ports: (definitions.get(step.id) ?? []).map((port) => ({
              id: port.id,
              width: PORT_SIZE,
              height: PORT_SIZE,
              layoutOptions: {
                'elk.port.side': port.side === 'left' ? 'WEST' : 'EAST',
              },
            })),
          }
        }),
      ],
      edges: edges.map((edge) => ({
        id: edge.id,
        sources: [sourcePortId(edge.id)],
        targets: [targetPortId(edge.id)],
      })),
    }

    const result = await elk.layout(graph)
    const laidOutNodes = result.children ?? []
    const laidOutEdges = result.edges ?? []
    const portSideById = new Map(
      [...definitions.values()].flat().map((port) => [port.id, port.side] as const),
    )

    const nodes: GraphLayoutNode[] = laidOutNodes.map((node) => {
      const size = nodeSize(currentCase, node.id)
      const fallbackPorts = defaultPortsForNode(size.height, definitions.get(node.id) ?? [])
      const fallbackById = new Map(fallbackPorts.map((port) => [port.id, port]))

      return {
        id: node.id,
        position: {
          x: node.x ?? 0,
          y: node.y ?? 0,
        },
        ports: (node.ports ?? []).map((port) => ({
          id: port.id,
          side: portSideById.get(port.id) ?? fallbackById.get(port.id)?.side ?? 'left',
          offset: (port.y ?? fallbackById.get(port.id)?.offset ?? size.height / 2) + (port.height ?? 0) / 2,
        })),
      }
    })

    const nodeById = new Map(nodes.map((node) => [node.id, node]))
    const elkEdgeById = new Map(laidOutEdges.map((edge) => [edge.id, edge]))
    const routedEdges: GraphLayoutEdge[] = edges.map((edge) => {
      const sourceHandle = sourcePortId(edge.id)
      const targetHandle = targetPortId(edge.id)
      const sourceNode = nodeById.get(edge.source)
      const targetNode = nodeById.get(edge.target)
      const sourcePort = sourceNode?.ports.find((port) => port.id === sourceHandle)
      const targetPort = targetNode?.ports.find((port) => port.id === targetHandle)
      const sourceSize = nodeSize(currentCase, edge.source)
      const sourcePoint = {
        x: (sourceNode?.position.x ?? 0) + sourceSize.width,
        y: (sourceNode?.position.y ?? 0) + (sourcePort?.offset ?? sourceSize.height / 2),
      }
      const targetPoint = {
        x: targetNode?.position.x ?? 0,
        y: (targetNode?.position.y ?? 0) + (targetPort?.offset ?? nodeSize(currentCase, edge.target).height / 2),
      }
      const elkPath = pathFromSections(
        elkEdgeById.get(edge.id)?.sections,
        sourcePoint,
        targetPoint,
      )

      return {
        id: edge.id,
        source: edge.source,
        target: edge.target,
        sourceHandle,
        targetHandle,
        path: elkPath || (
          sourceNode && targetNode && sourcePort && targetPort
            ? fallbackPath(sourceNode.position, nodeSize(currentCase, edge.source), sourcePort, targetNode.position, targetPort)
            : ''
        ),
      }
    })

    return { nodes, edges: routedEdges, usedFallback: false }
  } catch (error) {
    console.error('ELK layout failed; using the case fallback positions.', error)
    return createFallbackLayout(currentCase, edges, definitions)
  }
}
