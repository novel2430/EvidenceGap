import { Handle, Position } from '@xyflow/react'
import type { GraphPort } from '../graph/claimGraphLayout'

export function GraphHandles({ ports }: { ports: GraphPort[] }) {
  return (
    <>
      {ports.map((port) => (
        <Handle
          id={port.id}
          key={port.id}
          type={port.side === 'left' ? 'target' : 'source'}
          position={port.side === 'left' ? Position.Left : Position.Right}
          style={{ top: port.offset }}
        />
      ))}
    </>
  )
}
