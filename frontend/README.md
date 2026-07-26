# EvidenceGap Frontend

React + TypeScript + Vite frontend for the EvidenceGap backend.

The previous synthetic V0 case model and its unsupported presentation logic have been removed. The current frontend intentionally contains only the reusable workspace shell while the real backend contracts and API integration are introduced.

## Retained foundation

- `react-resizable-panels` for the adjustable workspace layout.
- `@xyflow/react` for the claim and inference graph surface.
- `elkjs` remains available for graph layout once backend presentation data is connected.
- Separate regions for run history, graph navigation, evidence inspection, and run summaries.

The frontend must render backend-provided analysis results and must not independently invent claim states, gap categories, conclusions, or evidence metadata.

## Development

```bash
pnpm install
pnpm dev
```

The project requires Node.js 24 and pnpm 11.


## Backend API

The API client lives in `src/api.ts`. Its default base URL is:

```text
https://www.next.zju.edu.cn/novel-evidencegap/back/
```

Override it for a deployment or local development by creating `.env.local`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8030/
```

A caller may also create an isolated client with a runtime URL:

```ts
import { createEvidenceGapApi } from './api'

const api = createEvidenceGapApi({ baseUrl: 'http://127.0.0.1:8030/' })
```

The client is not connected to the React workspace yet. This module only defines typed access to the current backend endpoints.
