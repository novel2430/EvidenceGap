# EvidenceGap Backend Runtime

EvidenceGap now runs as a constrained LangGraph Evidence Agent Harness by
default. The public `EvidenceGapEngine`, FastAPI URLs, request schemas, progress
stages, and presentation bundle remain unchanged. Set
`EVIDENCEGAP_AGENT_ENABLED=false` to use the retained fixed workflow.

The Agent Controller sees a compact, JSON-serializable Evidence Workspace and
chooses one structured action: `SEARCH`, `RESOLVE`, `ABSTAIN`, or `FINISH`.
`SEARCH` is the only high-level tool exposed to it. That tool wraps the existing
BM25 + MedCPT + BMRetriever retrieval, RRF, Cross-Encoder reranking, article
evidence Judge, deterministic claim aggregation, and final evidence graph. A
controller query affects retrieval and reranking only; the canonical claim still
defines the Claim ID, Judge hypothesis, aggregation, and downstream bundle.

LangGraph owns explicit state, conditional routing, the Controller → Tool →
Workspace loop, and SQLite checkpointing. Ordinary code validates every action,
enforces total/per-claim budgets and duplicate-query guards, selects a real
successful attempt, and supplies deterministic fallback decisions. The
Controller never creates the formal medical verdict.

Each successful run adds these auditable files under `<run>/agent/`:

```text
workspace.json
action_trace.jsonl
agent_manifest.json
execution_graph.mmd
checkpoints.sqlite
```

Run the network/GPU-free smoke test (it executes the real compiled graph with a
fake controller and fake search implementation):

```bash
source .venv/bin/activate
python -m pip install -e './backend[test,agent-dev]'
python -m evidencegap_backend.agent.demo_fake
```

Open the same graph topology in LangGraph Studio from the repository root:

```bash
langgraph dev
```

Useful Agent settings are available in `config.json` under `agent`, or through:

```text
EVIDENCEGAP_AGENT_ENABLED
EVIDENCEGAP_AGENT_MAX_STEPS
EVIDENCEGAP_AGENT_TOTAL_SEARCH_BUDGET
EVIDENCEGAP_AGENT_PER_CLAIM_SEARCH_BUDGET
EVIDENCEGAP_AGENT_CONTROLLER_RETRY_COUNT
EVIDENCEGAP_AGENT_CHECKPOINT_ENABLED
EVIDENCEGAP_AGENT_CONTROLLER_PROVIDER
EVIDENCEGAP_AGENT_CONTROLLER_MODEL
EVIDENCEGAP_AGENT_CONTROLLER_API_KEY_ENV
EVIDENCEGAP_AGENT_CONTROLLER_BASE_URL
EVIDENCEGAP_AGENT_CONTROLLER_MAX_TOKENS
EVIDENCEGAP_AGENT_CONTROLLER_THINKING
```

Studio is a development/trace surface only; FastAPI remains the production
entry point and no second business pipeline or server contract is introduced.

`backend/` is the independent online implementation of the current EvidenceGap
07.7 statement-to-presentation pipeline. It does not import `src/evidencegap`,
does not depend on the old CLI, and does not use subprocesses.

The stable entry point is `EvidenceGapEngine`:

```python
from pathlib import Path
from evidencegap_backend import BackendConfig, EvidenceGapEngine

engine = EvidenceGapEngine(
    BackendConfig(
        workspace_root=Path("/path/to/EvidenceGap"),
        provider="deepseek",
        model="deepseek-v4-pro",
        device="cuda:0",
    )
)

engine.load()  # Load models, indexes, Stanza, and the article store once.
try:
    first = engine.analyze_statement(
        statement="Vitamin D supplementation prevents respiratory infections.",
        run_name="example_1",
        language="English",
    )
    second = engine.analyze_statement(
        statement="Vitamin D supplementation reduces fracture risk.",
        run_name="example_2",
        language="English",
    )
    print(engine.runtime_status)
finally:
    engine.close()
```

## Runtime lifecycle

`EvidenceGapEngine.load()` now creates one long-lived `RuntimeResources` object
that owns and reuses:

- the memory-mapped BM25 index;
- the MedCPT query encoder and FAISS index;
- the BMRetriever query encoder and FAISS index;
- the MedCPT cross-encoder;
- the Stanza citation sentence splitter;
- a persistent DuckDB article store with a bounded in-memory article cache.

Calls to `analyze_statement()` are serialized because these GPU-backed resources
are shared by the engine. `close()` releases model references, closes DuckDB,
and clears CUDA caches where applicable. `runtime_status` exposes load counts,
resource identities, analysis counts, dense query counts, and article-cache
statistics for lifecycle verification.

The 07.7 algorithm and final presentation contract are unchanged: argument-
preserving decomposition, three-source article retrieval with RRF, MedCPT
cross-encoder reranking, Stanza citation sentence materialization, article-level
LLM evidence judgment, deterministic claim aggregation, evidence graph
construction, inference-gap analysis, and optional localization.

Pipeline stages still persist their normal artifacts for traceability and final
validation, but active engine execution passes the main stage values in memory
instead of repeatedly re-reading the same Parquet and JSON artifacts.

Phase C adds a thin FastAPI wrapper without changing the runtime pipeline.

## Phase C: FastAPI wrapper

The HTTP layer is intentionally thin. It creates one `EvidenceGapEngine`, loads
it once in the FastAPI lifespan, and uses one in-process worker to call only:

```python
engine.analyze_statement(...)
```

There is no subprocess, Celery, Redis, or second pipeline implementation. API
run state is stored atomically under `artifacts/v1/api_runs/`, while the normal
07.7 artifacts remain under the engine's configured artifact root.

Install the API and test dependencies:

```bash
python -m pip install -e './backend[test,agent-dev]'
```

Minimal environment:

```bash
export EVIDENCEGAP_WORKSPACE_ROOT="$PWD"
export EVIDENCEGAP_PROVIDER=deepseek
export EVIDENCEGAP_MODEL=deepseek-v4-pro
export DEEPSEEK_API_KEY='...'
```

Run one Uvicorn worker. Multiple workers would each load their own models and
indexes:

```bash
CUDA_VISIBLE_DEVICES=0 \
uvicorn evidencegap_backend.api.app:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1
```

Submit an analysis:

```bash
curl -i -X POST http://127.0.0.1:8000/api/v1/runs \
  -H 'Content-Type: application/json' \
  -d '{
    "statement": "Vitamin D supplementation prevents respiratory infections.",
    "language": "English"
  }'
```

Poll the returned `Location`:

```bash
curl http://127.0.0.1:8000/api/v1/runs/<run_id>
```

The status is `queued`, `running`, `succeeded`, or `failed`. While a run is
active, the response also reports the current pipeline stage and claim-level
progress. A succeeded response contains the complete presentation bundle in
`result`. `/health` reports the worker and Engine lifecycle state without
exposing model paths.

Phase 8.2 read and delivery endpoints:

```text
GET  /api/v1/runs?limit=20&cursor=<run_id>
GET  /api/v1/runs/<run_id>/articles/<article_node_id>
GET  /api/v1/runs/<run_id>/exports/result.json
GET  /api/v1/runs/<run_id>/exports/report.md
POST /api/v1/runs/<run_id>/localizations
GET  /api/v1/runs/<run_id>/localizations
GET  /api/v1/runs/<run_id>/localizations/<localization_id>
```

Article context is rebuilt from the long-lived article store and exact evidence
offsets are returned only after the source-text fingerprint is verified. Markdown
reports are rendered deterministically; they do not call an LLM or alter the
formal verdict. Localization variants reuse the saved statement and inference-gap
artifacts and never overwrite the source run.

Useful API environment variables:

```text
EVIDENCEGAP_API_RUN_STORE_ROOT
EVIDENCEGAP_API_MAX_QUEUE_SIZE
EVIDENCEGAP_API_MAX_STATEMENT_CHARS
EVIDENCEGAP_API_VALIDATE_RESOURCES
EVIDENCEGAP_CORS_ORIGINS
```

`EVIDENCEGAP_CORS_ORIGINS` is a comma-separated allowlist, for example:

```bash
export EVIDENCEGAP_CORS_ORIGINS='http://localhost:5173,http://127.0.0.1:5173'
```

## Configuration file

At startup the API looks for `<workspace_root>/config.json`. A different file can
be selected with `EVIDENCEGAP_CONFIG=/absolute/path/config.json`. Copy the
repository-level example to start:

```bash
cp config.example.json config.json
```

Configuration precedence is:

```text
explicit Python BackendConfig / ApiConfig
> environment variables
> config.json
> built-in defaults
```

The JSON file can select a different provider/model for each LLM stage:

```text
statement_decomposition
article_evidence
inference_gap
localization
agent_controller
```

Global environment variables such as `EVIDENCEGAP_MODEL` override all JSON
stage models. Stage-specific variables override the global value, for example:

```bash
export EVIDENCEGAP_MODEL=deepseek-v4-flash
export EVIDENCEGAP_DECOMPOSITION_MODEL=deepseek-v4-pro
export EVIDENCEGAP_INFERENCE_GAP_MODEL=deepseek-v4-pro
```

Prompts can be placed directly in JSON with `prompt.system` and
`prompt.additional_instructions`, but files are easier to maintain:

```json
{
  "llm": {
    "stages": {
      "statement_decomposition": {
        "prompt": {
          "system_file": "config/prompts/statement_decomposition.txt",
          "additional_instructions_file": "config/prompts/decomposition_extra.txt",
          "version": "statement-decomposition-custom-v1"
        }
      }
    }
  }
}
```

The default prompts are ordinary package files under
`backend/evidencegap_backend/prompts/`, so they can be reviewed and versioned
without searching Python modules. Prompt paths from `config.json` are resolved
relative to the configuration file. `system_file` replaces the packaged default
system prompt; `additional_instructions_file` appends to it. Environment
overrides use names such as:

```text
EVIDENCEGAP_DECOMPOSITION_PROMPT_FILE
EVIDENCEGAP_ARTICLE_EVIDENCE_PROMPT_FILE
EVIDENCEGAP_INFERENCE_GAP_PROMPT_FILE
EVIDENCEGAP_LOCALIZATION_PROMPT_FILE
```

Configuration is resolved once at process startup. Restart Uvicorn after
changing models, prompts, resource paths, runtime settings, or pipeline depths.
Every completed pipeline run stores the secret-free resolved configuration in
`resolved_config.json`. Stage manifests also record provider, model,
`prompt_version`, `prompt_sha256`, and `prompt_source`; API key values are never
persisted.
