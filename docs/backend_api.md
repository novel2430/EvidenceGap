# EvidenceGap Backend API 文件

**API 版本：** `0.5.0`  
**Presentation Bundle Schema：** `1.1.0`  
**Presentation Bundle Contract：** `phase077.presentation-bundle.v1`  
**Analysis Context Schema：** `1.0.0`  
**Run Store Schema：** `1.1.0`  
**Localization Store Schema：** `1.0.0`  
**文件基準：** Phase 8.1、Phase 8.2、Localization Hotfix、Inference Gap Combined Hotfix 均已套用並通過真實 Pipeline 驗收。

---

## 1. API 定位

EvidenceGap Backend 接收一段生醫論述，執行以下流程：

```text
Statement
→ Biomedical Claim Decomposition
→ Article Retrieval
→ Article Reranking
→ Evidence Sentence Materialization
→ Article-level Evidence Judgment
→ Deterministic Claim Aggregation
→ Evidence Graph Construction
→ Inference Gap Analysis
→ Presentation Bundle
→ Optional Localization
```

API 的正式輸出是可追溯的生醫證據分析結果，包括：

- 從原文拆出的 Claim；
- Claim 在原文中的精確位置；
- 每個 Claim 的文章級支持、反駁或資訊不足結果；
- 文章檢索與重排 provenance；
- 精確證據句與文章文字位置；
- Claim 之間的 inference step；
- Scope Gap 與 Causal Gap；
- Gap 對下游 Claim 和終端結論的影響；
- 方法學邊界與執行耗時；
- 可選的多語言 presentation variant。

### 方法學邊界

目前結果只代表：

> 對本次 Pipeline 所檢索、重排並評估的 Top Articles 進行的分析。

它不是：

- systematic review；
- meta-analysis；
- clinical recommendation；
- 最終醫學真理；
- 對所有文獻的窮盡性搜尋。

這些邊界會由後端寫入每個成功結果的 `analysis_context`，不應由前端自行改寫。

---

## 2. Runtime 架構與限制

### 2.1 單 Engine、單 Worker

FastAPI lifespan 會建立並載入一個 `EvidenceGapEngine`。所有分析與 localization job 由同一個 in-process queue 串行執行。

```text
FastAPI Process
├── EvidenceGapEngine（載入一次）
├── RunManager
└── One in-process worker thread
    ├── Analysis Job
    └── Localization Job
```

正式啟動時應使用：

```bash
--workers 1
```

多個 Uvicorn worker 會各自載入一套模型、索引與 GPU 資源，並產生彼此獨立的 queue，不是目前設計的使用方式。

### 2.2 Queue 行為

- Analysis 與 localization 共用同一個 queue。
- 同一時間只執行一個 job。
- Queue 滿時，提交端點返回 `503 Service Unavailable`。
- 目前沒有 job priority。
- 目前沒有取消、暫停、恢復或刪除 endpoint。
- 目前沒有 WebSocket 或 SSE；client 應輪詢狀態端點。

### 2.3 持久化與重啟

Run state 以原子 JSON 寫入 filesystem。

服務重啟後：

- 已成功或已失敗的 run 仍可讀取；
- `queued` 或 `running` 的 run 會被標記為 `failed`；
- `queued` 或 `running` 的 localization 也會被標記為 `failed`；
- 系統不會從中間 stage 自動續跑。

### 2.4 認證與部署

目前 API **沒有內建 authentication、authorization 或 rate limiting**。

不要直接暴露到公網。正式部署應至少在外層加入：

- Reverse proxy；
- TLS；
- 身分驗證；
- Request size limit；
- Rate limit；
- Network allowlist。

---

## 3. 啟動方式

### 3.1 安裝

```bash
python -m pip install -e './backend[test]'
```

### 3.2 最小環境

```bash
export EVIDENCEGAP_WORKSPACE_ROOT="$PWD"
export EVIDENCEGAP_CONFIG="$PWD/config.json"
export DEEPSEEK_API_KEY='...'
```

### 3.3 啟動

```bash
PYTHONPATH=backend \
CUDA_VISIBLE_DEVICES=0 \
python -m uvicorn evidencegap_backend.api.app:create_app \
  --factory \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 1 \
  --log-level info
```

### 3.4 Framework 文件端點

FastAPI 預設提供：

```text
GET /docs
GET /redoc
GET /openapi.json
```

---

## 4. API 設定

### 4.1 設定優先順序

```text
Explicit Python BackendConfig / ApiConfig
> Environment Variables
> config.json
> Built-in Defaults
```

### 4.2 API 相關設定

`config.json`：

```json
{
  "api": {
    "run_store_root": "artifacts/v1/api_runs",
    "max_queue_size": 16,
    "max_statement_chars": 20000,
    "validate_resources": true,
    "cors_origins": [
      "http://localhost:5173",
      "http://127.0.0.1:5173"
    ]
  }
}
```

對應環境變數：

| 環境變數 | 說明 | 預設值 |
|---|---|---:|
| `EVIDENCEGAP_API_RUN_STORE_ROOT` | API run filesystem store | `<workspace>/artifacts/v1/api_runs` |
| `EVIDENCEGAP_API_MAX_QUEUE_SIZE` | Analysis 與 localization 共用 queue 上限 | `16` |
| `EVIDENCEGAP_API_MAX_STATEMENT_CHARS` | Service 層 statement 長度上限 | `20000` |
| `EVIDENCEGAP_API_VALIDATE_RESOURCES` | 啟動時是否完整驗證模型與索引資源 | `true` |
| `EVIDENCEGAP_CORS_ORIGINS` | 逗號分隔的 CORS allowlist | 空 |

注意：`RunCreateRequest.statement` 的 Pydantic hard limit 是 `100000` 字元；實際請求還會受到 `max_statement_chars` 限制。預設有效上限為 `20000` 字元。

---

## 5. 通用約定

### 5.1 Base URL

以下範例假設：

```bash
BASE=http://127.0.0.1:8000
```

### 5.2 Content Type

JSON request：

```http
Content-Type: application/json
```

JSON response：

```http
Content-Type: application/json
```

Markdown export：

```http
Content-Type: text/markdown; charset=utf-8
```

大於 1000 bytes 的回應可能由 GZip middleware 壓縮。

### 5.3 ID 格式

```text
run_id          = run_<32 lowercase hex chars>
localization_id = loc_<32 lowercase hex chars>
```

例如：

```text
run_87788653525a41d3982a35aecd7f7a3c
loc_f851e0c9743b4fc1911bc8388b7ebcad
```

`claim_id`、`article_node_id`、`evidence_id` 與 `inference_step_id` 是 Pipeline contract ID，不應由 client 解析其內部生成規則。

### 5.4 時間格式

所有 API timestamp 都是 ISO 8601 / RFC 3339 UTC 時間字串，例如：

```text
2026-07-26T04:42:21.338927+00:00
```

### 5.5 Character Offset

所有 `character_start` / `character_end` 都使用 Python slicing 語意：

```text
[start, end)
```

即：

- `character_start` 包含；
- `character_end` 不包含。

驗證方式：

```python
text[character_start:character_end]
```

### 5.6 Job Status

Analysis 與 localization 共用以下狀態：

```text
queued
running
succeeded
failed
```

Job 執行失敗時，查詢端點仍然返回 HTTP `200`，實際失敗由：

```json
{
  "status": "failed",
  "error": {
    "code": "PIPELINE_FAILED",
    "message": "..."
  }
}
```

表示。

---

## 6. Endpoint 總覽

| Method | Path | 用途 |
|---|---|---|
| `GET` | `/health` | Runtime、worker 與 queue 狀態 |
| `POST` | `/api/v1/runs` | 提交新的完整分析 |
| `GET` | `/api/v1/runs` | 列出歷史 runs |
| `GET` | `/api/v1/runs/{run_id}` | 查詢 run 狀態與完整結果 |
| `GET` | `/api/v1/runs/{run_id}/articles/{article_node_id}` | 取得完整文章 canonical context 與 evidence spans |
| `GET` | `/api/v1/runs/{run_id}/exports/result.json` | 下載正式 presentation JSON |
| `GET` | `/api/v1/runs/{run_id}/exports/report.md` | 下載 deterministic Markdown 報告 |
| `POST` | `/api/v1/runs/{run_id}/localizations` | 從成功 run 建立語言版本 |
| `GET` | `/api/v1/runs/{run_id}/localizations` | 列出該 run 的語言版本 |
| `GET` | `/api/v1/runs/{run_id}/localizations/{localization_id}` | 查詢 localization 狀態與結果 |

---

# 7. Endpoint 詳細說明

## 7.1 Health

```http
GET /health
```

### Response `200`

```json
{
  "status": "ok",
  "engine_loaded": true,
  "worker_alive": true,
  "active_run_id": null,
  "queued_runs": 0,
  "load_count": 1,
  "analysis_runs": 12
}
```

### 欄位

| 欄位 | 類型 | 說明 |
|---|---|---|
| `status` | string | 固定為 `ok` |
| `engine_loaded` | boolean | Engine 是否已載入 |
| `worker_alive` | boolean | in-process worker thread 是否存活 |
| `active_run_id` | string/null | 當前 job 對應的 source run ID；執行 localization 時仍顯示 source run ID |
| `queued_runs` | integer | Queue 中等待的 job 數，包含 analysis 和 localization |
| `load_count` | integer | Engine 載入次數；正常單 process lifespan 應為 `1` |
| `analysis_runs` | integer | Engine 已執行的完整 analysis 次數，不包含 localization |

### curl

```bash
curl -fsS "$BASE/health" | jq
```

---

## 7.2 建立 Analysis Run

```http
POST /api/v1/runs
```

完整非同步分析。請求被接受後立即返回 `202`，client 需輪詢 `Location`。

### Request

```json
{
  "statement": "Vitamin D supplementation may reduce the risk of respiratory infections.",
  "language": "English"
}
```

### Request 欄位

| 欄位 | 必填 | 類型 | 約束 | 說明 |
|---|---:|---|---|---|
| `statement` | 是 | string | trim 後不可空；Pydantic 最大 100000；預設 service 最大 20000 | 待分析的生醫論述，可包含多個 Claim 與顯式推論 |
| `language` | 否 | string | trim 後不可空；最大 100 | Presentation output language；省略時使用 Backend `default_language` |

未知欄位會被拒絕，因為 request schema 使用 `extra="forbid"`。

### Response `202 Accepted`

Header：

```http
Location: /api/v1/runs/run_<id>
```

Body：

```json
{
  "run_id": "run_5957943b4b084e88b0491b7d4235c5c9",
  "status": "queued",
  "created_at": "2026-07-26T04:06:03.128540+00:00"
}
```

### curl

```bash
curl -i -X POST "$BASE/api/v1/runs" \
  -H 'Content-Type: application/json' \
  -d '{
    "statement": "Vitamin D supplementation may reduce the risk of respiratory infections.",
    "language": "English"
  }'
```

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `422` | 空白 statement、長度超限、language 空白或 request schema 不合法 |
| `503` | Job queue 已滿 |

---

## 7.3 查詢 Run

```http
GET /api/v1/runs/{run_id}
```

### Response `200`：通用結構

```json
{
  "run_id": "run_...",
  "status": "running",
  "language": "English",
  "created_at": "...",
  "started_at": "...",
  "finished_at": null,
  "progress": {
    "stage": "claim_analysis",
    "stage_index": 2,
    "total_stages": 5,
    "message": "Analyzed 1 of 3 claims",
    "completed_units": 1,
    "total_units": 3,
    "updated_at": "..."
  },
  "execution_summary": null,
  "error": null,
  "result": null
}
```

### Progress Stage

Stage ID 與位置固定如下：

| Index | `stage` | 說明 |
|---:|---|---|
| 1 | `statement_decomposition` | 拆解可驗證生醫 Claim 與顯式 inference step |
| 2 | `claim_analysis` | 對每個 Claim 執行 retrieval、reranking、evidence judgment 與 aggregation |
| 3 | `statement_bundle` | 合併 Claims、Articles、Evidence 與 argument graph |
| 4 | `inference_gap_analysis` | 判定 Scope Gap 與 Causal Gap |
| 5 | `output_generation` | 建立或本地化 presentation bundle |

只有 `claim_analysis` 通常會填入：

```json
{
  "completed_units": 1,
  "total_units": 3
}
```

其他 stage 通常為 `null`。Stage 很快時，輪詢 client 可能看不到每一個中間狀態。

### Response `200`：成功

成功後：

- `status = succeeded`
- `finished_at` 非空
- `execution_summary` 非空
- `error = null`
- `result` 為完整 Presentation Bundle

```json
{
  "run_id": "run_...",
  "status": "succeeded",
  "language": "English",
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "progress": {
    "stage": "output_generation",
    "stage_index": 5,
    "total_stages": 5,
    "message": "Preparing the presentation bundle",
    "completed_units": null,
    "total_units": null,
    "updated_at": "..."
  },
  "execution_summary": {
    "total_seconds": 38.427103,
    "stages": {
      "statement_decomposition": {"seconds": 2.516832},
      "claim_analysis": {"seconds": 14.287664},
      "statement_bundle": {"seconds": 0.081442},
      "inference_gap_analysis": {"seconds": 18.774882},
      "output_generation": {"seconds": 2.766283}
    }
  },
  "error": null,
  "result": {
    "schema_version": "1.1.0",
    "contract_id": "phase077.presentation-bundle.v1"
  }
}
```

### Response `200`：失敗

```json
{
  "run_id": "run_...",
  "status": "failed",
  "language": "English",
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "progress": {
    "stage": "inference_gap_analysis",
    "stage_index": 4,
    "total_stages": 5,
    "message": "Checking scope and causal inference gaps",
    "completed_units": null,
    "total_units": null,
    "updated_at": "..."
  },
  "execution_summary": null,
  "error": {
    "code": "PIPELINE_FAILED",
    "message": "..."
  },
  "result": null
}
```

### Run Error Codes

| Code | 說明 |
|---|---|
| `PIPELINE_FAILED` | 可預期的 Pipeline / provider / artifact validation 錯誤 |
| `INTERNAL_ERROR` | 未預期的 server exception；公開訊息不含內部 traceback |
| `SERVICE_SHUTDOWN` | 服務關閉前 job 尚在 queue 中，未開始執行 |
| `SERVICE_RESTARTED` | Process 重啟時發現 job 原本仍是 queued/running |

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `404` | `run_id` 不存在或格式不合法 |

### curl 輪詢

```bash
while true; do
  BODY=$(curl -fsS "$BASE/api/v1/runs/$RUN_ID")
  STATUS=$(jq -r '.status' <<<"$BODY")
  jq '{status, progress, error}' <<<"$BODY"
  [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]] && break
  sleep 2
done
```

---

## 7.4 Run History

```http
GET /api/v1/runs?limit=20&cursor=<run_id>
```

### Query Parameters

| 參數 | 必填 | 預設 | 約束 | 說明 |
|---|---:|---:|---|---|
| `limit` | 否 | `20` | `1..100` | 本頁最多回傳數量 |
| `cursor` | 否 | null | 必須是列表中存在的 run ID | 從該 run 的下一筆開始讀取 |

排序順序：

```text
created_at DESC, run_id DESC
```

`cursor` 是上一頁最後一個 `run_id`，不是 opaque token。

### Response `200`

```json
{
  "runs": [
    {
      "run_id": "run_...",
      "statement_preview": "GLP-1 receptor agonists reduce HbA1c...",
      "language": "English",
      "status": "succeeded",
      "created_at": "...",
      "started_at": "...",
      "finished_at": "...",
      "total_seconds": 38.427103,
      "summary": {
        "total_claims": 3,
        "evidence_states": {
          "SUPPORTED": 2,
          "REFUTED": 0,
          "CONFLICTED": 1,
          "INSUFFICIENT": 0,
          "ERROR": 0
        },
        "total_inference_steps": 1,
        "gaps": {
          "SCOPE_GAP": 1,
          "CAUSAL_GAP": 1
        },
        "articles": 30,
        "evidence": 52
      },
      "error": null
    }
  ],
  "next_cursor": "run_..."
}
```

### 行為

- `statement_preview` 會合併空白並截到最多 240 字元。
- `summary` 只在成功 run 完成時保存；queued/running/failed 可能為 `null`。
- `total_seconds` 來自成功 run 的 `execution_summary`。
- 損壞而無法讀取的個別 history record 會被跳過，不讓整個列表失效。

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `400` | cursor 格式不合法，或 cursor 不存在 |
| `422` | `limit` 不在 `1..100` |

---

## 7.5 Full Article Context

```http
GET /api/v1/runs/{run_id}/articles/{article_node_id}
```

取得某個成功 run 中某篇 Article 的完整 canonical text、section ranges 和精確 evidence spans。

`article_node_id` 可能包含 `:`，client 應進行 URL encoding。

### Response `200`

```json
{
  "article_node_id": "article:70941860ad6dd1feaf9c",
  "article_id": "pmid:12345678",
  "claim_id": "claim_...",
  "pmid": "12345678",
  "title": "Article title",
  "canonical_text": "Article title\n\nBackground...\n\nResults...",
  "source_text_fingerprint": "<sha256>",
  "fingerprint_verified": true,
  "sections": [
    {
      "sentence_type": "title",
      "section": "title",
      "section_index": 0,
      "character_start": 0,
      "character_end": 13
    },
    {
      "sentence_type": "abstract",
      "section": "results",
      "section_index": 1,
      "character_start": 15,
      "character_end": 214
    }
  ],
  "evidence_spans": [
    {
      "evidence_id": "claim_...:evidence:...",
      "claim_id": "claim_...",
      "section": "results",
      "section_index": 1,
      "sentence_index": 7,
      "character_start": 105,
      "character_end": 168,
      "text": "The intervention significantly reduced the primary outcome."
    }
  ]
}
```

### 可信度檢查

後端會：

1. 確認 `article_node_id` 屬於該 run；
2. 依 `article_id` 從 long-lived article store 重建文章；
3. 使用和原分析相同的 canonicalization；
4. 重算 `source_text_fingerprint`；
5. 比對 analysis artifact 中保存的 fingerprint；
6. 驗證每個 evidence span 都在範圍內；
7. 驗證 `canonical_text[start:end] == evidence.text`。

若文章來源已變更，或 offset 無法信任，API 不會靜默返回錯誤高亮，而是返回 `409 Conflict`。

`fingerprint_verified`：

- `true`：artifact 中有 fingerprint，且與重建文本一致；
- `false`：舊 artifact 沒有保存 fingerprint；offset 仍會逐句檢查，但缺少跨版本 fingerprint 證明。

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `404` | run 不存在 |
| `409` | run 尚未成功、article 不屬於該 run、文章來源 fingerprint 改變、offset 或 evidence text 不一致 |

### curl

```bash
ARTICLE_NODE_PATH=$(python -c \
  'import sys, urllib.parse; print(urllib.parse.quote(sys.argv[1], safe=""))' \
  "$ARTICLE_NODE_ID")

curl -fsS \
  "$BASE/api/v1/runs/$RUN_ID/articles/$ARTICLE_NODE_PATH" \
  | jq
```

---

## 7.6 JSON Export

```http
GET /api/v1/runs/{run_id}/exports/result.json
```

下載成功 run 的正式 Presentation Bundle。

### Response `200`

```http
Content-Type: application/json
Content-Disposition: attachment; filename="evidencegap-run_<id>.json"
```

Body 和：

```text
GET /api/v1/runs/{run_id}
→ .result
```

完全一致。

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `404` | run 不存在 |
| `409` | run 尚未成功或 result artifact 不可用 |

### curl

```bash
curl -fSL \
  "$BASE/api/v1/runs/$RUN_ID/exports/result.json" \
  -o "evidencegap-${RUN_ID}.json"
```

---

## 7.7 Markdown Export

```http
GET /api/v1/runs/{run_id}/exports/report.md
```

由後端 deterministic renderer 建立 Markdown 報告。

### 特性

- 不呼叫 LLM；
- 不新增文章；
- 不重新解讀 verdict；
- 不修改生醫內容；
- 使用 presentation bundle 的 display text；
- 包含 execution summary（若存在）。

### 報告章節

```text
EvidenceGap Analysis
Statement
Analysis Summary
Claims
Supporting Articles
Refuting Articles
Insufficient-Evidence Articles
Evidence Sentences
Inference Gaps
Methodological Boundary
Execution Summary
```

### Response `200`

```http
Content-Type: text/markdown; charset=utf-8
Content-Disposition: attachment; filename="evidencegap-run_<id>.md"
```

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `404` | run 不存在 |
| `409` | run 尚未成功，或 presentation bundle 不完整 |

---

## 7.8 建立 Localization Variant

```http
POST /api/v1/runs/{run_id}/localizations
```

從一個成功的 source run 產生新的 presentation language variant。

### Request

```json
{
  "language": "繁體中文（台灣）"
}
```

### Response `202 Accepted`

Header：

```http
Location: /api/v1/runs/{run_id}/localizations/{localization_id}
```

Body：

```json
{
  "localization_id": "loc_f851e0c9743b4fc1911bc8388b7ebcad",
  "source_run_id": "run_87788653525a41d3982a35aecd7f7a3c",
  "language": "繁體中文（台灣）",
  "status": "queued",
  "created_at": "..."
}
```

### Localization 實際執行內容

Localization 只重用 source run 已保存的：

```text
Statement Bundle
+
Inference Gap Bundle
→ Output Module
→ New Presentation Bundle
```

不重新執行：

- Claim decomposition；
- retrieval；
- reranking；
- sentence materialization；
- article evidence judgment；
- claim aggregation；
- inference gap analysis。

Source run 保持 immutable，不會被新語言版本覆蓋。

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `404` | source run 不存在 |
| `409` | source run 尚未成功，或 source artifact 不可用 |
| `422` | language 空白、過長或 request schema 不合法 |
| `503` | 共用 job queue 已滿 |

### 注意

- 每次 POST 都建立新的 `localization_id`；目前沒有 idempotency key。
- Localization 沒有 stage-level progress，只提供 `queued/running/succeeded/failed`。
- Localization 與 analysis 共用單 worker，因此可能需要等待前面的分析工作。

---

## 7.9 列出 Localizations

```http
GET /api/v1/runs/{run_id}/localizations
```

### Response `200`

```json
{
  "localizations": [
    {
      "localization_id": "loc_...",
      "source_run_id": "run_...",
      "language": "繁體中文（台灣）",
      "status": "succeeded",
      "created_at": "...",
      "started_at": "...",
      "finished_at": "...",
      "error": null,
      "result": null
    }
  ]
}
```

列表按：

```text
created_at DESC, localization_id DESC
```

為了避免列表 payload 過大，此 endpoint 不載入每個 localization 的完整 result，因此 `result` 為 `null`。需要結果時使用單筆查詢端點。

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `404` | source run 不存在 |

---

## 7.10 查詢 Localization

```http
GET /api/v1/runs/{run_id}/localizations/{localization_id}
```

### Response `200`：執行中

```json
{
  "localization_id": "loc_...",
  "source_run_id": "run_...",
  "language": "繁體中文（台灣）",
  "status": "running",
  "created_at": "...",
  "started_at": "...",
  "finished_at": null,
  "error": null,
  "result": null
}
```

### Response `200`：成功

```json
{
  "localization_id": "loc_...",
  "source_run_id": "run_...",
  "language": "繁體中文（台灣）",
  "status": "succeeded",
  "created_at": "...",
  "started_at": "...",
  "finished_at": "...",
  "error": null,
  "result": {
    "schema_version": "1.1.0",
    "contract_id": "phase077.presentation-bundle.v1",
    "output_language": "繁體中文（台灣）",
    "localized": true
  }
}
```

### Localization Error Codes

| Code | 說明 |
|---|---|
| `LOCALIZATION_FAILED` | Output Module、provider 格式、artifact validation 等可預期錯誤 |
| `INTERNAL_ERROR` | 未預期的 server exception |
| `SERVICE_SHUTDOWN` | 服務關閉時 localization 尚在 queue |
| `SERVICE_RESTARTED` | Process 重啟時 localization 尚為 queued/running |

### HTTP 錯誤

| Status | 條件 |
|---:|---|
| `404` | source run 或 localization 不存在 |

### 目前限制

目前沒有：

```text
/localizations/{localization_id}/exports/result.json
/localizations/{localization_id}/exports/report.md
```

Localization 成功後，client 應從查詢 response 的 `.result` 保存或匯出。

---

# 8. Presentation Bundle Contract

成功 analysis 的：

```text
GET /api/v1/runs/{run_id}
→ result
```

以及成功 localization 的：

```text
GET /api/v1/runs/{run_id}/localizations/{localization_id}
→ result
```

都使用同一個 Presentation Bundle contract。

## 8.1 Top-level

```json
{
  "schema_version": "1.1.0",
  "contract_id": "phase077.presentation-bundle.v1",
  "output_language": "English",
  "localized": false,
  "source_statement_bundle_sha256": "...",
  "source_inference_gap_analysis_sha256": "...",
  "analysis_context": {},
  "statement": {},
  "claims": [],
  "inference_steps": [],
  "articles": [],
  "evidence": [],
  "summary": {}
}
```

| 欄位 | 說明 |
|---|---|
| `schema_version` | Presentation schema version |
| `contract_id` | 固定 contract identity |
| `output_language` | Display text 的目標語言 |
| `localized` | 目標語言是否需要翻譯；English aliases 通常為 `false` |
| `source_statement_bundle_sha256` | Source statement bundle checksum |
| `source_inference_gap_analysis_sha256` | Source gap bundle checksum |
| `analysis_context` | 本次實際方法與能力邊界 |
| `statement` | 原始 statement 與 display text |
| `claims` | Claim-level 結果 |
| `inference_steps` | Claim 間推論與 gaps |
| `articles` | 每個 Claim 的 Top Articles |
| `evidence` | 選中的 evidence sentences |
| `summary` | 可由其他欄位 deterministic 重算的統計摘要 |

---

## 8.2 `analysis_context`

```json
{
  "schema_version": "1.0.0",
  "scope": "retrieved_top_articles",
  "is_systematic_review": false,
  "is_clinical_recommendation": false,
  "is_final_medical_truth": false,
  "aggregation_method": "deterministic_article_count",
  "uses_confidence_weighting": false,
  "retrieval_methods": ["BM25", "MedCPT", "BMRetriever"],
  "fusion_method": "reciprocal_rank_fusion",
  "reranker": "MedCPT Cross-Encoder",
  "source_depth": 100,
  "dense_nprobe": 1024,
  "rrf_k": 60,
  "rerank_depth": 100,
  "article_top_k": 10,
  "max_evidence_sentences_per_article": 5
}
```

重要語意：

- `aggregation_method = deterministic_article_count`：Claim verdict 依文章 stance 是否存在來決定；
- `uses_confidence_weighting = false`：Article confidence 不參與 Claim verdict 加權；
- `article_top_k`：每個 Claim 最終評估的文章上限；
- 這些欄位來自該 run 的實際 `PipelineConfig`。

---

## 8.3 `statement`

```json
{
  "statement_id": "statement_...",
  "original_text": "...",
  "source_language": "English",
  "analysis_status": "completed",
  "display_text": "..."
}
```

### `analysis_status`

```text
completed
partial_failure
failed
```

- `completed`：所有 Claim 完成；或 statement 沒有可分析 Claim；
- `partial_failure`：部分 Claim 完成、部分失敗；
- `failed`：有 Claims，但全部 Claim 失敗。

`display_text` 是前端應顯示的版本；英文輸出時通常等於 `original_text`，localization 時為翻譯文字。

---

## 8.4 `claims[]`

```json
{
  "claim_id": "claim_...",
  "source_text": "原文中的連續片段",
  "source_spans": [
    {
      "character_start": 0,
      "character_end": 72
    }
  ],
  "canonical_claim_en": "Canonical English biomedical claim.",
  "analysis_status": "completed",
  "verdict": "supported",
  "article_counts": {
    "total": 10,
    "support": 7,
    "refute": 0,
    "insufficient": 3
  },
  "rationale": "The retrieved direct evidence supports the claim...",
  "scope": "retrieved_top_articles",
  "boundary": {
    "is_pipeline_final_verdict": true,
    "is_final_medical_truth": false,
    "description": "The verdict summarizes the retrieved Top Articles..."
  },
  "article_node_ids": ["article:..."],
  "error": null,
  "evidence_state": "SUPPORTED",
  "argument_role": "PREMISE",
  "premise_inference_step_ids": ["inference_..."],
  "conclusion_inference_step_ids": [],
  "display_text": "...",
  "display_rationale": "..."
}
```

### Claim Verdict

Claim `verdict` 只在 `analysis_status=completed` 時存在：

| `verdict` | 規則 |
|---|---|
| `supported` | 至少一篇 support，沒有 refute |
| `refuted` | 至少一篇 refute，沒有 support |
| `mixed` | 同時存在 support 與 refute |
| `insufficient` | 沒有 support，也沒有 refute |

### Presentation Evidence State

| `evidence_state` | 來源 |
|---|---|
| `SUPPORTED` | `verdict=supported` |
| `REFUTED` | `verdict=refuted` |
| `CONFLICTED` | `verdict=mixed` |
| `INSUFFICIENT` | `verdict=insufficient` |
| `ERROR` | Claim analysis failed |

### Argument Role

| `argument_role` | 說明 |
|---|---|
| `PREMISE` | 是某個 inference step 的 premise，但不是其他 step 的 conclusion |
| `INTERMEDIATE` | 同時是某個 step 的 conclusion 與另一個 step 的 premise |
| `CONCLUSION` | 是 step conclusion，但不再作為下游 premise |
| `STANDALONE` | 不在任何顯式 inference step 中 |

### Source Spans

`source_spans` 是後端 deterministic exact matching 的結果，而非 LLM 猜測 offset。同一片段在原文重複出現時可能有多個 span。

### Claim Error

Claim-level `error` 是 string 或 `null`：

- completed Claim：`error = null`；
- failed Claim：`verdict/article_counts/rationale/scope/boundary` 等通常為 `null`，`evidence_state = ERROR`，`error` 保存失敗原因。

---

## 8.5 `inference_steps[]`

```json
{
  "inference_step_id": "inference_...",
  "premise_claim_ids": ["claim_a", "claim_b"],
  "conclusion_claim_id": "claim_c",
  "impact": {
    "direct_conclusion_claim_id": "claim_c",
    "downstream_claim_ids": ["claim_c", "claim_d"],
    "downstream_inference_step_ids": ["inference_next"],
    "terminal_claim_ids": ["claim_d"],
    "affects_terminal_conclusion": true,
    "cycle_detected": false
  },
  "gaps": [
    {
      "gap_type": "SCOPE_GAP",
      "detection_method": "llm",
      "reason_en": "The conclusion broadens the population...",
      "display_reason": "..."
    }
  ]
}
```

### Gap Type

```text
SCOPE_GAP
CAUSAL_GAP
```

沒有偵測到 gap 時，`gaps` 是空陣列。

### Gap Detection 與 Impact 的責任分工

- `gaps`：由 inference-gap LLM 判定；
- `impact`：由後端 deterministic graph traversal 計算。

### `impact`

| 欄位 | 說明 |
|---|---|
| `direct_conclusion_claim_id` | 此 inference step 的直接 conclusion |
| `downstream_claim_ids` | 從直接 conclusion 可到達的 Claims，包含直接 conclusion 本身 |
| `downstream_inference_step_ids` | 下游可到達的 inference steps |
| `terminal_claim_ids` | 沒有再向下推論的可到達 Claims |
| `affects_terminal_conclusion` | 是否可到達至少一個 terminal Claim |
| `cycle_detected` | Argument graph traversal 是否發現 cycle |

---

## 8.6 `articles[]`

```json
{
  "article_node_id": "article:...",
  "claim_id": "claim_...",
  "article_id": "pmid:12345678",
  "pmid": "12345678",
  "rank": 1,
  "retrieval_trace": {
    "bm25": {"rank": 12, "score": 8.31},
    "medcpt": {"rank": 3, "score": 0.84},
    "bmretriever": {"rank": null, "score": null},
    "fusion": {"rank": 4, "rrf_score": 0.047},
    "cross_encoder": {"score": 0.913},
    "final_article_rank": 1
  },
  "title": "Article title",
  "rationale": "The article reports direct evidence...",
  "stance": "support",
  "confidence": 0.91,
  "probabilities": {
    "support": 0.91,
    "refute": 0.03,
    "insufficient": 0.06
  },
  "evidence_ids": ["claim_...:evidence:..."],
  "provider": "deepseek",
  "model": "deepseek-v4-flash",
  "model_fingerprint": "...",
  "prompt_version": "...",
  "display_title": "...",
  "display_rationale": "..."
}
```

### Article Stance

```text
support
refute
insufficient
```

### Confidence 的正確語意

`confidence` 是該 Article 的 predicted stance probability：

```text
confidence == probabilities[stance]
```

它不是：

- Claim 為真的機率；
- 醫學真值 confidence；
- 多篇文章合併後的概率；
- 校準過的 clinical decision confidence。

### Retrieval Trace

- BM25、MedCPT、BMRetriever 可能個別沒有召回該文章，因此該方法的 `rank` 和 `score` 會同時為 `null`；
- `fusion.rank`、`fusion.rrf_score`、`cross_encoder.score` 和 `final_article_rank` 對 Top Article 必須存在；
- `rank` 是 Claim 內排序，不是跨所有 Claims 的全域排名。

---

## 8.7 `evidence[]`

```json
{
  "evidence_id": "claim_...:evidence:...",
  "source_node_id": "evidence:...",
  "claim_id": "claim_...",
  "article_node_id": "article:...",
  "article_id": "pmid:12345678",
  "pmid": "12345678",
  "label": "results:7",
  "text": "Evidence sentence text.",
  "source_evidence_id": "...",
  "sentence_id": "...",
  "sentence_index": 7,
  "sentence_index_within_section": 2,
  "section": "results",
  "section_index": 1,
  "character_start": 153,
  "character_end": 219,
  "source_text_fingerprint": "<sha256>",
  "splitter_fingerprint": "...",
  "display_text": "..."
}
```

### 語意

- `text`：正式分析使用的英文 evidence sentence；
- `display_text`：前端顯示文字，localization 時可被翻譯；
- `character_start/end`：相對於 Article Context API 返回的 `canonical_text`；
- `sentence_index`：文章 canonical sentence 的全域 index；
- `sentence_index_within_section`：section 內 index；
- `source_text_fingerprint`：canonical article text checksum；
- `splitter_fingerprint`：sentence splitting configuration / implementation identity。

`insufficient` Article 不會有 selected evidence，因此也不會產生對應 evidence rows。

---

## 8.8 `summary`

```json
{
  "total_claims": 3,
  "evidence_states": {
    "SUPPORTED": 2,
    "REFUTED": 0,
    "CONFLICTED": 1,
    "INSUFFICIENT": 0,
    "ERROR": 0
  },
  "argument_roles": {
    "PREMISE": 2,
    "INTERMEDIATE": 0,
    "CONCLUSION": 1,
    "STANDALONE": 0
  },
  "total_inference_steps": 1,
  "gaps": {
    "SCOPE_GAP": 1,
    "CAUSAL_GAP": 1
  },
  "articles": 30,
  "evidence": 52
}
```

`summary` 是由 bundle 內容 deterministic 重算並驗證的快取統計，不是額外的模型判斷。

`articles` 是所有 Claims 的 Article records 總數。例如 3 Claims、每個 Top-10，通常是 30；同一 PMID 若被不同 Claim 命中，會存在多個 Claim-specific Article node。

---

# 9. HTTP Error 格式

## 9.1 FastAPI / Route Error

一般 HTTP 錯誤：

```json
{
  "detail": "run not found"
}
```

Pydantic `422` 通常是：

```json
{
  "detail": [
    {
      "type": "string_too_short",
      "loc": ["body", "statement"],
      "msg": "String should have at least 1 character",
      "input": ""
    }
  ]
}
```

## 9.2 Job Error

已接受的 async job 若執行失敗，HTTP 查詢仍是 `200`：

```json
{
  "status": "failed",
  "error": {
    "code": "PIPELINE_FAILED",
    "message": "DeepSeek output was truncated..."
  }
}
```

Client 必須同時檢查：

```text
HTTP status
+
job.status
```

---

# 10. Filesystem Persistence Layout

預設 API store：

```text
artifacts/v1/api_runs/
└── run_<id>/
    ├── request.json
    ├── status.json
    ├── result.json                     # succeeded 後存在
    ├── localizations/
    │   └── loc_<id>/
    │       ├── request.json
    │       ├── status.json
    │       └── result.json             # succeeded 後存在
    └── localization_artifacts/
        └── loc_<id>/
            ├── presentation_bundle.json
            ├── request.json
            └── run_manifest.json
```

完整 Pipeline artifacts 仍保存於 Backend `artifact_root`，例如：

```text
artifacts/v1/pipeline/statement_run/run_<id>/
```

API store 的 `status.json` 會保存內部 artifact 路徑，以支援 Article Context 與 Localization Variant。

不要讓前端直接讀 filesystem artifact。公開 client 應只使用 API。

---

# 11. 完整 Client Flow

```text
1. GET /health
2. POST /api/v1/runs
3. 讀取 Location 或 run_id
4. 每 1–2 秒 GET /api/v1/runs/{run_id}
5. status=succeeded 後使用 result
6. 依 article_node_id 載入 Full Article Context
7. 視需要下載 JSON / Markdown
8. 視需要 POST localization
9. 輪詢 localization status
```

### Bash 範例

```bash
BASE=http://127.0.0.1:8000

CREATE=$(curl -fsS -X POST "$BASE/api/v1/runs" \
  -H 'Content-Type: application/json' \
  -d '{
    "statement": "Vitamin D supplementation may reduce the risk of respiratory infections.",
    "language": "English"
  }')

RUN_ID=$(jq -r '.run_id' <<<"$CREATE")

while true; do
  BODY=$(curl -fsS "$BASE/api/v1/runs/$RUN_ID")
  STATUS=$(jq -r '.status' <<<"$BODY")
  jq '{status, progress, error}' <<<"$BODY"
  [[ "$STATUS" == "succeeded" || "$STATUS" == "failed" ]] && break
  sleep 2
done

if [[ "$STATUS" == "succeeded" ]]; then
  curl -fSL "$BASE/api/v1/runs/$RUN_ID/exports/result.json" \
    -o "evidencegap-${RUN_ID}.json"

  curl -fSL "$BASE/api/v1/runs/$RUN_ID/exports/report.md" \
    -o "evidencegap-${RUN_ID}.md"
fi
```

---

# 12. Client 實作建議

## 12.1 不要把 Confidence 顯示成總體可信度

可以顯示：

```text
Article stance confidence: 0.91
```

不要顯示：

```text
This medical claim is 91% true.
```

## 12.2 Evidence Balance 使用 Count，不使用假概率

應使用：

```text
Support 6 / Refute 2 / Insufficient 2
```

不要將其換算成「60% 真實」。

## 12.3 先顯示 Claim / Inference Graph，再按需載入 Article Context

主 graph 建議只放：

```text
Claim ↔ Inference Step ↔ Claim
```

文章與證據透過右側 inspector 及 Article Context endpoint 按需載入，避免一次把大量 article/evidence node 放進主圖。

## 12.4 使用後端提供的 Methodological Boundary

前端應直接讀：

```text
result.analysis_context
claim.boundary
```

不要自行宣稱 systematic review、clinical recommendation 或 final medical truth。

## 12.5 保留三種文字

Claim 可同時展示：

```text
source_text
canonical_claim_en
display_text
```

其中：

- `source_text`：原文 exact quote；
- `canonical_claim_en`：實際進入生醫檢索與分析的 canonical English Claim；
- `display_text`：使用者所選語言的展示文字。

---

# 13. 目前未提供的 API

目前沒有：

- Run cancellation；
- Run deletion；
- Run retry / resume；
- WebSocket / SSE progress；
- API authentication；
- Rate limiting；
- Idempotency key；
- Article search endpoint；
- PMID 任意查詢 endpoint；
- Localization export endpoint；
- Per-article partial retry；
- User / project / tenant 隔離；
- Systematic review protocol；
- Study quality / risk-of-bias assessment；
- PICO、sample size、dose、follow-up 等結構化 study metadata API。

---

# 14. 版本與相容性

| 項目 | 目前版本 |
|---|---|
| FastAPI app | `0.5.0` |
| Presentation Bundle schema | `1.1.0` |
| Presentation Bundle contract | `phase077.presentation-bundle.v1` |
| Analysis Context schema | `1.0.0` |
| Run status store | `1.1.0` |
| Localization status store | `1.0.0` |

Client 應至少檢查：

```json
{
  "schema_version": "1.1.0",
  "contract_id": "phase077.presentation-bundle.v1"
}
```

目前 contract ID 保持不變，但 schema 可能隨 additive enrichment 提升。Client 不應只依賴欄位出現順序，應依欄位名稱解析。

---

# 15. 驗收狀態

Phase 8.1 / 8.2 已使用真實後端完成以下驗收：

- 3 Claims；
- 30 Article records；
- 1 Inference Step；
- Claim source spans；
- Retrieval trace；
- Gap impact；
- Analysis context；
- Execution summary；
- Run history；
- Full article context；
- Fingerprint verification；
- Evidence offset verification；
- JSON export；
- Markdown export；
- 繁體中文 localization；
- Source run immutability；
- Negative API paths。


