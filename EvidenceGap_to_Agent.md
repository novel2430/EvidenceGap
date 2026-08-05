# EvidenceGap Agent 化改造完整实作方案

> 文档定位：供本项目后续实作与 Coding Agent 直接使用  
> 改造分支目标：快速完成可运行、可展示、可合理称为 Agent / Agent Harness / LangGraph 架构的版本  
> 核心约束：保持现有前端请求 API 与最终展示数据契约不变，不重写已经验证过的检索、文章判断、聚合与 Gap Analysis 能力

---

## 1. 改造目标

本次改造不是重新设计 EvidenceGap 的医学证据算法，也不是为了生产级稳定性进行大规模平台工程。

真正目标是：

1. 保留现有 EvidenceGap 已经完成的确定性证据流水线；
2. 在 Claim 分析阶段加入一个能够根据当前证据状态动态决策的 Evidence Controller；
3. 使用 LangGraph 管理显式状态、条件路由、循环与 checkpoint；
4. 将现有检索与证据判断能力包装为受约束的高层 Tool；
5. 产生清楚可读的 Action Trace、Evidence Workspace 与执行图；
6. 保持现有 FastAPI endpoint、请求格式和主要响应格式不变；
7. 让项目可以准确描述为：

> EvidenceGap 是一个基于 LangGraph 的受约束 Evidence Agent Harness。Agent Controller 围绕显式 Evidence Workspace 动态选择待处理 Claim、生成检索 Query、调用证据搜索 Tool，并依据工具结果决定继续搜索、接受当前证据、承认证据不足或结束任务。底层检索、文章级 Judge、Claim 聚合与 Gap Analysis 继续由可审计的确定性模块执行。

本次改造追求的是：

```text
快速
＋ 能运行
＋ 能看到动态行动
＋ 能展示执行轨迹
＋ 技术叙事成立
```

而不是：

```text
高并发
＋ 分布式
＋ 完整长期记忆
＋ 任意 Tool 自主调用
＋ 生产级容错
```

---

## 2. 当前项目结构判断

当前 backend 已经非常适合进行“调度层 Agent 化”。

现有关键调用链为：

```text
FastAPI
→ RunManager
→ EvidenceGapEngine.analyze_statement()
→ run_statement_pipeline()
→ Statement Decomposition
→ run_statement_analysis()
→ 对每个 Claim 调用 run_analysis()
→ Statement Bundle
→ Inference Gap Analysis
→ Output Module
```

主要文件映射如下：

| 当前职责 | 当前文件 |
|---|---|
| FastAPI endpoints | `backend/evidencegap_backend/api/app.py` |
| 后台队列与任务执行 | `backend/evidencegap_backend/api/run_manager.py` |
| API run 状态与结果持久化 | `backend/evidencegap_backend/api/run_store.py` |
| 后端统一入口 | `backend/evidencegap_backend/engine.py` |
| 当前固定端到端调度 | `backend/evidencegap_backend/pipeline/statement_run.py` |
| Statement 拆解 | `backend/evidencegap_backend/pipeline/statement_decomposition.py` |
| 多 Claim 固定循环 | `backend/evidencegap_backend/pipeline/statement_analysis.py` |
| 单 Claim 完整分析 | `backend/evidencegap_backend/pipeline/analysis.py` |
| 三路检索与 RRF / Cross-Encoder | `backend/evidencegap_backend/pipeline/retrieval_adapters.py` |
| Article Evidence Judge | `backend/evidencegap_backend/pipeline/article_evidence.py` |
| Claim 聚合 | `backend/evidencegap_backend/pipeline/claim_aggregation.py` |
| Evidence Graph | `backend/evidencegap_backend/pipeline/final_graph.py` |
| Statement Bundle | `backend/evidencegap_backend/pipeline/statement_bundle.py` |
| 推理缺口分析 | `backend/evidencegap_backend/pipeline/inference_gap_analysis.py` |
| 最终展示与语言转换 | `backend/evidencegap_backend/output/presentation.py` |
| LLM structured transport | `backend/evidencegap_backend/stance/llm_judge.py` |
| 常驻模型、索引与资源 | `backend/evidencegap_backend/resources.py` |

当前架构的重要优势是：

- FastAPI 只是薄封装；
- `EvidenceGapEngine` 是稳定统一入口；
- `run_analysis()` 已经接近一个高层“证据分析 Tool”；
- GPU 模型和索引已经由 `RuntimeResources` 常驻复用；
- 每个阶段已经有明确 Artifact、manifest、hash 与 validator；
- 前端只依赖现有 API 和 presentation bundle，不关心内部如何调度。

因此，本次不需要重写 backend，只需要替换内部 orchestration。

---

## 3. 总体改造边界

### 3.1 必须保持不变

以下外部行为应保持不变：

```text
POST /api/v1/runs
GET  /api/v1/runs
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/articles/{article_node_id}
GET  /api/v1/runs/{run_id}/exports/result.json
GET  /api/v1/runs/{run_id}/exports/report.md
POST /api/v1/runs/{run_id}/localizations
```

保持：

- `EvidenceGapEngine.analyze_statement()` 方法签名；
- `StatementAnalysisResult` 返回结构；
- `RunManager` 调用方式；
- `RunStatusResponse`；
- presentation bundle 的现有核心字段；
- Article Context 精确 offset 行为；
- 前端请求代码。

可以向 `execution_summary` 或 Artifact 中新增 Agent 信息，因为现有 API 将其视为开放字典；但不要删除或改名现有字段。

### 3.2 继续作为确定性模块保留

以下能力不交给 Controller 自由决定内部执行方式：

- BM25；
- MedCPT；
- BMRetriever；
- RRF Fusion；
- Cross-Encoder reranking；
- Sentence materialization；
- Article Evidence Judge；
- Claim Aggregation；
- Final Graph 构建；
- Statement Bundle；
- Inference Gap Analysis；
- Localization；
- Schema validation；
- Artifact hash 与 manifest。

Agent 只决定“下一步做什么”，不重新实现“每一步内部怎么算”。

### 3.3 第一版明确不做

为了快速完成，MVP 不做：

- 通用 ReAct Agent；
- 将所有函数注册成 Tool；
- 多 Agent 协作；
- 长期跨任务 Memory；
- 向量化 Agent Memory；
- 自动修改 Prompt 或模型；
- 动态修改检索器参数；
- Redis、Celery、PostgreSQL；
- 前端 API 重构；
- Human-in-the-loop 页面；
- Claim SPLIT；
- 多轮检索结果的复杂证据级融合。

其中 `SPLIT` 可作为后续扩展，但不进入第一版主路径。

---

## 4. 目标架构

```text
现有 FastAPI
    ↓
RunManager
    ↓
EvidenceGapEngine
    ↓
EvidenceGap Agent Harness
    ├─ LangGraph Runtime
    ├─ Evidence Controller
    ├─ Evidence Workspace
    ├─ Action Validator
    ├─ Search Tool Adapter
    ├─ Budget / Stop Policy
    ├─ SQLite Checkpointer
    └─ Action Trace
          ↓
现有确定性 Evidence Pipeline
    ├─ Statement Decomposition
    ├─ Retrieval / Reranking
    ├─ Article Judge
    ├─ Claim Aggregation
    ├─ Final Graph
    ├─ Statement Bundle
    ├─ Gap Analysis
    └─ Output Module
```

### 4.1 为什么这可以称为 Agent

本版本具备 Agent 所需的核心结构：

1. 有一个由 LLM 驱动的 Controller；
2. 有显式、持续更新的环境状态；
3. 有受约束的 Action Space；
4. 有 Tool 调用；
5. Tool 结果会反馈到 Workspace；
6. Controller 会根据新状态再次决策；
7. 执行不是预先写死的一次性直线；
8. 有预算、重复检查与终止规则；
9. 不同输入可以产生不同的行动序列。

固定 Workflow 是：

```text
Claim 1 → 固定搜索 → 固定判断
Claim 2 → 固定搜索 → 固定判断
Claim 3 → 固定搜索 → 固定判断
```

Agent Harness 是：

```text
查看整个 Workspace
→ 选择当前最值得处理的 Claim
→ 生成 Query
→ 调用 Tool
→ 查看结果
→ 决定继续搜索 / 接受结果 / 证据不足
→ 再次查看整个 Workspace
→ 直到 FINISH
```

### 4.2 为什么这可以称为 Harness

Harness 不只是 LangGraph 本身，而是围绕 Controller 的完整运行外围：

```text
EvidenceGap Agent Harness
├─ Controller Prompt
├─ Structured Action Schema
├─ Evidence Workspace
├─ Tool Boundary
├─ Runtime Context
├─ Action Legality Validation
├─ Query De-duplication
├─ Step Budget
├─ Search Budget
├─ Per-claim Attempt Limit
├─ Stop Policy
├─ Checkpoint
├─ Trace
├─ Artifact Validation
└─ Deterministic Evidence Modules
```

LangGraph 是 Harness 的 orchestration runtime，不是 EvidenceGap 算法的主体。

---

## 5. 第一版 Action Space

第一版使用四个 Action：

```text
SEARCH
RESOLVE
ABSTAIN
FINISH
```

### 5.1 SEARCH

含义：

> 选择一个尚未终止的 Claim，使用新 Query 调用 Evidence Search Tool。

示例：

```json
{
  "action": "SEARCH",
  "claim_id": "claim_xxx",
  "query": "GLP-1 receptor agonist diabetic microvascular complications randomized trial",
  "reason": "Existing evidence only covers HbA1c reduction and does not directly cover complication prevention."
}
```

约束：

- Claim 必须存在；
- Claim 不能已终止；
- Query 不能为空；
- Query 不能与该 Claim 之前的 normalized query 重复；
- 全局 search budget 必须大于 0；
- 该 Claim 的 attempts 不可超过限制。

### 5.2 RESOLVE

含义：

> 接受当前最佳搜索 attempt 的确定性 Pipeline 结果，不再为该 Claim 搜索。

Controller 不得自行写入 `supported`、`refuted`、`mixed` 或 `insufficient`。

最终 verdict 必须来自当前选择的 `run_analysis()` 产物。

示例：

```json
{
  "action": "RESOLVE",
  "claim_id": "claim_xxx",
  "query": null,
  "reason": "The current attempt contains direct article-level evidence and additional search is unlikely to change the claim-level evidence state."
}
```

约束：

- 至少存在一个成功 attempt；
- 由普通程序选择或验证 `best_attempt_id`；
- Controller 只决定停止搜索，不得改 verdict。

### 5.3 ABSTAIN

含义：

> 当前证据仍有限，但继续搜索的边际价值过低、预算不足或多轮 Query 没有获得新的直接证据，因此停止。

ABSTAIN 不是运行失败。

第一版建议采用以下映射：

- 若已有成功 attempt：选择当前最佳 attempt，Claim 仍作为 `completed` 写入现有 Statement Analysis Contract；其 verdict 仍来自该 attempt；
- Agent terminal reason 额外记录为 `abstained`；
- 若没有任何成功 attempt：才写成现有 contract 中的 `failed`。

这样无需修改下游 `statement_bundle` 对 `completed / failed` 的现有契约。

### 5.4 FINISH

含义：

> 所有 Claim 均进入 terminal 状态，可以构建 Statement Bundle、执行 Gap Analysis 和 Output Module。

合法条件：

```text
所有 Claim ∈ {resolved, abstained, failed}
```

若仍存在 unresolved Claim，FINISH 必须被 validator 拒绝。

---

## 6. 第一版不实现 SPLIT 的原因

`SPLIT` 会修改 Argument Graph，并影响：

- Claim ID；
- source text 与 source spans；
- inference step；
- statement_result；
- statement_bundle；
- graph validation；
- 原始论述与新 Claim 的可追溯关系。

第一版加入 SPLIT 会把“调度层 Agent 化”扩大成“领域数据契约重构”。

因此建议：

```text
Agent MVP：SEARCH / RESOLVE / ABSTAIN / FINISH
后续扩展：SPLIT + Graph Patch
```

不实现 SPLIT 不影响 Agent 架构成立，因为动态 Claim 选择、Query 生成、Tool 调用、循环与终止已经存在。

---

## 7. Evidence Workspace 设计

LangGraph State 的中心不是聊天消息，而是显式 Evidence Workspace。

### 7.1 状态设计原则

Workspace 必须：

- 可 JSON 序列化；
- 可写入 SQLite checkpoint；
- 不包含 GPU 模型；
- 不包含 FAISS index；
- 不包含 BM25 backend；
- 不包含 callback；
- 不包含打开的数据库连接；
- 不依赖完整聊天消息历史。

`RuntimeResources`、BackendConfig、progress callback 等非序列化对象放在 Harness Runtime Context 中，通过 node closure 或 runner object 使用，不能写进 State。

### 7.2 建议 State

```python
class EvidenceWorkspace(TypedDict):
    schema_version: str
    run_name: str
    statement: str
    language: str
    statement_id: str | None

    decomposition: dict[str, Any] | None
    claims: dict[str, dict[str, Any]]
    claim_order: list[str]

    active_claim_id: str | None
    pending_decision: dict[str, Any] | None
    last_action_result: dict[str, Any] | None

    step_count: int
    max_steps: int
    remaining_search_budget: int

    terminal_claim_count: int
    status: str
    finish_reason: str | None

    action_trace: list[dict[str, Any]]
    final_outputs: dict[str, Any]
```

### 7.3 单 Claim Workspace

```python
class ClaimWorkspace(TypedDict):
    claim_id: str
    source_text: str
    source_spans: list[dict[str, int]]
    canonical_claim_en: str

    status: str
    # unresolved | resolved | abstained | failed

    search_history: list[str]
    normalized_queries: list[str]
    seen_article_ids: list[str]

    attempts: list[dict[str, Any]]
    best_attempt_id: str | None

    remaining_problem: str | None
    terminal_reason: str | None
```

### 7.4 Attempt Record

```python
class SearchAttempt(TypedDict):
    attempt_id: str
    claim_id: str
    query: str
    normalized_query: str
    artifact_dir: str
    graph_bundle_path: str

    verdict: str
    article_counts: dict[str, int]
    article_ids: list[str]
    new_article_ids: list[str]
    direct_evidence_articles: int

    utility_score: int
    status: str
    error: str | None
```

### 7.5 为什么不使用 MessagesState

EvidenceGap 不是聊天机器人。

如果仅保存 messages，会产生：

- Controller 依赖长文本历史；
- 已搜 Query 难以可靠去重；
- Claim 状态难以验证；
- checkpoint 恢复后难以确定当前事实状态；
- 模型可能把旧 reasoning 当成事实；
- Artifact 与 Agent State 难以对应。

因此消息可以作为调试信息，但不能成为系统真实状态来源。

---

## 8. Controller Schema

建议新增严格 Pydantic Schema：

```python
from enum import StrEnum
from pydantic import BaseModel, ConfigDict, Field


class AgentAction(StrEnum):
    SEARCH = "SEARCH"
    RESOLVE = "RESOLVE"
    ABSTAIN = "ABSTAIN"
    FINISH = "FINISH"


class AgentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: AgentAction
    claim_id: str | None = None
    query: str | None = None
    reason: str = Field(min_length=1, max_length=2000)
```

Controller 输出必须使用现有 `call_structured_llm()`，不必引入新的 LangChain model provider。

### 8.1 Controller 输入

Controller 每次只接收整理过的摘要：

- 原始 Statement；
- Claim 与 inference relationships；
- 每个 Claim 的状态；
- 已使用 Query；
- 每个 attempt 的 verdict 与 article count；
- 新增 article 数；
- 当前 remaining problem；
- 剩余 step/search budget；
- 上一个 Action 的结果。

不要将所有文章全文、所有 prompt、完整 action log 再次塞给 Controller。

### 8.2 Controller 不得做的事

Prompt 中明确禁止：

- 自己产生医学结论；
- 改写 Pipeline verdict；
- 假装已经调用 Tool；
- 返回不存在的 Claim ID；
- 重复 Query；
- 直接修改 Workspace；
- 修改检索算法参数；
- 输出 Action Schema 之外的字段。

### 8.3 Controller Prompt 文件

新增：

```text
backend/evidencegap_backend/prompts/agent_controller.txt
```

建议 Prompt 结构：

```text
Role
→ Evidence Controller，不是医学回答者

Goal
→ 在预算内改善各 Claim 的可核验证据覆盖

Available actions
→ SEARCH / RESOLVE / ABSTAIN / FINISH

State summary
→ 由程序注入

Rules
→ verdict 来自 Pipeline；不可重复搜索；不可越权

Output contract
→ 严格 JSON
```

---

## 9. Search Tool 设计

### 9.1 Tool 的对外语义

Agent 只看到一个高层 Tool：

```python
def search_evidence(
    *,
    claim_id: str,
    canonical_claim: str,
    query: str,
    attempt_id: str,
) -> SearchAttempt:
    ...
```

Tool 内部固定执行：

```text
Query
→ BM25
＋ MedCPT
＋ BMRetriever
→ RRF
→ Cross-Encoder
→ Top Articles
→ Sentence Materialization
→ Article Evidence Judge（针对 canonical claim）
→ Claim Aggregation
→ Final Graph
```

### 9.2 最关键的底层接口改造

当前 `run_analysis()` 的 `claim` 同时承担：

1. Claim identity；
2. Retrieval query；
3. Cross-Encoder query；
4. Judge hypothesis。

Agent 化后必须拆开：

```python
def run_analysis(
    root: Path,
    *,
    claim: str,
    retrieval_query: str | None = None,
    ...
) -> dict[str, Any]:
```

内部：

```python
claim_text = _clean_claim(claim)
query_text = _clean_claim(retrieval_query or claim_text)
claim_id = runtime_claim_id(claim_text)
```

其中：

- `claim_text` 始终是 canonical claim；
- `claim_id` 始终由 canonical claim 产生；
- BM25 / Dense / Cross-Encoder 使用 `query_text`；
- Article Judge 继续针对 `claim_text`；
- Claim Aggregation 与 Final Graph 继续针对 `claim_text`。

这可以避免 Agent 改写 Query 后改变 Claim identity 或判断目标。

### 9.3 Retrieval Adapter 改造

建议将：

```python
retrieve_runtime_articles(
    claim_id=claim_id,
    claim_text=claim_text,
)
```

改成：

```python
retrieve_runtime_articles(
    claim_id=claim_id,
    claim_text=claim_text,
    query_text=query_text,
)
```

内部使用：

```text
BM25 search(query_text)
Dense encode(query_text)
Cross-Encoder(query_text, article)
```

Artifact 同时记录：

```json
{
  "claim_id": "...",
  "claim_text": "canonical claim",
  "query_text": "agent generated retrieval query",
  "query_text_sha256": "..."
}
```

不可用 `query_text` 替换现有 `claim_text` 字段，否则下游 identity validator 会混乱。

### 9.4 多轮 Attempt Artifact

建议目录：

```text
<run_artifact>/statement_analysis/
├─ request.json
├─ statement_result.json
├─ run_manifest.json
└─ attempts/
   ├─ <claim_id>/
   │  ├─ search_001/
   │  ├─ search_002/
   │  └─ search_003/
   └─ ...
```

每个 `search_XXX` 都是一个完整合法的 `run_analysis()` Artifact。

Statement Analysis 最终结果中的：

```text
phase07_artifact_dir
graph_bundle_path
verdict
```

指向选中的最佳 attempt。

现有 `validate_statement_analysis_artifact()` 并不要求 Phase07 artifact 必须位于固定的 `claims/<claim_id>` 目录，只要路径存在、validator 通过且 claim identity 一致即可。因此可以保持现有下游契约。

---

## 10. Best Attempt 选择

第一版不进行复杂跨 attempt 证据融合。

每次 SEARCH 形成一个独立完整的 deterministic analysis attempt。Workspace 决定哪个 attempt 成为该 Claim 的最终结果。

建议使用确定性 utility：

```python
direct = support_articles + refute_articles
new_articles = len(new_article_ids)
non_insufficient_bonus = 1 if verdict != "insufficient" else 0

utility_score = (
    direct * 100
    + non_insufficient_bonus * 20
    + new_articles
)
```

排序：

```text
utility_score 高者优先
→ direct evidence articles 高者优先
→ 新文章较多者优先
→ 较早 attempt 优先，保证稳定
```

这项 utility 只表示“本次检索产生多少直接可判断证据”，不表示医学真实性。

MVP 中：

- Controller 决定是否继续搜索；
- 普通程序确定 `best_attempt_id`；
- RESOLVE / ABSTAIN 使用 `best_attempt_id`；
- Controller 不直接指定最终 verdict。

后续可以再实现跨 attempt article union 与重新聚合，但不是第一版必要条件。

---

## 11. LangGraph Execution Graph

### 11.1 建议节点

```text
START
  ↓
initialize_run
  ↓
decompose_statement
  ↓
initialize_workspace
  ↓
controller
  ↓
validate_decision
  ↓
route_action
  ├─ SEARCH  → execute_search → update_after_search ─┐
  ├─ RESOLVE → resolve_claim ────────────────────────┤
  ├─ ABSTAIN → abstain_claim ────────────────────────┤
  └─ FINISH  → finalize_statement_analysis           │
                                                    │
                 ┌───────────────────────────────────┘
                 ↓
              controller

finalize_statement_analysis
  ↓
build_statement_bundle
  ↓
inference_gap_analysis
  ↓
output_generation
  ↓
finalize_run
  ↓
END
```

### 11.2 节点职责

#### initialize_run

- 验证 statement / language / budget；
- 建立 run artifact root；
- 写入现有 `request.json`；
- 建立 `agent/` 与 `statement_analysis/attempts/`；
- 初始化 checkpoint path。

#### decompose_statement

- 调用现有 `run_statement_decomposition()`；
- 不改变其 Prompt、Artifact 与 validator；
- 将 decomposition bundle 写入 Workspace。

#### initialize_workspace

- 依据 decomposition 建立 ClaimWorkspace；
- 所有 Claim 初始为 `unresolved`；
- 初始化 search budget 与 step budget。

#### controller

- 建立紧凑 state summary；
- 调用 existing structured LLM transport；
- 返回 `AgentDecision`；
- 不执行实际动作。

#### validate_decision

确定性检查：

- schema；
- action 与字段组合；
- claim 是否存在；
- claim 状态；
- query 是否重复；
- budget；
- FINISH 是否允许。

若非法：

1. 记录 rejected decision；
2. 可进行一次 correction prompt；
3. 第二次仍非法时执行 deterministic fallback，避免无限循环。

#### execute_search

- 扣除 search budget；
- 生成 attempt ID；
- 调用 `search_evidence()`；
- 将结果写入 `last_action_result`。

#### update_after_search

- 更新 search history；
- 更新 seen article IDs；
- 计算 new article IDs；
- 计算 utility；
- 更新 best attempt；
- 形成 `remaining_problem`；
- 写 action trace；
- 回到 Controller。

#### resolve_claim

- 将 Claim 标记为 resolved；
- 固定 best attempt；
- 记录 terminal reason；
- 更新 terminal count；
- 回到 Controller。

#### abstain_claim

- 若有成功 attempt，固定 best attempt 并标记 abstained；
- 若无成功 attempt，标记 failed；
- 回到 Controller。

#### finalize_statement_analysis

- 从 decomposition 与各 Claim 的 chosen attempt 构建现有 `statement_result.json`；
- 写现有 contract 的 `request.json` 与 `run_manifest.json`；
- 返回 `statement_result` 与 `graphs_by_claim`；
- 执行现有 validator。

#### build_statement_bundle

- 直接复用现有 `run_statement_bundle()`；
- 使用 in-memory handoff；
- 不修改 contract。

#### inference_gap_analysis

- 复用现有 `run_inference_gap_analysis()`。

#### output_generation

- 复用现有 `run_output_module()`。

#### finalize_run

- 写 root `run_manifest.json`；
- 写 `agent/workspace.final.json`；
- 写 Agent manifest；
- 验证完整 pipeline artifact；
- 返回与原 `run_statement_pipeline()` 相同的结果结构。

---

## 12. Runtime Context 与 Checkpoint 边界

### 12.1 Runtime Context

新增非 checkpoint 对象：

```python
@dataclass
class AgentRuntimeContext:
    root: Path
    config: BackendConfig
    resources: RuntimeResources
    artifact_dir: Path
    progress_callback: ProgressCallback | None
    llm_stages: Mapping[str, LLMStageConfig]
    pipeline_config: PipelineConfig
```

它通过：

- runner instance；
- node closure；
- 或 LangGraph runtime context；

提供给节点。

不要把它放入 EvidenceWorkspace。

### 12.2 SQLite Checkpoint

每个 API run 使用独立 checkpoint：

```text
<run_artifact>/agent/checkpoints.sqlite
```

LangGraph invocation config：

```python
config = {
    "configurable": {
        "thread_id": run_name,
    }
}
```

当前后端是单 worker、同步执行，并由 Engine `_run_lock` 保护，因此第一版使用同步 `SqliteSaver` 足够。

### 12.3 Checkpoint 的展示价值

可以真实说明：

- 每个 LangGraph step 都保存 Workspace；
- 能检查任一步的状态；
- 能查看 Action 前后的差异；
- 可以支持后续 resume、interrupt 和 time-travel debugging；
- 第一版可以只实现 checkpoint，不必立刻开放恢复 API。

---

## 13. 建议新增目录

```text
backend/evidencegap_backend/agent/
├─ __init__.py
├─ contracts.py
├─ workspace.py
├─ controller.py
├─ decision_validation.py
├─ tools.py
├─ attempt_selection.py
├─ nodes.py
├─ graph.py
├─ runner.py
├─ artifacts.py
├─ tracing.py
└─ validation.py
```

### 文件职责

#### contracts.py

- `AgentAction`；
- `AgentDecision`；
- Workspace TypedDict；
- SearchAttempt；
- TraceEvent；
- Agent manifest constants。

#### workspace.py

- 从 decomposition 初始化 Workspace；
- compact controller summary；
- Claim 状态变更函数；
- terminal count；
- query normalization。

#### controller.py

- 读取 `agent_controller.txt`；
- 调用 `call_structured_llm()`；
- provider response normalization；
- Pydantic validation；
- correction request。

#### decision_validation.py

- Action 合法性；
- budget；
- duplicate query；
- FINISH guard；
- deterministic fallback。

#### tools.py

- `search_evidence()`；
- 调用改造后的 `run_analysis()`；
- 将 graph bundle 转为 SearchAttempt。

#### attempt_selection.py

- utility score；
- best attempt 选择；
- tie break。

#### nodes.py

- 所有 LangGraph Node 函数或 Node factory。

#### graph.py

- `StateGraph(EvidenceWorkspace)`；
- nodes；
- fixed edges；
- conditional edges；
- compile factory；
- Mermaid export。

#### runner.py

- `run_agent_statement_pipeline()`；
- 建立 runtime context；
- 建立 SqliteSaver；
- invoke graph；
- 返回与原 pipeline 相同的结果。

#### artifacts.py

- Agent request / manifest；
- final workspace；
- action trace；
- execution graph；
- Statement Analysis artifact materialization。

#### tracing.py

- append-only JSONL；
- action lifecycle；
- tool result summary；
- rejected decision；
- budget change。

#### validation.py

- Agent artifact validator；
- Workspace terminal state validator；
- Action trace consistency；
- chosen attempt identity。

---

## 14. Engine 接入方式

### 14.1 保留 Legacy Pipeline

不要删除：

```python
run_statement_pipeline()
```

新增：

```python
run_agent_statement_pipeline()
```

在 `EvidenceGapEngine.analyze_statement()` 中选择：

```python
runner = (
    run_agent_statement_pipeline
    if cfg.agent.enabled
    else run_statement_pipeline
)
```

这样可以：

- 快速比较 Workflow 与 Agent；
- 出现问题时回退；
- 展示 Agent 化收益；
- 保留当前测试基线。

### 14.2 方法返回保持一致

`run_agent_statement_pipeline()` 必须返回当前代码期待的字段：

```text
status
artifact_status
run_name
artifact_dir
statement_id
analysis_status
output_language
localized
execution_summary
counts
empty_claims
statement_bundle_path
inference_gap_analysis_path
presentation_bundle_path
presentation_bundle
```

因此：

- `RunManager` 不改；
- `RunStore` 不改；
- `StatementAnalysisResult` 不改；
- 前端请求不改。

---

## 15. Config 设计

建议新增：

```python
@dataclass(frozen=True)
class AgentConfig:
    enabled: bool = True
    max_steps: int = 20
    search_budget: int = 8
    max_searches_per_claim: int = 3
    controller_correction_attempts: int = 1
    checkpoint_enabled: bool = True
```

在 `BackendConfig` 增加：

```python
agent: AgentConfig
agent_controller_llm: LLMStageConfig | None = None
```

兼容规则：

- 旧配置没有 `agent` 时使用默认值；
- `agent_controller_llm` 未设置时继承 `article_evidence_llm` 的 provider、model、base URL 与认证；
- Controller 建议使用较小 max tokens，例如 1200–2000；
- Controller 不需要 thinking 模式也能工作；若要展示，可允许 DeepSeek thinking，但不把 reasoning 写入公开 trace。

配置示例：

```json
{
  "agent": {
    "enabled": true,
    "max_steps": 20,
    "search_budget": 8,
    "max_searches_per_claim": 3,
    "controller_correction_attempts": 1,
    "checkpoint_enabled": true
  },
  "llm": {
    "agent_controller": {
      "provider": "deepseek",
      "model": "deepseek-v4-pro",
      "max_tokens": 1600,
      "thinking": false
    }
  }
}
```

配置文件实际结构应遵循当前 `api/config.py` 的既有样式，不为 Agent 另建第二套配置系统。

---

## 16. Progress 与前端兼容

现有 API `RunStage` 只有五个阶段：

```text
statement_decomposition
claim_analysis
statement_bundle
inference_gap_analysis
output_generation
```

第一版不要修改枚举，避免前端同步改造。

LangGraph Agent Loop 全部映射为：

```text
stage = claim_analysis
```

动态 message 示例：

```text
Agent step 1: searching evidence for Claim C2
Agent step 2: C2 search returned 3 new articles
Agent step 3: resolving Claim C2 with attempt search_001
Agent step 4: searching evidence for Claim C1
```

`completed_units`：terminal Claim 数。  
`total_units`：总 Claim 数。

这样前端不改代码，也能在现有 Progress 面板中直接看到 Agent 动态行动。

---

## 17. Agent Artifact 与可展示信息

建议每个 run 新增：

```text
<run_artifact>/agent/
├─ request.json
├─ run_manifest.json
├─ workspace.initial.json
├─ workspace.final.json
├─ action_trace.jsonl
├─ execution_graph.mmd
├─ checkpoints.sqlite
└─ controller/
   ├─ step_001_request.json
   ├─ step_001_response.json
   ├─ step_002_request.json
   └─ step_002_response.json
```

### 17.1 action_trace.jsonl 示例

```json
{
  "event_id": "event_0003",
  "step": 3,
  "event_type": "action_completed",
  "action": "SEARCH",
  "claim_id": "claim_xxx",
  "query": "GLP-1 receptor agonist microvascular complications randomized trial",
  "reason": "Existing evidence only covers glycemic control.",
  "attempt_id": "search_002",
  "verdict": "mixed",
  "article_counts": {
    "total": 10,
    "support": 2,
    "refute": 1,
    "insufficient": 7
  },
  "new_article_count": 6,
  "remaining_search_budget": 4
}
```

### 17.2 Agent run manifest

至少记录：

```text
contract_id
schema_version
run_name
thread_id
controller provider/model
max_steps
initial_search_budget
consumed_search_budget
total_actions
action_counts
rejected_decisions
total_attempts
resolved_claims
abstained_claims
failed_claims
checkpoint path
action trace path
final workspace hash
```

### 17.3 Root run manifest

现有 root `run_manifest.json` 中增加：

```json
{
  "run_type": "agentic_evidencegap_end_to_end",
  "execution": {
    "orchestration": "langgraph",
    "agent_harness": {
      "enabled": true,
      "controller": "evidence_controller",
      "actions": ["SEARCH", "RESOLVE", "ABSTAIN", "FINISH"],
      "checkpoint": "sqlite"
    }
  },
  "agent": {
    "artifact_dir": "...",
    "run_manifest": {"path": "...", "sha256": "..."},
    "action_trace": {"path": "...", "sha256": "..."},
    "final_workspace": {"path": "...", "sha256": "..."}
  }
}
```

不要把 Agent artifact 加入当前 `stages` mapping，因为现有 validator 要求 `stages` 恰好等于既有五阶段集合。Agent metadata 应成为独立顶层字段。

---

## 18. LangGraph Studio 与执行图

### 18.1 根目录 langgraph.json

可选新增：

```json
{
  "dependencies": ["./backend"],
  "graphs": {
    "evidencegap_agent": "./backend/evidencegap_backend/agent/studio.py:graph"
  },
  "env": ".env"
}
```

### 18.2 Studio 注意事项

实际 EvidenceGap graph 依赖：

- BackendConfig；
- RuntimeResources；
- 大型模型；
- 索引；
- run-specific artifact path。

因此建议分成：

1. `build_agent_graph(context, checkpointer)`：真实 backend 使用；
2. `studio.py`：从环境加载配置并创建可调试 graph，或提供使用 mock tool 的结构展示 graph。

Studio 是额外展示入口，不应替代现有 FastAPI。

### 18.3 无 Studio 也必须输出 Mermaid

在 graph compile 后输出：

```python
graph.get_graph().draw_mermaid()
```

写入：

```text
agent/execution_graph.mmd
```

即使 Studio 环境不方便，仍然可以用 Mermaid 清楚展示 LangGraph 执行结构。

---

## 19. 环境与安装

项目结构假定：

```text
EvidenceGap/
├─ .venv/
├─ backend/
│  └─ pyproject.toml
├─ frontend/
└─ ...
```

虚拟环境位于根目录不影响 editable install。激活根目录 venv 后，从根目录执行：

```bash
source .venv/bin/activate

(
  cd backend
  python -m pip install -e '.[test,agent-dev]'
)
```

这里使用子 shell，安装结束后仍回到项目根目录。

也可以：

```bash
source .venv/bin/activate
cd backend
python -m pip install -e '.[test,agent-dev]'
cd ..
```

验证：

```bash
python - <<'PY'
import pydantic
import langgraph
import langchain_core

from langgraph.graph import StateGraph
from langgraph.checkpoint.sqlite import SqliteSaver

print("pydantic:", pydantic.__version__)
print("langchain-core:", langchain_core.__version__)
print("StateGraph:", StateGraph)
print("SqliteSaver:", SqliteSaver)
print("EvidenceGap agent environment: OK")
PY
```

说明：

- `.venv` 在根目录，只决定当前 Python interpreter；
- editable project 在 `backend/`，所以 `pip install -e` 必须针对 backend project；
- 最稳妥方式是在子 shell 中切到 `backend` 后使用标准 `'.[extras]'` 语法。

---

## 20. pyproject 依赖原则

正式依赖：

```toml
"pydantic>=2.10,<3",
"langgraph>=1.2,<2",
"langchain-core>=1.5,<2",
"langgraph-checkpoint-sqlite>=3.1,<4",
```

开发 extra：

```toml
agent-dev = [
  "langgraph-cli[inmem]>=0.4,<0.5",
]
```

不建议第一版加入：

```text
完整 langchain
langchain-openai
langchain-anthropic
create_agent
langgraph-prebuilt ReAct Agent
LangServe
Celery
Redis
```

项目已有自己的 provider、structured JSON、retry、prompt 与 artifact 机制，应继续复用。

---

## 21. 分阶段实作顺序

下面顺序以“最快看到 Agent 效果”为优先，不按传统长期工程顺序展开。

### Phase 01：Agent Contract 与 LangGraph 骨架

目标：

- 新增 `agent/` package；
- 定义 Workspace、Action、Decision；
- 建立 LangGraph 节点和条件路由；
- 先使用 fake controller 与 fake search tool 跑通循环；
- 写 SQLite checkpoint；
- 输出 Mermaid；
- 验证 budget 与终止。

范围：

```text
contracts.py
workspace.py
decision_validation.py
nodes.py
graph.py
tracing.py
```

验收：

- Graph 至少产生 `SEARCH → SEARCH → RESOLVE → FINISH`；
- duplicate query 会被拒绝；
- search budget 为 0 时无法 SEARCH；
- unresolved claim 存在时无法 FINISH；
- checkpoint DB 真实生成；
- state 可以 JSON dump。

这一阶段不接真实检索，避免一开始被 GPU 与 Artifact 细节拖慢。

### Phase 02：接入真实 Evidence Search Tool

目标：

- 给 `run_analysis()` 增加 `retrieval_query`；
- 给 retrieval adapter 增加 `query_text`；
- Judge 与聚合继续使用 canonical claim；
- 每次 SEARCH 写独立 attempt Artifact；
- 从 Final Graph 提取 attempt summary；
- 实现 best attempt utility。

重点文件：

```text
pipeline/analysis.py
pipeline/retrieval_adapters.py
agent/tools.py
agent/attempt_selection.py
```

验收：

- canonical claim 不变；
- 两个不同 Query 产生相同 claim_id；
- Retrieval Artifact 同时保存 claim_text 与 query_text；
- Judge request 仍使用 canonical claim；
- 不同 attempt 可独立通过 `validate_analysis_artifact()`；
- Workspace 能看到新增文章和 evidence counts。

### Phase 03：Controller 与 Agent Statement Analysis

目标：

- 新增 Controller Prompt；
- 复用 `call_structured_llm()`；
- 实现结构化 Decision；
- 实现 correction 与 fallback；
- 将 chosen attempts materialize 为现有 Statement Analysis Contract；
- 调用现有 Statement Bundle。

重点文件：

```text
prompts/agent_controller.txt
agent/controller.py
agent/artifacts.py
agent/validation.py
pipeline/statement_analysis.py（可抽共用 helper，但不要破坏 legacy）
```

验收：

- Controller 会针对不同 Workspace 输出不同动作；
- 至少一个真实案例发生 query rewrite；
- Agent Statement Analysis Artifact 通过现有 validator；
- `run_statement_bundle()` 无需改 contract 即可消费结果；
- ABSTAIN 不会篡改 verdict。

### Phase 04：完整 Pipeline 与 Engine 接入

目标：

- 实现 `run_agent_statement_pipeline()`；
- 复用 Gap Analysis 与 Output Module；
- 加入 `AgentConfig`；
- Engine 根据配置选择 Agent / Legacy；
- 保持 API 不变；
- 将动态行动显示在现有 Progress message。

重点文件：

```text
agent/runner.py
engine.py
config.py
api/config.py
config.example.json
pipeline/statement_run.py（保留 legacy）
```

验收：

- 前端不改，仍能创建和读取 run；
- presentation bundle 通过原 validator；
- article context API 正常；
- localization 正常；
- Agent Artifact 与 checkpoint 生成；
- Legacy 模式仍可运行。

### Phase 05：展示与实验整理

目标：

- `langgraph.json`；
- Studio 或 Mermaid 展示；
- Agent action trace summary；
- 固定 Workflow vs Agent 的小规模比较；
- 选出适合截图的输入。

建议报告：

```text
reports/agent/agent_demo_cases.md
reports/agent/workflow_vs_agent.json
reports/agent/workflow_vs_agent.md
```

比较指标：

- search attempt 数；
- unique query 数；
- 新增 unique article 数；
- direct evidence article 数；
- insufficient article 比例；
- resolved / abstained Claim 数；
- token / API request 数；
- latency；
- Agent action trace。

---

## 22. 建议 Coding Agent 的 Commit 边界

### Commit 1

```text
feat(agent): add workspace contracts and langgraph skeleton
```

只完成 State、Action、Graph、fake nodes、tests。

### Commit 2

```text
feat(agent): separate retrieval query from canonical claim
```

只完成 `retrieval_query` 数据流与相关 Artifact / tests。

### Commit 3

```text
feat(agent): add evidence search tool and attempt selection
```

接真实 `run_analysis()`，但尚未切 Engine。

### Commit 4

```text
feat(agent): add structured evidence controller
```

Controller、Prompt、validation、fallback。

### Commit 5

```text
feat(agent): materialize agent claim results into statement contracts
```

完成 Statement Analysis bridge、Bundle handoff。

### Commit 6

```text
feat(agent): route engine through langgraph harness
```

完整 pipeline、config、API regression。

### Commit 7

```text
docs(agent): add studio config traces and comparison demo
```

展示产物与文档。

拆分 Commit 可以避免 Coding Agent 一次修改大量 pipeline 代码而难以审查。

---

## 23. 测试计划

### 23.1 Unit Tests

新增：

```text
backend/tests/agent/
├─ test_workspace.py
├─ test_decision_validation.py
├─ test_attempt_selection.py
├─ test_graph_routing.py
├─ test_controller_contract.py
└─ test_agent_artifacts.py
```

必须覆盖：

- Query normalization；
- duplicate query；
- 不存在 Claim；
- 已 resolved Claim 再 SEARCH；
- budget exhausted；
- per-claim attempt limit；
- FINISH guard；
- deterministic utility tie break；
- Workspace JSON serialization；
- trace step 连续性；
- chosen attempt claim identity。

### 23.2 Graph Tests

使用 fake controller 和 fake tool：

```text
Case A：SEARCH → RESOLVE → FINISH
Case B：SEARCH → SEARCH → ABSTAIN → FINISH
Case C：非法重复 SEARCH → correction → RESOLVE
Case D：预算耗尽 → ABSTAIN → FINISH
Case E：两个 Claim 交错处理
```

### 23.3 Pipeline Regression

现有 tests 必须继续通过：

```text
test_api.py
test_engine.py
test_phase81_contract.py
test_phase82_article_context.py
test_phase9a_article_applicability.py
test_phase9b_structured_gaps.py
```

新增 Agent API integration：

- POST endpoint 不变；
- status polling 不变；
- succeeded result 可被前端 contract 消费；
- execution summary 可含额外 Agent 字段；
- article context offsets 不变；
- localization 可以基于 Agent run 执行。

### 23.4 Real Smoke Test

至少使用一个多层推理输入：

```text
GLP-1 receptor agonists reduce HbA1c in adults with type 2 diabetes.
Reduced HbA1c is associated with lower microvascular risk.
Therefore, GLP-1 receptor agonists prevent all diabetic complications
in every person with diabetes.
```

希望观察：

- Controller 先处理基础事实 Claim；
- 对“prevent all diabetic complications”生成更具体的新 Query；
- 某个 Claim 进行第二轮搜索；
- 最终部分 Claim RESOLVE，部分 Claim ABSTAIN 或以 insufficient 结果终止；
- Action Trace 清楚显示为什么继续或停止。

---

## 24. 完成标准

本次 Agent 化完成时，必须同时满足：

1. 使用 `langgraph.graph.StateGraph`；
2. 有显式 EvidenceWorkspace；
3. 有结构化 AgentDecision；
4. 有 SEARCH / RESOLVE / ABSTAIN / FINISH；
5. Controller 根据 Workspace 动态选择行动；
6. SEARCH 调用真实 Evidence Tool；
7. 至少支持同一 Claim 多个不同 Query attempt；
8. canonical claim 与 retrieval query 已分离；
9. 有预算和重复 Query guard；
10. 有循环和条件路由；
11. 有 SQLite checkpoint；
12. 有 JSONL Action Trace；
13. 有 Mermaid Execution Graph；
14. 原 FastAPI endpoints 不变；
15. 原 presentation bundle 可正常输出；
16. 原 Article Context API 正常；
17. 原 frontend 不需要修改即可运行；
18. Legacy Workflow 可以保留用于比较；
19. 至少一个真实案例可看到非固定行动序列；
20. 可以从 Artifact 证明 Agent 做过哪些行动，而不是只靠口头描述。

---

## 25. 项目展示时的技术表述

推荐表述：

> EvidenceGap 原本是一条固定的医学证据 RAG Workflow。改造后，我们保留了经过实验验证的多路检索、Cross-Encoder、文章级证据判断和 Claim 聚合模块，并在其上实现了一个基于 LangGraph 的受约束 Evidence Agent Harness。
>
> Agent 以 Evidence Workspace 作为显式短期记忆，通过结构化 Controller 在 SEARCH、RESOLVE、ABSTAIN 与 FINISH 之间动态决策。SEARCH 是一个高层 Evidence Tool，内部封装 BM25、MedCPT、BMRetriever、RRF、Cross-Encoder 和文章级 Judge。每次 Tool 结果会更新 Workspace，Controller 再根据新证据决定下一步。
>
> Harness 另外提供 Query 去重、搜索预算、动作合法性验证、SQLite checkpoint、Action Trace 与 Artifact 血缘。LangGraph 负责 State、Node、Conditional Edge、Loop 和持久化，而医学证据结论仍由现有可审计 Pipeline 产生，Controller 不直接篡改 verdict。

简历版：

> 将固定医学证据 RAG Workflow 改造为 LangGraph Evidence Agent Harness：设计显式 Evidence Workspace、结构化 Action Space 与 Evidence Search Tool，支持动态 Query 重写、多轮证据检索、预算终止、SQLite checkpoint 和可审计 Action Trace，并保持原 FastAPI 与前端契约兼容。

---

## 26. 不建议的错误实现

### 错误 1：只把现有五个步骤改名成 LangGraph Node

```text
Decompose → Search → Judge → Gap → Output
```

若没有 Controller、Action、反馈循环与动态路由，本质仍是固定 Workflow，只是使用 LangGraph 画图。

### 错误 2：把所有函数都注册成 Tool

这会让模型控制：

- RRF；
- Cross-Encoder；
- Judge；
- Aggregation；
- Artifact 写入。

既增加失控风险，也破坏现有实验与模块边界。

### 错误 3：让 Controller 直接输出 verdict

Controller 只能决定是否继续行动，不能覆盖正式 evidence verdict。

### 错误 4：在 State 中放 RuntimeResources

会导致 checkpoint 无法可靠序列化，并把大型模型对象与业务状态混在一起。

### 错误 5：为了 Agent 改掉 API

本次改造重点是 backend orchestration，不需要让前端适配另一套 Agent Server API。

### 错误 6：第一版就做 SPLIT 和证据融合

会显著放大 scope，拖慢最需要展示的 Agent Loop、Trace 与 LangGraph 部分。

### 错误 7：只写 Trace，不让 Action 影响结果

Agent 的 SEARCH Query 必须真实进入 Retrieval；chosen attempt 必须真实成为下游 Statement Bundle 的来源。否则 Trace 只是表演层。

---

## 27. 后续可选扩展

第一版完成后，可以选择增加：

### 27.1 SPLIT + Graph Patch

允许 Controller 将过宽 Claim 拆分，并通过普通程序验证 source span 与 inference graph patch。

### 27.2 多 Attempt Evidence Union

将多轮检索得到的 unique articles 合并后重新执行 Judge / Aggregation，而不是只选最佳 attempt。

### 27.3 Human-in-the-loop

在高风险 SPLIT、预算追加或结果发布前使用 LangGraph interrupt。

### 27.4 Resume API

利用 checkpoint 对中断 run 进行恢复，但保持现有 API 可兼容地扩展。

### 27.5 Planner / Reviewer 双 Controller

仅在需要展示 Multi-Agent 时增加，不应成为第一版必需条件。

### 27.6 LangSmith Trace

若环境允许，可以追加外部 observability；本地 JSONL 与 SQLite 仍应保留，不能让项目依赖云端才能审计。

---

## 28. 最终实施结论

本次最合适的改造不是：

```text
把 EvidenceGap 重写成自由 ReAct Agent
```

而是：

```text
保留当前 EvidenceGap 确定性医学证据模块
＋
在 Claim Analysis 调度层引入 Evidence Controller
＋
使用 LangGraph 管理 Workspace、Node、Conditional Edge 与 Loop
＋
将 run_analysis 包装为高层 Evidence Search Tool
＋
分离 canonical claim 与 retrieval query
＋
加入多轮 search attempt、预算、去重、终止、checkpoint 与 trace
＋
将最终 chosen attempt 桥接回现有 Statement Analysis Contract
＋
继续复用 Statement Bundle、Gap Analysis、Output 与 FastAPI
```

这是目前最快、最直接、最容易看到效果，同时又能在技术上合理称为 Agent Architecture、Agent Harness 和 LangGraph Runtime 的实现方向。

---

## 29. 本文依据

本方案依据以下项目材料与当前代码结构整理：

- `EvidenceGap_Agent化改造方向设计.md`；
- `EvidenceGap_V1_推進規劃書.md`；
- 当前 `backend/evidencegap_backend/engine.py`；
- 当前 `pipeline/statement_run.py`；
- 当前 `pipeline/statement_analysis.py`；
- 当前 `pipeline/analysis.py`；
- 当前 `pipeline/retrieval_adapters.py`；
- 当前 API、Artifact 与 validator 实现。

