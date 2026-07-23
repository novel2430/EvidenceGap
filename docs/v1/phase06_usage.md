# EvidenceGap V1 Phase 06：06.1 Schema + 06.2 DeBERTa Zero-shot Baseline

## 1. 安装与模型

保留项目既有 CUDA-compatible PyTorch：

```bash
pip install -r requirements/v1-phase06.txt
```

下载 verifier：

```bash
python scripts/download_v1_models.py \
  --root . \
  --model verifier-deberta-v3-base
```

模型目录：

```text
models/v1/verifier-deberta-v3-base/
```

Runner 只接受 safetensors，不回退到 `pytorch_model.bin`。

## 2. 从 Phase 05 Dev Top-5 建立输入

默认读取 frozen Dev ranking 与 `dev_full` canonical artifact：

```bash
python scripts/run_v1_phase06.py prepare-phase05 \
  --root . \
  --split dev \
  --top-k 5 \
  --run-name phase05_rrf_top5_dev
```

输出：

```text
artifacts/v1/stance_verification/inputs/phase05_rrf_top5_dev/
├── stance_inputs.parquet
└── run_manifest.json
```

该步骤验证：

- query/paper/pool fingerprint 一致；
- 原始 `sentence_index`、`sentence_type` 与 `sentence_text` 未改变；
- Phase 05 ranking checksum 与 manifest 一致；
- 不从 EvidenceBench aspect gold 推导 stance label。

## 3. 运行真实 Phase 05 输入的 zero-shot baseline

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase06.py zero-shot \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/phase05_rrf_top5_dev/stance_inputs.parquet \
  --device cuda:0 \
  --batch-size 16 \
  --amp fp16 \
  --run-name deberta_zero_shot_phase05_top5_dev
```

输出：

```text
artifacts/v1/stance_verification/zero_shot/deberta_zero_shot_phase05_top5_dev/
├── stance_predictions.parquet
└── run_manifest.json
```

EvidenceBench 没有 stance gold，因此该 run 只报告 label distribution、mean confidence、margin 与吞吐量，不伪造 Macro-F1。

## 4. HealthFC external expert evaluation

先建立相同 schema 的 HealthFC bundle 输入：

```bash
python scripts/run_v1_phase06.py prepare-healthfc \
  --root . \
  --run-name healthfc_eval
```

再运行相同模型：

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase06.py zero-shot \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/healthfc_eval/stance_inputs.parquet \
  --device cuda:0 \
  --batch-size 16 \
  --amp fp16 \
  --run-name deberta_zero_shot_healthfc
```

HealthFC run 会额外输出：

```text
reports/v1/stance_zero_shot_deberta_zero_shot_healthfc.json
reports/v1/stance_zero_shot_deberta_zero_shot_healthfc.md
```

指标：

```text
Accuracy
Macro-F1
Balanced Accuracy
Per-class Precision / Recall / F1
Confusion Matrix
```

该数字只能表述为 expert gold evidence bundle evaluation，不能冒充 Phase 05 单句准确率。

## 5. 验证 artifact

```bash
python scripts/run_v1_phase06.py validate \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/healthfc_eval/stance_inputs.parquet
```

```bash
python scripts/run_v1_phase06.py validate \
  --root . \
  --prediction-path artifacts/v1/stance_verification/zero_shot/deberta_zero_shot_healthfc/stance_predictions.parquet \
  --run-name deberta_zero_shot_healthfc
```

## 6. CPU smoke

CPU 只能使用：

```bash
python scripts/run_v1_phase06.py zero-shot \
  --root . \
  --input-path <stance_inputs.parquet> \
  --device cpu \
  --amp none \
  --batch-size 4
```

## 7. Test 隔离

Phase 06 开发禁止读取 Phase 05 test artifact。最终冻结评测才允许：

```bash
python scripts/run_v1_phase06.py prepare-phase05 \
  --root . \
  --split test \
  --allow-test \
  --top-k 5
```

## 8. Phase 06.3：LLM Structured Stance Judge

Phase 06.3 使用同一份英文 prompt 和同一份 JSON 输出契约接入：

```text
DeepSeek Chat Completions JSON mode
Anthropic Messages API Structured Outputs
```

API key 只从环境变量读取，不写入命令、artifact、cache 或 report：

```bash
export DEEPSEEK_API_KEY='...'
export ANTHROPIC_API_KEY='...'
```

默认模型：

```text
deepseek  -> deepseek-v4-pro
anthropic -> claude-sonnet-4-6
```

可随时使用 `--model` 覆盖，不需要修改代码。

### 8.1 先跑小规模 smoke

建议先在 HealthFC 的前 30 条上确认 key、模型权限、JSON schema 和费用：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/healthfc_eval/stance_inputs.parquet \
  --provider deepseek \
  --limit 30 \
  --request-batch-size 5 \
  --run-name deepseek_healthfc_smoke30
```

Claude：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/healthfc_eval/stance_inputs.parquet \
  --provider anthropic \
  --limit 30 \
  --request-batch-size 5 \
  --run-name claude_healthfc_smoke30
```

### 8.2 跑完整 HealthFC

DeepSeek：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/healthfc_eval/stance_inputs.parquet \
  --provider deepseek \
  --request-batch-size 8 \
  --run-name deepseek_v4_pro_healthfc
```

Claude Sonnet 4.6：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/healthfc_eval/stance_inputs.parquet \
  --provider anthropic \
  --request-batch-size 8 \
  --run-name claude_sonnet_46_healthfc
```

Claude Sonnet 5：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/healthfc_eval/stance_inputs.parquet \
  --provider anthropic \
  --model claude-sonnet-5 \
  --request-batch-size 5 \
  --max-retries 1 \
  --run-name claude_sonnet5_healthfc_full
```

Sonnet 5 在省略 `thinking` 字段时会默认启用 adaptive thinking。Phase 06 的固定结构化分类请求会自动加入：

```json
{"thinking": {"type": "disabled"}}
```

该行为只针对精确模型 ID `claude-sonnet-5`；Sonnet 4.6 与其他 Anthropic 模型保持原请求格式。请求配置会进入 model fingerprint、cache fingerprint 与 run report。

若使用其他模型：

```bash
--model <provider-model-id>
```

如果自己的 key 放在其他环境变量：

```bash
--api-key-env MY_PRIVATE_API_KEY
```

不要直接把 API key 作为 CLI 参数，避免进入 shell history。

### 8.3 输出

```text
artifacts/v1/stance_verification/llm_judge/<run_name>/
├── stance_predictions.parquet
└── run_manifest.json

reports/v1/stance_llm_<run_name>.json
reports/v1/stance_llm_<run_name>.md
```

每条 prediction 除了三分类概率，还包括：

```text
rationale             英文单句解释
evidence_type         direct_result / background / method / ...
requires_context      是否依赖缺失上下文
provider              deepseek / anthropic
provider_request_id   API request ID（若 provider 返回）
prompt_version        固定 prompt 版本
raw_response_sha256   原始 API response checksum
```

`evidence_type` 属于辅助展示字段。Prompt version `phase06_llm_stance_v2` 会明确列出七个合法值，并要求模型严格七选一；重叠时使用以下优先级：

```text
safety
> statistical_uncertainty
> population_or_scope
> direct_result
> method
> background
> mixed_or_other
```

常见同义词仍会自动规范化，未知值会降级为 `mixed_or_other`，不会触发整批 API 重试。由于 prompt version、prompt hash 和 response schema 都进入 cache key，v1 cache 不会被 v2 误用。

LLM 输出中的概率是模型自我报告概率，不应宣称已经完成统计校准。

### 8.4 JSON 回覆契约

两家 provider 都必须返回同一结构：

```json
{
  "results": [
    {
      "input_id": "exact-input-id",
      "label": "support",
      "probabilities": {
        "support": 0.9,
        "refute": 0.02,
        "insufficient": 0.08
      },
      "rationale": "The reported outcome directly supports the claim.",
      "evidence_type": "direct_result",
      "requires_context": false
    }
  ]
}
```

Runner 会拒绝并重试以下回覆：

- 缺少或重复 `input_id`；
- 数量或顺序与请求不一致；
- `evidence_type` 同义词会正规化，未知值会降级为 `mixed_or_other`；
- label 为空或非法但 probabilities 完整时会本地恢复；
- label 与 probability argmax 不一致时会本地 reconcile；
- 概率缺失、不在 `[0, 1]` 或不接近总和 1；
- rationale 为空；
- truncated、empty 或 non-JSON response。

### 8.5 Cache 与费用保护

每个成功 batch 会缓存到：

```text
artifacts/v1/stance_verification/llm_cache/<provider>/<model>/
```

cache key 包含：

```text
provider + model + prompt + JSON schema + inputs + request parameters
```

相同请求重跑会使用 cache，不重复调用 API。Report 分开记录：

```text
api_requests
cache_hits
billed_usage
cached_usage
retry_count
```

如果只想重新生成最终 Parquet/report，可以删除对应 `llm_judge/<run_name>`，保留 `llm_cache`。

### 8.6 DeepSeek thinking mode

默认关闭 thinking，以降低分类任务的延迟与费用。需要测试 reasoning 时显式加入：

```bash
--thinking
```

该参数只适用于 `--provider deepseek`。

### 8.7 跑 Phase 05 Dev evidence（06.6）

Phase 05 是句子级输入。正式 06.6 输入默认加入目标句前后各一条原始 canonical sentence，仅用于解析指代和比较关系；目标 `sentence_text` 与原始 `sentence_index` 不变。

```bash
python scripts/run_v1_phase06.py prepare-phase05 \
  --root . \
  --split dev \
  --top-k 5 \
  --context-window 1 \
  --run-name phase05_rrf_top5_ctx1_dev
```

输出：

```text
artifacts/v1/stance_verification/inputs/phase05_rrf_top5_ctx1_dev/
├── stance_inputs.parquet
└── run_manifest.json
```

#### 不扣费的全量执行规划

`--dry-run` 不读取 API key、不调用 provider，也不创建 prediction artifact：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/phase05_rrf_top5_ctx1_dev/stance_inputs.parquet \
  --provider deepseek \
  --model deepseek-v4-pro \
  --request-batch-size 5 \
  --dry-run
```

规划会报告：

```text
selected queries / rows
estimated API requests
rows-per-query distribution
rank gaps / duplicate sentence indices
claim / evidence / context character counts
selection checksum
```

#### Query 级 Smoke-100

不要再用 `--limit 100`：它表示 100 行，可能切断某个 query 的完整 Top-5。06.6 应使用确定性 query 抽样：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/phase05_rrf_top5_ctx1_dev/stance_inputs.parquet \
  --provider deepseek \
  --model deepseek-v4-pro \
  --query-sample-size 100 \
  --query-sample-seed 20260722 \
  --request-batch-size 5 \
  --max-retries 1 \
  --run-name deepseek_v4_pro_phase05_top5_ctx1_dev_smoke100
```

该命令会保留抽中 query 的完整 Top-5，约产生 500 条判断和 100 个 API requests。先检查 stance 分布、英文 rationale、`evidence_type` 与代表案例；这不是新增人工标注。

#### Full Dev

Smoke 输出正常后运行全部 Dev：

```bash
python scripts/run_v1_phase06.py llm-judge \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/phase05_rrf_top5_ctx1_dev/stance_inputs.parquet \
  --provider deepseek \
  --model deepseek-v4-pro \
  --request-batch-size 5 \
  --max-retries 1 \
  --run-name deepseek_v4_pro_phase05_top5_ctx1_dev
```

相同参数重跑会复用成功 batch cache。EvidenceBench 没有 stance gold，因此该 run 只输出预测分布、解释、evidence type、请求费用与完整 artifact validation，不报告 Macro-F1。冻结配置记录在：

```text
configs/v1/phase06_stance_verification_frozen.json
```

### LLM 输出容错与费用保护

- `evidence_type` 同义词会正规化，未知值降级为 `mixed_or_other`。
- label 与有效 probabilities 的 argmax 不一致时，以明确 label 为主并本地调整概率。
- label 为空或不是三类之一，但三类 probabilities 完整有效时，使用 probability argmax 恢复 label。
- 上述本地修复都会进入 report 计数，不会重新请求 API。
- 缺少完整概率、缺少 rationale、ID/数量错误或无法解析 JSON 仍会触发 retry。

### 8.8 从中断的 Full Dev cache 导出 partial artifact

如果 Full Dev 在完成前中止，不需要继续请求 API，也不需要删除 cache。使用与原运行完全相同的 input、provider、model、batch size、max tokens、base URL 与 thinking 设置执行：

```bash
python scripts/run_v1_phase06.py export-cache \
  --root . \
  --input-path artifacts/v1/stance_verification/inputs/phase05_rrf_top5_ctx1_dev/stance_inputs.parquet \
  --provider deepseek \
  --model deepseek-v4-pro \
  --request-batch-size 5 \
  --run-name deepseek_v4_pro_phase05_top5_ctx1_dev_partial2520
```

该命令：

- 不读取 API key；
- 不调用 DeepSeek 或 Anthropic；
- 按当前 Prompt 与请求参数重新计算 exact request fingerprint；
- 只读取真正匹配当前输入的 cache；
- 只导出完整 query group，排除不完整 Top-5；
- 生成正式 Parquet、manifest、JSON report 和 Markdown summary。

输出：

```text
artifacts/v1/stance_verification/llm_judge/<run_name>/
├── stance_predictions.parquet
└── run_manifest.json

reports/v1/stance_llm_<run_name>.json
reports/v1/stance_llm_<run_name>.md
```

查看总体覆盖率与 stance/evidence type 分布：

```bash
cat reports/v1/stance_llm_deepseek_v4_pro_phase05_top5_ctx1_dev_partial2520.md
```

验证导出的 prediction artifact：

```bash
python scripts/run_v1_phase06.py validate \
  --root . \
  --prediction-path artifacts/v1/stance_verification/llm_judge/deepseek_v4_pro_phase05_top5_ctx1_dev_partial2520/stance_predictions.parquet \
  --run-name deepseek_v4_pro_phase05_top5_ctx1_dev_partial2520
```

查看实际预测案例：

```bash
python - <<'PY'
import pyarrow.parquet as pq

path = (
    "artifacts/v1/stance_verification/llm_judge/"
    "deepseek_v4_pro_phase05_top5_ctx1_dev_partial2520/"
    "stance_predictions.parquet"
)
rows = pq.read_table(
    path,
    columns=[
        "query_id",
        "claim_text",
        "paper_id",
        "sentence_index",
        "evidence_rank",
        "evidence_text",
        "predicted_label",
        "confidence",
        "evidence_type",
        "rationale",
    ],
).slice(0, 20).to_pylist()

for row in rows:
    print("=" * 100)
    print(f"Query: {row['query_id']}  Rank: {row['evidence_rank']}")
    print("Claim:", row["claim_text"])
    print("Evidence:", row["evidence_text"])
    print(
        "Prediction:",
        row["predicted_label"],
        f"confidence={row['confidence']:.3f}",
        f"type={row['evidence_type']}",
    )
    print("Rationale:", row["rationale"])
PY
```
