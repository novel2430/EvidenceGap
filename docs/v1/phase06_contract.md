# EvidenceGap V1 Phase 06：Stance Verification Contract

## 1. 任务边界

Phase 06 接收 Claim 与已经选出的 evidence，输出三分类方向：

```text
support / refute / insufficient
```

Phase 06 不重新检索文章、不重新切句、不把 Phase 05 relevance score 解释为 stance probability，也不负责跨文章最终 Verdict。

正式三分类任务 ID：

```text
STANCE-EVIDENCE-3
```

Schema version：

```text
1.0.0
```

## 2. 统一输入

`StanceInputRecord` 同时支持两种 evidence unit：

```text
sentence
bundle
```

关键字段：

| 字段 | 说明 |
|---|---|
| `input_id` | 稳定的 stance 输入 ID |
| `input_fingerprint` | Claim、Evidence 与可选上下文的 SHA-256 |
| `claim_id` / `query_id` | Claim 与上游 query 的稳定关联 |
| `claim_text` | 待验证 Claim |
| `paper_id` | 来源论文；HealthFC bundle 可为空 |
| `sentence_index` | Phase 05 原始 zero-based index；bundle 可为空 |
| `evidence_rank` | Phase 05 final rank；不能当作 stance 标签 |
| `evidence_text` | 原始 evidence，不重新切分或改写 |
| `evidence_unit` | `sentence` 或 `bundle` |
| `retrieval_model` / `retrieval_score` | 上游 provenance |
| `source_artifact_sha256` | 上游 ranking/manifest checksum |
| `gold_label` | 可空；EvidenceBench 为 null，HealthFC 有三分类 gold |
| `source_locator_json` | 可回溯到原始 artifact/row 的 JSON |

`context_before` 与 `context_after` 被保留为可空字段，供后续 LLM judge 或 context-aware verifier 使用；06.2 baseline 只使用 `evidence_text`。

## 3. 统一输出

`StancePredictionRecord` 完整保留输入字段，并增加：

```text
run_name
model_name
model_fingerprint
stance_input_artifact_sha256
predicted_label
probability_support
probability_refute
probability_insufficient
confidence
probability_margin
abstained
```

三项概率必须有限、位于 `[0, 1]` 且总和为 1。`predicted_label` 必须是概率 argmax。Zero-shot baseline 不执行拒绝判断，因此 `abstained=false`。

## 4. DeBERTa Zero-shot NLI 契约

模型：

```text
cross-encoder/nli-deberta-v3-base
```

输入顺序不可颠倒：

```text
premise    = evidence_text
hypothesis = claim_text
```

模型标签由本地 `config.id2label` 读取并严格验证，不写死任意未知顺序：

```text
entailment    → support
contradiction → refute
neutral       → insufficient
```

若模型 config 无法明确识别这三个标签，runner 直接失败。

## 5. 数据职责

### Phase 05 Dev Top-5

用于生成真实 Pipeline 的句子级 stance 输入。没有 stance gold，不报告准确率，只保存预测分布与置信度诊断。

### HealthFC

使用 raw `en_top_sentences` 作为完整 evidence bundle，不猜测 delimiter，也不重新分句。用于 external expert bundle-level 三分类评测。

### Test 隔离

`prepare-phase05 --split test` 默认失败。只有最终冻结评测时才能显式传入：

```text
--allow-test
```

## 6. LLM Structured Judge 扩展

Phase 06.3 继续写入同一个 `StancePredictionRecord`，并使用以下 backward-compatible nullable 字段：

```text
rationale
evidence_type
requires_context
provider
provider_request_id
raw_response_sha256
prompt_version
```

其中 `evidence_type` 只允许：

```text
direct_result
background
method
population_or_scope
safety
statistical_uncertainty
mixed_or_other
```

Prompt version `phase06_llm_stance_v2` 明确要求模型从上述七类中严格单选，并按以下优先级处理重叠类别：

```text
safety
> statistical_uncertainty
> population_or_scope
> direct_result
> method
> background
> mixed_or_other
```

`mixed_or_other` 只作为最后兜底。Provider 返回的常见同义词仍会规范化到正式 taxonomy；未知的 `evidence_type` 会降级为 `mixed_or_other`，不会让已成功的 stance 判断整批失败或重复计费。

LLM judge 返回的 `probability_*` 是模型自我报告概率，只用于排序、展示和比较，不宣称已经完成 calibration。

LLM judge 的正式判断仍然只有：

```text
support / refute / insufficient
```

`rationale` 只解释结构化标签，不能修改标签，也不能引入 evidence 中不存在的来源或事实。

### LLM label/probability reconciliation

The explicit `label` is treated as the primary stance decision. If an LLM returns valid probabilities whose argmax disagrees with that label, the runner reconciles the probabilities locally, records the event in `probability_reconciled_rows` and `probability_reconciliation_counts`, and does not retry the API request. This prevents auxiliary probability inconsistencies from consuming repeated API calls.
