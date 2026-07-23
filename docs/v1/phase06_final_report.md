# EvidenceGap V1 Phase 06 Final Report

**Phase：** 06 — Stance Verification  
**状态：** Completed / Frozen  
**完成日期：** 2026-07-23

---

## 1. Executive Summary

Phase 06 已完成医疗 Claim 与候选 evidence 之间的三分类 stance 判断，并将结果整理为可追溯、可供 Evidence Graph 与前端直接消费的正式 artifact。

正式任务：

```text
Claim + Evidence
→ support / refute / insufficient
```

Phase 06 不重新检索文章、不重新切句、不把 Phase 05 relevance score 解释为 stance probability，也不负责跨文章最终医学 Verdict。

Phase 06 首先实现 DeBERTa-v3 zero-shot NLI baseline，随后实现 DeepSeek 与 Anthropic 的 structured LLM judge，并在 HealthFC 750 条专家标注 evidence bundle 上完成外部比较。

最终冻结配置：

```text
Primary judge     DeepSeek V4 Pro
Fast mode         DeepSeek V4 Flash
Prompt            phase06_llm_stance_v2
Formal labels     support / refute / insufficient
Evidence taxonomy 7-class auxiliary type
Graph contract    phase06.graph-ready.v1.1
```

HealthFC 最终结果：

| 方法 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| DeBERTa-v3 zero-shot | 0.4987 | 0.2641 | 0.3107 |
| DeepSeek V4 Flash | 0.7440 | 0.7392 | 0.7820 |
| **DeepSeek V4 Pro** | **0.7600** | **0.7506** | **0.7899** |
| Claude Sonnet 5 | 0.7040 | 0.7087 | 0.7643 |

DeepSeek V4 Pro 在当前 Prompt 与 HealthFC 标签口径下取得最佳观察结果，因此被选为 Phase 06 主模型。

此外，Phase 06 已对 Phase 05 frozen Dev Top-5 evidence 运行大规模实际推理，并导出：

```text
2,520 / 4,373 complete queries
12,600 / 21,865 sentence judgments
57.63% Dev coverage
```

该结果明确记录为 **large-scale partial Dev inference artifact**，不冒充 Full Dev，也不用于报告 sentence-level accuracy，因为 EvidenceBench 没有 stance gold。

---

## 2. Task Contract

正式任务 ID：

```text
STANCE-EVIDENCE-3
```

### 2.1 Input

Phase 06 接收统一 `StanceInputRecord`，支持：

```text
sentence
bundle
```

关键字段：

- `input_id`
- `input_fingerprint`
- `claim_id`
- `query_id`
- `claim_text`
- `paper_id`
- `sentence_index`
- `sentence_type`
- `evidence_rank`
- `evidence_text`
- `context_before`
- `context_after`
- `retrieval_model`
- `retrieval_score`
- `source_run_name`
- `source_artifact_sha256`
- `gold_label`
- `source_locator_json`

Phase 05 sentence-level 输入必须保留原始 zero-based `sentence_index` 和未经改写的 `sentence_text`。相邻上下文只用于解析指代、比较对象与局部语义，不能替代目标 evidence sentence。

### 2.2 Output

统一 `StancePredictionRecord` 在完整保留输入 provenance 的基础上增加：

```text
predicted_label
probability_support
probability_refute
probability_insufficient
confidence
probability_margin
abstained
rationale
evidence_type
requires_context
provider
provider_request_id
raw_response_sha256
prompt_version
model_fingerprint
```

正式 stance 标签仅有：

```text
support
refute
insufficient
```

LLM 自报概率用于排序、展示与诊断，不宣称已经完成统计校准，也不能解释为医学可信度。

---

## 3. Auxiliary Evidence Taxonomy

LLM judge 同时输出一个前端与错误分析用途的 evidence type：

```text
direct_result
background
method
population_or_scope
safety
statistical_uncertainty
mixed_or_other
```

Prompt 要求严格七选一，并使用以下优先级处理重叠类别：

```text
safety
> statistical_uncertainty
> population_or_scope
> direct_result
> method
> background
> mixed_or_other
```

该字段不是正式 stance 标签。Provider 返回常见同义词时会本地正规化；未知辅助类型降级为 `mixed_or_other`，不会因为非核心字段漂移而重复请求 API。

---

## 4. Implemented Systems

### 4.1 DeBERTa Zero-shot Baseline

模型：

```text
cross-encoder/nli-deberta-v3-base
```

输入顺序：

```text
premise    = evidence_text
hypothesis = claim_text
```

NLI 映射：

```text
entailment    → support
contradiction → refute
neutral       → insufficient
```

模型标签从本地 `config.id2label` 读取并验证，不写死未知类别顺序。

### 4.2 LLM Structured Judge

支持 Provider：

```text
DeepSeek
Anthropic
```

DeepSeek 使用 JSON mode，并在本地执行严格 schema 校验与容错恢复；Anthropic 使用原生 Structured Outputs JSON Schema。

正式英文 Prompt 强调：

```text
Relevance is not support.
A lack of support is not automatically a refutation.
Judge only from the supplied evidence.
Do not use outside medical knowledge.
Match population, intervention/exposure, comparator, outcome, direction and scope.
Do not convert association into causation.
```

Prompt 与 rationale 均使用英文。

### 4.3 Reliability and Cost Controls

实现包括：

- request-level persistent cache；
- deterministic request fingerprint；
- interruption-safe resume；
- per-batch token usage tracking；
- malformed auxiliary field normalization；
- label/probability reconciliation；
- blank/invalid label recovery from valid probability argmax；
- partial cache export without API calls；
- query-level deterministic sampling；
- API-free dry-run planning。

若明确 `label` 与自报概率 argmax 不一致，以 label 为正式判断，本地调整概率并记录 reconciliation，不重新收费请求。

---

## 5. HealthFC External Evaluation

### 5.1 Evaluation Data

HealthFC 全部 750 条 expert-annotated samples 用作外部评测，不参与主要训练或 Prompt 标签拟合。

Gold label 分布：

| Label | Count |
|---|---:|
| support | 202 |
| refute | 125 |
| insufficient | 423 |

输入使用 HealthFC gold evidence bundle，而不是 Phase 05 sentence-level retrieval output。因此该评测代表 expert bundle-level stance/generalization，不等价于 Phase 05 单句准确率。

HealthFC stance input artifact：

```text
artifacts/v1/stance_verification/inputs/healthfc_eval/
├── stance_inputs.parquet
└── run_manifest.json
```

Input SHA-256：

```text
0a614c5943255d5b3a66464442bc39a7c9e998a98b7653cc73e5b1d2b3304722
```

---

## 6. HealthFC Final Results

### 6.1 Overall Comparison

| 方法 | Accuracy | Macro-F1 | Balanced Accuracy |
|---|---:|---:|---:|
| DeBERTa-v3 zero-shot | 0.49867 | 0.26406 | 0.31070 |
| DeepSeek V4 Flash | 0.74400 | 0.73920 | 0.78197 |
| **DeepSeek V4 Pro** | **0.76000** | **0.75065** | **0.78985** |
| Claude Sonnet 5 | 0.70400 | 0.70870 | 0.76427 |

相对 DeBERTa baseline，DeepSeek V4 Pro：

```text
Accuracy          +0.26133
Macro-F1          +0.48658
Balanced Accuracy +0.47915
```

### 6.2 DeBERTa Zero-shot

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| support | 0.15385 | 0.05941 | 0.08571 |
| refute | 0.27273 | 0.02400 | 0.04412 |
| insufficient | 0.54312 | 0.84870 | 0.66236 |

Prediction distribution：

```text
support       78
refute        11
insufficient 661
```

DeBERTa 几乎将所有样本判为 neutral/insufficient，并且平均 confidence 为 0.9744。该结果表明 generic zero-shot NLI 与专家医疗 stance 任务严重不匹配，只保留为传统 baseline。

### 6.3 DeepSeek V4 Flash

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| support | 0.95954 | 0.82178 | 0.88533 |
| refute | 0.43802 | 0.84800 | 0.57766 |
| insufficient | 0.85373 | 0.67612 | 0.75462 |

Confusion matrix，rows = gold，columns = prediction：

```text
                 support  refute  insufficient
support              166       4            32
refute                 2     106            17
insufficient            5     132           286
```

### 6.4 DeepSeek V4 Pro — Frozen Primary

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| support | 0.96429 | 0.80198 | 0.87568 |
| refute | 0.45532 | 0.85600 | 0.59444 |
| insufficient | 0.86744 | 0.71158 | 0.78182 |

Confusion matrix：

```text
                 support  refute  insufficient
support              162      10            30
refute                 2     107            16
insufficient            4     118           301
```

相对 Flash：

```text
Accuracy          +0.0160
Macro-F1          +0.0114
Balanced Accuracy +0.0079
```

Pro 将 `gold insufficient → predicted refute` 从 Flash 的 `132` 降至 `118`，并提高 insufficient recall 与 F1，因此被选为主模型。

### 6.5 Claude Sonnet 5

Sonnet 5 使用 Structured Outputs，且显式关闭 thinking。

| Label | Precision | Recall | F1 |
|---|---:|---:|---:|
| support | 0.97590 | 0.80198 | 0.88043 |
| refute | 0.38014 | 0.88800 | 0.53237 |
| insufficient | 0.87329 | 0.60284 | 0.71329 |

Confusion matrix：

```text
                 support  refute  insufficient
support              162      15            25
refute                 2     111            12
insufficient            2     166           255
```

Sonnet 5 对 support precision 与 refute recall 很高，但更倾向把 evidence uncertainty 或 lack of support 判为 refute，因此总体低于 DeepSeek V4 Flash 与 Pro。

---

## 7. Main Error Pattern

所有强 LLM judge 的主要错误均集中在：

```text
Gold insufficient
→ Predicted refute
```

计数：

| 模型 | Insufficient → Refute |
|---|---:|
| DeepSeek V4 Flash | 132 |
| **DeepSeek V4 Pro** | **118** |
| Claude Sonnet 5 | 166 |

这说明模型容易将：

```text
not proven
not statistically significant
uncertain
limited evidence
```

误解为对 affirmative Claim 的直接反驳。

Prompt v2 已明确区分 lack of support 与 refutation，但该边界仍是 Phase 06 的主要已知限制。

---

## 8. Frozen Configuration

```json
{
  "task": "stance_verification",
  "labels": ["support", "refute", "insufficient"],
  "primary_provider": "deepseek",
  "primary_model": "deepseek-v4-pro",
  "fast_model": "deepseek-v4-flash",
  "external_comparison_model": "claude-sonnet-5",
  "traditional_baseline": "cross-encoder/nli-deberta-v3-base",
  "prompt_version": "phase06_llm_stance_v2",
  "phase05_top_k": 5,
  "context_window": 1,
  "request_batch_size": 5,
  "max_retries": 1,
  "test_artifact_allowed": false
}
```

冻结依据：

```text
HealthFC expert-annotated external evaluation
```

Phase 05 Test artifact 未用于开发或模型选择。

---

## 9. Phase 05 Real Evidence Run

### 9.1 Input

来源：

```text
Phase 05 frozen Full Dev ranking
BMRetriever Top-20 + MedCPT Top-20
→ equal-weight RRF, k=10
→ Top-5 sentences
```

输入配置：

```text
split          dev
top_k          5
context_window ±1 original canonical sentence
model          DeepSeek V4 Pro
prompt         phase06_llm_stance_v2
```

### 9.2 Coverage

Full Dev 理论规模：

```text
4,373 queries
21,865 sentence judgments
```

实际完成并正式导出的 partial artifact：

```text
2,520 complete queries
12,600 sentence judgments
57.6263% query coverage
57.6263% row coverage
incomplete cached queries = 0
```

该 artifact 由完整 Top-5 query 构成，没有半个 query 或 rank gap。

### 9.3 Token Usage

已缓存实际推理用量：

```text
Input tokens   4,032,919
Output tokens  1,533,260
Total tokens   5,566,179
```

Partial export 本身：

```text
API requests = 0
```

### 9.4 Stance Distribution

| Stance | Count | Share |
|---|---:|---:|
| insufficient | 8,397 | 66.64% |
| support | 3,752 | 29.78% |
| refute | 451 | 3.58% |

该分布说明 Phase 05 的高相关候选句中，大量内容仍只是背景、方法或不足以给出方向。Phase 06 的核心产品价值正是避免把所有 retrieval hits 错当成支持或反驳证据。

### 9.5 Evidence Type Distribution

| Evidence type | Count | Share |
|---|---:|---:|
| direct_result | 5,090 | 40.40% |
| background | 4,611 | 36.60% |
| method | 2,198 | 17.44% |
| population_or_scope | 358 | 2.84% |
| mixed_or_other | 141 | 1.12% |
| statistical_uncertainty | 135 | 1.07% |
| safety | 67 | 0.53% |

`background + method` 占 `54.04%`，进一步证明相关句不等于 stance-bearing evidence。

### 9.6 Diagnostics

```text
mean confidence          0.88937
mean probability margin  0.81120
requires_context rows    349 / 12,600
abstained rows           0
validation               PASS
```

这些 confidence 来自 LLM 自报，只适合展示或相对排序。

---

## 10. Phase 06.6 Frozen Artifact

```text
artifacts/v1/stance_verification/llm_judge/
deepseek_v4_pro_phase05_top5_ctx1_dev_partial_cache/
├── stance_predictions.parquet
└── run_manifest.json
```

Prediction SHA-256：

```text
8ebf654713721a5f8b6c9c38b75266e6f4e666a418ef2b4bf36f9564aee931b9
```

Source stance input SHA-256：

```text
ca0777b0a4e0e66a21c1dd69665f223a712fa8795e5365a1e585102d9315cafb
```

Reports：

```text
reports/v1/
stance_llm_deepseek_v4_pro_phase05_top5_ctx1_dev_partial_cache.json
stance_llm_deepseek_v4_pro_phase05_top5_ctx1_dev_partial_cache.md
```

---

## 11. Graph-ready Export

Graph schema：

```text
schema_version  1.1.0
contract_id     phase06.graph-ready.v1.1
```

输出：

```text
artifacts/v1/stance_verification/graph_ready/
deepseek_v4_pro_phase05_top5_ctx1_dev_partial_graph/
├── query_summaries.parquet
├── paper_summaries.parquet
├── graph_nodes.parquet
├── graph_edges.parquet
├── graph_bundles.jsonl
└── run_manifest.json
```

### 11.1 Graph Structure

Nodes：

```text
Claim             2,520
Article           2,520
Evidence Sentence 12,600
Total             17,640
```

Edges：

```text
Claim → Article         retrieved_from   2,520
Article → Evidence      contains        12,600
Evidence → Claim        supports         3,752
Evidence → Claim        refutes            451
Evidence → Claim        insufficient     8,397
Total                                  27,720
```

Validation：

```text
status  PASS
graphs  2,520
bundles 2,520
API requests during export 0
```

### 11.2 Transparent Aggregation

Rank weight：

```text
rank_weight = 1 / Phase 05 evidence_rank
```

Stance mass：

```text
support_mass      = Σ rank_weight × probability_support
refute_mass       = Σ rank_weight × probability_refute
insufficient_mass = Σ rank_weight × probability_insufficient
```

以上只是透明证据质量摘要，不是最终医学 Verdict。

### 11.3 Directional Semantics

`directional_evidence_pattern`：

```text
support_only
refute_only
mixed
none
```

它只描述 hard-label 方向性证据是否存在，不表示整体结论。

分布：

| Pattern | Count | Share |
|---|---:|---:|
| support_only | 1,448 | 57.46% |
| refute_only | 145 | 5.75% |
| mixed | 171 | 6.79% |
| none | 756 | 30.00% |

`directional_mass_share`：

```text
(support_mass + refute_mass)
/
(support_mass + refute_mass + insufficient_mass)
```

该字段用于区分：

```text
有方向的证据偏支持
```

与：

```text
整体证据已经充分支持
```

例如：

```text
directional_evidence_pattern = support_only
mass_leader                 = insufficient
directional_mass_share      = 0.16
```

表示只有少量方向性证据偏支持，但整体仍由信息不足主导。

`directional_margin` 只描述 support 与 refute 两方之间的方向差异，不能显示为总体 confidence。若 pattern = `none`，前端不应显示支持／反驳方向。

---

## 12. Engineering Validation

Phase 06 已验证：

```text
HealthFC input schema                       PASS
Stance prediction schema                   PASS
Provider structured output validation      PASS
Persistent cache and resume                PASS
No duplicate input IDs                     PASS
Phase 05 original sentence indices         preserved
Phase 05 original sentence text            preserved
Complete-query partial cache export        PASS
Graph node/edge cardinality                PASS
Graph relation integrity                   PASS
Artifact checksums and manifests           present
Test artifact isolation                    preserved
```

Graph cardinality invariant：

```text
nodes = claims + articles + evidence
      = 2,520 + 2,520 + 12,600
      = 17,640

edges = retrieved_from + contains + stance
      = 2,520 + 12,600 + 12,600
      = 27,720
```

---

## 13. Architectural Decision

原始 V1 规划倾向使用可训练 DeBERTa verifier，并限制 LLM 只负责表达。

Phase 06 的正式实验显示：

```text
DeBERTa zero-shot Macro-F1  0.2641
DeepSeek V4 Pro Macro-F1    0.7506
```

因此 Phase 06 将 structured LLM judge 冻结为当前 V1 stance verifier。该变更基于 HealthFC 专家标注外部评测，而不是仅凭人工挑选 Demo。

LLM 输出仍受以下约束：

- 只能使用提供的 Claim 与 evidence；
- 必须输出固定 JSON schema；
- rationale 不能修改 label；
- 不得伪造来源；
- 所有预测均保存 provider、model、Prompt、request ID 与 response checksum；
- 后续最终 Verdict 必须使用透明聚合规则，而不是让 LLM 自由覆盖结构化结果。

---

## 14. Known Limitations

### 14.1 HealthFC 与实际句子输入存在粒度差异

HealthFC 使用 expert gold evidence bundle；Phase 05 实际输入是单句加相邻上下文。HealthFC 数字不能宣称为 Phase 05 sentence-level accuracy。

### 14.2 Phase 05 Dev 只完成 Partial Coverage

实际 stance artifact 覆盖 `57.63%` Dev，不是 Full Dev。后续报告必须保留 `partial` 标记。

### 14.3 没有 Phase 05 Sentence-level Stance Gold

EvidenceBench 提供 evidence sentence/aspect 标注，但没有 support/refute/insufficient gold，因此实际 Phase 05 输出只能做分布、案例与 Graph-ready validation，不能报告 accuracy 或 F1。

### 14.4 Refute Over-prediction

所有强 LLM 模型都存在将 uncertain/unsupported evidence 判为 refute 的倾向。DeepSeek V4 Pro 是比较中最优，但该问题没有完全消失。

### 14.5 LLM Probabilities 未校准

自报 confidence 与 probabilities 不能当作临床风险或统计可信度。Graph 聚合只能把它们视为透明启发式权重。

### 14.6 Graph Summary 不是最终 Verdict

`mass_leader`、`directional_evidence_pattern`、`directional_margin` 与 `directional_mass_share` 都是证据分布摘要。它们不能直接宣称医学 Claim 已被证实或推翻。

---

## 15. Phase Completion Decision

```text
Phase 06: COMPLETE
```

完成条件：

1. 统一 stance input/output schema：完成；
2. 传统 zero-shot baseline：完成；
3. LLM structured judge：完成；
4. HealthFC 专家数据外部评测：完成；
5. 模型与 Prompt 冻结：完成；
6. Phase 05 实际 evidence 大规模运行：完成，partial Dev 57.63%；
7. Graph-ready query/paper/node/edge export：完成；
8. Checksum、manifest、report 与 validation：完成；
9. Test isolation：保持；
10. 与后续 Evidence Graph 的稳定接口：完成。

---

## 16. Handoff to Phase 07

Phase 07 可以直接消费：

```text
stance_predictions.parquet
query_summaries.parquet
paper_summaries.parquet
graph_nodes.parquet
graph_edges.parquet
graph_bundles.jsonl
```

Phase 07 应负责：

```text
Article Retrieval
→ Article Reranking
→ Evidence Sentence Retrieval
→ Stance Verification
→ Multi-article aggregation
→ Evidence Graph / product pipeline
```

Phase 07 不应重新打开 Phase 06 模型选型，除非另行定义独立 ablation。

推荐正式配置：

```text
Primary stance judge  DeepSeek V4 Pro
Fast mode             DeepSeek V4 Flash
Prompt                phase06_llm_stance_v2
Graph contract        phase06.graph-ready.v1.1
```

前端展示时必须同时呈现：

- support/refute/insufficient evidence；
- 原始论文与 sentence index；
- rationale；
- evidence type；
- directional evidence pattern；
- directional mass share；
- information-insufficient mass；
- partial-run provenance。

不得只显示一个模糊的最终分数来掩盖证据不足或支持／反驳并存。

