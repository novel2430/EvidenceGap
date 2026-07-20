# EvidenceGap 人工标注规范 v0

## 1. 标注目标

将复杂结论转化为一张可审计的 Claim–Inference–Evidence Graph，并明确指出：

- 哪些内容是直接事实；
- 哪些内容是规则、计算结果或假设；
- 哪些结论依赖联合前提；
- 哪些推理步骤尚未完成；
- 当前证据最多允许表达多强的结论。

## 2. Atomic Claim

Atomic Claim 应满足：

1. 存在一个可以独立判断的核心命题；
2. 不应同时包含两个可以分别被支持或反驳的事实；
3. 时间、地区、人群、比较对象和强度属于 Claim 的范围；
4. “A 导致 B”通常至少需要区分：
   - A 是否发生；
   - B 是否发生；
   - A 与 B 是否存在因果关系。

### 不建议的 Claim

> 药物进入集采后提高了采购确定性并降低了推广收益，因此应转向零售。

它至少包含三个事实或推论和一个建议。

## 3. Claim 类型

- `OBSERVED_FACT`
  - 原始资料直接记载的事实。
- `COMPUTED_METRIC`
  - 由结构化数据或明确计算过程得到的指标。
- `DOMAIN_RULE`
  - 法规、政策、临床规则或明确的领域规则。
- `ASSUMPTION`
  - 推理需要，但当前尚未验证的前提。
- `INFERRED_CLAIM`
  - 由其他 Claim 通过 Inference Step 推导。
- `RECOMMENDATION`
  - 关于行动、资源配置或决策的主张。

## 4. Inference Step

Inference Step 是独立对象，不是普通图边。

它表示：

> 一组前提在某种推理规则下，共同支持一个结论。

至少标注：

- 前提 Claim；
- 结论 Claim；
- 推理类型；
- 规则说明；
- 是否存在隐含假设；
- 是否依赖专家判断。

## 5. Evidence

Evidence 必须是可定位的原始材料片段，而不是只保存整篇文档标题。

至少保存：

- Source ID；
- 来源类型；
- 原文；
- Evidence Span；
- 时间；
- 地区；
- 人群；
- 是否为 synthetic；
- 能够支持的最大范围。

“主题相关”不等于“构成支持”。

## 6. Evidence Binding

Evidence Item 与 Claim 之间通过 Evidence Binding 连接。

Binding Role：

- `SUPPORTING`
- `REFUTING`
- `CONTEXT`
- `RULE_SUPPORT`
- `SCOPE_LIMITATION`

同一 Evidence 可绑定多个 Claim；同一 Claim 可有多项支持和反驳 Evidence。

## 7. Verification 状态

- `SUPPORTED`
- `PARTIAL`
- `UNKNOWN`
- `CONTRADICTED`
- `CONFLICTED`
- `BLOCKED`

状态必须配合原因代码，例如：

- `NOT_EVALUATED`
- `NO_EVIDENCE_FOUND`
- `OUTSIDE_CORPUS`
- `PRIVATE_DATA_REQUIRED`
- `SCOPE_MISMATCH`
- `INFERENCE_RULE_MISSING`
- `DECISION_CONTEXT_MISSING`
- `CAUSALITY_NOT_ESTABLISHED`
- `CONFLICTING_EVIDENCE`
- `NOT_EMPIRICALLY_VERIFIABLE`

## 8. Gap 类型

- `RETRIEVAL_GAP`
- `EVIDENCE_GAP`
- `SCOPE_GAP`
- `INFERENCE_GAP`
- `ASSUMPTION_GAP`
- `CAUSAL_GAP`
- `DECISION_GAP`
- `PRIVATE_DATA_GAP`
- `CONFLICT_GAP`

Gap 必须关联具体 Claim 或 Inference Step，不能只写“证据不足”。

## 9. Recommendation 与 Decision Context

Recommendation 不能只因为若干事实为真就被视为已证明。

至少检查：

- 决策目标；
- 优化指标；
- 可选行动；
- 成本；
- 风险；
- 资源约束；
- 时间范围。

若缺失，应产生 `DECISION_CONTEXT_MISSING` 或 `DECISION_GAP`。

## 10. Safe Conclusion

安全结论只能执行以下弱化操作：

- 缩小时间、地区或人群范围；
- 从整体降为局部；
- 从因果降为相关；
- 从确定降为可能；
- 从立即执行降为进一步评估；
- 删除无支持子结论；
- 明确增加限制条件与缺失信息。

不得增加：

- 新事实；
- 新数字；
- 新因果关系；
- 新行动建议；
- 原始图中不存在的前提。

## 11. 双人标注流程

1. 两名标注者独立完成。
2. 不查看对方版本。
3. 自动检查 Schema。
4. 比较 Claim、类型、前提、Gap 与安全结论。
5. 记录原始分歧。
6. 仲裁。
7. 修改规范。
8. 重标受影响案例。

任何重大分歧都不得只在最终文件中被覆盖。
