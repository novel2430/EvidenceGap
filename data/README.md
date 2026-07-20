# EvidenceGap V0–V1 人工案例起始包

这是一套用于首轮产品与语义验证的人工案例数据。

## 内容

- `annotation_guideline_v0.md`
  - 第一版人工标注规范
  - 说明 Claim、Evidence、Inference Step、Gap 与 Safe Conclusion 的基本边界
- `v0_static_cases.yaml`
  - 3 个可直接用于 V0 前端静态展示的完整案例
- `v1_semantic_cases.yaml`
  - 10 个用于双人独立标注和语义分歧实验的案例
- `case_contract_draft.json`
  - V0/V1 共用的数据字段草案，不是最终 V2 Domain Model

## 使用建议

1. 先把 `v0_static_cases.yaml` 接入前端，确认图、Inspector 和 Gap Report 是否能被理解。
2. 再让两名标注者分别处理 `v1_semantic_cases.yaml`，不要先看参考答案。
3. 将两份标注结果进行比对，记录以下分歧：
   - Claim 拆分边界
   - Claim 类型
   - 必要前提
   - Inference Step
   - 主要 Gap
   - Safe Conclusion 强度
4. 完成仲裁后，再进入 V2 Domain Model 固化。

## 重要说明

- 所有医疗、政策、试验与商业材料均为 **synthetic**。
- 数据仅用于软件结构、交互和标注规范验证。
- 不能作为医学、政策或商业事实引用。
- 后续接入真实公开资料时，必须保留来源版本、适用时间、地区、人群与 Evidence Span。
