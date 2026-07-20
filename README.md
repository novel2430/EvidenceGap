# EvidenceGap

EvidenceGap 是医疗结论证据链审计与缺口分析系统。

当前开发阶段：V0 静态产品垂直切片。

## 目录

- `data/`：人工案例、临时契约、标注规范与布局数据
- `frontend/`：React 前端应用
- `backend/`：V2 后端领域模型预留位置
- `docs/`：架构、产品验证与设计决策文档

## 当前边界

V0 只使用明确标记为 synthetic 的人工案例，不包含真实医疗分析能力。

当前不实现：

- LLM
- 自动 Claim 拆解
- Evidence Retrieval
- NLI Verification
- 自动状态传播
- Gap Engine
- 数据库或正式 API
