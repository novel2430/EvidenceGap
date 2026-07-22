# EvidenceGap

EvidenceGap 是医疗 Claim 的文献检索、证据定位、立场判断与证据图分析系统。

当前分析流水线：

```text
Medical Claim
→ Article Retrieval
→ Article Reranking
→ Evidence Sentence Retrieval
→ Stance Verification
→ Evidence Graph
```

## 当前进度

- Phase 00–04：数据契约、文章语料、Sparse/Dense Retrieval 与 Article Reranking。
- Phase 05：EvidenceBench 固定候选池中的 Evidence Sentence Retrieval。
- Phase 06.1：统一的 sentence/bundle stance input/output schema。
- Phase 06.2：`cross-encoder/nli-deberta-v3-base` zero-shot NLI baseline。
- Phase 06.3：DeepSeek／Claude structured LLM stance judge。

Phase 06 使用：

```bash
python scripts/run_v1_phase06.py --help
```

详细契约与命令：

```text
docs/v1/phase06_contract.md
docs/v1/phase06_usage.md
```

## 目录

- `src/evidencegap/`：V1 离线分析模块。
- `scripts/`：各 Phase 的可执行入口。
- `configs/v1/`：冻结配置与 baseline 配置。
- `data/`：数据契约、manifest 与本地 raw/processed 数据位置。
- `frontend/`：Evidence Graph 产品展示壳。
- `backend/`：后续端到端 API 位置。
- `docs/v1/`：任务契约、实验使用说明与阶段报告。

## 当前边界

Phase 06.2 是 zero-shot NLI baseline，不是已经校准的医学最终 Verdict 模型。当前仍不实现：

- LLM Claim 拆解；
- LLM evidence judge；
- 跨文章证据质量加权；
- 最终 Verdict 聚合；
- 正式端到端 API。
