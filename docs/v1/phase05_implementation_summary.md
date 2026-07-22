# Phase 05 第一輪實作摘要

## 已完成

- manifest-aware EvidenceBench loader，按 `raw_locator` 以 `ijson` 串流抽取 records；小於 16 MiB 的 fixture 才允許標準庫 fallback。
- canonical EvidenceQuery artifact、ordered pool fingerprint、raw/download provenance 與 source-manifest stale detection。
- 嚴格 sentence order/index、sentence type、gold index、forward/reverse aspect mapping 與 paper/pool conflict validation。
- gold-free model-input projection；results evaluation fields 不進 canonical scorer input。
- Aspect Recall@5、official per-query Aspect Recall@Optimal、Results Aspect Recall@5、Sentence Precision@5、First-hit MRR。
- paper-local BM25 baseline，沿用 Phase 02 tokenization contract。
- MedCPT 與 BMRetriever sentence embedding/scoring backend，pool/query persistent cache。
- MedCPT Cross-Encoder sentence reranking，僅重排 retrieval Top-N，保持 candidate identity。
- deterministic multi-GPU sharding、device grouping、atomic shard、resume signature、stale/corrupt rejection。
- ranked sentence Parquet、run manifest、JSON report、score-direction diagnostic 與 artifact validator。
- `scripts/run_v1_phase05.py` 的 audit/prepare/bm25/dense/rerank/evaluate/diagnose/validate commands。
- `docs/v1/evaluation_contract.md` 與 `docs/v1/data_contract.md` 已固定 official Optimal 語意。

## 本地驗證

在不包含正式 raw dataset 與模型權重的交付環境中完成：

- 全部 Python production modules `compileall` 通過。
- Phase 05 全部 CLI subcommand `--help` 通過。
- synthetic EvidenceBench object raw JSON：audit、prepare、canonical validate 通過。
- mapping mismatch、short Top-5、source-manifest stale 均能正確拒絕。
- BM25 synthetic end-to-end：ranking、manifest、五項 evaluation、diagnose、resume、stale parameter 與 checksum corruption 驗證通過。
- deterministic fake Dense encoder end-to-end：2 shards（含 empty shard）、embedding cache、resume、ranking/evaluation 通過。
- deterministic fake Cross-Encoder end-to-end：2 shards、Top-N rerank、candidate-set preservation、ranking/evaluation 通過。

## 尚需在正式伺服器執行

交付 zip 不包含 EvidenceBench-100k raw files、CUDA 模型與正式 manifest artifacts，因此尚未在此環境執行真實 100-query GPU smoke。建議依 `phase05_usage.md` 順序執行：

```text
Dev 100 audit/prepare
→ BM25
→ MedCPT
→ BMRetriever
→ Cross-Encoder
→ diagnose
```

第一輪禁止使用 official Test 選型。
