# EvidenceGap V1 Phase 05 Final Report

**Phase：** 05 — Evidence Sentence Retrieval  
**狀態：** Completed / Frozen  
**完成日期：** 2026-07-22

---

## 1. Executive Summary

Phase 05 已完成在 EvidenceBench-100k 上的文章內證據句檢索與正式 Dev/Test 評估。

最終配置：

```text
BMRetriever Top-20
+
MedCPT Top-20
→ equal-weight RRF
→ k = 10
→ default evidence output Top-5
```

此配置在 Full Dev 的主要指標與最強單模型 BMRetriever 實質持平，同時改善 Recall@Optimal、Sentence Precision@5 與 First-hit MRR。配置在 Test 前凍結，並於 Full Test 20,000 queries 上四項指標全面優於 BMRetriever。

Cross-Encoder 在較大 Dev 樣本上穩定退化，已從最終 pipeline 淘汰。

---

## 2. Task Contract

```text
Input:
  hypothesis
  known paper candidate sentences

Output:
  ranked original zero-based sentence indices
```

正式輸出保留：

- `query_id`
- `paper_id`
- `sentence_index`
- `sentence_type`
- `sentence_text`
- `retrieval_model`
- `retrieval_score`
- `retrieval_rank`
- `final_score`
- `final_rank`

不重新切句、不刪除 heading、不修改原始 sentence index，也不將 gold/evaluation 欄位送入 scorer。

---

## 3. Evaluation Contract

正式主要指標：

- `Aspect Recall@5`
- `Aspect Recall@Optimal`

輔助指標：

- `Sentence Precision@5`
- `First-hit MRR`

正式 metrics 只對 `coverable_aspect_ids` 計算。宣告但沒有 sentence mapping 的 `unmapped_aspect_ids` 保留供 audit，不計入不可達分母。

`Results Aspect Recall@5` 在現有 EvidenceBench Dev/Test 中沒有 eligible queries，因此報告為 `null`，不參與最終選型。

---

## 4. Frozen Configuration

```json
{
  "task": "evidence_sentence_retrieval",
  "left_model": "bmretriever",
  "right_model": "medcpt",
  "left_depth": 20,
  "right_depth": 20,
  "fusion": "equal_weight_rrf",
  "rrf_k": 10,
  "left_weight": 1.0,
  "right_weight": 1.0,
  "cross_encoder": null,
  "default_evidence_top_k": 5,
  "retain_full_unique_union": true
}
```

Test 執行後不再修改以上參數。

---

## 5. Final Results

### 5.1 Full Dev

Full Dev：`4,373` queries，`4,127` eligible queries。

| 配置 | Aspect Recall@5 | Aspect Recall@Optimal | Sentence Precision@5 | First-hit MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.15218 | 0.10714 | 0.14921 | 0.30293 |
| MedCPT | 0.19946 | 0.14132 | 0.20572 | 0.37199 |
| BMRetriever | **0.21007** | 0.14899 | 0.20916 | 0.37808 |
| **Frozen RRF `k=10`** | 0.20969 | **0.15181** | **0.21464** | **0.38838** |

相對 BMRetriever：

- Aspect Recall@5：`-0.00038`（實質持平）；
- Aspect Recall@Optimal：`+0.00281`；
- Sentence Precision@5：`+0.00548`；
- First-hit MRR：`+0.01029`。

### 5.2 Full Test

Full Test：`20,000` queries，`18,838` eligible queries。

| 配置 | Aspect Recall@5 | Aspect Recall@Optimal | Sentence Precision@5 | First-hit MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.14936 | 0.10615 | 0.15060 | 0.30575 |
| MedCPT | 0.19317 | 0.14059 | 0.20206 | 0.36889 |
| BMRetriever | 0.20253 | 0.14701 | 0.21043 | 0.37838 |
| **Frozen RRF `k=10`** | **0.20603** | **0.15057** | **0.21623** | **0.38590** |

RRF `k=10` 相對 BMRetriever：

| 指標 | Absolute Δ | Relative Δ | Cluster-bootstrap 95% CI |
|---|---:|---:|---:|
| Aspect Recall@5 | +0.00350 | +1.73% | [+0.00107, +0.00591] |
| Aspect Recall@Optimal | +0.00356 | +2.42% | [+0.00155, +0.00564] |
| Sentence Precision@5 | +0.00581 | +2.76% | [+0.00385, +0.00767] |
| First-hit MRR | +0.00752 | +1.99% | [+0.00426, +0.01078] |

四項 CI 均高於零，證明 frozen configuration 成功泛化。

---

## 6. Complementarity Finding

Full Test Top-20 candidate-level aspect coverage：

```text
BMRetriever: 0.46450
MedCPT:      0.44763
Union:       0.53873
```

每個 eligible query 平均有：

- `1.103` 條 BMR-only gold sentence；
- `0.988` 條 MedCPT-only gold sentence。

RRF 的增益來源是兩個 dense retrievers 的真實互補，而不是 raw score blending。RRF 只使用 source ranks，因此不受兩個 embedding 模型分數尺度不一致影響。

---

## 7. Rejected Configurations

### BM25

保留為 sparse baseline；Full Test 四項指標均顯著低於 frozen RRF。

### MedCPT single model

前排 Precision 與 MRR 良好，但 aspect coverage 低於 BMR；融合後可提供互補候選。

### BMRetriever single model

最強單模型。Full Dev Aspect Recall@5 略高於 frozen RRF，但 Test 四項指標皆低於 frozen RRF。

### MedCPT Cross-Encoder

在 Smoke-100 曾對 BMR 候選出現小幅提升，但在 Dev-1000 與融合候選上全面退化。其 article-relevance 偏好不能穩定優化 sentence-level aspect coverage，因此不採用。

### RRF `k=20` / `k=60`

兩者在 Full Dev 的 Precision@5 與 MRR 略高，但 Aspect Recall@5 與 Recall@Optimal 低於 `k=10`。為兼顧主要指標與下游 evidence quality，凍結 `k=10`。

---

## 8. Frozen Artifacts

### Dev

```text
artifacts/v1/evidence_sentence_retrieval/fusion/
rrf_bmr20_medcpt20_k10_full_dev/
├── ranked_sentences.parquet
└── run_manifest.json
```

Ranking SHA-256：

```text
1788e0660a6490f62ca5ca7fa232fab33ad61c060619a0d0bcea10fd57487a4c
```

Metrics report：

```text
reports/v1/
evidence_sentence_retrieval_rrf_bmr20_medcpt20_k10_full_dev_dev.json
```

### Test

```text
artifacts/v1/evidence_sentence_retrieval/fusion/
rrf_bmr20_medcpt20_k10_full_test/
├── ranked_sentences.parquet
└── run_manifest.json
```

Ranking SHA-256：

```text
8f69d853d9ebf8c906538f828e58737b160d2cf62cb49183004fba23ae753bd0
```

Metrics and comparison reports：

```text
reports/v1/evidence_sentence_retrieval_rrf_bmr20_medcpt20_k10_full_test_test.json
reports/v1/evidence_sentence_rrf_k10_vs_bmretriever_full_test.json
reports/v1/evidence_sentence_rrf_k10_vs_medcpt_full_test.json
reports/v1/evidence_sentence_rrf_k10_vs_bm25_full_test.json
reports/v1/evidence_sentence_complementarity_full_test.json
```

不需要新增 `artifacts/.../final/` 目錄；具名 run directory、manifest、report 與 checksum 即構成 frozen artifact。

---

## 9. Validation Status

Full Test final ranking：

```text
rows:                    560,943
queries:                  20,000
duplicate_index_queries:       0
rank_gap_queries:              0
invalid_scores:                0
missing_queries:               0
wrong_depth_queries:           0
```

Test canonical：

```text
queries:                    20,000
unique_papers:              19,068
candidate_sentences:     3,522,617
coverable_aspects:          109,674
unmapped_aspects:            56,166
empty_coverable_queries:      1,162
canonical_sha256:
f75ccf5010318908d3625cc9edbfc1c7b370ee1ae982188a6c524f3e82620022
```

Python multiprocessing 偶爾產生 `resource_tracker` semaphore cleanup warning，但所有正式 artifacts 均通過 coverage、rank、score 與 checksum 驗證，未造成資料缺失。

---

## 10. Phase 06 Handoff

Phase 06 開發使用：

```text
artifacts/v1/evidence_sentence_retrieval/fusion/
rrf_bmr20_medcpt20_k10_full_dev/ranked_sentences.parquet
```

每個 query 預設讀取 `final_rank <= 5` 的 evidence sentences，組成：

```text
hypothesis
+
Top-5 evidence sentence texts
→ stance verification
```

Phase 06 必須：

- 使用 Dev artifact 進行開發與分析；
- 保留 `query_id`、`paper_id`、`sentence_index` 與 `sentence_text`；
- 不重新切句或重新編號；
- 將 sentence-level stance 與 query-level aggregation 分開保存；
- 只在最終評估時使用 Test artifact。

---

## 11. Completion Assessment

| 驗收項目 | 狀態 |
|---|---|
| 能覆蓋重要 aspects | PASS |
| 保留原始 sentence index | PASS |
| 證據句包含可直接引用文字 | PASS |
| Dev 選型與 Test 驗證分離 | PASS |
| 最終 artifact 可重現、可校驗 | PASS |
| Results Aspect Recall@5 | N/A：原始欄位無 eligible data |
| Position-only ablation | Not run；實作未硬編碼文章位置 |

**Phase 05 status：COMPLETED。**

