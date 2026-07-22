# EvidenceGap V1 Phase 05 Comparison Report

**任務：** Evidence Sentence Retrieval  
**資料集：** EvidenceBench-100k  
**完成日期：** 2026-07-22  
**最終選定配置：** BMRetriever Top-20 + MedCPT Top-20 → equal-weight RRF，`k=10`

---

## 1. 報告目的

Phase 05 的目標是在文章已知的條件下，依據醫療 hypothesis 對文章內原始候選句進行排序，輸出最可能承載證據的句子索引，供 Phase 06 Stance Verification 使用。

```text
Input:
  hypothesis
  paper_as_candidate_pool

Output:
  ranked original zero-based sentence indices
```

本報告整理：

1. 資料契約與評測口徑；
2. BM25、MedCPT、BMRetriever、Cross-Encoder 與 RRF 的比較；
3. Smoke-100、Dev-1000、Full Dev 與 Full Test 的實驗演進；
4. Cross-Encoder 淘汰原因；
5. RRF `k=10` 的最終選型依據。

---

## 2. 資料與評測契約

### 2.1 不變條件

Phase 05 嚴格保留 EvidenceBench 原始候選池：

- 不重新切句；
- 不刪除 section headings；
- 不對候選句做文字去重；
- 不重新編號；
- 排名輸出的 `sentence_index` 保持原始 zero-based index；
- gold/evaluation 欄位不進入模型 scorer。

### 2.2 Aspect schema 修正

EvidenceBench 中部分宣告的 aspects 沒有任何 sentence mapping。因此 canonical schema 採用：

```text
aspect_ids
coverable_aspect_ids
unmapped_aspect_ids
```

正式 aspect metrics 與 exact minimum set cover 只以 `coverable_aspect_ids` 為分母；無 mapping 的 aspects 保留在 canonical artifact 中供 audit，但不將不可達標註計入模型失分。

### 2.3 Optimal budget

原始資料沒有可直接使用的 `evidence_retrieval_at_optimal_evaluation`。Phase 05 以 coverable aspects 與 sentence mappings 執行 exact minimum set cover，為每個可評估 query 推導最小 evidence budget。

### 2.4 Test duplicate mapping

Full Test 在 `test_1987_aspect_23` 首次發現同一 aspect mapping list 內重複 sentence index。這是集合型標註中的冗餘值，而非候選句重複。

Canonicalization 採用保序去重，同時仍執行：

- index 範圍驗證；
- aspect ID 驗證；
- forward/reverse mapping 一致性驗證。

原始資料不被改寫。

### 2.5 Metrics

正式主要指標：

- **Aspect Recall@5**：Top-5 覆蓋的 coverable aspects 比例，按 query macro average；
- **Aspect Recall@Optimal**：在每個 query 的最小 set-cover budget 下計算 aspect recall。

輔助指標：

- **Sentence Precision@5**：Top-5 中連結任一 gold aspect 的句子比例；
- **First-hit MRR**：第一條 gold evidence sentence 的 reciprocal rank。

`Results Aspect Recall@5` 在本次 Dev/Test 皆為 `null`，原因是原始資料中的 `results_aspect_list_ids` 沒有可評估內容；此結果不是 scorer 故障，也不參與選型。

---

## 3. 實驗流程

```text
Dev-100 engineering smoke
→ Dev-1000 scale and direction check
→ Full Dev configuration selection
→ freeze BMR Top-20 + MedCPT Top-20, RRF k=10
→ Full Test once
```

Test 僅用於驗證 Dev 已凍結配置的泛化，不再依 Test 修改 source depth、RRF `k`、權重或模型選擇。

---

## 4. Dev-100：Baseline 與 Cross-Encoder Smoke

| 配置 | Aspect Recall@5 | Aspect Recall@Optimal | Sentence Precision@5 | First-hit MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.11736 | 0.07048 | 0.12979 | 0.24826 |
| MedCPT | 0.14504 | 0.11144 | **0.18085** | **0.33628** |
| BMRetriever | 0.16109 | 0.10836 | 0.17447 | 0.31673 |
| BMRetriever → MedCPT CE, depth 20 | **0.17370** | **0.11762** | 0.15957 | 0.32023 |
| MedCPT → MedCPT CE, depth 20 | 0.14556 | 0.10528 | 0.14468 | 0.30385 |

Smoke-100 初步顯示：

- BMRetriever 在 Aspect Recall@5 上是最佳單模型；
- MedCPT 在 Precision@5 與 MRR 上較好；
- Cross-Encoder 對 BMR 候選曾出現小樣本增益；
- Cross-Encoder 對 MedCPT 候選沒有有效增益。

此階段只能驗證工程路徑與產生假設，不能據此凍結模型。

---

## 5. Dev-1000：Cross-Encoder 反轉與融合訊號

| 配置 | Aspect Recall@5 | Aspect Recall@Optimal | Sentence Precision@5 | First-hit MRR |
|---|---:|---:|---:|---:|
| MedCPT | 0.20734 | 0.14846 | 0.21462 | 0.37814 |
| BMRetriever | 0.21044 | **0.15642** | 0.21144 | 0.37725 |
| BMRetriever → MedCPT CE, depth 20 | 0.18241 | 0.12664 | 0.18178 | 0.33325 |
| BMR Top-20 + MedCPT Top-20 → RRF `k=60` | **0.21474** | 0.15595 | **0.22267** | **0.39377** |

Dev-1000 顯示 Smoke-100 的 Cross-Encoder 增益沒有泛化：BMR + CE 四項指標全面低於原始 BMR。因此 Cross-Encoder 從正式候選中淘汰。

相反，RRF 在不使用 Cross-Encoder 的情況下同時改善 Recall@5、Precision@5 與 MRR，並基本保留 BMR 的 Recall@Optimal。

### 5.1 Dev-1000 candidate complementarity（Top-20）

| 指標 | BMRetriever | MedCPT | Union |
|---|---:|---:|---:|
| Candidate-level aspect coverage | 0.46558 | 0.46258 | **0.54110** |
| 平均 gold sentence 數 | 3.425 | 3.416 | **4.484** |

每個 query 平均仍有：

- `1.068` 條 BMR-only gold sentence；
- `1.059` 條 MedCPT-only gold sentence。

這證明兩個 dense retrievers 存在穩定互補性，而非單純排序噪音。

---

## 6. Full Dev：正式選型

Full Dev 包含 `4,373` queries，其中 `4,127` queries 具有非空 coverable aspects，可進入正式 aspect 評測。

### 6.1 Full Dev baseline 與 RRF `k=60`

| 配置 | Aspect Recall@5 | Aspect Recall@Optimal | Sentence Precision@5 | First-hit MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.15218 | 0.10714 | 0.14921 | 0.30293 |
| MedCPT | 0.19946 | 0.14132 | 0.20572 | 0.37199 |
| **BMRetriever** | **0.21007** | 0.14899 | 0.20916 | 0.37808 |
| BMR Top-20 + MedCPT Top-20 → RRF `k=60` | 0.20916 | **0.15129** | **0.21609** | **0.38920** |

RRF `k=60` 與 BMR 在主指標上實質持平，但 RRF 的 Precision@5 與 MRR 顯著較好。

### 6.2 RRF `k` ablation

固定：

```text
BMRetriever Top-20
MedCPT Top-20
equal weights
完整 unique union
```

只比較 `k`：

| 配置 | Aspect Recall@5 | Aspect Recall@Optimal | Sentence Precision@5 | First-hit MRR |
|---|---:|---:|---:|---:|
| BMRetriever | **0.21007** | 0.14899 | 0.20916 | 0.37808 |
| RRF `k=10` | 0.20969 | **0.15181** | 0.21464 | 0.38838 |
| RRF `k=20` | 0.20921 | 0.15137 | 0.21560 | 0.38893 |
| RRF `k=60` | 0.20916 | 0.15129 | **0.21609** | **0.38920** |

趨勢：

- 較小 `k` 更重視各 source 的前排，Aspect Recall@5 與 Recall@Optimal 較好；
- 較大 `k` 更平滑地獎勵 source agreement，Precision@5 與 MRR 略好；
- `k=10` 與 BMR 的主指標差距僅 `-0.00038`（約 `-0.18%`），同時改善其餘三項。

因此 Phase 05 選擇 `k=10` 作為完整 pipeline 的均衡配置。此選擇在 Test 前凍結。

### 6.3 Full Dev candidate complementarity（Top-20）

| 指標 | BMRetriever | MedCPT | Union |
|---|---:|---:|---:|
| Candidate-level aspect coverage | 0.46855 | 0.45448 | **0.54478** |
| 平均 gold sentence 數 | 3.354 | 3.285 | **4.350** |

Union 相對最佳單模型增加 `0.07623` absolute aspect coverage。每個 query 平均有 `1.064` 條 BMR-only 與 `0.996` 條 MedCPT-only gold sentences。

---

## 7. Full Test：凍結配置驗證

Full Test 包含：

- `20,000` queries；
- `18,838` eligible queries；
- `19,068` unique papers / candidate pools；
- `3,522,617` candidate sentences；
- RRF output `560,943` rows；
- `0` missing queries；
- `0` duplicate index queries；
- `0` rank gaps；
- `0` invalid scores。

### 7.1 Full Test metrics

| 配置 | Aspect Recall@5 | Aspect Recall@Optimal | Sentence Precision@5 | First-hit MRR |
|---|---:|---:|---:|---:|
| BM25 | 0.14936 | 0.10615 | 0.15060 | 0.30575 |
| MedCPT | 0.19317 | 0.14059 | 0.20206 | 0.36889 |
| BMRetriever | 0.20253 | 0.14701 | 0.21043 | 0.37838 |
| **RRF `k=10`** | **0.20603** | **0.15057** | **0.21623** | **0.38590** |

Dev 上凍結的 RRF `k=10` 在 Test 四項指標皆為最佳。

### 7.2 RRF `k=10` vs BMRetriever

Bootstrap 以 `systematic_review_id` 為 cluster 單位，共 `13,363` clusters、`10,000` resamples。

| 指標 | Absolute Δ | Relative Δ | 95% CI | Resamples with Δ > 0 |
|---|---:|---:|---:|---:|
| Aspect Recall@5 | +0.00350 | +1.73% | [+0.00107, +0.00591] | 99.75% |
| Aspect Recall@Optimal | +0.00356 | +2.42% | [+0.00155, +0.00564] | 99.97% |
| Sentence Precision@5 | +0.00581 | +2.76% | [+0.00385, +0.00767] | 100.00% |
| First-hit MRR | +0.00752 | +1.99% | [+0.00426, +0.01078] | 100.00% |

四項 CI 均高於零。RRF 不只維持主指標，而是在 Test 上全面、可信地優於最強單模型 BMRetriever。

### 7.3 RRF `k=10` vs MedCPT

| 指標 | Absolute Δ | Relative Δ | 95% CI |
|---|---:|---:|---:|
| Aspect Recall@5 | +0.01286 | +6.66% | [+0.01049, +0.01528] |
| Aspect Recall@Optimal | +0.00997 | +7.09% | [+0.00793, +0.01206] |
| Sentence Precision@5 | +0.01417 | +7.01% | [+0.01229, +0.01606] |
| First-hit MRR | +0.01701 | +4.61% | [+0.01377, +0.02044] |

### 7.4 RRF `k=10` vs BM25

| 指標 | Absolute Δ | Relative Δ | 95% CI |
|---|---:|---:|---:|
| Aspect Recall@5 | +0.05667 | +37.95% | [+0.05314, +0.06019] |
| Aspect Recall@Optimal | +0.04442 | +41.85% | [+0.04164, +0.04735] |
| Sentence Precision@5 | +0.06563 | +43.58% | [+0.06270, +0.06860] |
| First-hit MRR | +0.08016 | +26.22% | [+0.07501, +0.08517] |

### 7.5 Full Test complementarity（Top-20）

| 指標 | BMRetriever | MedCPT | Union |
|---|---:|---:|---:|
| Candidate-level aspect coverage | 0.46450 | 0.44763 | **0.53873** |
| 平均 gold sentence 數 | 3.385 | 3.270 | **4.373** |

Union 相對最佳單模型增加 `0.07423` absolute aspect coverage。每個 query 平均有：

- `1.103` 條 BMR-only gold sentence；
- `0.988` 條 MedCPT-only gold sentence。

互補性在 Dev-1000、Full Dev 與 Full Test 皆穩定重現。

---

## 8. Cross-Encoder 淘汰結論

MedCPT Cross-Encoder 是 article relevance reranker，沒有針對 sentence-level aspect diversity 或 Top-5 aspect coverage 訓練。

實驗中：

- Smoke-100 對 BMR 候選曾有小幅增益；
- Dev-1000 對 BMR 候選的四項指標全面下降；
- 對 MedCPT 候選與融合候選同樣退化；
- 融合後增加的候選互補性被 Cross-Encoder 排序破壞。

因此正式配置不使用 Cross-Encoder，且不在 Test 重新嘗試其他 rerank depth。

---

## 9. 最終選型

```text
BMRetriever Top-20
+
MedCPT Top-20
→ equal-weight Reciprocal Rank Fusion
→ RRF k = 10
→ retain full unique union ranking
→ downstream default Top-5
```

選型理由：

1. BMRetriever 與 MedCPT 在候選層有穩定互補性；
2. RRF 不依賴不可比較的 raw dense score scales；
3. Full Dev 主指標與 BMR 實質持平，其他指標更好；
4. Frozen configuration 在 Full Test 四項指標全面勝過 BMR；
5. 不需要 Cross-Encoder，排序 deterministic，可重現且易於審計。

---

## 10. 限制

1. `Results Aspect Recall@5` 因資料欄位不可用而無法評測；
2. 約三分之一宣告 aspects 沒有 sentence mapping，正式 metrics 只能以 coverable aspects 為分母；
3. 候選 union 的 coverage ceiling 明顯高於最終 Top-5，代表簡單 RRF 尚未完全利用 aspect diversity；
4. 沒有額外建立 position-only baseline，因此「不依靠文章位置」是由實作契約保障，而非獨立 position ablation 證明；
5. 多進程結束時偶爾出現 Python `resource_tracker` semaphore warning，但所有正式 run 均通過 rows、query coverage、rank continuity、score validity 與 checksum 驗證。

---

## 11. 結論

Phase 05 已完成 EvidenceBench 上可重現的 sentence-level evidence retrieval pipeline。最終混合配置在 Full Test 20,000 queries 上，相對 BM25、MedCPT 與 BMRetriever 均取得最佳整體結果，並保留原始 sentence index 與可追溯 artifact。

Phase 05 可正式關閉，後續由 Phase 06 使用 Dev ranking 的 Top-5 evidence sentences 開發 stance verifier；Test ranking 僅用於最終 pipeline 評估。

