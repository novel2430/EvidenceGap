# EvidenceGap V1 Phase 00｜Evaluation Contract

- Contract ID: `evidencegap.v1.evaluation-contract`
- Schema version: `1.0.0`
- Status: `FROZEN_FOR_V1_PHASE_01`
- Scope: 正式評測資料、track、指標、報告規則與禁止事項

## 1. 目的

本文件固定 EvidenceGap V1 各模組如何評測，避免：

- 用 incomplete judgments 得出過度確定的 Retrieval 結論；
- 將 synthetic 與 expert gold 混合；
- 用單一 Accuracy 掩蓋某一模組失敗；
- 在 test set 上選參數；
- 把前端 Demo 案例當作正式實驗。

## 2. 共通評測規則

1. 每次正式 run 必須記錄 task ID、dataset、split、model revision、contract version、seed 與 timestamp。
2. 所有 threshold、fusion weight、Top-K 與 calibration 參數只能在 train/dev 決定。
3. Test set 只用於最終一次或明確版本化的正式評測。
4. Synthetic 與 expert evaluation 分開報告。
5. 不得將四個模組平均成單一 `EvidenceGap Accuracy`。
6. 正式報告至少同時輸出 machine-readable JSON 與 human-readable Markdown。

建議 run metadata：

```json
{
  "run_id": "...",
  "task_id": "...",
  "dataset": "...",
  "split": "...",
  "contract_version": "1.0.0",
  "model_id": "...",
  "model_revision": "...",
  "seed": 0,
  "parameters": {},
  "created_at": "..."
}
```

---

# 3. AR-MEDFACT-JUDGED

## 3.1 Judged Candidate Ranking

### Evaluation unit

每條 Claim 與其所有已標註 candidate articles。

### Relevance grade

```text
2 = abs(synthetic_label) == 2
1 = abs(synthetic_label) == 1
0 = synthetic_label == 0
```

### Primary metrics

```text
nDCG@10
MRR
Recall@10
Recall@50
Recall@100
```

### Metric interpretation

- `nDCG@10`：前十名是否優先放入強相關文章。
- `MRR`：第一篇已知 relevant article 出現的位置。
- `Recall@K`：已知 relevant articles 在 Top-K 被召回的比例。

### Query eligibility

正式 aggregate metrics 只包含至少有一篇 `relevance_grade > 0` 的 Claim。無正例 Claim 另行統計，不得讓 metric implementation 靜默忽略。

## 3.2 Open-Corpus Retrieval

### Corpus

以唯一 `source_pmid` 去重後的 MedFact article corpus。

### Judgments

只有 MedFact 中出現過的 claim–article pairs 有 judgment；其他文章全部是 `UNJUDGED`。

### Primary metrics

```text
Known-positive Recall@10
Known-positive Recall@50
Known-positive Recall@100
```

### Required report warning

每份 open-corpus report 必須包含：

```text
Judgments are incomplete. Unlabeled documents are treated as unjudged,
not as confirmed negatives.
```

不得在 incomplete qrels 下將普通 Precision@K 解讀成真實 precision。

## 3.3 Tracks

所有結果分開報告：

```text
independent_source: claim_pmid != source_pmid
origin_source:      claim_pmid == source_pmid
overall
```

主結論以 `independent_source` 為準。

---

# 4. ESR-EVIDENCEBENCH

## 4.1 Evaluation unit

一個 hypothesis–paper datapoint。

Retriever 輸出 ranked sentence indices。

## 4.2 Aspect coverage

對預測 Top-K 句子集合 `S_K`，一個 aspect 被召回的條件是：

```text
aspect2sentence_indices[aspect] 與 S_K 的交集非空
```

## 4.3 Primary metrics

```text
Aspect Recall@5
Aspect Recall@Optimal
Results Aspect Recall@5
```

### Aspect Recall@5

Top-5 句子覆蓋的 gold aspects 比例。

### Aspect Recall@Optimal

在 contract 指定的 optimal selection budget／dataset evaluation convention 下計算。實作時必須將具體 budget 與算法寫入 run parameters，不能只報名稱。

### Results Aspect Recall@5

只針對 `results_aspect_list_ids` 計算；若該欄位為 null／empty，該 datapoint 不進此項 aggregate，但需報告 eligible case count。

## 4.4 Auxiliary metrics

```text
Sentence Precision@5
First-hit MRR
```

輔助指標不得取代 aspect metrics。

## 4.5 Split policy

```text
official train:
  訓練、開發、建立 grouped dev

official test:
  最終正式評測
```

Phase 01 若建立 dev，需按 `systematic_review_id` 分組，避免同一 review 的高度相關 hypotheses 跨 train/dev。

---

# 5. STANCE-MEDFACT-5

## 5.1 Label order

```text
-2 < -1 < 0 < +1 < +2
```

這是一個有序五分類任務。V1 baseline 可以使用 ordinary weighted cross entropy，但評測必須同時反映分類與 ordinal distance。

## 5.2 Primary metrics

```text
Macro-F1
Weighted-F1
Ordinal MAE
Confusion Matrix
```

### Ordinal MAE

```text
mean(abs(predicted_label - gold_label))
```

## 5.3 Calibration metrics

完成 temperature scaling 後建議報告：

```text
Expected Calibration Error (ECE)
Brier score
Coverage at confidence threshold
Selective accuracy / Macro-F1
```

Phase 00 只固定名稱，不固定 threshold。

## 5.4 Tracks

```text
independent_source
origin_source
overall
```

主結果使用 `independent_source`。

## 5.5 Required reports

- Zero-shot DeBERTa NLI baseline。
- MedFact five-class fine-tuned model。
- 各標籤 support count。
- Confusion matrix。
- Synthetic label limitation。

---

# 6. VERDICT-HEALTHFC-3

## 6.1 Gold labels

```text
0 = SUPPORTED
1 = NOT_ENOUGH_INFORMATION
2 = REFUTED
```

## 6.2 Primary metrics

```text
Macro-F1
Balanced Accuracy
Per-class Recall
Confusion Matrix
```

## 6.3 Auxiliary metrics

```text
Accuracy
Per-class Precision
Per-class F1
```

因為類別不平衡，普通 Accuracy 不得作唯一主指標。

## 6.4 Input boundary

正式 Verifier 輸入只有：

```text
en_claim
en_top_sentences
```

以下欄位不得進輸入：

```text
en_explanation
label
de_verdict
```

## 6.5 Evaluation role

HealthFC 是外部 expert evaluation：

- 不參與主要 fine-tuning。
- 不用於 threshold／temperature 選擇。
- 不評估 article retrieval。
- 不評估完整 sentence retrieval。

若需要對 HealthFC 做 calibration，必須另行定義不污染最終 expert test 的切分；V1 初版不做。

---

# 7. End-to-end Demo 評估邊界

完整產品 Pipeline 可以輸出：

```text
Claim
Article ranking
Evidence sentences
Per-evidence stance
Aggregated verdict
Evidence Graph
```

但三套資料沒有形成同一個端到端 benchmark，因此 V1 不提供單一端到端 accuracy。

End-to-end Demo 只報：

```text
Latency
Failure stage
Trace completeness
Source traceability
Qualitative cases
```

若未來新增真正端到端 expert benchmark，再另立 contract。

# 8. 正式報告最小結構

```text
1. Task definition
2. Dataset and split
3. Candidate universe / judgment completeness
4. Model and revision
5. Parameters selected on dev
6. Main metrics
7. Per-track / per-class metrics
8. Latency and resource use
9. Failure cases
10. Limitations
```

# 9. 禁止的報告方式

- 單一 EvidenceGap Accuracy。
- 將 synthetic MedFact 分數包裝成 expert medical accuracy。
- 將 HealthFC 結果解讀為 open-domain retrieval performance。
- 將 EvidenceBench 結果解讀為 support/refute performance。
- 對 open-corpus 未標註文章直接計算「真實負例 Precision」。
- 只展示人工挑選案例而不報固定 test metrics。
