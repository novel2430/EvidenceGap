# EvidenceGap V1 Phase 00｜Task Contract

- Contract ID: `evidencegap.v1.task-contract`
- Schema version: `1.0.0`
- Status: `FROZEN_FOR_V1_PHASE_01`
- Scope: EvidenceGap V1 的任務邊界、預測單位、資料責任與禁止事項

## 1. 目的

Phase 00 先固定 EvidenceGap V1 的機器學習任務，避免後續把 Retrieval、Evidence Selection、Stance Verification 與前端狀態混為同一件事。

V1 的產品流程是：

```text
Medical Claim
→ Article Retrieval
→ Article Reranking
→ Evidence Sentence Retrieval
→ Stance Verification
→ Evidence Graph
```

但正式實驗由四個彼此獨立的任務組成，各自使用不同資料集與評測指標。三套資料並不是同一批端到端樣本，不得把四組分數合成單一的「EvidenceGap Accuracy」。

## 2. 共通原則

1. 每個任務必須保存可追溯的資料來源 ID。
2. Retrieval relevance 與 stance direction 是不同標籤，不得共用一個欄位。
3. Synthetic supervision 與 expert annotation 必須分開報告。
4. 前端 `DemoCase`／Graph ViewModel 不等於底層分析資料模型。
5. Phase 00 不切 split、不建立索引、不跑模型、不轉換全量資料。
6. LLM API 可在後續負責敘述，但不得改寫正式 Verifier 判決。

---

# Task A｜AR-MEDFACT-JUDGED

## Task ID

`AR-MEDFACT-JUDGED`

## 名稱

MedFact judged claim–article ranking

## 目的

針對同一 Claim 下已有 synthetic judgment 的候選文章進行相關度排序，驗證 BM25、Dense Retriever 與 Reranker 的排序能力。

## 預測單位

一個 Claim 對一組**已有 judgment 的候選文章**。

## 輸入

```text
claim_id
claim_text
candidate_articles[]:
  article_id
  article_text
```

## 輸出

```text
ranked_articles[]:
  article_id
  retrieval_score
  rank
```

## Gold annotation

來源：MedFact-Synth `synthetic_label`。

```text
relevance_grade = abs(synthetic_label)

2 = synthetic_label ∈ {-2, +2}
1 = synthetic_label ∈ {-1, +1}
0 = synthetic_label = 0
```

`relevance_grade` 只描述文章是否具有直接或部分證據關係，不描述支持或反駁方向。

## Candidate universe

正式 judged ranking 的候選池只包含該 Claim 在 MedFact-Synth 中已有 judgment 的文章。

MedFact-Synth 並沒有對每條 Claim 標註全部 132 萬篇文章，因此 V1 必須分開兩條 track：

### Judged Candidate Ranking

- 候選文章全部有 judgment。
- 可正式計算 MRR、nDCG 與 Recall。

### Open-Corpus Retrieval

- 從約 132 萬個唯一 `source_pmid` 中檢索。
- 未標註文章是 `UNJUDGED`，不是可靠負例。
- 只能報 known-positive recall 等不完整 qrels 指標，並明確標註 judgments incomplete。

## Track

```text
independent_source:
  claim_pmid != source_pmid

origin_source:
  claim_pmid == source_pmid

overall:
  全部 judged pairs
```

正式主結果使用 `independent_source`；`origin_source` 只作 sanity check。

## 主要指標

Judged Candidate Ranking：

```text
nDCG@10
MRR
Recall@10
Recall@50
Recall@100
```

Open-Corpus Retrieval：

```text
Known-positive Recall@10
Known-positive Recall@50
Known-positive Recall@100
```

## Allowed datasets

- MedFact-Synth

## 禁止事項

- 不得把 132 萬文章庫中所有未標註文章都當成真正負例。
- 不得用 stance 正負方向直接當 retrieval score。
- 不得把 `claim_pmid == source_pmid` 與獨立來源結果混成唯一主分數。
- 不得使用 HealthFC 或 EvidenceBench 評估 open-corpus article retrieval。

## 已知限制

MedFact labels 是 synthetic judgment；正式 retrieval 分數證明的是對這套 synthetic qrels 的排序能力，不等於臨床專家檢索品質。

---

# Task B｜ESR-EVIDENCEBENCH

## Task ID

`ESR-EVIDENCEBENCH`

## 名稱

EvidenceBench hypothesis-to-evidence sentence ranking

## 目的

在文章已知的條件下，根據 hypothesis 對文章內候選句排序，找出能覆蓋重要 evidence aspects 的句子。

## 預測單位

一個 `hypothesis–paper` datapoint。

## 輸入

```text
query_id
hypothesis
paper_id
candidate_sentences[]
sentence_types[]
```

欄位來源：

```text
hypothesis                  ← hypothesis
candidate_sentences         ← paper_as_candidate_pool
sentence_types              ← sentence_types_in_candidate_pool
paper_id                    ← paper_id
```

## 輸出

```text
ranked_sentences[]:
  sentence_index
  score
  rank
```

## Gold annotation

```text
aspect_ids                   ← aspect_list_ids
aspect_text                  ← aspect_id2aspect
aspect_to_sentence_indices   ← aspect2sentence_indices
sentence_to_aspects          ← sentence_index2aspects
results_aspect_ids           ← results_aspect_list_ids
```

Gold 的核心不是唯一正確句，而是每個 aspect 可由哪些句子提供。因此正式評測以 aspect coverage 為主。

## 主要指標

```text
Aspect Recall@5
Aspect Recall@Optimal
Results Aspect Recall@5
```

輔助指標：

```text
Sentence Precision@5
First-hit MRR
```

## Split usage

- 官方 train：開發與參數選擇。
- 官方 test：最終正式結果。
- V1 Phase 01 可從官方 train 依 `systematic_review_id` 分組建立 dev。

## Index invariants

以下規則不可破壞：

1. `paper_as_candidate_pool` 的句子順序不可改變。
2. 不可重新分句。
3. 不可刪除 section headings。
4. 不可重新編號。
5. 所有 gold sentence indices 必須落在候選池範圍內。

## Allowed datasets

- EvidenceBench-100k

## 禁止事項

- 不得用 EvidenceBench 判斷支持或反駁方向。
- 不得只用單一 gold sentence accuracy 取代 aspect coverage。
- 不得用 test set 選擇融合權重、threshold 或模型超參數。

## 已知限制

文章已經給定，因此此任務不驗證 article retrieval；它只驗證文章內 evidence sentence ranking。

---

# Task C｜STANCE-MEDFACT-5

## Task ID

`STANCE-MEDFACT-5`

## 名稱

MedFact five-level claim–article stance classification

## 目的

判斷一篇文章對 Claim 的立場方向與強度。

## 預測單位

一個 MedFact claim–article pair。

## 輸入方向

```text
premise    = source
hypothesis = claim
```

輸入方向固定，不得任意交換。

## 輸出

```text
predicted_label ∈ {-2, -1, 0, +1, +2}
probabilities:
  -2
  -1
   0
  +1
  +2
confidence
stance_score
```

其中：

```text
stance_score = Σ probability(label) × label
confidence   = max(probabilities)
```

## Gold label

```text
-2 = STRONG_REFUTE
-1 = PARTIAL_REFUTE
 0 = NEUTRAL_OR_INSUFFICIENT
+1 = PARTIAL_SUPPORT
+2 = STRONG_SUPPORT
```

`0` 同時涵蓋不相關、未直接處理 Claim 與資訊不足，不應重命名為單純的 `NEUTRAL`。

## 主要指標

```text
Macro-F1
Weighted-F1
Ordinal MAE
Confusion Matrix
```

校準指標可在後續加入：

```text
ECE
Brier score
```

## Track

```text
independent_source
origin_source
overall
```

正式主結果使用 `independent_source`。

## Allowed datasets

- MedFact-Synth：主要訓練、開發與 synthetic test
- HealthFC：不得直接以五級標籤訓練

## 禁止事項

- 不得把 MedCPT Cross-Encoder relevance score 當 stance。
- 不得將 explanation／rationale 拼入模型輸入造成答案洩漏。
- 不得將 synthetic test 與 HealthFC expert evaluation 合併成一個結果。

## 已知限制

此任務的五級 gold label 是 synthetic supervision。它可用於大規模模型適配和壓力測試，但不能單獨證明 expert-level medical verification。

---

# Task D｜VERDICT-HEALTHFC-3

## Task ID

`VERDICT-HEALTHFC-3`

## 名稱

HealthFC expert evidence-to-verdict classification

## 目的

使用人工整理的醫療 evidence，驗證最終 Supported／Not Enough Information／Refuted 判決。

## 預測單位

一個 HealthFC claim–gold-evidence case。

## 輸入

```text
claim         = en_claim
evidence_text = en_top_sentences
```

`en_top_sentences` 在 raw CSV 中是文字欄位。Phase 00 不猜測其句子 delimiter，也不重新分句；canonical contract 先保存為 `evidence_text`。

## 輸出

```text
verdict ∈ {
  SUPPORTED,
  NOT_ENOUGH_INFORMATION,
  REFUTED
}
probabilities[3]
confidence
```

## Gold label

```text
0 = SUPPORTED
1 = NOT_ENOUGH_INFORMATION
2 = REFUTED
```

若使用五級 MedFact Verifier，三級機率可按以下方式折疊：

```text
REFUTED               = P(-2) + P(-1)
NOT_ENOUGH_INFORMATION = P(0)
SUPPORTED              = P(+1) + P(+2)
```

## 主要指標

```text
Macro-F1
Balanced Accuracy
Per-class Recall
Confusion Matrix
```

普通 Accuracy 只能作輔助指標。

## Dataset usage

HealthFC 全部作外部 expert evaluation。

## 禁止事項

- 不得把 `en_explanation` 拼入 Verifier 輸入。
- 不得把 HealthFC 作主要訓練集。
- 不得用 HealthFC 評估 article retrieval。
- 不得聲稱 HealthFC 已驗證完整文章內 sentence retrieval。

## 已知限制

HealthFC CSV 沒有完整候選文章池，因此它只能驗證 `gold evidence → verdict`，不能驗證前面的 retrieval stages。

---

# 3. Phase 00 完成條件

全部成立才視為 Phase 00 完成：

- 四個任務的預測單位、輸入、輸出與 gold 清楚。
- Retrieval relevance 與 stance label 分離。
- Judged ranking 與 open-corpus retrieval 分離。
- EvidenceBench sentence index invariants 明確。
- HealthFC label mapping 明確。
- HealthFC explanation 不進模型輸入。
- Synthetic 與 expert evaluation 分開。
- Canonical analysis records 與 frontend ViewModel 分開。
- Machine-readable mappings、label maps、examples 可由腳本產生並通過 validator。
