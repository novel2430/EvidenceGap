# EvidenceGap V1 推進規劃書

## 一、V1 目標

EvidenceGap V1 的目標不是直接做成完整醫療問答系統，而是建立並驗證一條可追溯的醫療證據分析流水線：

```text
Medical Claim
→ Article Retrieval
→ Article Reranking
→ Evidence Sentence Retrieval
→ Stance Verification
→ Evidence Graph
```

使用者輸入一條醫療 Claim 後，系統應能輸出：

* 相關醫學文章；
* 文章中的關鍵證據句；
* 證據對 Claim 的支持、反駁或資訊不足判斷；
* PMID、文章與證據來源；
* 可視化 Evidence Graph；
* 各模組的置信度與評測結果。

V1 不宣稱達到臨床決策系統標準，而是證明這套架構能在大規模資料、句子級證據與專家標註資料上有效運行。

---

## 二、現有資產

### 資料

| 資料集                |                               規模 | V1 主要角色                     |
| ------------------ | -------------------------------: | --------------------------- |
| MedFact-Synth      |    1,497,981 claim–article pairs | 文章檢索、文章級 stance、Verifier 訓練 |
| EvidenceBench-100k | 107,461 hypothesis–paper samples | 文章內證據句檢索                    |
| HealthFC           |      750 expert-annotated claims | 最終專家標註 Verdict 評估           |

### 模型

| 模型                     | 角色                           |
| ---------------------- | ---------------------------- |
| BM25                   | Sparse Retrieval baseline    |
| MedCPT Query Encoder   | Claim 編碼                     |
| MedCPT Article Encoder | Article／Sentence 編碼          |
| BMRetriever-410M       | 第二套 Dense Retrieval baseline |
| MedCPT Cross-Encoder   | Article／Sentence Reranking   |
| DeBERTa-v3 NLI         | Stance Verifier 起點           |

---

# Version 0｜資料與任務契約固定

## 目的

在寫模型與資料處理程式前，固定每個資料集的輸入、輸出、標籤與評測責任，避免後面反覆更換資料格式或混淆任務。

## 建議作法

定義三個正式 Task Contract。

### Task A：Article Retrieval

```text
Input:
claim_id
claim_text

Candidate:
source_pmid
source_text

Gold relevance:
abs(synthetic_label)
```

相關文章定義：

```text
synthetic_label ∈ {-2, -1, 1, 2}
```

不相關或資訊不足：

```text
synthetic_label = 0
```

### Task B：Evidence Sentence Retrieval

```text
Input:
hypothesis
paper_as_candidate_pool

Output:
ranked sentence indices

Gold:
aspect2sentence_indices
```

### Task C：Stance Verification

MedFact-Synth：

```text
source + claim
→ {-2, -1, 0, 1, 2}
```

HealthFC：

```text
gold evidence + claim
→ Supported / NEI / Refuted
```

同時定義共用資料物件：

```text
ClaimRecord
ArticleRecord
EvidenceRecord
StanceResult
RetrievalResult
```

## 產出

```text
docs/v1/task_contract.md
docs/v1/data_contract.md
```

## 驗收條件

* 每個欄位的來源和含義明確；
* 每個資料集只有清楚且有限的角色；
* 訓練、開發和測試資料用途明確；
* 不依賴尚未實作的前端或 LLM。

---

# Version 1｜正式資料切分與標準化

## 目的

建立可重現、無資料洩漏的 train／dev／test manifest，但不改寫原始資料。

## 建議作法

### MedFact-Synth

使用：

```text
group_id = hash(claim_pmid, normalized_claim)
```

按照完整 Claim 分組切分：

```text
Train：90%
Dev：5%
Test：5%
```

同一 Claim 的所有 source articles 必須位於同一 split。

每筆增加衍生欄位：

```text
group_id
split
is_origin_source
relevance_grade
stance_label
```

其中：

```text
is_origin_source =
claim_pmid == source_pmid
```

正式測試以：

```text
is_origin_source = false
```

為主。

### EvidenceBench

保留官方切分：

```text
Train：87,461
Test：20,000
```

可從 train 中按 systematic review 或 paper 分組抽出 dev，避免同一 review 的高度相似樣本跨集合。

### HealthFC

全部保留為外部 expert evaluation。

不使用 HealthFC 進行主要訓練。

## 產出

```text
data/processed/v1/manifests/
├── medfact_train.jsonl
├── medfact_dev.jsonl
├── medfact_test.jsonl
├── evidencebench_train.jsonl
├── evidencebench_dev.jsonl
├── evidencebench_test.jsonl
└── healthfc_eval.jsonl
```

原始大型文字可繼續留在 Parquet／JSON 中，manifest 只保存 ID、split、標籤與定位資訊，避免複製大量資料。

## 驗收條件

* 同一 MedFact Claim 不跨 split；
* EvidenceBench 官方 test 未被修改；
* HealthFC 未進入訓練集合；
* 所有 manifest 可由固定 seed 重建；
* 有統計報告證明無明顯資料遺失。

---

# Version 2｜文章庫與 BM25 Baseline

## 目的

先建立最簡單但可信的 Article Retrieval baseline，確認資料層和評測層可運作。

## 建議作法

從 MedFact-Synth 按 `source_pmid` 去重，建立約 132 萬篇文章摘要庫：

```text
document_id = source_pmid
title_and_abstract = source
```

建立 BM25 索引。

每條 Claim 搜尋 Top-100 articles。

相關度分級：

```text
2 = abs(label) == 2
1 = abs(label) == 1
0 = label == 0
```

主要評測：

```text
Recall@10
Recall@50
Recall@100
MRR
nDCG@10
```

測試結果分成：

```text
Independent-source track
Origin-source track
Overall track
```

## 產出

```text
artifacts/v1/article_corpus/
artifacts/v1/bm25_index/
reports/v1/article_retrieval_bm25.json
reports/v1/article_retrieval_bm25.md
```

## 驗收條件

* 能對 dev/test Claim 穩定返回 Top-K；
* 評測程式可重複執行；
* 查詢結果可回溯至 PMID 與原始文章；
* 不以人工挑選案例代替整體指標。

---

# Version 3｜Dense Article Retrieval

## 目的

驗證生醫 Dense Retrieval 是否比 BM25 更適合 Claim–Article Matching。

## 建議作法

分別建立：

```text
MedCPT Article index
BMRetriever Article index
```

查詢模型：

```text
MedCPT Query Encoder
BMRetriever query encoding
```

對相同 dev/test 集合評測：

```text
BM25
MedCPT
BMRetriever
```

可再測試簡單融合：

```text
BM25 score + Dense score
```

但融合權重只能在 dev 上選擇。

### 工程建議

* 文章向量離線批次計算；
* 使用 8 張 GPU 分片編碼；
* 先保存 float16 embeddings；
* 使用 FAISS 建立索引；
* 不在每次啟動時重新編碼；
* 索引 manifest 記錄模型 revision、資料 fingerprint 和向量維度。

## 產出

```text
artifacts/v1/medcpt_article_index/
artifacts/v1/bmretriever_article_index/
reports/v1/article_retrieval_comparison.md
```

## 驗收條件

* Dense 索引可以離線載入；
* 結果可與 BM25 公平比較；
* 能確認 Dense 是否真的改善 Recall 或 nDCG；
* 若 BMRetriever 沒有明顯價值，可在後續正式 Pipeline 中不使用。

---

# Version 4｜Article Reranking

## 目的

改善 Retriever Top-K 的排序品質，把最有證據價值的文章推到前面。

## 建議作法

第一階段：

```text
Retriever → Top-100
```

第二階段：

```text
MedCPT Cross-Encoder
→ Rerank Top-100
→ Return Top-10 / Top-20
```

比較：

```text
BM25 only
Dense only
Hybrid Retrieval
Hybrid + Cross-Encoder
```

主要關注：

```text
MRR
nDCG@10
Precision@10
Recall@10
```

Reranker 不負責支持或反駁判斷，其分數只表示相關性。

## 產出

```text
reports/v1/article_reranking.md
artifacts/v1/article_reranking_runs/
```

## 驗收條件

* Reranking 沒有降低 Recall；
* nDCG@10 或 MRR 有實際提升；
* 推理速度可以支援單次 Claim 查詢；
* Reranker 結果與 Verifier 結果分開保存。

---

# Version 5｜Evidence Sentence Retrieval

## 目的

在文章已知的條件下，找出真正支撐分析的證據句，避免把整篇摘要直接交給 Verifier。

## 建議作法

在 EvidenceBench 上比較：

```text
BM25 sentence retrieval
MedCPT sentence retrieval
BMRetriever sentence retrieval
Cross-Encoder reranking
```

輸入：

```text
hypothesis
paper_as_candidate_pool
```

輸出：

```text
Top-5 evidence sentence indices
```

正式評測以 aspect coverage 為主：

```text
Aspect Recall@5
Aspect Recall@Optimal
Results Aspect Recall@5
```

普通句子命中率只能作輔助指標，因為同一 aspect 可能有多個正確證據句。

### 與產品 Pipeline 的銜接

EvidenceBench 用於驗證模型能力。

實際 MedFact Pipeline 中，Retriever 找到文章後，再將文章摘要或全文分句，交給同一 Evidence Sentence Retriever。

## 產出

```text
reports/v1/evidence_sentence_retrieval.md
artifacts/v1/evidencebench_runs/
```

## 驗收條件

* 能覆蓋重要 aspects；
* Top-5 句子不是只依靠文章位置；
* 保留原始 sentence index；
* 證據句可被前端直接引用。

---

# Version 6｜Verifier Baseline 與微調

## 目的

建立真正負責支持、反駁和資訊不足判斷的模型。

## 建議作法

### 第一階段：Zero-shot baseline

使用原始 DeBERTa-v3 NLI：

```text
premise = source / evidence
hypothesis = claim
```

輸出：

```text
contradiction
entailment
neutral
```

先在 MedFact dev 和 HealthFC 上測零樣本結果。

### 第二階段：MedFact 五分類微調

將 classification head 改為：

```text
-2 / -1 / 0 / 1 / 2
```

第一版使用：

```text
Weighted Cross Entropy
max_length = 512
grouped train/dev/test
```

暫時不引入複雜 ordinal loss 或 multi-task learning。

### HealthFC 映射

```text
Refuted   = P(-2) + P(-1)
NEI       = P(0)
Supported = P(1) + P(2)
```

主要指標：

MedFact：

```text
Macro-F1
Weighted-F1
Ordinal MAE
Confusion Matrix
```

HealthFC：

```text
Macro-F1
Balanced Accuracy
Per-class Recall
Confusion Matrix
```

再使用 dev set 做 temperature scaling，獲得較可信的 confidence。

## 產出

```text
models/v1/verifier-medfact-5class/
reports/v1/verifier_zero_shot.md
reports/v1/verifier_finetuned.md
reports/v1/healthfc_external_eval.md
```

## 驗收條件

* Fine-tuned 模型優於原始 zero-shot；
* HealthFC 分數獨立報告；
* Verifier 能拒絕低信心判斷；
* Synthetic test 與 expert gold test 不混為同一結果。

---

# Version 7｜完整 Pipeline 與 Evidence Graph

## 目的

把已經獨立驗證的模組串成一條可執行的產品流水線。

## 建議作法

正式流程：

```text
1. 使用者輸入 Claim
2. Article Retriever 返回 Top-100
3. Article Reranker 返回 Top-10
4. Sentence Retriever 從每篇文章選 Top Evidence
5. Verifier 判斷每條證據的 stance
6. 聚合文章與證據結果
7. 建立 Evidence Graph
```

### Graph 最小結構

節點：

```text
Claim
Article
Evidence Sentence
Verdict
```

邊：

```text
Claim → Article        retrieved_from
Article → Evidence     contains
Evidence → Claim       supports / refutes / insufficient
Claim → Verdict        summarized_as
```

每條邊保存：

```text
retrieval_score
rerank_score
stance_probabilities
stance_score
model_revision
source_reference
```

### Verdict 聚合

第一版使用透明規則，不讓 LLM 自由決策：

```text
文章相關度
×
證據相關度
×
Verifier stance confidence
```

分別累積：

```text
support mass
refute mass
neutral mass
```

最終 Verdict 和各篇文章的判斷都要保留，不能只輸出一個總結。

## 產出

```text
src/evidencegap/pipeline/
src/evidencegap/graph/
artifacts/v1/demo_runs/
```

## 驗收條件

* 任意輸入 Claim 能完整跑通；
* 每個結果可以追溯到原始 PMID 和句子；
* 每個模組可以獨立替換；
* Pipeline 失敗時能指出是哪個階段失敗；
* 不需要依賴 LLM API 才能完成判決。

---

# Version 8｜API 與前端展示

## 目的

把分析結果做成容易理解、看起來專業的產品 Demo，而不是繼續堆模型。

## 建議作法

### 後端

使用簡單 API：

```text
POST /claims/analyze
GET  /runs/{run_id}
GET  /runs/{run_id}/graph
GET  /articles/{pmid}
```

單機 Demo 可以同步或簡單 background job，不需要一開始引入複雜分散式任務系統。

### 前端

核心頁面只有一個分析工作區：

```text
Claim Input
Article Ranking
Evidence Detail
Evidence Graph
Verdict Summary
```

建議布局：

```text
左側：文章與證據列表
中間：Evidence Graph
右側：Verdict、分數與文章詳情
```

區塊使用 Splitter，讓使用者自行調整大小。

Graph 重點呈現：

* 支持與反駁使用清楚的視覺區分；
* 節點顯示 PMID、文章、Evidence；
* 點擊節點可查看原始文字；
* 顯示各模組分數；
* 不用過度複雜的前端狀態管理。

### LLM API

LLM 只負責生成解釋：

```text
輸入：
Claim
Top Evidence
Verifier result
Confidence
Sources

輸出：
簡短總結
支持與反駁理由
不確定性
```

LLM 不得修改正式 Verdict，也不得產生不存在的來源。

## 產出

```text
backend/
frontend/
demo/
```

## 驗收條件

* 使用者能完成一次 Claim 分析；
* Graph、文章、Evidence 和 Verdict 互相連動；
* API 關閉時不影響離線評測；
* LLM API 不可用時，核心分析仍能正常輸出。

---

# Version 9｜最終評測與 V1 報告

## 目的

形成可以展示、審查和繼續研究的正式成果，而不只是「系統能跑」。

## 建議作法

正式報告分四個模組：

### Article Retrieval

```text
BM25
MedCPT
BMRetriever
Hybrid
Hybrid + Reranker
```

### Evidence Sentence Retrieval

```text
BM25
MedCPT
BMRetriever
Reranked
```

### Synthetic Article Stance

```text
Zero-shot DeBERTa
Fine-tuned DeBERTa
```

### Expert Verdict Evaluation

```text
HealthFC Macro-F1
Balanced Accuracy
Per-class Recall
Calibration
```

另外報告：

```text
Latency
GPU memory
Index size
Embedding generation cost
End-to-end runtime
Failure cases
```

挑選代表案例：

* 明確支持；
* 明確反駁；
* 支持與反駁並存；
* 資訊不足；
* Retrieval 失敗；
* Evidence 選錯；
* Verifier 判斷錯誤。

## 產出

```text
reports/v1/final_report.md
reports/v1/metrics.json
reports/v1/error_analysis.md
```

## 驗收條件

* 所有正式數字都來自固定 test set；
* 不使用人工挑選 Demo 代表整體性能；
* 清楚區分 synthetic 與 expert evaluation；
* 每個結論都能找到對應實驗和結果；
* V1 的能力邊界和失敗案例寫清楚。

---

# 三、建議推進順序

```text
V0  任務與資料 Contract
↓
V1  Split 與 Manifest
↓
V2  BM25 Article Retrieval
↓
V3  Dense Article Retrieval
↓
V4  Article Reranking
↓
V5  Evidence Sentence Retrieval
↓
V6  Verifier
↓
V7  End-to-End Pipeline + Graph
↓
V8  API + 前端
↓
V9  正式評測與報告
```

前六個版本主要解決「分析能力是否成立」。

後三個版本主要解決「如何組合、展示與交付」。

---

# 四、推進原則

## 1. 每個版本必須有獨立驗收結果

不能用「後面串起來就會好」作為當前階段的驗收方式。

## 2. 先 Baseline，再增加複雜度

順序固定為：

```text
BM25
→ Dense
→ Reranker
→ Fine-tuning
→ Fusion
```

不要一開始同時做多模型融合和複雜 loss。

## 3. 模組分數分開報告

禁止用一個模糊的總 Accuracy 掩蓋某個模組的失敗。

## 4. Synthetic 與 Expert Gold 嚴格分開

MedFact-Synth 可以提供規模，但 HealthFC 才是專家標註外部驗收。

## 5. 前端是展示層，不承擔分析邏輯

前端可以炫，但所有正式結果都必須由後端 Pipeline 產生。

## 6. LLM 只負責表達，不負責取代證據判斷

正式 Verdict 來自可評測、可校準的 Verifier。LLM API 只將結果寫成人類容易閱讀的解釋。

---

# 五、V1 完成標準

EvidenceGap V1 完成時，應同時具備：

1. 約 132 萬篇生醫文章的可查詢索引；
2. 可比較的 Sparse、Dense、Reranked Retrieval 結果；
3. EvidenceBench 句子級證據檢索結果；
4. MedFact 五級 stance Verifier；
5. HealthFC 專家標註外部評測；
6. 可追溯的端到端 Evidence Pipeline；
7. Claim、Article、Evidence、Verdict 組成的 Evidence Graph；
8. 可操作且具有展示效果的前端；
9. 完整實驗指標與錯誤分析；
10. 明確說明 V1 能證明及不能證明的內容。

