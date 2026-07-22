# EvidenceGap V1 Phase 00｜Data Contract

- Contract ID: `evidencegap.v1.data-contract`
- Schema version: `1.0.0`
- Status: `FROZEN_FOR_V1_PHASE_01`
- Scope: 原始資料欄位、canonical records、ID 與 provenance 規則

## 1. 目的

本文件定義三套 raw datasets 如何映射為 EvidenceGap V1 的 canonical analysis records。

Phase 00 只固定 contract 並建立少量 example，不改寫全量 raw data。正式 split、manifest 與標準化資料在 Phase 01 才建立。

## 2. Raw data 不可變規則

1. `data/raw/v1/` 視為 immutable input。
2. 不得覆寫 Parquet、JSON 或 CSV。
3. 衍生資料必須寫入 `data/processed/v1/`、`data/contracts/v1/`、`artifacts/v1/` 或 `reports/v1/`。
4. 所有衍生輸出必須保存 raw dataset、raw record ID／locator 與 contract version。
5. EvidenceBench 的候選句順序與 sentence index 不得變動。

## 3. Canonical ID 規則

### 3.1 Text normalization

只為穩定 ID 使用，不改寫實際模型輸入文字：

```text
normalized_text =
  Unicode NFKC
  → trim leading/trailing whitespace
  → collapse consecutive whitespace to one ASCII space
```

### 3.2 Stable text hash

```text
text_hash = SHA-256(normalized_text UTF-8) 的前 16 個 hex characters
```

### 3.3 Canonical IDs

```text
MedFact claim_id:
  medfact:{claim_pmid}:{claim_text_hash}

MedFact split_group_id:
  medfact-group:{claim_pmid}:{claim_text_hash}

MedFact article_id:
  pmid:{source_pmid}

MedFact judgment_id:
  medfact-pair:{idx}

EvidenceBench query_id:
  evidencebench:{raw_record_id}

HealthFC case_id:
  healthfc:{claim_text_hash}
```

若原始 source ID 缺失，Phase 01 必須顯式使用 fallback hash，並在 provenance 中記錄 `id_fallback=true`；不得靜默產生空 ID。

## 4. 共通 provenance

每個 canonical record 至少包含：

```json
{
  "dataset": "dataset_name",
  "contract_version": "1.0.0",
  "raw_locator": {
    "path": "relative/raw/path",
    "record_id": "raw record id or row id"
  }
}
```

對 Parquet 可額外保存 shard 名稱；對 EvidenceBench 保存頂層 object key；對 CSV 保存 zero-based row index。

---

# 5. MedFact-Synth mappings

## 5.1 Raw fields

| Canonical meaning | Raw field |
|---|---|
| row ID | `idx` |
| Claim source PMID | `claim_pmid` |
| Claim generation potential | `claim_potential` |
| Claim text | `claim` |
| Article PMID | `source_pmid` |
| Article title + abstract text | `source` |
| Five-level stance | `synthetic_label` |
| Labeling prompt/rationale material | `system_prompt`, `user_prompt`, `assistant_prompt` |

Prompt fields不進入 Retriever 或 Verifier 的正式輸入。

## 5.2 ClaimRecord

```json
{
  "record_type": "ClaimRecord",
  "claim_id": "medfact:30588127:<hash>",
  "dataset": "medfact_synth",
  "text": "...",
  "source_reference": {
    "type": "pmid",
    "value": "30588127"
  },
  "split_group_id": "medfact-group:30588127:<hash>",
  "contract_version": "1.0.0",
  "raw_locator": {
    "path": "data/raw/v1/medfact_synth/data/<shard>.parquet",
    "record_id": "0"
  }
}
```

## 5.3 ArticleRecord

```json
{
  "record_type": "ArticleRecord",
  "article_id": "pmid:30588127",
  "dataset": "medfact_synth",
  "pmid": "30588127",
  "text": "title and abstract-like source text",
  "contract_version": "1.0.0",
  "raw_locator": {
    "path": "data/raw/v1/medfact_synth/data/<shard>.parquet",
    "record_id": "0"
  }
}
```

Phase 00 不嘗試從 `source` 字串可靠拆出 title 與 abstract；原文完整保存。

## 5.4 ClaimArticleJudgment

```json
{
  "record_type": "ClaimArticleJudgment",
  "judgment_id": "medfact-pair:0",
  "claim_id": "medfact:30588127:<hash>",
  "article_id": "pmid:30588127",
  "stance_label": 2,
  "relevance_grade": 2,
  "is_origin_source": true,
  "claim_potential": "support",
  "dataset": "medfact_synth",
  "contract_version": "1.0.0"
}
```

衍生規則：

```text
stance_label    = int(synthetic_label)
relevance_grade = abs(stance_label)
is_origin_source = str(claim_pmid) == str(source_pmid)
```

---

# 6. EvidenceBench-100k mappings

## 6.1 Raw fields

| Canonical meaning | Raw field |
|---|---|
| Query text | `hypothesis` |
| Candidate sentences | `paper_as_candidate_pool` |
| Gold aspect IDs | `aspect_list_ids` |
| Aspect text | `aspect_id2aspect` |
| Aspect → sentence indices | `aspect2sentence_indices` |
| Sentence index → aspects | `sentence_index2aspects` |
| Results-only aspect IDs | `results_aspect_list_ids` |
| Official optimal evaluation budget | `evidence_retrieval_at_optimal_evaluation["optimal"]` when present; otherwise derive the same exact minimum set-cover budget from the gold aspect mappings |
| Candidate sentence type | `sentence_types_in_candidate_pool` |
| Paper ID | `paper_id` |
| Review grouping ID | `systematic_review_id` |

Raw evaluation/result fields：

```text
evidence_retrieval_at_optimal_evaluation
results_evidence_retrieval_at_5_evaluation
results_evidence_retrieval_at_optimal_evaluation
```

均不得進入 scorer 或模型輸入。Phase 05 canonical loader 優先從第一個欄位抽取 `optimal` 整數；若 100k raw record 缺少該欄位，則從已驗證一致的 `aspect2sentence_indices`／`sentence_index2aspects` 精確求解最小 set cover。結果保存為 evaluator-only 的 `optimal_sentence_budget`，並以 `optimal_sentence_budget_source` 標記 `raw` 或 `derived`；results evaluation 欄位不進 canonical model artifact。

## 6.2 EvidenceQueryRecord

```json
{
  "record_type": "EvidenceQueryRecord",
  "query_id": "evidencebench:train_0",
  "dataset": "evidencebench_100k",
  "hypothesis": "...",
  "paper_id": "pmc_1626393",
  "systematic_review_id": "18087584",
  "candidate_sentences": ["..."],
  "sentence_types": ["abstract"],
  "aspect_ids": ["train_0_aspect_0"],
  "aspect_text": {
    "train_0_aspect_0": "..."
  },
  "aspect_to_sentence_indices": {
    "train_0_aspect_0": [13, 119]
  },
  "sentence_to_aspects": {
    "13": ["train_0_aspect_0"]
  },
  "results_aspect_ids": null,
  "optimal_sentence_budget": 2,
  "pool_fingerprint": "<sha256 of paper_id and exact ordered sentence list>",
  "contract_version": "1.0.0",
  "raw_locator": {
    "path": "data/raw/v1/evidencebench_100k/evidencebench_100k_train_set.json",
    "record_id": "train_0"
  }
}
```

## 6.3 Index invariants

對每個 record：

```text
len(sentence_types) == len(candidate_sentences)

對所有 aspect_to_sentence_indices 中的 index：
0 <= index < len(candidate_sentences)

aspect IDs 必須存在於 aspect_text 或明確標記缺失
```

`sentence_to_aspects` 是反向索引。Validator 檢查它與 `aspect_to_sentence_indices` 的一致性；若資料集本身存在不一致，必須在驗證報告中列為 warning/error，不得靜默修復 raw data。

---

# 7. HealthFC mappings

## 7.1 Raw fields

| Canonical meaning | Raw field |
|---|---|
| English Claim | `en_claim` |
| Human explanation | `en_explanation` |
| Gold evidence text | `en_top_sentences` |
| Verdict label | `label` |
| Source URL | `url` |
| Authors | `authors` |
| Date | `date` |

德文欄位保留在 raw data，不作 V1 英文 Verifier 的正式輸入。

## 7.2 ExpertVerdictRecord

```json
{
  "record_type": "ExpertVerdictRecord",
  "case_id": "healthfc:<claim_hash>",
  "dataset": "healthfc",
  "claim": "...",
  "evidence_text": "...",
  "label_id": 0,
  "verdict": "SUPPORTED",
  "explanation": "...",
  "source_url": "...",
  "authors": "...",
  "date": "...",
  "contract_version": "1.0.0",
  "raw_locator": {
    "path": "data/raw/v1/healthfc/Datensatz.csv",
    "record_id": "0"
  }
}
```

`explanation` 屬於 reference metadata，只可用於錯誤分析、展示或未來 explanation evaluation；不得進 Verifier 輸入。

## 7.3 Label mapping

```text
0 → SUPPORTED
1 → NOT_ENOUGH_INFORMATION
2 → REFUTED
```

---

# 8. Canonical analysis records 與前端 ViewModel 分離

目前 repo 已有：

```text
data/contracts/case_contract_draft.json
frontend/src/types.ts
```

它們描述的是產品展示所需的 Claim／Inference／Evidence／Gap ViewModel。

Phase 00 新增的 canonical analysis records 描述的是：

```text
raw dataset
→ retrieval/evidence/verification input-output
```

兩者不得直接合併。後續完整 Pipeline 階段建立明確 adapter：

```text
RetrievalResult
EvidenceResult
StanceResult
        ↓ adapter
Evidence Graph / DemoCase ViewModel
```

前端 layout、position、selection、display label 等欄位不得進入 ML data contract。

---

# 9. Phase 01 可新增但 Phase 00 不產生的欄位

```text
split
split_seed
split_strategy
corpus_document_offset
embedding_shard
index_version
model_revision
```

Phase 00 只固定命名與邊界，不提前建立正式 split 或索引。

# 10. Machine-readable outputs

執行：

```bash
python scripts/bootstrap_v1_contracts.py --root .
```

產生：

```text
data/contracts/v1/
├── dataset_mappings.json
├── label_maps.json
└── examples/
    ├── medfact_claim_article.json
    ├── evidencebench_query.json
    └── healthfc_verdict.json
```

腳本必須可重複執行，且不修改 `data/raw/v1/`。
