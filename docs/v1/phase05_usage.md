# EvidenceGap V1 Phase 05：Evidence Sentence Retrieval

Phase 05 在「hypothesis 與 paper 已知」的條件下，對 EvidenceBench 提供的固定候選句進行排序。它不做文章檢索，也不判斷 support/refute stance。

```text
hypothesis + fixed ordered candidate sentences
→ ranked original zero-based sentence indices
→ aspect-based evaluation
```

## 1. 不可破壞的資料契約

- `paper_as_candidate_pool` 的文字、順序和句子邊界保持不變。
- 不刪除 section headings、空句、短句或重複文字。
- 不重新分句、deduplicate 或重新編號。
- `sentence_index` 永遠是 raw candidate pool 的原始 zero-based index。
- `len(sentence_types_in_candidate_pool)` 必須等於候選句數。
- forward `aspect2sentence_indices` 與 reverse `sentence_index2aspects` 必須完全一致。
- gold aspect、result aspect 與 raw evaluation result 不會進入 scorer 的 model-input projection。

正式 split source of truth：

```text
data/processed/v1/manifests/evidencebench_train.jsonl
data/processed/v1/manifests/evidencebench_dev.jsonl
data/processed/v1/manifests/evidencebench_test.jsonl
```

`--max-queries` 只截取 manifest 中的前 N 筆，因此 100/1000-query smoke subset 可重現。Raw JSON 由 `raw_locator` 定位，使用 `ijson` 一次串流掃描並只抽取需要的 records；不以 `json.load()` 載入完整資料集。

## 2. 安裝

先保留專案既有 CUDA-compatible PyTorch，再安裝：

```bash
pip install -r requirements/v1-phase05.txt
```

不要因 Phase 05 升級既有 PyTorch/CUDA。Cross-Encoder 只使用：

```text
models/v1/medcpt-cross/model.safetensors
```

不允許回退到 pickle-based `pytorch_model.bin`。

## 3. Canonical preparation 與 audit

100-query Dev audit：

```bash
python scripts/run_v1_phase05.py audit \
  --root . \
  --split dev \
  --max-queries 100
```

Materialize canonical artifact：

```bash
python scripts/run_v1_phase05.py prepare \
  --root . \
  --split dev \
  --max-queries 100
```

預設輸出：

```text
artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100/
├── queries.jsonl
└── canonical_manifest.json
```

Audit/report 會列出 candidate count、paper/pool conflict、空 aspect、空 results aspect，以及官方 per-query optimal budget 的最大值。這些 evaluation metadata 不會進入 model scoring。

## 4. 正式 evaluator 語意

### Aspect Recall@5

對具有非空 `aspect_ids` 的 query，若 Top-5 任一句落在某 aspect 的 gold sentence indices 中，該 aspect 視為 covered。先算 query-level recall，再 macro average。

### Aspect Recall@Optimal

每筆 query 優先使用 raw：

```text
evidence_retrieval_at_optimal_evaluation["optimal"]
```

若 raw record 缺少該欄位，loader 會由 gold aspect mappings 精確求解同一定義的最小 set-cover budget。該值作為 query cutoff K，再對 ranking Top-K 計算 aspect coverage。Audit 的 `optimal_budget_source_counts` 會分別統計 raw 與 derived 數量。K 只由 evaluator 使用；retriever 不會因 gold optimal 改變分數或排名。

若 run depth 小於某筆 query 的 K，evaluator 會 fail，而不是猜測或靜默降低 cutoff。可先用 `audit` 查看 `optimal_budget_max`，再設定足夠大的固定 `--top-k`。

### Results Aspect Recall@5

只納入 `results_aspect_ids` 非 null 且非空的 query。null/empty query 從 aggregate 排除，不計 0 分。

### Sentence Precision@5

Top-5 中具有至少一個 gold aspect 的句子數，除以實際返回句數。候選池少於 5 時分母為實際返回數；完全沒有 prediction 時定義為 0。

### First-hit MRR

第一個覆蓋任意 gold aspect 的句子 rank 取倒數；完全未命中為 0。

## 5. BM25 baseline

BM25 是 paper-local scoring，不建立全域 sentence index。Tokenization contract 與 Phase 02 相同：NFKC、lowercase、既有 token regex、無 stemming、無 stopword removal。

```bash
python scripts/run_v1_phase05.py bm25 \
  --root . \
  --split dev \
  --max-queries 100 \
  --top-k 50 \
  --run-name bm25_smoke100
```

相同 `pool_fingerprint` 的 hypotheses 會重用 tokenization、document frequency 與 BM25 pool statistics。

## 6. Dense sentence retrieval

### MedCPT

```text
query: MedCPT Query Encoder(exact hypothesis)
candidate: MedCPT Article Encoder(empty title, exact sentence)
pooling: CLS
similarity: inner product
normalization: false
```

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase05.py dense \
  --root . \
  --model medcpt \
  --split dev \
  --max-queries 100 \
  --devices 0 \
  --top-k 50 \
  --run-name medcpt_smoke100
```

### BMRetriever

```text
query: Phase 03 BMR_TASK + Query: hypothesis
candidate: Represent this passage\npassage: exact sentence
pooling: EOS + last token
similarity: inner product
normalization: false
```

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase05.py dense \
  --root . \
  --model bmretriever \
  --split dev \
  --max-queries 100 \
  --devices 0 \
  --top-k 50 \
  --run-name bmretriever_smoke100
```

Sentence embedding identity：

```text
model fingerprint + pool_fingerprint + original sentence_index + input format
```

Query embedding identity：

```text
model fingerprint + query_id + hypothesis SHA-256
```

同一 `paper_id` 若具有不同 ordered pool fingerprint，不會靜默合併。

## 7. Cross-Encoder reranking

輸入是：

```text
tokenizer(exact hypothesis, exact candidate sentence)
```

輸出是 raw single relevance logit，higher = more relevant。

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase05.py rerank \
  --root . \
  --split dev \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --candidate-path artifacts/v1/evidence_sentence_retrieval/runs/medcpt_smoke100/ranked_sentences.parquet \
  --candidate-run-name medcpt_smoke100 \
  --devices 0 \
  --rerank-depth 20 \
  --batch-size 16 \
  --run-name medcpt_cross_medcpt_smoke100
```

Cross-Encoder 只重排 candidate retrieval 的前 N 句。Top-N 外的句子保持原 retrieval 順序，不新增、不刪除、不因文字相同而合併 index。

## 8. Dev-only candidate fusion experiment

Fusion is intentionally limited to two fixed configurations. It never combines raw
MedCPT, BMRetriever, and Cross-Encoder scores because their scales are unrelated.
Gold annotations are used only by `complementarity` and `compare`; `fuse` itself
uses ranks only.

First inspect whether BMRetriever and MedCPT retrieve different useful sentences:

```bash
python scripts/run_v1_phase05.py complementarity \
  --root . \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --left artifacts/v1/evidence_sentence_retrieval/runs/bmretriever_smoke100/ranked_sentences.parquet \
  --right artifacts/v1/evidence_sentence_retrieval/runs/medcpt_smoke100/ranked_sentences.parquet \
  --left-name bmretriever \
  --right-name medcpt \
  --depths 5,10,20,50 \
  --report reports/v1/evidence_sentence_complementarity_smoke100.json
```

The report includes candidate overlap, model-only gold sentences, single-run
aspect coverage, and union aspect coverage. This is a Dev diagnostic only; it
must not be used to alter rankings per query.

### Configuration A: full Top-20 union, then score every union candidate

```bash
python scripts/run_v1_phase05.py fuse \
  --root . \
  --split dev \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --left artifacts/v1/evidence_sentence_retrieval/runs/bmretriever_smoke100/ranked_sentences.parquet \
  --right artifacts/v1/evidence_sentence_retrieval/runs/medcpt_smoke100/ranked_sentences.parquet \
  --left-name bmretriever \
  --right-name medcpt \
  --left-depth 20 \
  --right-depth 20 \
  --rrf-k 60 \
  --run-name bmr20_medcpt20_union_smoke100
```

Omitting `--output-depth` preserves every unique sentence from both Top-20
lists. RRF only gives the union a deterministic order; no candidate is filtered.
The union contains at most 40 sentences per query, so rerank all of them:

```bash
CUDA_VISIBLE_DEVICES=3 \
python scripts/run_v1_phase05.py rerank \
  --root . \
  --split dev \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --candidate-path artifacts/v1/evidence_sentence_retrieval/fusion/bmr20_medcpt20_union_smoke100/ranked_sentences.parquet \
  --candidate-run-name bmr20_medcpt20_union_smoke100 \
  --devices 0 \
  --rerank-depth 40 \
  --batch-size 16 \
  --max-length 512 \
  --amp fp16 \
  --run-name medcpt_cross_bmr20_medcpt20_union_smoke100
```

### Configuration B: Phase-04-style RRF Top-20, then Cross-Encoder

```bash
python scripts/run_v1_phase05.py fuse \
  --root . \
  --split dev \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --left artifacts/v1/evidence_sentence_retrieval/runs/bmretriever_smoke100/ranked_sentences.parquet \
  --right artifacts/v1/evidence_sentence_retrieval/runs/medcpt_smoke100/ranked_sentences.parquet \
  --left-name bmretriever \
  --right-name medcpt \
  --left-depth 50 \
  --right-depth 50 \
  --output-depth 20 \
  --rrf-k 60 \
  --run-name rrf_bmr50_medcpt50_top20_smoke100
```

Then rerank that fixed Top-20 with `--rerank-depth 20`.

Compare either challenger against the existing BMRetriever + Cross-Encoder run
using paired cluster bootstrap. The default bootstrap unit is
`systematic_review_id`, with `paper_id` as fallback, so related queries are not
silently treated as independent:

```bash
python scripts/run_v1_phase05.py compare \
  --root . \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --baseline artifacts/v1/evidence_sentence_retrieval/reranked/medcpt_cross_bmretriever_d20_smoke100/ranked_sentences.parquet \
  --challenger artifacts/v1/evidence_sentence_retrieval/reranked/medcpt_cross_bmr20_medcpt20_union_smoke100/ranked_sentences.parquet \
  --baseline-name bmr_ce_d20 \
  --challenger-name union_ce \
  --bootstrap-unit systematic_review \
  --bootstrap-samples 10000 \
  --report reports/v1/evidence_sentence_union_vs_bmr_ce_smoke100.json
```

The primary decision metric remains `Aspect Recall@5`. Smoke-100 is exploratory:
a positive mean delta is enough to justify Dev-1000, but configuration selection
must be confirmed on larger Dev. Do not inspect official Test complementarity or
bootstrap results before the final configuration is frozen.

## 9. Multi-GPU 與 device semantics

例如：

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 \
python scripts/run_v1_phase05.py dense \
  --root . \
  --model medcpt \
  --split dev \
  --max-queries 1000 \
  --devices 0,1,2,3 \
  --num-shards 8 \
  --top-k 50 \
  --run-name medcpt_smoke1000
```

`CUDA_VISIBLE_DEVICES` 生效後，程式內編號重新從 `cuda:0` 開始。Shards 會按 device 分組執行，避免同一張 GPU 同時載入多個 worker model。

Dense 以 `pool_fingerprint` deterministic 分 shard，使同一 sentence pool 的 hypotheses 落在同一 shard。Cross-Encoder 以 `sha256(query_id)` 分 shard。

## 10. Atomic shard 與 resume

每個 shard 都保存：

```text
input/canonical checksum
model fingerprint
parameters
shard index/count
output checksum
```

寫入採 temporary file + atomic rename。重跑時：

- 完整且 signature 相同的 shard直接重用；
- 只存在 parquet 或 metadata 的 orphan shard自動重算；
- signature 不同或 checksum 錯誤視為 stale/corrupt，必須用 `--force`；
- 不會只因檔案存在就視為成功。

## 11. Evaluate、diagnose、validate

```bash
python scripts/run_v1_phase05.py evaluate \
  --root . \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --run artifacts/v1/evidence_sentence_retrieval/runs/bm25_smoke100/ranked_sentences.parquet
```

Score direction diagnostic：

```bash
python scripts/run_v1_phase05.py diagnose \
  --root . \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --run artifacts/v1/evidence_sentence_retrieval/runs/medcpt_smoke100/ranked_sentences.parquet \
  --score-field retrieval_score
```

會輸出 gold/non-gold mean score 與 bounded pairwise accuracy。Diagnostic gold 僅用於結果分析，不改變 ranking。

Artifact validation：

```bash
python scripts/run_v1_phase05.py validate \
  --root . \
  --canonical-dir artifacts/v1/evidence_sentence_retrieval/canonical/dev_first_100 \
  --run artifacts/v1/evidence_sentence_retrieval/runs/bm25_smoke100/ranked_sentences.parquet \
  --run-name bm25_smoke100
```

## 12. Artifact layout

```text
artifacts/v1/evidence_sentence_retrieval/
├── canonical/
├── sentence_embeddings/
├── query_embeddings/
├── runs/
├── fusion/
└── reranked/
```

每列 ranking 至少保存：

```text
query_id
paper_id
pool_fingerprint
original sentence_index
sentence_type
sentence_text
retrieval model/score/rank
cross-encoder score
final rank
```

正式 identity 始終是：

```text
query_id + paper_id + original sentence_index
```

## 13. 建議第一輪執行順序

```text
Dev 100 audit
→ Dev 100 prepare
→ BM25 Dev 100
→ MedCPT Dev 100
→ BMRetriever Dev 100
→ Cross-Encoder Dev 100
→ diagnose
→ Dev 1000
→ full Dev
```

第一輪不要跑 official Test。模型、fixed top-k、rerank depth 與其他配置只能根據 Dev 決定；配置凍結後才執行一次正式 Test。
