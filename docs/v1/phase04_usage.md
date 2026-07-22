# EvidenceGap V1 Phase 04 Usage

Phase 04 consumes the saved Phase 02/03 TREC runs. It does not rebuild the corpus,
BM25 index, dense embeddings, or FAISS indexes.

## 1. Three-way RRF fusion smoke test

All source runs must use the same split and query limit.

```bash
python scripts/run_v1_phase04.py fuse \
  --root . \
  --split dev \
  --source bm25=bm25s_default \
  --source medcpt=medcpt_ivf-sq-fp16_nprobe1024 \
  --source bmretriever=bmretriever_ivf-sq-fp16_nprobe1024 \
  --method rrf \
  --rrf-k 60 \
  --run-name rrf_bm25_medcpt_bmretriever_k60 \
  --max-queries 100 \
  --force
```

The command writes:

```text
artifacts/v1/reranking/candidates/<split>_<run>.parquet
artifacts/v1/reranking/candidates/<split>_<run>.manifest.json
artifacts/v1/reranking/runs/*.trec
reports/v1/article_retrieval_<run>_<split>.json
reports/v1/article_retrieval_<run>_<split>.md
```

The candidate parquet keeps the full union before Top-100 truncation. It includes
source ranks, source scores, source mask, RRF score, and fused rank. The report
also records full-union known-positive recall, which is the candidate coverage
ceiling before reranking.

## 2. Deterministic union baseline

`union` orders candidates by best source rank, then source agreement, then rank
sum. It is a deterministic non-learned baseline; its TREC score only preserves
that ordering.

```bash
python scripts/run_v1_phase04.py fuse \
  --root . \
  --split dev \
  --source medcpt=medcpt_ivf-sq-fp16_nprobe1024 \
  --source bmretriever=bmretriever_ivf-sq-fp16_nprobe1024 \
  --method union \
  --run-name union_medcpt_bmretriever \
  --max-queries 100 \
  --force
```

## 3. Cross-encoder reranking

The MedCPT cross encoder receives `(claim, title + abstract)` pairs, truncated to
512 tokens. Its single raw logit is ranked descending; higher means more relevant.

After setting `CUDA_VISIBLE_DEVICES`, pass visible devices starting at `0`:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/run_v1_phase04.py rerank \
  --root . \
  --split dev \
  --candidate-path artifacts/v1/reranking/candidates/dev_rrf_bm25_medcpt_bmretriever_k60.parquet \
  --candidate-run-name rrf_bm25_medcpt_bmretriever_k60 \
  --run-name medcpt_cross_rrf3_k60 \
  --devices 0,1,2,3 \
  --num-shards 4 \
  --batch-size 16 \
  --amp fp16 \
  --rerank-depth 100 \
  --max-queries 100 \
  --force
```

Each process loads one model copy and scores one or more query shards. Input and
score shards are written atomically. Existing score shards are reused only when
the input checksum, model fingerprint, maximum length, and precision match.

The reranking report includes mean cross-encoder score by relevance grade and a
pairwise score-direction diagnostic. A non-monotonic mean on a very small smoke
sample is reported but does not silently reverse logits.

`--rerank-depth` limits expensive cross-encoder scoring to the leading fused
candidates. It defaults to `--top-k` and must be at least `--top-k`. When both
are 100, the reranker changes only the order of the original RRF Top-100; it
cannot introduce candidates from fusion ranks 101 onward or remove a Top-100
candidate. The full union remains in the reranked parquet, with unscored rows
placed after the reranked block in their original fusion order. The JSON report
records a per-track candidate-set preservation audit.

## 4. Recommended progression

```text
100-query dev smoke
→ 1,000-query dev run
→ full dev for one or two fixed configurations
→ freeze configuration
→ run test once
```

Use Dev only for source selection, RRF weights, candidate choices, and runtime
parameters. Do not tune on Test.

## 5. Compare reports

```bash
python scripts/run_v1_phase04.py compare \
  --root . \
  --split dev \
  --report reports/v1/article_retrieval_bm25_dev.json \
  --report reports/v1/article_retrieval_medcpt_ivf-sq-fp16_nprobe1024_dev.json \
  --report reports/v1/article_retrieval_bmretriever_ivf-sq-fp16_nprobe1024_dev.json \
  --report reports/v1/article_retrieval_rrf_bm25_medcpt_bmretriever_k60_dev.json \
  --report reports/v1/article_retrieval_medcpt_cross_rrf3_k60_dev.json \
  --output-stem article_retrieval_phase04_comparison_dev
```

## Torch 2.5 and MedCPT safe weights

The MedCPT Cross Encoder main branch contains a pickle-based
`pytorch_model.bin`. Current Transformers releases reject that file when
PyTorch is older than 2.6 because of CVE-2025-32434. Do not upgrade the fixed
CUDA environment only for this model. Download the verified safetensors
conversion instead:

```bash
python scripts/download_v1_models.py \
  --root . \
  --model medcpt-cross
```

The downloader retains the tokenizer/config files from the model's main
revision and fetches the equivalent `model.safetensors` from the verified
conversion commit. Phase 04 reranking intentionally requires safetensors.
