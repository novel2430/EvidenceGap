# V1 Phase 03 — Dense Article Retrieval

Phase 03 reuses the Phase 02 canonical corpus, Claim catalog, judgments and evaluator. It adds two independent dense baselines:

- MedCPT Query Encoder + Article Encoder
- BMRetriever-410M

Judged candidate ranking uses exact dot products against stored article embeddings. Open-corpus retrieval uses FAISS IVF. This keeps ANN approximation out of judged metrics.

## 1. Install dependencies

Use the already working CUDA-compatible PyTorch installation, then:

```bash
python -m pip install -r requirements/v1-phase03.txt
```

## 2. Prepare title/abstract inputs

Phase 02 normalized whitespace for BM25. This step recovers the selected raw source and preserves its title/abstract boundary without changing the Phase 02 corpus.

```bash
python scripts/run_v1_phase03.py prepare-inputs \
  --root . \
  --threads 16 \
  --memory-limit 64GB
```

Output:

```text
artifacts/v1/dense/article_inputs/
├── article_inputs.parquet
└── article_inputs_manifest.json
```

## 3. Encode MedCPT articles with eight GPUs

`--devices` uses visible CUDA indices. Completed shards are reused automatically; an interrupted shard is rebuilt without discarding completed shards.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/run_v1_phase03.py encode-articles \
  --root . \
  --model medcpt \
  --devices 0,1,2,3,4,5,6,7 \
  --num-shards 8 \
  --batch-size 64
```

If memory is insufficient, reduce `--batch-size` to 32 or 16.

Encode dev queries:

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase03.py encode-queries \
  --root . \
  --model medcpt \
  --split dev \
  --device cuda:0 \
  --batch-size 256
```

## 4. Build MedCPT FAISS index

The default index stores IVF vectors as FP16 scalar-quantized values while preserving inner-product search.

```bash
python scripts/run_v1_phase03.py build-index \
  --root . \
  --model medcpt \
  --index-type ivf-sq-fp16 \
  --nlist 4096 \
  --nprobe 64 \
  --train-size 200000 \
  --threads 16
```

## 5. Run a small MedCPT dev check

```bash
python scripts/run_v1_phase03.py run \
  --root . \
  --model medcpt \
  --split dev \
  --nprobe 64 \
  --top-k 100 \
  --max-queries 100
```

Then run the full dev set without `--max-queries`.

You can compare ANN search depth without rebuilding or re-encoding:

```bash
python scripts/run_v1_phase03.py run --root . --model medcpt --split dev --nprobe 32
python scripts/run_v1_phase03.py run --root . --model medcpt --split dev --nprobe 64
python scripts/run_v1_phase03.py run --root . --model medcpt --split dev --nprobe 128
```

## 6. Encode BMRetriever

BMRetriever follows its official input contract: task instruction for queries, passage prefix for documents, explicit EOS, last-token pooling, and unnormalized dot product.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/run_v1_phase03.py encode-articles \
  --root . \
  --model bmretriever \
  --devices 0,1,2,3,4,5,6,7 \
  --num-shards 8 \
  --batch-size 8
```

```bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/run_v1_phase03.py encode-queries \
  --root . \
  --model bmretriever \
  --split dev \
  --device cuda:0 \
  --batch-size 32
```

```bash
python scripts/run_v1_phase03.py build-index \
  --root . \
  --model bmretriever \
  --index-type ivf-sq-fp16 \
  --nlist 4096 \
  --nprobe 64 \
  --train-size 200000 \
  --threads 16
```

## 7. Validate

```bash
python scripts/run_v1_phase03.py validate \
  --root . \
  --model medcpt \
  --split dev
```

Repeat for BMRetriever.

## 8. Compare BM25 and dense runs

```bash
python scripts/run_v1_phase03.py compare \
  --root . \
  --split dev \
  --report reports/v1/article_retrieval_bm25_dev.json \
  --report reports/v1/article_retrieval_medcpt_ivf-sq-fp16_nprobe64_dev.json \
  --report reports/v1/article_retrieval_bmretriever_ivf-sq-fp16_nprobe64_dev.json
```

Only after selecting FAISS `nprobe` on dev should you encode test queries and run test.

## Artifact structure

```text
artifacts/v1/dense/
├── article_inputs/
├── medcpt/
│   ├── article_embeddings/
│   ├── query_embeddings/
│   └── faiss_index/
└── bmretriever/
    ├── article_embeddings/
    ├── query_embeddings/
    └── faiss_index/
```

## Evaluation boundary

- MedFact judged candidate pools contain at most five articles; headline judged metrics are MRR, nDCG@5, Top-1 positive rate and pairwise accuracy.
- Open-corpus results use incomplete judgments and report known-positive recall/HitRate only.
- Independent-source is the primary track.
- MedCPT and BMRetriever use their official unnormalized dot-product embedding rules.
