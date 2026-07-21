# EvidenceGap V1 Phase 02 Usage

## Install

```bash
python -m pip install -r requirements/v1-phase02.txt
```

## Recommended smoke test

```bash
python scripts/run_v1_phase02.py build-corpus \
  --root . \
  --output-dir artifacts/v1/article_corpus_quick \
  --quick-rows 50000

python scripts/run_v1_phase02.py build-index \
  --root . \
  --corpus-dir artifacts/v1/article_corpus_quick \
  --index-dir artifacts/v1/bm25_index_quick

python scripts/run_v1_phase02.py query \
  --root . \
  --index-dir artifacts/v1/bm25_index_quick \
  "The intervention reduces blood pressure"
```

The quick corpus contains only the first requested Phase 01 rows. It is for an
engineering smoke test, not a valid benchmark.

## Full corpus and baseline index

```bash
python scripts/run_v1_phase02.py build-corpus \
  --root . \
  --threads 16 \
  --memory-limit 32GB

python scripts/run_v1_phase02.py build-index \
  --root . \
  --k1 1.2 \
  --b 0.75

python scripts/run_v1_phase02.py validate --root .
```

`build-index` materializes token IDs during construction. BM25S supports mmap
when loading the completed index, but indexing the 1.32M-article corpus still
requires meaningful host RAM. Use the quick path first and monitor peak RSS on
the full build.

## Development evaluation

```bash
python scripts/run_v1_phase02.py run \
  --root . \
  --split dev \
  --top-k 100
```

For a small execution check:

```bash
python scripts/run_v1_phase02.py run \
  --root . \
  --split dev \
  --top-k 100 \
  --max-queries 100
```

## Final test evaluation

Run only after the implementation and parameters are fixed on dev:

```bash
python scripts/run_v1_phase02.py run \
  --root . \
  --split test \
  --top-k 100
```

## Output boundary

- `artifacts/v1/article_corpus/`: canonical engine-independent corpus and qrels.
- `artifacts/v1/bm25_index/`: BM25S index and exact tokenizer/index manifest.
- `artifacts/v1/article_retrieval_runs/`: TREC-style run files.
- `reports/v1/`: JSON and Markdown metrics.

Judged Candidate Ranking and Open-Corpus Known-positive Recall remain separate.
Unjudged open-corpus articles are never treated as confirmed negatives.
