# EvidenceGap V1 Phase 07.2 Retrieval Adapters

## Scope

Phase 07.2 connects the frozen Phase 04 and Phase 05 retrieval configurations to
arbitrary runtime claims. It does not run the Phase 06 stance judge.

```text
arbitrary English medical claim
→ Phase 04 runtime article retrieval
→ Stanza GENIA runtime sentence materialization
→ Phase 05 runtime sentence retrieval per article
→ Top-5 evidence candidates per Top-10 article
```

The frozen values are recorded in:

```text
configs/v1/phase07_retrieval_adapters_frozen.json
```

## Runtime command

After exposing the intended physical GPU with `CUDA_VISIBLE_DEVICES`, use the
process-local device number. For example, physical GPU 3 becomes `cuda:0`:

```bash
CUDA_VISIBLE_DEVICES=3 \
python scripts/run_v1_phase07.py retrieve-evidence \
  --root . \
  --claim "Vitamin D supplementation prevents respiratory infections." \
  --run-name vitamin_d_phase072 \
  --device cuda:0
```

Use `--force` only when intentionally replacing the whole run directory.

The runtime command expects the previously built Phase 02–04 assets at their
standard paths:

```text
artifacts/v1/article_corpus/
artifacts/v1/bm25_index/
artifacts/v1/dense/article_inputs/
artifacts/v1/dense/medcpt/faiss_index/
artifacts/v1/dense/bmretriever/faiss_index/
models/v1/medcpt-query/
models/v1/medcpt-article/
models/v1/bmretriever-410m/
models/v1/medcpt-cross/
models/v1/stanza/en/tokenize/genia.pt
```

All paths have CLI overrides when a local installation uses different locations.

## Frozen runtime flow

### Article retrieval

```text
BM25 Top-100
+ MedCPT Top-100 (FAISS nprobe=1024)
+ BMRetriever Top-100 (FAISS nprobe=1024)
→ equal-weight RRF, k=60
→ fixed RRF Top-100
→ MedCPT Cross-Encoder reranking over all 100
→ final Top-10 articles
```

The dense runtime queries use the same frozen `nprobe=1024` recorded by the
formal Phase 04 Dev/Test runs. The adapter preserves each source rank and score, the RRF score and rank, the
cross-encoder logit, and the final article rank. The cross encoder changes only
the order of the fixed RRF Top-100 candidate set.

### Runtime sentence materialization

The final Top-10 articles are passed through the existing
`phase07.runtime-sentence.v1` contract. Titles remain one separate sentence;
abstract sections are segmented by Stanza GENIA and retain source offsets.

### Sentence retrieval

For each article independently:

```text
BMRetriever Top-20
+ MedCPT Top-20
→ equal-weight RRF, k=10
→ Top-5 evidence candidates
```

Sentence candidates from different articles never compete in one common pool.
The adapter uses exact RuntimeSentence IDs and texts; it does not resplit or
deduplicate them and does not use the article-level cross encoder at sentence
level.

Before either sentence retriever runs, the frozen
`phase07.evidence-sentence-eligibility.v1` policy removes only:

- `sentence_type=title` rows; and
- sentences from recovered structured sections whose text has no sentence-final
  punctuation.

The second rule targets source-truncated fragments such as an inline
`CONCLUSION:` value cut off by the MedFact source. The RuntimeSentence remains
in the materialization artifact for provenance, but it cannot occupy an
Evidence Top-5 slot. Plain `section=abstract` text is not filtered by this
heuristic because no reliable section boundary exists.

## Artifacts

A run is stored under:

```text
artifacts/v1/pipeline/retrieval_adapters/<run-name>/
├── request.json
├── article_retrieval/
│   ├── article_candidates.parquet
│   ├── top_articles.parquet
│   ├── runtime_articles.jsonl
│   └── run_manifest.json
├── sentence_materialization/
│   ├── runtime_articles.parquet
│   ├── runtime_sentences.parquet
│   ├── runtime_sentences.jsonl
│   └── run_manifest.json
├── evidence_retrieval/
│   ├── sentence_rankings.parquet
│   ├── evidence_candidates.parquet
│   ├── evidence_candidates.jsonl
│   └── run_manifest.json
└── run_manifest.json
```

`evidence_candidates.parquet` is the direct input intended for the next Phase
07 step that adapts Phase 06 stance verification. It contains the original
claim, article/PMID provenance, RuntimeSentence identity and offsets, both
sentence-retriever ranks and scores, RRF score, and evidence rank within article.

## Manual validation

```bash
python scripts/run_v1_phase07.py validate-retrieval-adapters \
  --root . \
  --artifact-dir artifacts/v1/pipeline/retrieval_adapters/vitamin_d_phase072
```

Inspect the final article and sentence choices:

```bash
python - <<'PY'
import pyarrow.parquet as pq

articles = pq.read_table(
    "artifacts/v1/pipeline/retrieval_adapters/vitamin_d_phase072/"
    "article_retrieval/top_articles.parquet"
).to_pylist()
evidence = pq.read_table(
    "artifacts/v1/pipeline/retrieval_adapters/vitamin_d_phase072/"
    "evidence_retrieval/evidence_candidates.parquet"
).to_pylist()

for article in articles:
    print(article["final_article_rank"], article["pmid"], article["title"])
    for row in evidence:
        if row["article_id"] == article["article_id"]:
            print("  ", row["evidence_rank_within_article"], row["sentence_text"])
PY
```

A successful validation guarantees:

- ten unique articles with contiguous final ranks;
- stable sentence identities and offset round trips;
- evidence candidates cover every Top-10 article;
- evidence ranks are contiguous within each article;
- selected evidence maps exactly back to the RuntimeSentence artifact;
- titles and unterminated structured-section fragments are absent from both
  sentence rankings and final evidence;
- output checksums and stage manifests are intact.

## Resource behavior

The adapters load heavy models sequentially on one process-visible GPU. The
BMRetriever query and passage encoders share the same in-memory checkpoint copy,
which avoids loading the 410M model twice on an 11 GB RTX 2080 Ti. Models are
released between major stages before the next model family is loaded.
