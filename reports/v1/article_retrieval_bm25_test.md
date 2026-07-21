# BM25 Article Retrieval — test

Run: `bm25s_default`  
Source: `live_retrieval`

Queries in split: 26,303
Queries with eligible judgments: 26,303
Queries evaluated in this run: 26,303
Excluded without eligible judgments: 0

| Track | Eligible | Excl. no candidates | Excl. no positive | MRR | nDCG@3 | nDCG@5 | Top-1 positive | Mean Top-1 grade | Pairwise acc. | KP Recall@10 | KP Recall@100 | HitRate@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | 20386 | 298 | 5619 | 0.9044 | 0.8827 | 0.8947 | 0.8143 | 1.2568 | 0.6957 | 0.5554 | 0.8026 | 0.8731 |
| origin | 13677 | 12603 | 23 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.9634 | n/a | 0.9959 | 0.9994 | 0.9994 |
| overall | 25298 | 0 | 1005 | 0.9600 | 0.9260 | 0.9414 | 0.9226 | 1.6902 | 0.8099 | 0.6998 | 0.8682 | 0.9466 |

## Failure cases

Independent-track failures: 10,002

Path: `reports/v1/article_retrieval_bm25_test_failures.jsonl`

A failure record is emitted when judged Top-1 is not positive, graded relevance is inverted, no known positive appears in Top-K, or the first known positive appears after rank 50.

## Interpretation boundary

Judged Candidate Ranking is evaluated only on annotated MedFact pairs.
Open-Corpus results report known-positive recall under incomplete judgments; unjudged articles are not confirmed negatives.
The independent-source track is the primary result.
Judged Recall@10/50 remain in JSON only for compatibility and should not be used as headline metrics because candidate pools contain at most five documents.
