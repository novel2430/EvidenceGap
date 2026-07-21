# BM25 Article Retrieval — dev

Run: `bm25s_default`  
Source: `saved_trec`

Queries in split: 26,644
Queries with eligible judgments: 26,644
Queries evaluated in this run: 26,644
Excluded without eligible judgments: 0

| Track | Eligible | Excl. no candidates | Excl. no positive | MRR | nDCG@3 | nDCG@5 | Top-1 positive | Mean Top-1 grade | Pairwise acc. | KP Recall@10 | KP Recall@100 | HitRate@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| independent | 20641 | 286 | 5717 | 0.9006 | 0.8800 | 0.8920 | 0.8071 | 1.2514 | 0.6884 | 0.5520 | 0.8043 | 0.8732 |
| origin | 13758 | 12856 | 30 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.9644 | n/a | 0.9943 | 0.9997 | 0.9997 |
| overall | 25598 | 0 | 1046 | 0.9579 | 0.9236 | 0.9398 | 0.9184 | 1.6851 | 0.8043 | 0.6960 | 0.8690 | 0.9454 |

## Failure cases

Independent-track failures: 10,193

Path: `reports/v1/article_retrieval_bm25_dev_failures.jsonl`

A failure record is emitted when judged Top-1 is not positive, graded relevance is inverted, no known positive appears in Top-K, or the first known positive appears after rank 50.

## Interpretation boundary

Judged Candidate Ranking is evaluated only on annotated MedFact pairs.
Open-Corpus results report known-positive recall under incomplete judgments; unjudged articles are not confirmed negatives.
The independent-source track is the primary result.
Judged Recall@10/50 remain in JSON only for compatibility and should not be used as headline metrics because candidate pools contain at most five documents.
