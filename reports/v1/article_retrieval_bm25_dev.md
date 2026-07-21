# BM25 Article Retrieval — dev

Run: `bm25s_default`

| Track | Queries | MRR | nDCG@10 | Recall@10 | KP Recall@100 | HitRate@100 |
|---|---:|---:|---:|---:|---:|---:|
| independent | 20641 | 0.9006 | 0.8920 | 1.0000 | 0.8043 | 0.8732 |
| origin | 13758 | 1.0000 | 1.0000 | 1.0000 | 0.9997 | 0.9997 |
| overall | 25598 | 0.9579 | 0.9398 | 1.0000 | 0.8690 | 0.9454 |

## Interpretation boundary

Judged Candidate Ranking is evaluated only on annotated MedFact pairs.
Open-Corpus results report known-positive recall under incomplete judgments; unjudged articles are not confirmed negatives.
The independent-source track is the primary result.
