# EvidenceGap V1 Phase 04 Article Retrieval Comparison

## 1. Executive Summary

Phase 04 evaluates whether a hybrid candidate pipeline can preserve MedCPT's near-complete open-corpus recall while improving the front-ranked article quality beyond the strongest Phase 03 single retriever, BMRetriever.

The selected configuration is:

```text
BM25 Top-100
+ MedCPT Top-100
+ BMRetriever Top-100
→ Reciprocal Rank Fusion, k = 60
→ Fixed RRF Top-100 candidate set
→ MedCPT Cross-Encoder reranking within Top-100
```

On the official MedFact Test independent-source track, the final configuration achieved:

```text
MRR                              0.9301
nDCG@5                           0.9158
Top-1 positive rate              86.37%
Pairwise ordering accuracy       75.88%
Known-positive Recall@100        99.9975%
HitRate@100                      100.00%
```

Compared with the Phase 03 best single retriever, BMRetriever, the final Phase 04 pipeline improved all principal front-ranking metrics while increasing Known-positive Recall@100 from 95.23% to 99.9975% and HitRate@100 from 97.37% to 100%.

The Dev and Test results are nearly identical, providing no evidence of material Dev-specific overfitting.

---

## 2. Research Question

Phase 04 is designed to answer:

> Can the system preserve MedCPT's near-complete open-corpus recall while converting that recall advantage into better Top-1, nDCG and graded relevance ranking than BMRetriever?

The pre-defined target thresholds were:

```text
Known-positive Recall@100  >= 99.5%
Top-1 positive rate        > 83.76%
nDCG@5                     > 0.9045
Pairwise accuracy          > 72.37%
```

The final Test result passed all four thresholds.

---

## 3. Evaluation Contract

### 3.1 Primary track

The primary result is the MedFact independent-source track:

```text
claim_pmid != source_pmid
```

Origin-source results are retained only as a lexical-shortcut sanity check. Overall results are supplementary and must not replace the independent-source result.

### 3.2 Open-corpus metrics

The 1.32-million-article corpus has incomplete qrels. Therefore, open-corpus evaluation reports only:

```text
Known-positive Recall@K
HitRate@K
```

Unjudged articles are not treated as confirmed negatives. Precision and accuracy are not reported for open-corpus retrieval.

### 3.3 Judged ranking metrics

Each claim usually has only one to five judged candidates, so judged Recall@10 and Recall@50 saturate and are not used for model comparison.

The formal judged ranking metrics are:

```text
MRR
nDCG@3
nDCG@5
Top-1 positive rate
Mean Top-1 relevance grade
Pairwise ordering accuracy
```

---

## 4. Fixed Final Configuration

The final configuration was selected using Dev only and then applied unchanged to Test.

### Candidate generators

```text
BM25 Top-100
MedCPT Top-100
BMRetriever Top-100
```

### Fusion

```text
Method: Reciprocal Rank Fusion
RRF k: 60
Source weights: 1.0 / 1.0 / 1.0
```

### Candidate policy

```text
RRF determines the final Top-100 candidate set.
The Cross-Encoder reranks only these Top-100 candidates.
The candidate membership is not allowed to change.
```

This policy avoids allowing documents ranked below RRF Top-100 to displace already-retrieved known positives.

### Reranker

```text
Model: models/v1/medcpt-cross
Input pair: claim, article title + abstract
Score: raw single relevance logit, higher is more relevant
Max length: 512
Precision: FP16
Batch size: 16 per GPU
GPUs: 4
Rerank depth: 100
Final top-k: 100
```

---

## 5. Main Test Comparison

All values below are from the MedFact Test independent-source track.

| Model | MRR | nDCG@5 | Top-1 positive | Mean Top-1 grade | Pairwise accuracy | Recall@10 | Recall@100 | HitRate@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.9044 | 0.8947 | 81.43% | — | 69.57% | 55.54% | 80.26% | 87.31% |
| MedCPT | 0.8717 | 0.8683 | 75.12% | — | 61.48% | 99.97% | 99.98% | 100.00% |
| BMRetriever | 0.9165 | 0.9045 | 83.76% | — | 72.37% | 76.10% | 95.23% | 97.37% |
| Three-way RRF | 0.9181 | 0.9048 | 84.06% | 1.3016 | 72.33% | 85.90% | 99.9975% | 100.00% |
| **RRF + MedCPT Cross-Encoder** | **0.9301** | **0.9158** | **86.37%** | **1.3376** | **75.88%** | **85.07%** | **99.9975%** | **100.00%** |

### 5.1 Final pipeline vs BMRetriever

The final Phase 04 pipeline improved over the Phase 03 best single retriever by:

```text
MRR                         +0.0136
nDCG@5                      +0.0113
Top-1 positive rate         +2.62 percentage points
Pairwise accuracy           +3.51 percentage points
Known-positive Recall@10    +8.97 percentage points
Known-positive Recall@100   +4.77 percentage points
HitRate@100                 +2.63 percentage points
```

The final pipeline therefore improves both front-ranked evidence quality and open-corpus known-positive coverage.

### 5.2 Retriever roles

The three retrieval systems remain complementary:

```text
MedCPT:
near-complete high-recall candidate generation

BMRetriever:
strong standalone dense ranking

BM25:
lexical matching for terminology, abbreviations, drug names,
gene names and numerical expressions
```

The final result does not support removing any of the three retrievers from the Phase 04 candidate pipeline.

---

## 6. RRF vs Cross-Encoder Ablation

The following table isolates the effect of Cross-Encoder reranking on the fixed RRF Top-100 candidate set.

| Test independent | Three-way RRF | RRF + Cross-Encoder | Delta |
|---|---:|---:|---:|
| MRR | 0.9181 | **0.9301** | +0.0120 |
| nDCG@5 | 0.9048 | **0.9158** | +0.0109 |
| Top-1 positive | 84.06% | **86.37%** | +2.31 pp |
| Mean Top-1 grade | 1.3016 | **1.3376** | +0.0360 |
| Pairwise accuracy | 72.33% | **75.88%** | +3.55 pp |
| Known-positive Recall@10 | **85.90%** | 85.07% | -0.83 pp |
| Known-positive Recall@50 | **99.28%** | 98.84% | -0.44 pp |
| Known-positive Recall@100 | 99.9975% | **99.9975%** | 0 |
| HitRate@100 | 100.00% | **100.00%** | 0 |
| Failure cases | 8,034 | **7,189** | -845 |

Failure cases decreased by 845, or approximately 10.5%.

### Interpretation

The Cross-Encoder substantially improves front-ranked and graded relevance ordering. Known-positive Recall@10 and Recall@50 decrease slightly, which indicates that the reranker prioritizes stronger relevance grades rather than maximizing the number of all known positives within shallow cutoffs.

This trade-off is acceptable for Phase 04 because:

1. the complete Top-100 candidate set is preserved;
2. Known-positive Recall@100 and HitRate@100 do not decline;
3. MRR, nDCG, Top-1 positive rate and pairwise ordering all improve;
4. later sentence retrieval and verification stages operate on the retained candidate articles.

No weighted blending or further Test-driven tuning is justified by these results.

---

## 7. Dev-Test Generalization

| Final Cross-Encoder configuration | Dev | Test |
|---|---:|---:|
| MRR | 0.9300 | 0.9301 |
| nDCG@5 | 0.9148 | 0.9158 |
| Top-1 positive | 86.40% | 86.37% |
| Mean Top-1 grade | 1.3433 | 1.3376 |
| Pairwise accuracy | 75.71% | 75.88% |
| Known-positive Recall@10 | 85.08% | 85.07% |
| Known-positive Recall@50 | 98.68% | 98.84% |
| Known-positive Recall@100 | 99.9976% | 99.9975% |
| HitRate@100 | 100.00% | 100.00% |

The principal Dev and Test metrics are nearly identical. In particular, Top-1 positive rate, Recall@10 and Recall@100 differ only at the level of rounding. The results provide no evidence of material Dev-specific overfitting from the Phase 04 configuration selection process.

---

## 8. Cross-Encoder Score Diagnostics

Test judged candidates show a monotonic relationship between relevance grade and the Cross-Encoder score.

| Relevance grade | Rows | Mean score | Standard deviation | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| 0 | 25,516 | 4.0769 | 7.9603 | -15.9922 | 16.0469 |
| 1 | 17,113 | 10.0484 | 5.8356 | -15.0156 | 16.0469 |
| 2 | 30,518 | 13.1273 | 4.5954 | -15.5859 | 16.0469 |

```text
Mean score monotonic by grade: true
Pairwise grade comparisons: 62,472
Pairwise score-direction accuracy: 82.26%
```

These diagnostics support the intended score semantics:

```text
higher score = stronger article relevance to the claim
```

The result also confirms that the model is not merely separating relevant from irrelevant articles; it captures useful relevance-strength ordering between grades 1 and 2.

---

## 9. Candidate Preservation Contract

The final reranking implementation must not alter RRF Top-100 membership.

Test diagnostics:

| Track | Baseline Top-100 rows | Reranked Top-100 rows | Missing | Added | Preserved |
|---|---:|---:|---:|---:|---:|
| Independent | 2,038,600 | 2,038,600 | 0 | 0 | true |
| Origin | 1,367,700 | 1,367,700 | 0 | 0 | true |
| Overall | 2,529,800 | 2,529,800 | 0 | 0 | true |

This contract explains why the final Cross-Encoder result has exactly the same Recall@100 as RRF.

The full-union known-positive ceiling and RRF Top-100 result are also equal on the independent track:

```text
Full-union Known-positive Recall: 99.9975%
RRF Top-100 Known-positive Recall: 99.9975%
CE Top-100 Known-positive Recall:  99.9975%
```

RRF therefore retains every known positive recovered by the complete three-way union, except for the same extremely small fraction already absent from the union itself.

---

## 10. Engineering Results

### Test workload

```text
Test claims                         26,303
Independent eligible queries       20,386
Unique claim-article pairs scored  2,538,910
Source candidate rows              13,898,111
Output candidate rows              13,898,111
Candidate rows lost                0
```

### Inference environment

```text
Hardware       4 x RTX 2080 Ti, 11 GB
Precision      FP16
Batch size     16 per GPU
Devices        cuda:0, cuda:1, cuda:2, cuda:3
Max length     512
Rerank depth   100
```

### Final artifacts

```text
reports/v1/article_retrieval_rrf_bm25_medcpt_bmretriever_k60_full_dev.json
reports/v1/article_retrieval_rrf_bm25_medcpt_bmretriever_k60_full_test.json
reports/v1/article_retrieval_medcpt_cross_rrf3_k60_d100_full_dev.json
reports/v1/article_retrieval_medcpt_cross_rrf3_k60_d100_full_test.json

artifacts/v1/reranking/candidates/
artifacts/v1/reranking/reranked_candidates/
artifacts/v1/reranking/runs/
```

The Python multiprocessing semaphore warnings printed at process shutdown did not correspond to missing rows, failed shards or incomplete reports. They remain a minor engineering cleanup item rather than a result-validity issue.

---

## 11. Limitations

1. **Incomplete open-corpus qrels.** Unjudged articles cannot be treated as confirmed negatives. Open-corpus precision and accuracy are therefore not reported.
2. **Small judged pools.** Judged Recall@10 and Recall@50 saturate because each claim normally has only one to five judged articles.
3. **Synthetic judgments.** MedFact-Synth provides scale but does not replace expert clinical evaluation.
4. **Shallow known-positive recall trade-off.** Cross-Encoder reranking slightly reduces Known-positive Recall@10 and Recall@50 while improving stronger relevance ranking.
5. **Article-level evaluation only.** Phase 04 does not establish that a highly ranked article contains the best extractable evidence sentence or that its stance is correctly classified.
6. **Large duplicated analytical outputs.** Candidate parquet files preserve multiple tracks and judged/open views, which produces substantial logical duplication. Storage normalization can be considered later without changing the retrieval method.

---

## 12. Final Conclusion

Phase 04 confirms that the three retrievers provide complementary signals. MedCPT supplies near-complete known-positive recall, BMRetriever supplies strong dense ranking, and BM25 contributes lexical matches that remain valuable for biomedical terminology.

Three-way RRF converts these signals into a high-recall Top-100 candidate set. MedCPT Cross-Encoder reranking then improves the internal ordering of this fixed candidate set without changing its membership.

On the official Test independent-source track, the final configuration achieves:

```text
MRR                              0.9301
nDCG@5                           0.9158
Top-1 positive rate              86.37%
Pairwise ordering accuracy       75.88%
Known-positive Recall@100        99.9975%
HitRate@100                      100.00%
```

It exceeds the Phase 03 best single retriever, BMRetriever, in every principal reported ranking and open-corpus recall metric. Dev and Test results are highly consistent.

Phase 04 is therefore complete. The selected article retrieval and reranking configuration should be frozen and carried forward into Phase 05 Evidence Sentence Retrieval without further Test-driven tuning.

