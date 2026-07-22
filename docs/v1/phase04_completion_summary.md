# EvidenceGap V1 Phase 04 Completion Summary

## Status

```text
Phase 04: COMPLETE
```

Phase 04 has completed Dev selection and official Test evaluation. The final configuration is fixed and should not be tuned further using Test results.

---

## Final Pipeline

```text
BM25 Top-100
+ MedCPT Top-100
+ BMRetriever Top-100
→ RRF, k = 60
→ Fixed RRF Top-100 candidate set
→ MedCPT Cross-Encoder reranking, depth = 100
```

Inference configuration:

```text
Input          claim + article title and abstract
Max length     512
Precision      FP16
Batch size     16 per GPU
GPUs           4 x RTX 2080 Ti
Final top-k    100
```

The Cross-Encoder changes only the internal order of RRF Top-100. It cannot add or remove candidates from the final Top-100 set.

---

## Official Test Result

Primary track:

```text
MedFact independent-source
claim_pmid != source_pmid
```

| Metric | Final result |
|---|---:|
| MRR | 0.9301 |
| nDCG@5 | 0.9158 |
| Top-1 positive rate | 86.37% |
| Mean Top-1 relevance grade | 1.3376 |
| Pairwise ordering accuracy | 75.88% |
| Known-positive Recall@10 | 85.07% |
| Known-positive Recall@50 | 98.84% |
| Known-positive Recall@100 | 99.9975% |
| HitRate@100 | 100.00% |

---

## Comparison with Phase 03 Best Single Retriever

| Test independent | BMRetriever | Phase 04 final | Improvement |
|---|---:|---:|---:|
| MRR | 0.9165 | **0.9301** | +0.0136 |
| nDCG@5 | 0.9045 | **0.9158** | +0.0113 |
| Top-1 positive | 83.76% | **86.37%** | +2.62 pp |
| Pairwise accuracy | 72.37% | **75.88%** | +3.51 pp |
| Recall@10 | 76.10% | **85.07%** | +8.97 pp |
| Recall@100 | 95.23% | **99.9975%** | +4.77 pp |
| HitRate@100 | 97.37% | **100.00%** | +2.63 pp |

Phase 04 therefore achieves the intended combination of high open-corpus recall and stronger front-ranked article quality.

---

## RRF vs Cross-Encoder

| Test independent | RRF | RRF + Cross-Encoder |
|---|---:|---:|
| MRR | 0.9181 | **0.9301** |
| nDCG@5 | 0.9048 | **0.9158** |
| Top-1 positive | 84.06% | **86.37%** |
| Pairwise accuracy | 72.33% | **75.88%** |
| Recall@10 | **85.90%** | 85.07% |
| Recall@50 | **99.28%** | 98.84% |
| Recall@100 | 99.9975% | **99.9975%** |
| HitRate@100 | 100.00% | **100.00%** |

Cross-Encoder reranking improves front-ranked and graded relevance quality. Recall@10 and Recall@50 decline slightly, but the fixed Top-100 contract preserves Recall@100 and HitRate@100 exactly.

---

## Dev-Test Consistency

| Final configuration | Dev | Test |
|---|---:|---:|
| MRR | 0.9300 | 0.9301 |
| nDCG@5 | 0.9148 | 0.9158 |
| Top-1 positive | 86.40% | 86.37% |
| Pairwise accuracy | 75.71% | 75.88% |
| Recall@10 | 85.08% | 85.07% |
| Recall@100 | 99.9976% | 99.9975% |

The Dev and Test results are almost identical. No material Dev-specific overfitting was observed.

---

## Validation Contracts

```text
Independent Top-100 candidate set preserved   PASS
Origin Top-100 candidate set preserved        PASS
Overall Top-100 candidate set preserved       PASS
Missing reranked candidate rows               0
Added reranked candidate rows                 0
Source candidate rows                         13,898,111
Output candidate rows                         13,898,111
Unique claim-article pairs scored             2,538,910
```

Cross-Encoder score diagnostics on Test:

```text
Grade 0 mean score                 4.0769
Grade 1 mean score                10.0484
Grade 2 mean score                13.1273
Mean score monotonic by grade     true
Pairwise score-direction accuracy 82.26%
```

---

## Frozen Configuration

The following choices are now frozen:

```text
Retrievers        BM25 + MedCPT + BMRetriever
Retriever depth   100 per source
Fusion            equal-weight RRF
RRF k             60
Candidate top-k   100
Reranker          MedCPT Cross-Encoder
Rerank depth      100
Max length        512
Precision         FP16
Batch size        16 per GPU
```

Do not use Test to change RRF k, source weights, candidate depth, rerank depth, batch shape or the reranker model.

---

## Formal Result Files

```text
reports/v1/article_retrieval_rrf_bm25_medcpt_bmretriever_k60_full_dev.json
reports/v1/article_retrieval_rrf_bm25_medcpt_bmretriever_k60_full_test.json
reports/v1/article_retrieval_medcpt_cross_rrf3_k60_d100_full_dev.json
reports/v1/article_retrieval_medcpt_cross_rrf3_k60_d100_full_test.json
```

Primary reusable artifacts:

```text
artifacts/v1/reranking/candidates/
artifacts/v1/reranking/reranked_candidates/
artifacts/v1/reranking/runs/
```

---

## Remaining Minor Technical Debt

1. Rename or clarify the report field `queries_with_eligible_judgments` where it represents the total split query count.
2. Clean up the Python multiprocessing semaphore warning printed during interpreter shutdown.
3. Consider normalizing duplicated candidate parquet views if storage size becomes a practical issue.
4. Preserve the current evaluator caveats so failure-case counts are not interpreted as simple retrieval misses.

None of these items changes the Phase 04 result or blocks Phase 05.

---

## Handoff to Phase 05

Phase 05 should treat the following as the fixed article-level input:

```text
Retriever candidate set:
RRF Top-100

Preferred front-ranked articles:
MedCPT Cross-Encoder order over the fixed RRF Top-100
```

Phase 05 should focus on sentence-level evidence retrieval and must not reopen Phase 04 model selection unless a separate, explicitly scoped ablation is planned.
