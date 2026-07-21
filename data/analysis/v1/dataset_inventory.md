# EvidenceGap V1 Dataset Inventory

Generated: 2026-07-21T04:57:31.290200+00:00

## MedFact-Synth

- Rows: 1,497,981
- Shards: 17
- Unique claim PMIDs: 426,440
- Unique source PMIDs: 1,320,128
- Claim PMID = source PMID: 273,374 (18.25%)

Labels:

```json
{
  "-1.0": 111819,
  "-2.0": 266667,
  "0.0": 545603,
  "1.0": 228047,
  "2.0": 345845
}
```

Preliminary role: large synthetic **claim–article stance** corpus. It is useful for scale and stress-testing, but cannot be the sole final medical-quality proof.

## EvidenceBench-100k

- Rows: 107,461
- Train: 87,461
- Test: 20,000
- Common fields: aspect2sentence_indices, aspect_id2aspect, aspect_list_ids, hypothesis, paper_as_candidate_pool, paper_id, results_aspect_list_ids, results_evidence_retrieval_at_5_evaluation, results_evidence_retrieval_at_optimal_evaluation, sentence_index2aspects, sentence_types_in_candidate_pool, systematic_review_id

Preliminary role: **hypothesis-to-evidence-sentence retrieval/extraction**. Review the samples and field names before fixing the exact input/output contract.

## HealthFC

- Rows: 750
- Duplicate English claims: 0
- Missing English explanations: 9

Labels:

```json
{
  "0": 202,
  "1": 423,
  "2": 125
}
```

Preliminary role: small expert-annotated **final evaluation set**, not the main training or indexing corpus.

## Decision gate

Before writing model code, fix these five items:

1. Exact meaning of every label.
2. Exact EvidenceBench hypothesis, paper-text, candidate-sentence and gold-evidence fields.
3. Which dataset evaluates article retrieval.
4. Which dataset evaluates evidence-sentence extraction.
5. Which dataset evaluates final stance/verdict.

After reviewing this report, write a one-page V1 task contract. Do not clean or transform the datasets before that contract is fixed.
