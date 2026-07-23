# EvidenceGap V1 Phase 06.7｜Graph-ready stance export

Phase 06.7 converts sentence-level stance predictions into deterministic artifacts for the downstream Evidence Graph and frontend. It does not call an LLM and does not create the final medical Verdict.

## Input

A validated Phase 06 stance prediction artifact, for example:

```text
artifacts/v1/stance_verification/llm_judge/
deepseek_v4_pro_phase05_top5_ctx1_dev_partial_cache/
stance_predictions.parquet
```

The input must contain sentence evidence with stable `query_id`, `paper_id`, original `sentence_index`, Phase 05 `evidence_rank`, stance probabilities, rationale, evidence type, and model provenance.

## Run

```bash
python scripts/run_v1_phase06.py export-graph \
  --root . \
  --prediction-path \
    artifacts/v1/stance_verification/llm_judge/deepseek_v4_pro_phase05_top5_ctx1_dev_partial_cache/stance_predictions.parquet \
  --run-name deepseek_v4_pro_phase05_top5_ctx1_dev_partial_graph
```

This command makes zero API requests.

## Outputs

```text
artifacts/v1/stance_verification/graph_ready/<run_name>/
├── query_summaries.parquet
├── paper_summaries.parquet
├── graph_nodes.parquet
├── graph_edges.parquet
├── graph_bundles.jsonl
└── run_manifest.json
```

Reports:

```text
reports/v1/stance_graph_<run_name>.json
reports/v1/stance_graph_<run_name>.md
```

### Query and paper summaries

Each summary retains evidence counts and soft stance mass:

```text
rank_weight = 1 / Phase 05 evidence_rank
stance_mass(label) = Σ rank_weight × LLM_probability(label)
```

The transparent `directional_evidence_pattern` field describes only whether explicit support and/or refute evidence is present:

```text
support_only
refute_only
mixed
none
```

`directional_evidence_pattern` must not be interpreted as the overall evidence conclusion. For example, `support_only` can coexist with `mass_leader = insufficient` when one sentence supports the claim but most retrieved evidence is insufficient.

Two separate directional fields are exported:

```text
directional_margin
= abs(support_mass - refute_mass)
  / (support_mass + refute_mass)

directional_mass_share
= (support_mass + refute_mass)
  / (support_mass + refute_mass + insufficient_mass)
```

`directional_margin` describes balance only between support and refute. `directional_mass_share` describes how much of the total weighted evidence is directional at all. `mass_leader` and these directional fields are display-oriented summaries. They are not a final medical Verdict, and the LLM probabilities are not calibrated.

### Graph nodes

```text
claim
article
evidence
```

Evidence nodes retain the original sentence text/index, Phase 05 rank and score, stance, confidence, evidence type, rationale/context metadata, model fingerprint, and source provenance.

### Graph edges

```text
Claim    → Article   retrieved_from
Article  → Evidence  contains
Evidence → Claim     supports / refutes / insufficient
```

Stance edges contain the predicted stance probability, reciprocal-rank weight, stance mass, model fingerprint, and source checksums.

### JSONL bundles

`graph_bundles.jsonl` contains one complete graph object per query. It is intended for lightweight backend/frontend loading without joining Parquet files. Every bundle explicitly declares:

```json
{
  "boundary": {
    "is_final_medical_verdict": false
  }
}
```

Study quality, cross-article weighting, claim decomposition, and the final Verdict node remain downstream responsibilities.

## Validation

The exporter rejects:

- duplicate input IDs;
- non-sentence evidence;
- missing paper/rank linkage;
- rank gaps within a query-paper group;
- duplicate sentence indices;
- inconsistent claim text within one query;
- dangling graph edges;
- evidence nodes without exactly one `contains` and one stance edge.
