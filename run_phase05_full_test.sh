#!/usr/bin/env bash
set -euo pipefail

# Run from the EvidenceGap repository root.
#
# Usage:
#   bash run_phase05_full_test.sh
#   bash run_phase05_full_test.sh logs/phase05_full_test.log
#
# Optional environment overrides:
#   GPU_IDS=2,3,4,5
#   DEVICE_IDS=0,1,2,3
#   NUM_SHARDS=8
#   MEDCPT_BATCH_SIZE=64
#   BMR_BATCH_SIZE=8
#   TOP_K=50
#   BOOTSTRAP_SAMPLES=10000

ROOT="${ROOT:-.}"
GPU_IDS="${GPU_IDS:-2,3,4,5}"
DEVICE_IDS="${DEVICE_IDS:-0,1,2,3}"
NUM_SHARDS="${NUM_SHARDS:-8}"
MEDCPT_BATCH_SIZE="${MEDCPT_BATCH_SIZE:-64}"
BMR_BATCH_SIZE="${BMR_BATCH_SIZE:-8}"
TOP_K="${TOP_K:-50}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-10000}"

LOG_FILE="${1:-logs/phase05_full_test_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$LOG_FILE")"

# Show output in the terminal and save stdout/stderr to one file.
exec > >(tee -a "$LOG_FILE") 2>&1

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNNER="scripts/run_v1_phase05.py"
CANONICAL_DIR="artifacts/v1/evidence_sentence_retrieval/canonical/test_full"

BM25_RUN="bm25_full_test"
MEDCPT_RUN="medcpt_full_test"
BMR_RUN="bmretriever_full_test"
RRF_RUN="rrf_bmr20_medcpt20_k10_full_test"

BM25_PATH="artifacts/v1/evidence_sentence_retrieval/runs/${BM25_RUN}/ranked_sentences.parquet"
MEDCPT_PATH="artifacts/v1/evidence_sentence_retrieval/runs/${MEDCPT_RUN}/ranked_sentences.parquet"
BMR_PATH="artifacts/v1/evidence_sentence_retrieval/runs/${BMR_RUN}/ranked_sentences.parquet"
RRF_PATH="artifacts/v1/evidence_sentence_retrieval/fusion/${RRF_RUN}/ranked_sentences.parquet"

section() {
  printf '\n\n============================================================\n'
  printf '%s\n' "$1"
  printf '============================================================\n'
}

run() {
  printf '\n+ '
  printf '%q ' "$@"
  printf '\n'
  "$@"
}

section "Phase 05 Full Test"
echo "Started at: $(date --iso-8601=seconds)"
echo "Repository: $(pwd)"
echo "Log file: $LOG_FILE"
echo "CUDA_VISIBLE_DEVICES: $GPU_IDS"
echo "Logical devices: $DEVICE_IDS"
echo "Shards: $NUM_SHARDS"
echo "Frozen fusion: BMR Top-20 + MedCPT Top-20, equal-weight RRF, k=10"

section "1/9 Prepare Full Test canonical"
run "$PYTHON_BIN" "$RUNNER" prepare \
  --root "$ROOT" \
  --split test

section "2/9 BM25 Full Test"
run "$PYTHON_BIN" "$RUNNER" bm25 \
  --root "$ROOT" \
  --split test \
  --canonical-dir "$CANONICAL_DIR" \
  --top-k "$TOP_K" \
  --run-name "$BM25_RUN"

section "3/9 MedCPT Full Test"
run env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  "$PYTHON_BIN" "$RUNNER" dense \
  --root "$ROOT" \
  --model medcpt \
  --split test \
  --canonical-dir "$CANONICAL_DIR" \
  --devices "$DEVICE_IDS" \
  --num-shards "$NUM_SHARDS" \
  --batch-size "$MEDCPT_BATCH_SIZE" \
  --amp fp16 \
  --top-k "$TOP_K" \
  --run-name "$MEDCPT_RUN"

section "4/9 BMRetriever Full Test"
run env CUDA_VISIBLE_DEVICES="$GPU_IDS" \
  "$PYTHON_BIN" "$RUNNER" dense \
  --root "$ROOT" \
  --model bmretriever \
  --split test \
  --canonical-dir "$CANONICAL_DIR" \
  --devices "$DEVICE_IDS" \
  --num-shards "$NUM_SHARDS" \
  --batch-size "$BMR_BATCH_SIZE" \
  --amp fp16 \
  --top-k "$TOP_K" \
  --run-name "$BMR_RUN"

section "5/9 Frozen RRF k=10 Full Test"
run "$PYTHON_BIN" "$RUNNER" fuse \
  --root "$ROOT" \
  --split test \
  --canonical-dir "$CANONICAL_DIR" \
  --left "$BMR_PATH" \
  --right "$MEDCPT_PATH" \
  --left-name bmretriever \
  --right-name medcpt \
  --left-depth 20 \
  --right-depth 20 \
  --rrf-k 10 \
  --run-name "$RRF_RUN"

section "6/9 Compare RRF vs BMRetriever"
run "$PYTHON_BIN" "$RUNNER" compare \
  --root "$ROOT" \
  --canonical-dir "$CANONICAL_DIR" \
  --baseline "$BMR_PATH" \
  --challenger "$RRF_PATH" \
  --baseline-name bmretriever \
  --challenger-name rrf_bmr20_medcpt20_k10 \
  --bootstrap-unit systematic_review \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --report reports/v1/evidence_sentence_rrf_k10_vs_bmretriever_full_test.json

section "7/9 Compare RRF vs MedCPT"
run "$PYTHON_BIN" "$RUNNER" compare \
  --root "$ROOT" \
  --canonical-dir "$CANONICAL_DIR" \
  --baseline "$MEDCPT_PATH" \
  --challenger "$RRF_PATH" \
  --baseline-name medcpt \
  --challenger-name rrf_bmr20_medcpt20_k10 \
  --bootstrap-unit systematic_review \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --report reports/v1/evidence_sentence_rrf_k10_vs_medcpt_full_test.json

section "8/9 Compare RRF vs BM25"
run "$PYTHON_BIN" "$RUNNER" compare \
  --root "$ROOT" \
  --canonical-dir "$CANONICAL_DIR" \
  --baseline "$BM25_PATH" \
  --challenger "$RRF_PATH" \
  --baseline-name bm25 \
  --challenger-name rrf_bmr20_medcpt20_k10 \
  --bootstrap-unit systematic_review \
  --bootstrap-samples "$BOOTSTRAP_SAMPLES" \
  --report reports/v1/evidence_sentence_rrf_k10_vs_bm25_full_test.json

section "9/9 Full Test complementarity"
run "$PYTHON_BIN" "$RUNNER" complementarity \
  --root "$ROOT" \
  --canonical-dir "$CANONICAL_DIR" \
  --left "$BMR_PATH" \
  --right "$MEDCPT_PATH" \
  --left-name bmretriever \
  --right-name medcpt \
  --depths 5,10,20,50 \
  --report reports/v1/evidence_sentence_complementarity_full_test.json

section "Phase 05 Full Test completed"
echo "Finished at: $(date --iso-8601=seconds)"
echo "Full log: $LOG_FILE"
echo "Final ranking: $RRF_PATH"
echo "Reports:"
echo "  reports/v1/evidence_sentence_retrieval_${BM25_RUN}_test.json"
echo "  reports/v1/evidence_sentence_retrieval_${MEDCPT_RUN}_test.json"
echo "  reports/v1/evidence_sentence_retrieval_${BMR_RUN}_test.json"
echo "  reports/v1/evidence_sentence_retrieval_${RRF_RUN}_test.json"
echo "  reports/v1/evidence_sentence_rrf_k10_vs_bmretriever_full_test.json"
echo "  reports/v1/evidence_sentence_rrf_k10_vs_medcpt_full_test.json"
echo "  reports/v1/evidence_sentence_rrf_k10_vs_bm25_full_test.json"
echo "  reports/v1/evidence_sentence_complementarity_full_test.json"
