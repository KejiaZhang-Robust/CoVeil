#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${LARGE_MODEL:?Set LARGE_MODEL to a model path or Hugging Face identifier.}"
: "${SMALL_MODEL:?Set SMALL_MODEL to a model path or Hugging Face identifier.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET="${DATASET:-dataset/MEDPRIV.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-outputs/edge_side/results.jsonl}"

ARGS=(
  --input_file "$DATASET"
  --output_file "$OUTPUT_FILE"
  --large_model "$LARGE_MODEL"
  --small_model "$SMALL_MODEL"
  --alpha "${ALPHA:-0.5}"
  --privacy_weight "${PRIVACY_WEIGHT:-1.0}"
  --candidate_topk "${CANDIDATE_TOPK:-50}"
  --max_new_tokens "${MAX_NEW_TOKENS:-160}"
  --record_steps "${RECORD_STEPS:-160}"
  --trace_topk "${TRACE_TOPK:-100}"
  --device "${DEVICE:-auto}"
)
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi

"$PYTHON_BIN" leakage_fusion_SM_fusion_defense.py "${ARGS[@]}"
