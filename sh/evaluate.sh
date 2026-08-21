#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

: "${RESULTS:?Set RESULTS to a fusion results JSONL file.}"
TOKENIZER="${TOKENIZER:-${LARGE_MODEL:-}}"
: "${TOKENIZER:?Set TOKENIZER or LARGE_MODEL to a tokenizer path or identifier.}"

PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/evaluation}"
mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" evaluate_leakage.py \
  --input_jsonl "$RESULTS" \
  --tokenizer "$TOKENIZER" \
  --output_json "$OUTPUT_DIR/evidence_recall.json" \
  --ks "${REPORT_KS:-10,50,100}" \
  --auc_k "${AUC_K:-100}"

if [[ -n "${ATTACKER_MODEL:-}" ]]; then
  "$PYTHON_BIN" evaluate_leakage_reconstruct.py \
    --input_jsonl "$RESULTS" \
    --attacker_model "$ATTACKER_MODEL" \
    --output_jsonl "$OUTPUT_DIR/private_context_inversion.jsonl" \
    --topk_per_step "${INVERSION_TOPK:-100}" \
    --max_new_tokens "${INVERSION_MAX_NEW_TOKENS:-256}" \
    --device "${DEVICE:-auto}"
fi
