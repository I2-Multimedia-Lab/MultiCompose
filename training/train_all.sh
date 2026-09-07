#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${MODEL_NAME:?Set MODEL_NAME to a local model path or model identifier}"
: "${CONFIG_FILE:?Set CONFIG_FILE to a concept TSV based on configs/concepts.example.tsv}"

args=(
  --config "${CONFIG_FILE}"
  --model "${MODEL_NAME}"
  --output-root "${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/train}"
  --max-train-steps "${MAX_TRAIN_STEPS:-1500}"
  --save-steps "${SAVE_STEPS:-200}"
  --resolution "${RESOLUTION:-512}"
  --learning-rate "${LEARNING_RATE:-1e-5}"
  --run
)
if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  args+=(--continue-on-error)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/training/train_batch.py" "${args[@]}"
