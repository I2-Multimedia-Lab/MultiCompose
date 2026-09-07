#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${MODEL_NAME:?Set MODEL_NAME to a local model path or model identifier}"
: "${MAPPING_FILE:?Set MAPPING_FILE to your concept TSV}"
: "${CASES_FILE:?Set CASES_FILE to an inference TSV based on configs/inference_cases.example.tsv}"

args=(
  --cases "${CASES_FILE}"
  --mapping "${MAPPING_FILE}"
  --model "${MODEL_NAME}"
  --output-root "${OUTPUT_ROOT:-${PROJECT_ROOT}/runs/inference}"
  --steps "${N_TIMESTEPS:-50}"
  --resampling-steps "${RESAMPLING_STEPS:-10}"
  --guidance "${GUIDANCE_SCALE:-0.8}"
  --t-cond "${T_COND:-0.3}"
  --run
)
if [[ "${CONTINUE_ON_ERROR:-0}" == "1" ]]; then
  args+=(--continue-on-error)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/inference/infer_batch.py" "${args[@]}"
