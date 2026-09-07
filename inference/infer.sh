#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${MODEL_NAME:?Set MODEL_NAME to a local model path or model identifier}"
: "${MAPPING_FILE:?Set MAPPING_FILE to your concept TSV}"
: "${THEME1:?Set THEME1 to the first concept_id}"
: "${THEME2:?Set THEME2 to the second concept_id}"
: "${BACKGROUND:?Set BACKGROUND to the background concept_id}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/inference/infer_one_pair.py" \
  --theme1 "${THEME1}" \
  --theme2 "${THEME2}" \
  --background "${BACKGROUND}" \
  --attr1 "${ATTR1:-attribute_one}" \
  --attr2 "${ATTR2:-attribute_two}" \
  --action "${ACTION:-standing}" \
  --mapping "${MAPPING_FILE}" \
  --model "${MODEL_NAME}" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/runs/inference}" \
  --guidance "${GUIDANCE_SCALE:-0.8}" \
  --t-cond "${T_COND:-0.3}" \
  --steps "${N_TIMESTEPS:-50}" \
  --resampling-steps "${RESAMPLING_STEPS:-10}" \
  --seed "${SEED:-1234}" \
  --external-boxes "${EXTERNAL_BOXES:-120,220,500,960+540,220,920,960}" \
  --run
