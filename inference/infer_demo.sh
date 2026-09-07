#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

args=(
  --theme1 "${THEME1:-subject_a}"
  --theme2 "${THEME2:-subject_b}"
  --background "${BACKGROUND:-background_scene}"
  --attr1 "${ATTR1:-attribute_one}"
  --attr2 "${ATTR2:-attribute_two}"
  --action "${ACTION:-standing}"
  --mapping "${MAPPING_FILE:-${PROJECT_ROOT}/configs/concepts.example.tsv}"
  --model "${MODEL_NAME:-/path/to/models/stable-diffusion-xl-base-1.0}"
  --steps "${N_TIMESTEPS:-4}"
  --resampling-steps "${RESAMPLING_STEPS:-1}"
)
if [[ "${RUN_INFER:-0}" == "1" ]]; then
  args+=(--run)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/inference/infer_one_pair.py" "${args[@]}"
