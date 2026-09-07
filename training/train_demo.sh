#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

args=(
  --concept-id "${CONCEPT_ID:-subject_a}"
  --dataset-dir "${DATASET_DIR:-/path/to/datasets/subject_a}"
  --instance-prompt "${INSTANCE_PROMPT:-Photo of a <subject_a> object.}"
  --modifier-token "${MODIFIER_TOKEN:-<subject_a>}"
  --model "${MODEL_NAME:-/path/to/models/stable-diffusion-xl-base-1.0}"
  --max-train-steps "${MAX_TRAIN_STEPS:-3}"
  --save-steps "${SAVE_STEPS:-3}"
)
if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
  args+=(--run)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/training/train_one_concept.py" "${args[@]}"
