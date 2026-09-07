#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${MODEL_NAME:?Set MODEL_NAME to a local model path or model identifier}"
: "${DATASET_DIR:?Set DATASET_DIR to the concept image directory}"
: "${CONCEPT_ID:?Set CONCEPT_ID}"
: "${INSTANCE_PROMPT:?Set INSTANCE_PROMPT}"
: "${MODIFIER_TOKEN:?Set MODIFIER_TOKEN}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/training/train_one_concept.py" \
  --concept-id "${CONCEPT_ID}" \
  --dataset-dir "${DATASET_DIR}" \
  --output-dir "${OUTPUT_DIR:-${PROJECT_ROOT}/runs/train/${CONCEPT_ID}}" \
  --instance-prompt "${INSTANCE_PROMPT}" \
  --modifier-token "${MODIFIER_TOKEN}" \
  --model "${MODEL_NAME}" \
  --resolution "${RESOLUTION:-512}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --learning-rate "${LEARNING_RATE:-1e-5}" \
  --max-train-steps "${MAX_TRAIN_STEPS:-1500}" \
  --save-steps "${SAVE_STEPS:-200}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-4}" \
  --run
