#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${GENERATED_IMAGE:?Set GENERATED_IMAGE}"
: "${SUBJECT_A_REF:?Set SUBJECT_A_REF}"
: "${SUBJECT_B_REF:?Set SUBJECT_B_REF}"
: "${BACKGROUND_REF:?Set BACKGROUND_REF}"
: "${SUBJECT_A:?Set SUBJECT_A}"
: "${SUBJECT_B:?Set SUBJECT_B}"
: "${BACKGROUND:?Set BACKGROUND}"
: "${PROMPT:?Set PROMPT}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation/vlm_evaluator.py" \
  --generated-image "${GENERATED_IMAGE}" \
  --subject-a-ref "${SUBJECT_A_REF}" \
  --subject-b-ref "${SUBJECT_B_REF}" \
  --background-ref "${BACKGROUND_REF}" \
  --subject-a "${SUBJECT_A}" \
  --subject-b "${SUBJECT_B}" \
  --background "${BACKGROUND}" \
  --prompt "${PROMPT}" \
  --endpoint "${VLM_ENDPOINT:-http://127.0.0.1:8000/v1/chat/completions}" \
  --model "${VLM_MODEL:-your-vlm-model}" \
  --out-json "${OUT_JSON:-${PROJECT_ROOT}/runs/evaluation/vlm.json}" \
  --run
