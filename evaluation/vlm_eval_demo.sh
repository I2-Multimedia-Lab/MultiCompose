#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

args=(
  --generated-image "${GENERATED_IMAGE:-/path/to/generated.png}"
  --subject-a-ref "${SUBJECT_A_REF:-/path/to/subject_a/0.jpg}"
  --subject-b-ref "${SUBJECT_B_REF:-/path/to/subject_b/0.jpg}"
  --background-ref "${BACKGROUND_REF:-/path/to/background/0.jpg}"
  --subject-a "${SUBJECT_A:-subject_a}"
  --subject-b "${SUBJECT_B:-subject_b}"
  --background "${BACKGROUND:-background_scene}"
  --prompt "${PROMPT:-photo of subject_a wearing attribute_one and subject_b wearing attribute_two, background_scene}"
  --out-json "${OUT_JSON:-${PROJECT_ROOT}/runs/evaluation_demo.json}"
)
if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  args+=(--run)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation/vlm_evaluator.py" "${args[@]}"
