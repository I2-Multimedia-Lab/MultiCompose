#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

args=(
  --generated-image "${GENERATED_IMAGE:-/path/to/generated.png}"
  --subject-a-ref-dir "${SUBJECT_A_REF_DIR:-/path/to/subject_a}"
  --subject-b-ref-dir "${SUBJECT_B_REF_DIR:-/path/to/subject_b}"
  --background-ref-dir "${BACKGROUND_REF_DIR:-/path/to/background}"
  --subject-a "${SUBJECT_A:-subject_a}"
  --subject-b "${SUBJECT_B:-subject_b}"
  --attr-a "${ATTR_A:-attribute_one}"
  --attr-b "${ATTR_B:-attribute_two}"
  --background "${BACKGROUND:-background_scene}"
  --boxes "${BOXES:-}"
  --prompt "${PROMPT:-photo of subject_a wearing attribute_one and subject_b wearing attribute_two, background_scene}"
  --out-json "${OUT_JSON:-${PROJECT_ROOT}/runs/traditional_evaluation_demo.json}"
)
if [[ "${RUN_EVAL:-0}" == "1" ]]; then
  args+=(--run)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation/traditional_evaluator.py" "${args[@]}"
