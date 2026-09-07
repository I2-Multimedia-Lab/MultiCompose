#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${GENERATED_IMAGE:?Set GENERATED_IMAGE}"
: "${SUBJECT_A_REF_DIR:?Set SUBJECT_A_REF_DIR}"
: "${SUBJECT_B_REF_DIR:?Set SUBJECT_B_REF_DIR}"
: "${BACKGROUND_REF_DIR:?Set BACKGROUND_REF_DIR}"
: "${SUBJECT_A:?Set SUBJECT_A}"
: "${SUBJECT_B:?Set SUBJECT_B}"
: "${ATTR_A:?Set ATTR_A}"
: "${ATTR_B:?Set ATTR_B}"
: "${BACKGROUND:?Set BACKGROUND}"
: "${PROMPT:?Set PROMPT}"

args=(
  --generated-image "${GENERATED_IMAGE}"
  --subject-a-ref-dir "${SUBJECT_A_REF_DIR}"
  --subject-b-ref-dir "${SUBJECT_B_REF_DIR}"
  --background-ref-dir "${BACKGROUND_REF_DIR}"
  --subject-a "${SUBJECT_A}"
  --subject-b "${SUBJECT_B}"
  --attr-a "${ATTR_A}"
  --attr-b "${ATTR_B}"
  --background "${BACKGROUND}"
  --prompt "${PROMPT}"
  --clip-model "${CLIP_MODEL:-openai/clip-vit-base-patch32}"
  --identity-model "${IDENTITY_MODEL:-facebook/dinov2-base}"
  --device "${DEVICE:-auto}"
  --out-json "${OUT_JSON:-${PROJECT_ROOT}/runs/evaluation/traditional.json}"
  --run
)
if [[ -n "${BOXES:-}" ]]; then
  args+=(--boxes "${BOXES}")
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  args+=(--local-files-only)
fi
exec "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation/traditional_evaluator.py" "${args[@]}"
