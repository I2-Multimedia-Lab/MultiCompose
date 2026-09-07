#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${VLM_IMAGE:?Set VLM_IMAGE to the layout preview image}"
: "${SUBJECT_A:?Set SUBJECT_A for box detection}"
: "${SUBJECT_B:?Set SUBJECT_B for box detection}"
: "${MODEL_NAME:?Set MODEL_NAME to a local model path or model identifier}"
: "${MAPPING_FILE:?Set MAPPING_FILE to your concept TSV}"
: "${THEME1:?Set THEME1 to the first concept_id}"
: "${THEME2:?Set THEME2 to the second concept_id}"
: "${BACKGROUND:?Set BACKGROUND to the background concept_id}"

box_args=(
  --run
  --image "${VLM_IMAGE}"
  --subjects "${SUBJECT_A}" "${SUBJECT_B}"
  --width "${IMAGE_WIDTH:-1024}"
  --height "${IMAGE_HEIGHT:-1024}"
  --margin "${BOX_MARGIN:-0.04}"
  --format external
)
if [[ -n "${VLM_BOX_JSON:-}" ]]; then
  box_args+=(--out-json "${VLM_BOX_JSON}")
fi
BOXES="$(${PYTHON_BIN} "${PROJECT_ROOT}/box/vlm_layout_boxes.py" "${box_args[@]}")"
echo "[infer-vlm] boxes=${BOXES}"
export EXTERNAL_BOXES="${BOXES}"
exec bash "${PROJECT_ROOT}/inference/infer.sh"
