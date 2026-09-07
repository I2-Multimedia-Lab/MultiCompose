#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EVALUATOR="${PROJECT_ROOT}/source/evaluation/batch_vlm_evaluator.py"

RESULTS_ROOT="${RESULTS_ROOT:-/path/to/generated-results}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/runs/vlm_batch_eval}"
REF_IMAGE_DIRS="${REF_IMAGE_DIRS:-/path/to/reference-images:/path/to/datasets}"
MAX_WORKERS="${MAX_WORKERS:-5}"
MAX_IMAGES="${MAX_IMAGES:-0}"

if [[ "${RUN_EVAL:-0}" != "1" ]]; then
  echo "[batch-vlm-eval] dry-run; no API request was sent"
  echo "RESULTS_ROOT=${RESULTS_ROOT}"
  echo "OUTPUT_DIR=${OUTPUT_DIR}"
  echo "REF_IMAGE_DIRS=${REF_IMAGE_DIRS}"
  echo "GROUP_FILTER=${GROUP_FILTER:-<all>}"
  echo "MAX_IMAGES=${MAX_IMAGES} MAX_WORKERS=${MAX_WORKERS}"
  echo "Set RUN_EVAL=1 and VLM_API_KEY to execute."
  exit 0
fi

if [[ -z "${VLM_API_KEY:-${DASHSCOPE_API_KEY:-}}" ]]; then
  echo "RUN_EVAL=1 requires VLM_API_KEY" >&2
  exit 2
fi

export RESULTS_ROOT OUTPUT_DIR REF_IMAGE_DIRS MAX_WORKERS MAX_IMAGES
export REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-90}"
export ONLY_FUSION="${ONLY_FUSION:-1}"
export GROUP_FILTER="${GROUP_FILTER:-}"
export SUBJECT_FILTER="${SUBJECT_FILTER:-}"
export RESUME_WITH_FILTER="${RESUME_WITH_FILTER:-0}"
exec "${PYTHON_BIN}" "${EVALUATOR}"
