#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${VLM_JSON:?Set VLM_JSON to the VLM evaluation JSON}"
: "${TRADITIONAL_JSON:?Set TRADITIONAL_JSON to the traditional evaluation JSON}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation/combine_evaluations.py" \
  --vlm-json "${VLM_JSON}" \
  --traditional-json "${TRADITIONAL_JSON}" \
  --vlm-weight "${VLM_WEIGHT:-0.5}" \
  --traditional-weight "${TRADITIONAL_WEIGHT:-0.5}" \
  --out-json "${OUT_JSON:-${PROJECT_ROOT}/runs/evaluation/combined.json}"
