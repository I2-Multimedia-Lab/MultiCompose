#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VLM_JSON="${VLM_JSON:-${PROJECT_ROOT}/evaluation/examples/vlm_output.schema.json}"
TRADITIONAL_JSON="${TRADITIONAL_JSON:-${PROJECT_ROOT}/evaluation/examples/traditional_output.schema.json}"

exec "${PYTHON_BIN}" "${PROJECT_ROOT}/evaluation/combine_evaluations.py" \
  --vlm-json "${VLM_JSON}" \
  --traditional-json "${TRADITIONAL_JSON}" \
  --vlm-weight "${VLM_WEIGHT:-0.5}" \
  --traditional-weight "${TRADITIONAL_WEIGHT:-0.5}" \
  --out-json "${OUT_JSON:-${PROJECT_ROOT}/runs/combined_evaluation_demo.json}"
