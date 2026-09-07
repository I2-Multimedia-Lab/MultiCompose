#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[box-demo] normalized JSON:"
"${PYTHON_BIN}" "${PROJECT_ROOT}/box/vlm_layout_boxes.py" \
  --subjects cat dog --width 1024 --height 1024 --demo --margin 0
echo "[box-demo] sampler external-boxes string:"
"${PYTHON_BIN}" "${PROJECT_ROOT}/box/vlm_layout_boxes.py" \
  --subjects cat dog --width 1024 --height 1024 --demo --margin 0 --format external
