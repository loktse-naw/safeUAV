#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/root/SafeUAV/pytorch_code}"
PYTHON_BIN="/usr/local/miniconda3/envs/py312/bin/python"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
"$PYTHON_BIN" scripts/run_d1.py \
  --config configs/prototype.yaml \
  --output results/d1_remote \
  --benchmark results/benchmark_remote.json

