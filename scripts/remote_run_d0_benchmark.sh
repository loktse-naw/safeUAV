#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/root/SafeUAV/pytorch_code}"
PYTHON_BIN="/usr/local/miniconda3/envs/py312/bin/python"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
"$PYTHON_BIN" scripts/run_d0.py --config configs/base.yaml --output results/d0_remote
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/benchmark.py --config configs/prototype.yaml --output results/benchmark_remote.json --transitions 10000

