#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/root/SafeUAV/pytorch_code}"
PYTHON_BIN="/usr/local/miniconda3/envs/py312/bin/python"

cd "$PROJECT_DIR"
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -c "import torch, gymnasium, stable_baselines3; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0), 'gymnasium': gymnasium.__version__, 'sb3': stable_baselines3.__version__})"

