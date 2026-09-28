#!/usr/bin/env bash
set -euo pipefail

# 项目目录：$1 > $SAFEUAV_DIR > $HOME/SafeUAV/pytorch_code
# 用 $HOME 前缀后，root（/root/...）与普通账号（如 u2025110281 的 /home/...）都能正确解析。
PROJECT_DIR="${1:-${SAFEUAV_DIR:-$HOME/SafeUAV/pytorch_code}}"

# 解释器：$SAFEUAV_PYTHON > 本机 venv > 旧共享 conda 路径
if [ -n "${SAFEUAV_PYTHON:-}" ]; then
  PYTHON_BIN="$SAFEUAV_PYTHON"
elif [ -x "$HOME/venvs/safeuav/bin/python" ]; then
  PYTHON_BIN="$HOME/venvs/safeuav/bin/python"
elif [ -x /usr/local/miniconda3/envs/py312/bin/python ]; then
  PYTHON_BIN=/usr/local/miniconda3/envs/py312/bin/python
else
  echo "ERROR: 未找到 Python 解释器，请设置 SAFEUAV_PYTHON" >&2
  exit 1
fi

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src"
"$PYTHON_BIN" scripts/run_d0.py --config configs/base.yaml --output results/d0_remote
"$PYTHON_BIN" -m pytest -q
"$PYTHON_BIN" scripts/benchmark.py --config configs/prototype.yaml --output results/benchmark_remote.json --transitions 10000

