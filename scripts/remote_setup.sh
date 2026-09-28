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
"$PYTHON_BIN" -m pip install -r requirements.txt
"$PYTHON_BIN" -c "import torch, gymnasium, stable_baselines3; print({'torch': torch.__version__, 'cuda': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0), 'gymnasium': gymnasium.__version__, 'sb3': stable_baselines3.__version__})"

