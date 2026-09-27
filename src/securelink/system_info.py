from __future__ import annotations

import os
import platform
import subprocess
import sys
from typing import Any

import gymnasium
import numpy
import pandas
import psutil
import stable_baselines3
import torch


def collect_system_info() -> dict[str, Any]:
    gpu_smi = "unavailable"
    try:
        gpu_smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    memory = psutil.virtual_memory()
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "memory_total_gib": memory.total / 1024**3,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "gymnasium": gymnasium.__version__,
        "stable_baselines3": stable_baselines3.__version__,
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "torch_cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": gpu_smi,
        "pid": os.getpid(),
    }

