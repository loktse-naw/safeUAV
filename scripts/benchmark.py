from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import load_config
from securelink.environment import SecureLinkEnv
from securelink.evaluation import write_json
from securelink.system_info import collect_system_info


def make_env(config: dict, run_seed: int, worker_id: int = 0):
    def factory():
        env = SecureLinkEnv(
            config,
            method="B11",
            attack_rule=config["training"]["attack_rule"],
            stream_context={"purpose": "benchmark", "run_seed": run_seed, "worker_id": worker_id},
        )
        return env

    return factory


def sample_benchmark(config: dict, n_envs: int, vector_type: str, transitions: int) -> dict:
    factories = [make_env(config, 9000, index) for index in range(n_envs)]
    if vector_type == "subproc":
        env = SubprocVecEnv(factories, start_method="spawn")
    else:
        env = DummyVecEnv(factories)
    env.reset()
    rng = np.random.default_rng(404)
    iterations = int(np.ceil(transitions / n_envs))
    start = time.perf_counter()
    for _ in range(iterations):
        actions = rng.uniform(-1.0, 1.0, size=(n_envs, 3)).astype(np.float32)
        env.step(actions)
    elapsed = time.perf_counter() - start
    env.close()
    completed = iterations * n_envs
    return {
        "kind": "environment_sampling",
        "vector_type": vector_type,
        "n_envs": n_envs,
        "transitions": completed,
        "seconds": elapsed,
        "transitions_per_second": completed / elapsed,
    }


def train_benchmark(config: dict, device: str, timesteps: int) -> dict:
    env = DummyVecEnv([make_env(config, 9901)])
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    model = PPO(
        "MlpPolicy",
        env,
        seed=9901,
        device=device,
        n_steps=int(config["training"]["n_steps"]),
        batch_size=int(config["training"]["batch_size"]),
        n_epochs=int(config["training"]["n_epochs"]),
        learning_rate=float(config["training"]["learning_rate"]),
        gamma=float(config["training"]["gamma"]),
        gae_lambda=float(config["training"]["gae_lambda"]),
        ent_coef=float(config["training"]["ent_coef"]),
        policy_kwargs={"net_arch": list(config["training"]["policy_layers"])},
        verbose=0,
    )
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps, progress_bar=False)
    elapsed = time.perf_counter() - start
    rss_after = process.memory_info().rss
    cuda_peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    actual_device = str(model.device)
    env.close()
    return {
        "kind": "ppo_training",
        "requested_device": device,
        "actual_device": actual_device,
        "timesteps": timesteps,
        "seconds": elapsed,
        "timesteps_per_second": timesteps / elapsed,
        "rss_before_mib": rss_before / 1024**2,
        "rss_after_mib": rss_after / 1024**2,
        "cuda_peak_allocated_mib": cuda_peak / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "prototype.yaml"))
    parser.add_argument("--output", default=str(ROOT / "results" / "benchmark.json"))
    parser.add_argument("--transitions", type=int, default=20000)
    args = parser.parse_args()
    config = load_config(args.config)
    rows = []
    for n_envs in (1, 4, 8):
        rows.append(sample_benchmark(config, n_envs, "dummy", args.transitions))
    if os.name != "nt":
        for n_envs in (4, 8):
            rows.append(sample_benchmark(config, n_envs, "subproc", args.transitions))
    benchmark_steps = int(config["training"]["benchmark_timesteps"])
    rows.append(train_benchmark(config, "cpu", benchmark_steps))
    if torch.cuda.is_available():
        rows.append(train_benchmark(config, "cuda", benchmark_steps))
    result = {"system": collect_system_info(), "benchmarks": rows}
    write_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
