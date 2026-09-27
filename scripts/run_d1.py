from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import config_digest, load_config, save_resolved_config
from securelink.environment import SecureLinkEnv
from securelink.evaluation import evaluate_policy, write_json
from securelink.system_info import collect_system_info


def source_digest() -> str:
    digest = hashlib.sha256()
    paths = sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "scripts").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _env_factory(config: dict, method: str, attack_rule: str, run_seed: int, worker_id: int):
    def factory():
        env = SecureLinkEnv(
            config,
            method=method,
            attack_rule=attack_rule,
            stream_context={"purpose": "train", "run_seed": run_seed, "worker_id": worker_id},
        )
        return Monitor(env)

    return factory


def train_model(
    config: dict,
    method: str,
    seed: int,
    output: Path,
    device: str,
    n_envs: int,
) -> tuple[PPO, dict]:
    set_random_seed(seed)
    factories = [
        _env_factory(config, method, str(config["training"]["attack_rule"]), seed, rank)
        for rank in range(n_envs)
    ]
    if n_envs > 1 and os.name != "nt":
        env = SubprocVecEnv(factories, start_method="spawn")
        vector_type = "subproc"
    else:
        env = DummyVecEnv(factories)
        vector_type = "dummy"
    log_dir = output / "training_logs" / f"{method}_seed{seed}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = configure(str(log_dir), ["csv"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    process = psutil.Process(os.getpid())
    rss_before = process.memory_info().rss
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
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
    model.set_logger(logger)
    start = time.perf_counter()
    model.learn(total_timesteps=int(config["training"]["total_timesteps"]), progress_bar=False)
    elapsed = time.perf_counter() - start
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{method}_seed{seed}"
    model.save(model_path)
    rss_after = process.memory_info().rss
    timing = {
        "method": method,
        "train_seed": seed,
        "n_envs": n_envs,
        "vector_type": vector_type,
        "requested_timesteps": int(config["training"]["total_timesteps"]),
        "actual_timesteps": int(model.num_timesteps),
        "timesteps": int(model.num_timesteps),
        "seconds": elapsed,
        "timesteps_per_second": int(model.num_timesteps) / elapsed,
        "requested_device": device,
        "actual_device": str(model.device),
        "rss_before_mib": rss_before / 1024**2,
        "rss_after_mib": rss_after / 1024**2,
        "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0.0
        ),
        "model_path": str(model_path.with_suffix(".zip")),
        "log_path": str(log_dir / "progress.csv"),
    }
    env.close()
    return model, timing


def choose_runtime(config: dict, benchmark_path: Path | None) -> tuple[str, int]:
    requested = str(config["training"].get("device", "auto"))
    n_envs = 1
    if benchmark_path and benchmark_path.exists():
        with benchmark_path.open("r", encoding="utf-8") as handle:
            benchmark = json.load(handle)
        training = [row for row in benchmark["benchmarks"] if row["kind"] == "ppo_training"]
        if training:
            fastest = max(training, key=lambda row: row["timesteps_per_second"])
            requested = str(fastest["requested_device"])
        sampling = [row for row in benchmark["benchmarks"] if row["kind"] == "environment_sampling"]
        if sampling:
            fastest_sampling = max(sampling, key=lambda row: row["transitions_per_second"])
            if fastest_sampling["vector_type"] == "subproc" and os.name != "nt":
                n_envs = int(fastest_sampling["n_envs"])
    return requested, n_envs


def plot_results(episodes: pd.DataFrame, steps: pd.DataFrame, output: Path) -> None:
    metrics = ["asr_bpshz", "slot_sop", "task_completed", "alice_total_energy_j"]
    summary = episodes.groupby(["method", "train_seed"], dropna=False)[metrics].mean().reset_index()
    summary.to_csv(output / "method_seed_summary.csv", index=False)
    overall = episodes.groupby("method")[metrics].agg(["mean", "std", "count"])
    overall.to_csv(output / "method_overall_summary.csv")

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, metric in zip(axes.flat, metrics):
        grouped = episodes.groupby("method")[metric]
        means = grouped.mean()
        stds = grouped.std().fillna(0.0)
        axis.bar(means.index, means.values, yerr=stds.values, capsize=3)
        axis.set_title(metric)
        axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output / "d1_metrics.png", dpi=180)
    plt.close(figure)

    selected = steps[(steps["episode_index"] == 0)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (method, train_seed), group in selected.groupby(["method", "train_seed"], dropna=False):
        axes[0].plot(group["x_m"], group["y_m"], alpha=0.7, label=f"{method}-{train_seed}")
        axes[1].plot(group["slot"], group["alice_power_w"], alpha=0.7, label=f"{method}-{train_seed}")
    axes[0].set(xlabel="x (m)", ylabel="y (m)", title="Test trajectory, first held-out episode")
    axes[0].axis("equal")
    axes[1].set(xlabel="slot", ylabel="Alice power (W)", title="Power curve, first held-out episode")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[1].legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(output / "trajectory_and_power.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "prototype.yaml"))
    parser.add_argument("--output", default=str(ROOT / "results" / "d1"))
    parser.add_argument("--benchmark", default=str(ROOT / "results" / "benchmark.json"))
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    save_resolved_config(config, output / "resolved_config.yaml")
    methods = args.methods or list(config["training"]["methods"])
    train_seeds = [int(value) for value in config["training"]["train_seeds"]]
    test_start = int(config["training"].get("development_seeds_start", config["training"].get("test_seeds_start")))
    test_count = int(config["training"].get("development_episodes", config["training"].get("test_episodes")))
    test_seeds = list(range(test_start, test_start + test_count))
    benchmark_path = Path(args.benchmark).resolve() if args.benchmark else None
    benchmark_device, benchmark_n_envs = choose_runtime(config, benchmark_path)
    device = args.device or benchmark_device
    n_envs = args.n_envs or benchmark_n_envs

    episode_frames: list[pd.DataFrame] = []
    step_frames: list[pd.DataFrame] = []
    timings: list[dict] = []
    for method in methods:
        method_seeds: list[int | None] = [None] if method == "B00" else train_seeds
        for seed in method_seeds:
            if method == "B00":
                model = None
                timings.append(
                    {
                        "method": method,
                        "train_seed": None,
                        "n_envs": 0,
                        "vector_type": "none",
                        "requested_timesteps": 0,
                        "actual_timesteps": 0,
                        "timesteps": 0,
                        "seconds": 0.0,
                        "timesteps_per_second": None,
                        "requested_device": device,
                        "actual_device": "none",
                        "rss_before_mib": None,
                        "rss_after_mib": None,
                        "cuda_peak_allocated_mib": 0.0,
                        "model_path": None,
                        "log_path": None,
                    }
                )
                predictor = None
            else:
                assert seed is not None
                model, timing = train_model(config, method, seed, output, device, n_envs)
                timings.append(timing)

                def predictor(observation, active=model):
                    action, _ = active.predict(observation, deterministic=True)
                    return action

            episodes, steps = evaluate_policy(
                config,
                method,
                str(config["training"]["attack_rule"]),
                test_seeds,
                predictor,
                seed,
                purpose="development",
            )
            episode_frames.append(episodes)
            step_frames.append(steps)
    episode_frame = pd.concat(episode_frames, ignore_index=True)
    step_frame = pd.concat(step_frames, ignore_index=True)
    episode_frame.to_csv(output / "raw_episodes.csv", index=False)
    step_frame.to_csv(output / "raw_steps.csv", index=False)
    timing_frame = pd.DataFrame(timings)
    timing_frame.to_csv(output / "training_runtime.csv", index=False)
    plot_results(episode_frame, step_frame, output)
    manifest = {
        "config_digest": config_digest(config),
        "implementation_version": config.get("implementation", {}).get("version", "unknown"),
        "source_digest": source_digest(),
        "methods": methods,
        "train_seeds": train_seeds,
        "development_seeds": test_seeds,
        "final_test_seeds": {
            "start": config.get("evaluation", {}).get("final_test_seeds_start"),
            "episodes": config.get("evaluation", {}).get("final_test_episodes"),
            "status": config.get("evaluation", {}).get("final_test_status", "not_defined"),
        },
        "attack_rule": config["training"]["attack_rule"],
        "device": device,
        "n_envs": n_envs,
        "budget": {
            "requested_training_transitions": int(sum(row["requested_timesteps"] for row in timings)),
            "actual_training_transitions": int(sum(row["actual_timesteps"] for row in timings)),
            "development_transitions": int(len(step_frame)),
            "final_test_transitions": 0,
        },
        "system": collect_system_info(),
        "files": {
            "raw_episodes": str(output / "raw_episodes.csv"),
            "raw_steps": str(output / "raw_steps.csv"),
            "runtime": str(output / "training_runtime.csv"),
            "summary": str(output / "method_seed_summary.csv"),
        },
    }
    write_json(manifest, output / "manifest.json")
    print(
        json.dumps(
            {"output": str(output), "methods": methods, "device": device, "n_envs": n_envs}, indent=2
        )
    )


if __name__ == "__main__":
    main()
