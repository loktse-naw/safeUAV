"""Fine-grid, true-channel myopic diagnostic; never an implementable attacker."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import config_digest, load_config, save_resolved_config
from securelink.environment import SecureLinkEnv
from securelink.evaluation import run_episode
from securelink.physics import db_to_linear, receiver_noise_w
from run_attack_pressure_scan import power_grid


class FinePrivilegedEnv(SecureLinkEnv):
    def _willie_power(self) -> float:
        """Use the current hidden gains and previous Alice power on the frozen 189-point grid."""
        assert self._current_gains is not None
        peak = float(self.config["power"]["willie_peak_w"])
        cap = min(peak, self.willie_remaining_j / self.tx_s)
        if cap <= 0:
            return 0.0
        grid = power_grid()
        grid = np.unique(np.append(grid[grid <= cap], cap))
        gains = self._current_gains
        alice = float(self.prev_alice_power_w or self.config["power"]["alice_fixed_w"])
        noise = receiver_noise_w(self.config)
        beta = db_to_linear(float(self.config["power"]["self_interference_db"]))
        bob = np.log1p(alice * gains["g_ab"] / (gains["g_wb"] * grid + noise))
        willie = np.log1p(alice * gains["g_aw"] / (beta * grid + noise))
        rates = np.maximum(0.0, (bob - willie) / np.log(2.0))
        return float(grid[int(np.argmin(rates))])


def main() -> None:
    config = load_config(ROOT / "configs" / "motion_residual_v1.yaml")
    output = ROOT / "results" / "attack_privileged_fine_v1"
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output / "resolved_config.yaml")
    model_paths = {
        seed: ROOT / "results" / "motion_residual_v1_32768" / "models" / f"B10_seed{seed}.zip"
        for seed in (11001, 11002, 11003)
    }
    models = {seed: PPO.load(path, device="cpu") for seed, path in model_paths.items()}
    episode_rows = []
    step_rows = []
    for defense, seed, model in [("B00", 0, None)] + [
        ("residual_B10", seed, models[seed]) for seed in models
    ]:
        env = FinePrivilegedEnv(
            config,
            method="B00" if model is None else "B10",
            attack_rule="privileged_myopic_fine_189",
            stream_context={"purpose": "attack_development", "run_seed": 0, "worker_id": 0},
        )
        for scenario in range(30001, 30031):
            predictor = None if model is None else lambda obs, active=model: active.predict(obs, deterministic=True)[0]
            episode, steps = run_episode(env, scenario, predictor)
            labels = {
                "defense": defense,
                "train_seed": seed,
                "attack_label": "privileged_myopic_fine_189",
                "attack_class": "privileged_reference",
            }
            episode.update(labels)
            episode_rows.append(episode)
            for step in steps:
                step.update(labels)
                step_rows.append(step)
        print(f"{defense} seed {seed} complete", flush=True)
    episodes = pd.DataFrame(episode_rows)
    steps = pd.DataFrame(step_rows)
    episodes.to_csv(output / "raw_episodes.csv", index=False)
    steps.to_csv(output / "raw_steps.csv", index=False)
    summary = episodes.groupby("defense")[[
        "asr_bpshz", "slot_sop", "episode_sop", "task_completed", "willie_energy_j",
        "alice_radiated_energy_j", "propulsion_energy_j", "executed_hard_violations",
    ]].mean()
    summary["episodes"] = episodes.groupby("defense").size()
    summary.to_csv(output / "summary.csv")
    manifest = {
        "config_digest": config_digest(config),
        "model_sha256": {str(seed): hashlib.sha256(path.read_bytes()).hexdigest() for seed, path in model_paths.items()},
        "scenarios": list(range(30001, 30031)),
        "grid_points_before_budget_cap": len(power_grid()),
        "episode_rows": len(episodes),
        "step_rows": len(steps),
        "all_episodes_100_slots": bool((episodes.slots_recorded == 100).all()),
        "executed_hard_violations": int(episodes.executed_hard_violations.sum()),
        "all_willie_energy_within_6j": bool((episodes.willie_energy_j <= 6.0 + 1e-8).all()),
        "final_test_transitions": 0,
        "note": "Current true hidden gains and prior Alice power; fine-grid myopic numeric reference, not an implementable rule or episode optimum.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("episode_rows", "step_rows", "executed_hard_violations")}, indent=2))


if __name__ == "__main__":
    main()
