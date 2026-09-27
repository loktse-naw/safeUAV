"""Replay frozen v2 direct-motion models against the current direct branch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import load_config
from securelink.evaluation import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen", default=str(ROOT / "results" / "d1_v2_motion_joint"))
    parser.add_argument("--output", default=str(ROOT / "results" / "motion_direct_compat_v1"))
    args = parser.parse_args()
    frozen = Path(args.frozen).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "controlled_rerun.yaml")
    old_episodes = pd.read_csv(frozen / "raw_episodes.csv")
    old_steps = pd.read_csv(frozen / "raw_steps.csv")
    episode_parts = []
    step_parts = []
    for method in ("B10", "B11"):
        for seed in (11001, 11002, 11003):
            model = PPO.load(frozen / "models" / f"{method}_seed{seed}.zip", device="cpu")

            def predict(observation, active=model):
                return active.predict(observation, deterministic=True)[0]

            episodes, steps = evaluate_policy(
                config, method, "uniform", list(range(20001, 20031)), predict, seed
            )
            episode_parts.append(episodes)
            step_parts.append(steps)
    new_episodes = pd.concat(episode_parts, ignore_index=True)
    new_steps = pd.concat(step_parts, ignore_index=True)
    keys = ["method", "train_seed", "scenario_seed"]
    episode_columns = ["asr_bpshz", "slot_sop", "episode_sop", "task_completed", "alice_total_energy_j", "alice_radiated_energy_j", "fallback_takeover_slots", "executed_hard_violations"]
    step_columns = ["proposed_dx_m", "proposed_dy_m", "executed_dx_m", "executed_dy_m", "alice_power_w", "secrecy_rate_bpshz", "slot_energy_j"]
    old_e = old_episodes[old_episodes.method.isin(("B10", "B11"))].set_index(keys).sort_index()
    new_e = new_episodes.set_index(keys).sort_index()
    old_s = old_steps[old_steps.method.isin(("B10", "B11"))].set_index(keys + ["slot"]).sort_index()
    new_s = new_steps.set_index(keys + ["slot"]).sort_index()
    assert old_e.index.equals(new_e.index) and old_s.index.equals(new_s.index)
    deviations = {
        "episodes": {name: float(np.max(np.abs(old_e[name] - new_e[name]))) for name in episode_columns},
        "steps": {name: float(np.max(np.abs(old_s[name] - new_s[name]))) for name in step_columns},
    }
    tolerances = {
        "asr_bpshz": 1e-6,
        "slot_sop": 0.0,
        "episode_sop": 0.0,
        "task_completed": 0.0,
        "alice_total_energy_j": 1e-3,
        "alice_radiated_energy_j": 1e-6,
        "fallback_takeover_slots": 0.0,
        "executed_hard_violations": 0.0,
        "proposed_dx_m": 1e-5,
        "proposed_dy_m": 1e-5,
        "executed_dx_m": 1e-5,
        "executed_dy_m": 1e-5,
        "alice_power_w": 1e-6,
        "secrecy_rate_bpshz": 1e-6,
        "slot_energy_j": 1e-3,
    }
    summary = {
        "frozen_result": str(frozen),
        "episode_rows": len(new_e),
        "step_rows": len(new_s),
        "maximum_absolute_deviation": deviations,
        "tolerances": tolerances,
        "compatible_within_tolerance": all(
            value <= tolerances[name] for group in deviations.values() for name, value in group.items()
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    new_episodes.to_csv(output / "replayed_episodes.csv", index=False)
    new_steps.to_csv(output / "replayed_steps.csv", index=False)
    print(json.dumps(summary, indent=2))
    if not summary["compatible_within_tolerance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
