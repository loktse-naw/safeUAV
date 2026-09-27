"""Evaluate the fixed v2 path and the dynamic zero-residual path without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import load_config, save_resolved_config
from securelink.evaluation import evaluate_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "results" / "motion_baselines_v1"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "configs" / "motion_residual_v1.yaml")
    save_resolved_config(config, output / "resolved_config.yaml")
    seeds = list(range(20001, 20031))
    fixed_e, fixed_s = evaluate_policy(config, "B00", "uniform", seeds)
    zero_e, zero_s = evaluate_policy(config, "B10", "uniform", seeds)
    fixed_e["comparison_method"] = "B00_fixed_initial"
    zero_e["comparison_method"] = "R0_dynamic_zero"
    fixed_s["comparison_method"] = "B00_fixed_initial"
    zero_s["comparison_method"] = "R0_dynamic_zero"
    episodes = pd.concat([fixed_e, zero_e], ignore_index=True)
    steps = pd.concat([fixed_s, zero_s], ignore_index=True)
    episodes.to_csv(output / "raw_episodes.csv", index=False)
    steps.to_csv(output / "raw_steps.csv", index=False)
    key = ["scenario_seed", "slot"]
    f = fixed_s.set_index(key).sort_index()
    z = zero_s.set_index(key).sort_index()
    path_offset = np.hypot(f.x_m - z.x_m, f.y_m - z.y_m)
    power_difference = np.abs(f.alice_power_w - z.alice_power_w)
    summary = {
        "evaluation_seeds": seeds,
        "episodes_per_method": len(fixed_e),
        "fixed": {name: float(fixed_e[name].mean()) for name in ("asr_bpshz", "slot_sop", "episode_sop", "task_completed", "propulsion_energy_j", "alice_radiated_energy_j", "fallback_takeover_slots")},
        "dynamic_zero": {name: float(zero_e[name].mean()) for name in ("asr_bpshz", "slot_sop", "episode_sop", "task_completed", "propulsion_energy_j", "alice_radiated_energy_j", "fallback_takeover_slots")},
        "mean_path_offset_m": float(path_offset.mean()),
        "max_path_offset_m": float(path_offset.max()),
        "max_power_difference_w": float(power_difference.max()),
        "path_stepwise_identical_within_1e_8_m": bool((path_offset <= 1e-8).all()),
        "zero_hard_violations": int(zero_e.executed_hard_violations.sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
