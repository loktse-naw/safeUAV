"""Evaluate fixed defense policies against feasible rule attacks and a separate privileged reference."""

from __future__ import annotations

import argparse
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


SCENARIOS = list(range(30001, 30031))
TRAIN_SEEDS = (11001, 11002, 11003)
RULES = (
    ("silent", "none", "implementable"),
    ("uniform_0p3", "uniform", "implementable"),
    ("frontload_peak", "frontload", "implementable"),
    ("random_bangbang", "random_bangbang", "implementable"),
    ("low_constant", "low_constant", "implementable"),
    ("privileged_myopic_reference", "privileged_myopic_reference", "privileged_reference"),
)


def bootstrap_paired(values: np.ndarray, seed: int) -> tuple[float, float]:
    """Crossed seed/scenario resampling; each scenario draw is shared by all seeds."""

    rng = np.random.default_rng(seed)
    trials = np.empty(5_000)
    n_seeds, n_scenarios = values.shape
    for index in range(len(trials)):
        seed_index = rng.integers(0, n_seeds, n_seeds)
        scenario_index = rng.integers(0, n_scenarios, n_scenarios)
        trials[index] = values[np.ix_(seed_index, scenario_index)].mean()
    return float(np.quantile(trials, 0.025)), float(np.quantile(trials, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "motion_residual_v1.yaml"))
    parser.add_argument("--scan", default=str(ROOT / "results" / "attack_pressure_scan_v1" / "metadata.json"))
    parser.add_argument("--models", default=str(ROOT / "results" / "motion_residual_v1_32768" / "models"))
    parser.add_argument("--output", default=str(ROOT / "results" / "attack_episode_rules_v1"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    save_resolved_config(config, output / "resolved_config.yaml")
    scan = json.loads(Path(args.scan).read_text(encoding="utf-8"))
    low_power = float(scan["low_power_rule_w"])
    if not 0.0 < low_power <= float(config["power"]["willie_peak_w"]):
        raise ValueError("Frozen low-power rule is outside Willie peak limit")
    if scan["config_digest"] != config_digest(config):
        raise ValueError("Scan and episode configuration digests differ")

    models = {seed: PPO.load(Path(args.models) / f"B10_seed{seed}.zip", device="cpu") for seed in TRAIN_SEEDS}
    model_hashes = {
        str(seed): hashlib.sha256((Path(args.models) / f"B10_seed{seed}.zip").read_bytes()).hexdigest()
        for seed in TRAIN_SEEDS
    }
    episode_rows: list[dict] = []
    step_rows: list[dict] = []
    defenses = [("B00", 0, None)] + [("residual_B10", seed, models[seed]) for seed in TRAIN_SEEDS]
    for defense, train_seed, model in defenses:
        method = "B00" if model is None else "B10"

        def predictor(observation, active=model):
            return active.predict(observation, deterministic=True)[0]

        for rule_label, internal_rule, rule_class in RULES:
            attack_policy = (lambda allowed_observation: low_power) if rule_label == "low_constant" else None
            env = SecureLinkEnv(
                config,
                method=method,
                attack_rule=internal_rule,
                attack_policy=attack_policy,
                stream_context={"purpose": "attack_development", "run_seed": 0, "worker_id": 0},
            )
            for scenario_seed in SCENARIOS:
                episode, steps = run_episode(env, scenario_seed, None if model is None else predictor)
                episode.update(
                    {
                        "defense": defense,
                        "train_seed": train_seed,
                        "attack_label": rule_label,
                        "attack_class": rule_class,
                    }
                )
                episode_rows.append(episode)
                for step in steps:
                    step.update(
                        {
                            "defense": defense,
                            "train_seed": train_seed,
                            "attack_label": rule_label,
                            "attack_class": rule_class,
                        }
                    )
                    step_rows.append(step)
            print(f"{defense} seed {train_seed}: {rule_label} complete", flush=True)
    episodes = pd.DataFrame(episode_rows)
    steps = pd.DataFrame(step_rows)
    episodes.to_csv(output / "raw_episodes.csv", index=False)
    steps.to_csv(output / "raw_steps.csv", index=False)

    measures = [
        "asr_bpshz",
        "slot_sop",
        "episode_sop",
        "task_completed",
        "willie_energy_j",
        "alice_radiated_energy_j",
        "propulsion_energy_j",
        "executed_hard_violations",
    ]
    summary = episodes.groupby(["defense", "attack_label", "attack_class"])[measures].mean().reset_index()
    summary["episodes"] = episodes.groupby(["defense", "attack_label", "attack_class"]).size().values
    summary.to_csv(output / "rule_summary.csv", index=False)
    by_seed = episodes.groupby(["defense", "train_seed", "attack_label", "attack_class"])[measures].mean().reset_index()
    by_seed.to_csv(output / "seed_rule_summary.csv", index=False)
    steps["phase"] = pd.cut(steps.slot, bins=[-1, 32, 66, 99], labels=["early", "middle", "late"])
    phase = steps.groupby(["defense", "attack_label", "phase"], observed=True)[["secrecy_rate_bpshz", "willie_power_w", "alice_power_w"]].mean().reset_index()
    phase.to_csv(output / "phase_summary.csv", index=False)

    index = ["defense", "train_seed", "scenario_seed"]
    uniform = episodes[episodes.attack_label == "uniform_0p3"][index + ["asr_bpshz"]].rename(
        columns={"asr_bpshz": "uniform_asr_bpshz"}
    )
    paired = episodes.merge(uniform, on=index, validate="many_to_one")
    paired["asr_minus_uniform"] = paired.asr_bpshz - paired.uniform_asr_bpshz
    paired[index + ["attack_label", "attack_class", "asr_minus_uniform", "willie_energy_j"]].to_csv(
        output / "paired_episode_differences.csv", index=False
    )
    interval_rows = []
    for (defense, rule_label, rule_class), group in paired.groupby(["defense", "attack_label", "attack_class"]):
        matrix = group.pivot(index="train_seed", columns="scenario_seed", values="asr_minus_uniform").sort_index().sort_index(axis=1).to_numpy()
        low, high = bootstrap_paired(matrix, 78001 + len(interval_rows))
        interval_rows.append(
            {
                "defense": defense,
                "attack_label": rule_label,
                "attack_class": rule_class,
                "asr_minus_uniform": float(matrix.mean()),
                "exploratory_crossed_bootstrap_low": low,
                "exploratory_crossed_bootstrap_high": high,
                "training_seeds": len(matrix),
                "scenarios": matrix.shape[1],
            }
        )
    pd.DataFrame(interval_rows).to_csv(output / "paired_rule_intervals.csv", index=False)
    manifest = {
        "config_digest": config_digest(config),
        "scan_metadata_sha256": hashlib.sha256(Path(args.scan).read_bytes()).hexdigest(),
        "model_sha256": model_hashes,
        "low_power_rule_w": low_power,
        "scenarios": SCENARIOS,
        "training_seeds": list(TRAIN_SEEDS),
        "rule_labels": [row[0] for row in RULES],
        "implementable_rules": [row[0] for row in RULES if row[2] == "implementable"],
        "privileged_rule": "privileged_myopic_reference",
        "episode_rows": len(episodes),
        "step_rows": len(steps),
        "final_test_transitions": 0,
        "all_episodes_100_slots": bool((episodes.slots_recorded == 100).all()),
        "executed_hard_violations": int(episodes.executed_hard_violations.sum()),
        "all_willie_energy_within_6j": bool((episodes.willie_energy_j <= 6.0 + 1e-8).all()),
        "low_rule_allowed_observation_only": True,
        "note": "The privileged myopic rule reads simulator true current gains and is not an information-matched attacker or an episode oracle.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("episode_rows", "step_rows", "low_power_rule_w", "executed_hard_violations")}, indent=2))


if __name__ == "__main__":
    main()
