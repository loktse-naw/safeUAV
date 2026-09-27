from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import config_digest, load_config, save_resolved_config
from securelink.environment import SecureLinkEnv
from securelink.physics import channel_gains, secrecy_metrics
from securelink.system_info import collect_system_info


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, sort_keys=True, default=_json_default)


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def old_power_diagnostic(old_results: Path, output: Path, config: dict) -> dict:
    steps_path = old_results / "raw_steps.csv"
    if not steps_path.exists():
        return {"status": "not_available", "path": str(steps_path)}
    frame = pd.read_csv(steps_path)
    tx_s = float(config["time"]["slot_s"]) * float(config["time"]["tx_duty"])
    budget = float(config["power"]["alice_radiated_budget_j"])
    bob = np.asarray(config["scenario"]["bob_m"], dtype=float)
    rows: list[dict] = []
    for keys, group in frame[frame["method"].isin(["B01", "B11"])].groupby(
        ["method", "train_seed", "episode_index"], dropna=False
    ):
        group = group.sort_values("slot").copy()
        cumulative = (group["alice_power_w"] * tx_s).cumsum().to_numpy()
        exhausted = np.flatnonzero(cumulative >= budget - 1e-9)
        distances = np.hypot(group["x_m"].to_numpy() - bob[0], group["y_m"].to_numpy() - bob[1])
        favorable_index = int(np.argmin(distances))
        third = np.array_split(np.arange(len(group)), 3)
        rows.append(
            {
                "method": keys[0],
                "train_seed": int(keys[1]),
                "episode_index": int(keys[2]),
                "exhaustion_slot": int(group.iloc[exhausted[0]]["slot"]) if exhausted.size else math.nan,
                "budget_exhausted": int(bool(exhausted.size)),
                "front_power_mean_w": float(group.iloc[third[0]]["alice_power_w"].mean()),
                "middle_power_mean_w": float(group.iloc[third[1]]["alice_power_w"].mean()),
                "back_power_mean_w": float(group.iloc[third[2]]["alice_power_w"].mean()),
                "front_asr_bpshz": float(group.iloc[third[0]]["secrecy_rate_bpshz"].mean()),
                "middle_asr_bpshz": float(group.iloc[third[1]]["secrecy_rate_bpshz"].mean()),
                "back_asr_bpshz": float(group.iloc[third[2]]["secrecy_rate_bpshz"].mean()),
                "closest_bob_slot": int(group.iloc[favorable_index]["slot"]),
                "closest_bob_distance_m": float(distances[favorable_index]),
                "remaining_budget_at_closest_bob_j": float(max(0.0, budget - cumulative[favorable_index])),
                "zero_power_fraction": float((group["alice_power_w"] <= 1e-12).mean()),
            }
        )
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "old_power_episode_diagnostic.csv", index=False)
    summary = (
        detail.groupby("method")
        .agg(
            episodes=("episode_index", "count"),
            exhausted_fraction=("budget_exhausted", "mean"),
            exhaustion_slot_mean=("exhaustion_slot", "mean"),
            front_power_mean_w=("front_power_mean_w", "mean"),
            middle_power_mean_w=("middle_power_mean_w", "mean"),
            back_power_mean_w=("back_power_mean_w", "mean"),
            front_asr_bpshz=("front_asr_bpshz", "mean"),
            middle_asr_bpshz=("middle_asr_bpshz", "mean"),
            back_asr_bpshz=("back_asr_bpshz", "mean"),
            remaining_budget_at_closest_bob_j=("remaining_budget_at_closest_bob_j", "mean"),
            zero_power_fraction=("zero_power_fraction", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "old_power_summary.csv", index=False)
    return {
        "status": "complete",
        "episode_rows": len(detail),
        "summary": summary.to_dict(orient="records"),
        "limitation": "Old rows contain executed power only; proposal-to-execution clipping is unavailable.",
    }


def _power_to_action(power_w: float, config: dict, env: SecureLinkEnv) -> float:
    peak = float(config["power"]["alice_peak_w"])
    mapping = str(config["power"].get("action_mapping", "legacy_affine"))
    target = float(np.clip(power_w, 0.0, peak))
    if mapping == "legacy_affine":
        return 2.0 * target / peak - 1.0
    remaining_slots = max(1, env.n_slots - env.slot)
    center = min(peak, env.radiated_remaining_j / (env.tx_s * remaining_slots))
    if target >= center:
        return 0.0 if peak <= center + 1e-15 else (target - center) / (peak - center)
    return -1.0 if center <= 1e-15 else target / center - 1.0


def _fixed_path_episode(config: dict, seed: int, strategy: str) -> tuple[dict, list[dict]]:
    method = "B00" if strategy == "fixed_0p2" else "B01"
    env = SecureLinkEnv(
        config,
        method=method,
        attack_rule="uniform",
        stream_context={"purpose": "power_diagnostic", "run_seed": 0, "worker_id": 0},
    )
    _, _ = env.reset(seed=seed)
    terminated = False
    while not terminated:
        if strategy in {"fixed_0p2", "uniform_budget"}:
            target = 0.2
        elif strategy == "frontload_budget":
            target = 1.0 if env.slot < 20 else 0.0
        elif strategy == "legacy_zero_action":
            target = 0.5
        elif strategy == "centered_zero_action":
            target = env.radiated_remaining_j / max(env.tx_s * (env.n_slots - env.slot), 1e-12)
        else:
            raise ValueError(strategy)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        if method == "B01":
            action[0] = _power_to_action(target, config, env)
        _, _, terminated, _, info = env.step(action)
    return dict(info["episode_summary"]), [dict(row) for row in env.step_records]


def fixed_path_power_diagnostic(config: dict, output: Path) -> dict:
    start = int(config["evaluation"]["development_seeds_start"])
    count = int(config["evaluation"]["development_episodes"])
    seeds = list(range(start, start + count))
    rows: list[dict] = []
    steps: list[dict] = []
    variants: list[tuple[str, dict]] = [
        ("fixed_0p2", config),
        ("uniform_budget", config),
        ("frontload_budget", config),
    ]
    legacy = json.loads(json.dumps(config))
    legacy["power"]["action_mapping"] = "legacy_affine"
    centered = json.loads(json.dumps(config))
    centered["power"]["action_mapping"] = "remaining_budget_centered"
    variants.extend([("legacy_zero_action", legacy), ("centered_zero_action", centered)])
    for strategy, variant in variants:
        for seed in seeds:
            summary, episode_steps = _fixed_path_episode(variant, seed, strategy)
            summary["strategy"] = strategy
            rows.append(summary)
            for row in episode_steps:
                row["strategy"] = strategy
                steps.append(row)
    episodes = pd.DataFrame(rows)
    step_frame = pd.DataFrame(steps)
    episodes.to_csv(output / "fixed_path_power_episodes.csv", index=False)
    step_frame.to_csv(output / "fixed_path_power_steps.csv", index=False)
    metrics = episodes.groupby("strategy").agg(
        episodes=("scenario_seed", "count"),
        asr_mean=("asr_bpshz", "mean"),
        asr_std=("asr_bpshz", "std"),
        slot_sop=("slot_sop", "mean"),
        episode_sop=("episode_sop", "mean"),
        radiated_energy_j=("alice_radiated_energy_j", "mean"),
        total_energy_j=("alice_total_energy_j", "mean"),
    ).reset_index()
    metrics.to_csv(output / "fixed_path_power_summary.csv", index=False)
    return {"seeds": seeds, "summary": metrics.to_dict(orient="records")}


def _paired_minimum_classification(values: np.ndarray, powers: np.ndarray) -> tuple[int, str, float, float]:
    means = values.mean(axis=0)
    index = int(np.argmin(means))
    differences = values - values[:, [index]]
    standard_errors = differences.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    tied = differences.mean(axis=0) <= 1.96 * standard_errors + 1e-6
    if tied[0] and tied[-1]:
        label = "tie_or_zero_rate_platform"
    elif index == 0:
        label = "silent"
    elif index == len(powers) - 1:
        label = "peak"
    else:
        label = "interior"
    paired_delta = values[:, -1] - values[:, index]
    return index, label, float(paired_delta.mean()), float(paired_delta.std(ddof=1) / math.sqrt(len(paired_delta)))


def monte_carlo_power_scan(config: dict, output: Path) -> dict:
    samples = int(config["diagnostics"]["monte_carlo_samples"])
    root_seed = int(config["diagnostics"]["monte_carlo_seed"])
    grid_points = int(config["diagnostics"]["willie_power_grid_points"])
    powers_w = np.linspace(0.0, float(config["power"]["willie_peak_w"]), grid_points)
    target = float(config["metric"]["secrecy_target_bpshz"])
    reference = SecureLinkEnv(config, method="B00", attack_rule="none").reference_nodes
    positions = {
        "start": reference[0],
        "path_q1": reference[25],
        "path_mid": reference[50],
        "path_q3": reference[75],
        "last_tx": reference[99],
    }
    aggregate_rows: list[dict] = []
    minimum_rows: list[dict] = []
    group_id = 0
    for position_name, position in positions.items():
        for alice_power in (0.05, 0.2, 1.0):
            seed_sequence = np.random.SeedSequence([root_seed, group_id])
            children = seed_sequence.spawn(samples)
            secrecy = np.empty((samples, grid_points), dtype=float)
            for sample_index, child in enumerate(children):
                gains = channel_gains(position, config, np.random.default_rng(child), fading=True)
                for power_index, willie_power in enumerate(powers_w):
                    secrecy[sample_index, power_index] = secrecy_metrics(
                        alice_power, float(willie_power), gains, config
                    )["secrecy_rate_bpshz"]
            for power_index, willie_power in enumerate(powers_w):
                values = secrecy[:, power_index]
                aggregate_rows.append(
                    {
                        "position": position_name,
                        "alice_power_w": alice_power,
                        "willie_power_w": willie_power,
                        "samples": samples,
                        "paired_group": group_id,
                        "mean_asr_bpshz": float(values.mean()),
                        "asr_standard_error": float(values.std(ddof=1) / math.sqrt(samples)),
                        "sop": float(np.mean(values < target)),
                        "sop_standard_error": float(
                            math.sqrt(max(0.0, np.mean(values < target) * (1.0 - np.mean(values < target)) / samples))
                        ),
                    }
                )
            minimum_index, label, delta, delta_se = _paired_minimum_classification(secrecy, powers_w)
            minimum_rows.append(
                {
                    "position": position_name,
                    "alice_power_w": alice_power,
                    "samples": samples,
                    "paired_group": group_id,
                    "minimizing_willie_power_w": float(powers_w[minimum_index]),
                    "classification": label,
                    "minimum_mean_asr_bpshz": float(secrecy[:, minimum_index].mean()),
                    "peak_minus_minimum_asr": delta,
                    "peak_minus_minimum_standard_error": delta_se,
                }
            )
            group_id += 1
    aggregate = pd.DataFrame(aggregate_rows)
    minima = pd.DataFrame(minimum_rows)
    aggregate.to_csv(output / "willie_fading_scan_aggregate.csv", index=False)
    minima.to_csv(output / "willie_fading_scan_minima.csv", index=False)

    alice_powers = np.asarray(config["diagnostics"]["alice_power_grid_w"], dtype=float)
    episode_values = np.empty((samples, len(alice_powers)), dtype=float)
    episode_sop = np.empty_like(episode_values)
    for sample_index, child in enumerate(np.random.SeedSequence([root_seed, 999]).spawn(samples)):
        slot_children = child.spawn(config["time"]["slots"])
        rates = np.empty((config["time"]["slots"], len(alice_powers)), dtype=float)
        for slot, slot_child in enumerate(slot_children):
            gains = channel_gains(reference[slot], config, np.random.default_rng(slot_child), fading=True)
            for power_index, alice_power in enumerate(alice_powers):
                rates[slot, power_index] = secrecy_metrics(
                    float(alice_power), float(config["power"]["willie_uniform_w"]), gains, config
                )["secrecy_rate_bpshz"]
        episode_values[sample_index] = rates.mean(axis=0)
        episode_sop[sample_index] = (rates < target).mean(axis=0)
    alice_rows = []
    for index, alice_power in enumerate(alice_powers):
        alice_rows.append(
            {
                "alice_power_w": alice_power,
                "samples": samples,
                "mean_asr_bpshz": float(episode_values[:, index].mean()),
                "asr_standard_error": float(episode_values[:, index].std(ddof=1) / math.sqrt(samples)),
                "mean_slot_sop": float(episode_sop[:, index].mean()),
                "slot_sop_standard_error": float(episode_sop[:, index].std(ddof=1) / math.sqrt(samples)),
            }
        )
    alice_frame = pd.DataFrame(alice_rows)
    alice_frame.to_csv(output / "alice_power_path_fading_scan.csv", index=False)
    counts = minima["classification"].value_counts().to_dict()
    return {
        "samples_per_group": samples,
        "root_seed": root_seed,
        "paired_exogenous_fading": True,
        "willie_groups": len(minima),
        "classification_counts": counts,
        "classification_is_group_fraction_not_state_probability": True,
        "alice_path_summary": alice_rows,
    }


def repair_old_accounting(old_results: Path, output: Path) -> dict:
    runtime_path = old_results / "training_runtime.csv"
    if not runtime_path.exists():
        return {"status": "not_available", "path": str(runtime_path)}
    runtime = pd.read_csv(runtime_path)
    rows = []
    for _, row in runtime.iterrows():
        requested = int(row.get("timesteps", 0))
        actual = 0
        if row["method"] != "B00":
            progress_path = old_results / "training_logs" / f"{row['method']}_seed{int(row['train_seed'])}" / "progress.csv"
            progress = pd.read_csv(progress_path)
            actual = int(progress["time/total_timesteps"].dropna().iloc[-1])
        rows.append(
            {
                "method": row["method"],
                "train_seed": int(row["train_seed"]),
                "requested_training_transitions": requested,
                "actual_training_transitions": actual,
                "difference": actual - requested,
                "source": "training_runtime.csv + progress.csv",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "old_training_accounting_corrected.csv", index=False)
    learning = frame[frame["method"] != "B00"]
    result = {
        "status": "complete",
        "requested_training_transitions": int(learning["requested_training_transitions"].sum()),
        "actual_training_transitions": int(learning["actual_training_transitions"].sum()),
        "old_raw_evaluation_transitions": int(len(pd.read_csv(old_results / "raw_steps.csv"))),
        "fair_evaluation_transitions_without_duplicate_b00": 30000,
        "note": "Old files are preserved; this is a derived correction ledger.",
    }
    write_json(result, output / "old_training_accounting_corrected.json")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "acceptance.yaml"))
    parser.add_argument("--output", default=str(ROOT / "results" / "acceptance_v2"))
    parser.add_argument("--old-results", default=str(ROOT / "results" / "d1_remote"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    save_resolved_config(config, output / "resolved_config.yaml")
    report = {
        "implementation_version": config["implementation"]["version"],
        "config_digest": config_digest(config),
        "old_power_diagnostic": old_power_diagnostic(Path(args.old_results), output, config),
        "fixed_path_power_diagnostic": fixed_path_power_diagnostic(config, output),
        "monte_carlo_power_scan": monte_carlo_power_scan(config, output),
        "old_accounting_correction": repair_old_accounting(Path(args.old_results), output),
        "observation_contract": SecureLinkEnv.observation_contract(),
        "system": collect_system_info(),
    }
    write_json(report, output / "acceptance_diagnostics.json")
    print(json.dumps({"output": str(output), "config_digest": report["config_digest"]}, indent=2))


if __name__ == "__main__":
    main()
