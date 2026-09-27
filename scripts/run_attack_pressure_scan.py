"""Paired low-power Willie scan with an independent confirmation sample."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import config_digest, load_config, save_resolved_config
from securelink.environment import SecureLinkEnv
from securelink.physics import channel_gains, db_to_linear, receiver_noise_w


SAMPLES = 10_000
BATCH = 1_000
ROOT_SEED = 74001
PRACTICAL_TIE_BPSHZ = 0.005
POSITIONS = ("start", "path_q1", "path_mid", "path_q3", "last_tx")
ALICE_POWERS = (0.05, 0.2, 1.0)
SI_DB = (-120.0, -110.0, -100.0)


def power_grid() -> np.ndarray:
    coarse = np.arange(0.0, 1.0000001, 0.02)
    fine_near_zero = np.arange(0.0001, 0.0100001, 0.0001)
    fine_low = np.arange(0.011, 0.0500001, 0.001)
    return np.unique(np.round(np.concatenate((coarse, fine_near_zero, fine_low)), 8))


def source_gains(
    position: np.ndarray, config: dict, source_id: int, phase_id: int
) -> np.ndarray:
    """Draw the three fading links once; every power point shares these draws."""

    rng = np.random.default_rng(np.random.SeedSequence([ROOT_SEED, source_id, phase_id]))
    normals = rng.normal(size=(SAMPLES, 6))
    k = db_to_linear(float(config["channel"]["rician_k_db"]))
    los = math.sqrt(k / (k + 1.0))
    scatter = math.sqrt(1.0 / (2.0 * (k + 1.0)))
    fad_ab = np.abs(los + scatter * (normals[:, 0] + 1j * normals[:, 1])) ** 2
    fad_aw = np.abs(los + scatter * (normals[:, 2] + 1j * normals[:, 3])) ** 2
    fad_wb = np.abs((normals[:, 4] + 1j * normals[:, 5]) / math.sqrt(2.0)) ** 2
    base = channel_gains(position, config, fading=False)
    return np.stack((base["g_ab"] * fad_ab, base["g_aw"] * fad_aw, base["g_wb"] * fad_wb), axis=1)


def rates_for_grid(gains: np.ndarray, alice_power: float, powers: np.ndarray, config: dict) -> np.ndarray:
    noise = receiver_noise_w(config)
    si = db_to_linear(float(config["power"]["self_interference_db"]))
    ab = gains[:, [0]]
    aw = gains[:, [1]]
    wb = gains[:, [2]]
    gamma_b = alice_power * ab / (wb * powers[None, :] + noise)
    gamma_w = alice_power * aw / (si * powers[None, :] + noise)
    return np.maximum(0.0, np.log1p(gamma_b) / math.log(2.0) - np.log1p(gamma_w) / math.log(2.0))


def grid_statistics(
    gains: np.ndarray, alice_power: float, powers: np.ndarray, config: dict, candidate_index: int | None
) -> dict[str, np.ndarray]:
    count = len(gains)
    sums = np.zeros(len(powers))
    sums_sq = np.zeros(len(powers))
    outages = np.zeros(len(powers))
    zeros = np.zeros(len(powers))
    delta_sum = np.zeros(len(powers))
    delta_sq = np.zeros(len(powers))
    threshold = float(config["metric"]["secrecy_target_bpshz"])
    for start in range(0, count, BATCH):
        rates = rates_for_grid(gains[start : start + BATCH], alice_power, powers, config)
        sums += rates.sum(axis=0)
        sums_sq += np.square(rates).sum(axis=0)
        outages += (rates < threshold).sum(axis=0)
        zeros += (rates <= 1e-12).sum(axis=0)
        if candidate_index is not None:
            differences = rates - rates[:, [candidate_index]]
            delta_sum += differences.sum(axis=0)
            delta_sq += np.square(differences).sum(axis=0)
    means = sums / count
    variance = np.maximum(0.0, (sums_sq - count * means**2) / (count - 1))
    delta_mean = delta_sum / count
    delta_variance = np.maximum(0.0, (delta_sq - count * delta_mean**2) / (count - 1))
    return {
        "mean": means,
        "se": np.sqrt(variance / count),
        "sop": outages / count,
        "zero_fraction": zeros / count,
        "paired_delta": delta_mean,
        "paired_delta_se": np.sqrt(delta_variance / count),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "motion_residual_v1.yaml"))
    parser.add_argument("--output", default=str(ROOT / "results" / "attack_pressure_scan_v1"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    save_resolved_config(config, output / "resolved_config.yaml")
    powers = power_grid()
    if not np.allclose(powers[[np.argmin(abs(powers - v)) for v in (0.0, 0.3, 1.0)]], (0.0, 0.3, 1.0)):
        raise AssertionError("Reference power points missing")
    reference = SecureLinkEnv(config, method="B00", attack_rule="none").reference_nodes
    positions = dict(zip(POSITIONS, (reference[0], reference[25], reference[50], reference[75], reference[99])))
    groups = []
    for position_index, name in enumerate(POSITIONS):
        for alice_index, alice_power in enumerate(ALICE_POWERS):
            groups.append((name, alice_power, -110.0, position_index * len(ALICE_POWERS) + alice_index))
    source_mid_02 = 2 * len(ALICE_POWERS) + 1
    groups.extend(("path_mid", 0.2, si, source_mid_02) for si in (-120.0, -100.0))

    aggregate_rows: list[dict] = []
    group_rows: list[dict] = []
    samples_archive: dict[str, np.ndarray] = {}
    sample_cache: dict[tuple[int, int], np.ndarray] = {}
    paired_path = output / "validation_paired_samples.csv.gz"
    with gzip.open(paired_path, "wt", newline="", encoding="utf-8") as raw_handle:
        raw_writer = csv.writer(raw_handle)
        raw_writer.writerow(("group_id", "sample_id", "selected_w", "rate_selected", "rate_silent", "rate_uniform_0p3", "rate_peak"))
        for group_id, (name, alice_power, si_db, source_id) in enumerate(groups):
            variant = copy.deepcopy(config)
            variant["power"]["self_interference_db"] = si_db
            for phase_id in (0, 1):
                cache_key = (source_id, phase_id)
                if cache_key not in sample_cache:
                    sample_cache[cache_key] = source_gains(positions[name], variant, source_id, phase_id)
                    samples_archive[f"source{source_id}_{'select' if phase_id == 0 else 'confirm'}"] = sample_cache[cache_key]
            selection = grid_statistics(sample_cache[(source_id, 0)], alice_power, powers, variant, None)
            selected_index = int(np.argmin(selection["mean"]))
            confirmation = grid_statistics(sample_cache[(source_id, 1)], alice_power, powers, variant, selected_index)
            selected_power = float(powers[selected_index])
            tie = (
                (confirmation["paired_delta"] <= PRACTICAL_TIE_BPSHZ)
                | (confirmation["paired_delta"] - 1.96 * confirmation["paired_delta_se"] <= 0.0)
            )
            plateau = confirmation["zero_fraction"] >= 0.99
            reference_indices = {label: int(np.argmin(abs(powers - value))) for label, value in (("silent", 0.0), ("uniform_0p3", 0.3), ("peak", 1.0))}
            row = {
                "group_id": group_id,
                "position": name,
                "alice_power_w": alice_power,
                "self_interference_db": si_db,
                "shared_fading_source_id": source_id,
                "selected_willie_power_w": selected_power,
                "selection_mean_asr": float(selection["mean"][selected_index]),
                "confirmation_mean_asr": float(confirmation["mean"][selected_index]),
                "confirmation_best_grid_power_diagnostic_w": float(powers[int(np.argmin(confirmation["mean"]))]),
                "confirmation_tie_min_w": float(powers[tie].min()),
                "confirmation_tie_max_w": float(powers[tie].max()),
                "confirmation_tie_point_count": int(tie.sum()),
                "zero_rate_platform_min_w": float(powers[plateau].min()) if plateau.any() else math.nan,
                "zero_rate_platform_max_w": float(powers[plateau].max()) if plateau.any() else math.nan,
                "zero_rate_platform_point_count": int(plateau.sum()),
                "selected_zero_rate_fraction": float(confirmation["zero_fraction"][selected_index]),
            }
            for label, index in reference_indices.items():
                delta = float(confirmation["paired_delta"][index])
                se = float(confirmation["paired_delta_se"][index])
                row[f"{label}_minus_selected_asr"] = delta
                row[f"{label}_minus_selected_ci_low"] = delta - 1.96 * se
                row[f"{label}_minus_selected_ci_high"] = delta + 1.96 * se
                row[f"{label}_mean_asr"] = float(confirmation["mean"][index])
            group_rows.append(row)
            for phase, stats in (("select", selection), ("confirm", confirmation)):
                for index, power in enumerate(powers):
                    aggregate_rows.append(
                        {
                            "group_id": group_id,
                            "position": name,
                            "alice_power_w": alice_power,
                            "self_interference_db": si_db,
                            "phase": phase,
                            "willie_power_w": float(power),
                            "samples": SAMPLES,
                            "mean_asr_bpshz": float(stats["mean"][index]),
                            "asr_standard_error": float(stats["se"][index]),
                            "slot_sop": float(stats["sop"][index]),
                            "zero_rate_fraction": float(stats["zero_fraction"][index]),
                            "paired_delta_from_selected": float(stats["paired_delta"][index]) if phase == "confirm" else math.nan,
                            "paired_delta_standard_error": float(stats["paired_delta_se"][index]) if phase == "confirm" else math.nan,
                        }
                    )
            confirm_gains = sample_cache[(source_id, 1)]
            selected_rates = rates_for_grid(confirm_gains, alice_power, np.array([selected_power]), variant)[:, 0]
            silent_rates = rates_for_grid(confirm_gains, alice_power, np.array([0.0]), variant)[:, 0]
            uniform_rates = rates_for_grid(confirm_gains, alice_power, np.array([0.3]), variant)[:, 0]
            peak_rates = rates_for_grid(confirm_gains, alice_power, np.array([1.0]), variant)[:, 0]
            raw_writer.writerows(
                (group_id, i, selected_power, selected_rates[i], silent_rates[i], uniform_rates[i], peak_rates[i])
                for i in range(SAMPLES)
            )
            print(f"group {group_id + 1}/{len(groups)}: {name}, Alice={alice_power}, SI={si_db}, selected={selected_power:.6g} W", flush=True)
    np.savez_compressed(output / "raw_paired_channel_gains.npz", **samples_archive)
    aggregate = pd.DataFrame(aggregate_rows)
    group_frame = pd.DataFrame(group_rows)
    aggregate.to_csv(output / "scan_points.csv", index=False)
    group_frame.to_csv(output / "scan_groups.csv", index=False)
    nominal = group_frame[(group_frame.self_interference_db == -110.0) & (group_frame.alice_power_w == 0.2)]
    low_constant = float(max(0.001, np.round(nominal.selected_willie_power_w.median() / 0.001) * 0.001))
    metadata = {
        "config_digest": config_digest(config),
        "root_seed": ROOT_SEED,
        "selection_and_confirmation_samples_per_group": SAMPLES,
        "batch_size": BATCH,
        "groups": len(groups),
        "unique_fading_sources_per_phase": len(sample_cache) // 2,
        "grid_points": len(powers),
        "grid_w": powers.tolist(),
        "low_power_rule_w": low_constant,
        "low_power_rule_selection": "Median of the five nominal-SI, Alice-0.2-W selected powers, rounded to the nearest 0.001 W with 0.001 W floor; frozen before episode evaluation.",
        "practical_tie_tolerance_bpshz": PRACTICAL_TIE_BPSHZ,
        "tie_rule": "Independent-confirmation paired delta from selected <= 0.005 bit/s/Hz OR its 95% normal-approximation lower bound <= 0.",
        "zero_rate_platform_rule": "At least 99% of confirmation fading samples have zero clipped secrecy rate at the grid point.",
        "final_test_used": False,
        "training_used": False,
        "reference_points_w": [0.0, 0.3, 1.0],
        "noise_w": receiver_noise_w(config),
        "noise_over_self_interference_w": {str(si): receiver_noise_w(config) / db_to_linear(si) for si in SI_DB},
        "raw_channel_gain_archive_sha256": hashlib.sha256((output / "raw_paired_channel_gains.npz").read_bytes()).hexdigest(),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: metadata[key] for key in ("groups", "grid_points", "low_power_rule_w", "noise_over_self_interference_w")}, indent=2))


if __name__ == "__main__":
    main()
