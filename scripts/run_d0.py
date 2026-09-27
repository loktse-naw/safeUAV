from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import config_digest, load_config, save_resolved_config
from securelink.environment import ScenarioConfigurationError, SecureLinkEnv
from securelink.evaluation import dataframe_digest, evaluate_policy, run_episode, write_json
from securelink.geometry import segment_clear_of_circle
from securelink.physics import channel_gains, db_to_linear, receiver_noise_w, secrecy_metrics
from securelink.system_info import collect_system_info


def independent_formula(p_a: float, p_w: float, gains: dict[str, float], config: dict, no_eve: bool = False) -> dict:
    noise_dbm = (
        config["channel"]["noise_psd_dbm_hz"]
        + 10.0 * math.log10(config["channel"]["bandwidth_hz"])
        + config["channel"]["noise_figure_db"]
    )
    noise = 10.0 ** ((noise_dbm - 30.0) / 10.0)
    gamma_b = p_a * gains["g_ab"] / (p_w * gains["g_wb"] + noise)
    gamma_w = 0.0 if no_eve else p_a * gains["g_aw"] / (
        10.0 ** (config["power"]["self_interference_db"] / 10.0) * p_w + noise
    )
    c_b = math.log2(1.0 + gamma_b)
    c_w = math.log2(1.0 + gamma_w)
    return {"gamma_b": gamma_b, "gamma_w": gamma_w, "c_b": c_b, "c_w": c_w, "rs": max(0.0, c_b - c_w)}


def numerical_checks(config: dict) -> list[dict]:
    cases: list[dict] = []
    positions = {
        "nominal": np.array([650.0, 350.0]),
        "near_willie": np.array(config["scenario"]["willie_m"], dtype=float),
    }
    gains_nominal = channel_gains(positions["nominal"], config, fading=False)
    definitions = [
        ("interference_off", 0.2, 0.0, False, gains_nominal),
        ("eavesdropper_off", 0.2, 0.3, True, gains_nominal),
        ("zero_transmit_power", 0.0, 0.3, False, gains_nominal),
        (
            "negative_capacity_difference_clipped",
            0.2,
            0.0,
            False,
            channel_gains(positions["near_willie"], config, fading=False),
        ),
    ]
    for name, p_a, p_w, no_eve, gains in definitions:
        implementation = secrecy_metrics(p_a, p_w, gains, config, disable_eavesdropper=no_eve)
        independent = independent_formula(p_a, p_w, gains, config, no_eve)
        error = max(
            abs(implementation["gamma_b"] - independent["gamma_b"]),
            abs(implementation["gamma_w"] - independent["gamma_w"]),
            abs(implementation["secrecy_rate_bpshz"] - independent["rs"]),
        )
        passed = error <= 1e-10 * max(1.0, implementation["gamma_b"], implementation["gamma_w"])
        if name == "zero_transmit_power":
            passed = passed and implementation["secrecy_rate_bpshz"] == 0.0
        if name == "eavesdropper_off":
            passed = passed and implementation["capacity_w_bpshz"] == 0.0
        if name == "negative_capacity_difference_clipped":
            passed = passed and implementation["capacity_difference_bpshz"] < 0.0 and implementation["secrecy_rate_bpshz"] == 0.0
        cases.append(
            {
                "case": name,
                "passed": bool(passed),
                "absolute_error": error,
                "implementation": implementation,
                "independent": independent,
                "noise_w": receiver_noise_w(config),
            }
        )
    return cases


def power_scan(config: dict, output: Path) -> tuple[pd.DataFrame, dict]:
    safe_path = SecureLinkEnv(config, method="B00", attack_rule="none").reference_nodes
    positions = {
        "start": np.asarray(config["scenario"]["start_m"], dtype=float),
        "near_bob": np.asarray(config["scenario"]["bob_m"], dtype=float),
        "near_willie": np.asarray(config["scenario"]["willie_m"], dtype=float),
        "goal": np.asarray(config["scenario"]["goal_m"], dtype=float),
        "path_q1": safe_path[25],
        "path_mid": safe_path[50],
        "path_q3": safe_path[75],
    }
    rows: list[dict] = []
    for beta_db in (-130.0, -120.0, -110.0, -100.0, -90.0):
        scan_config = copy.deepcopy(config)
        scan_config["power"]["self_interference_db"] = beta_db
        for position_name, position in positions.items():
            gains = channel_gains(position, scan_config, fading=False)
            for p_a in (0.0, 0.05, 0.2, 1.0):
                for p_w in np.linspace(0.0, 1.0, 51):
                    metrics = secrecy_metrics(p_a, float(p_w), gains, scan_config)
                    rows.append(
                        {
                            "position": position_name,
                            "x_m": position[0],
                            "y_m": position[1],
                            "alice_power_w": p_a,
                            "willie_power_w": float(p_w),
                            "self_interference_db": beta_db,
                            **metrics,
                        }
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "power_scan.csv", index=False)
    nonzero = frame[frame["alice_power_w"] > 0]
    groups = []
    for keys, group in nonzero.groupby(["position", "alice_power_w", "self_interference_db"]):
        min_row = group.loc[group["secrecy_rate_bpshz"].idxmin()]
        max_row = group.loc[group["willie_power_w"].idxmax()]
        groups.append(
            {
                "position": keys[0],
                "alice_power_w": keys[1],
                "self_interference_db": keys[2],
                "minimizing_willie_power_w": min_row["willie_power_w"],
                "minimum_secrecy_rate_bpshz": min_row["secrecy_rate_bpshz"],
                "max_power_secrecy_rate_bpshz": max_row["secrecy_rate_bpshz"],
                "max_power_is_grid_minimum": bool(
                    abs(max_row["secrecy_rate_bpshz"] - min_row["secrecy_rate_bpshz"]) <= 1e-12
                ),
            }
        )
    minima = pd.DataFrame(groups)
    minima.to_csv(output / "power_scan_minima.csv", index=False)
    nominal = frame[
        (frame["position"] == "near_bob")
        & (frame["alice_power_w"] == 0.2)
        & (frame["self_interference_db"] == -110.0)
    ]
    fig, axis = plt.subplots(figsize=(6.4, 4.0))
    axis.plot(nominal["willie_power_w"], nominal["secrecy_rate_bpshz"], marker="o", markersize=2)
    axis.set(xlabel="Willie power (W)", ylabel="Secrecy rate (bit/s/Hz)", title="D0 fixed-geometry power scan")
    axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "power_scan_nominal.png", dpi=180)
    plt.close(fig)
    summary = {
        "groups": len(minima),
        "max_power_grid_minimum_fraction": float(minima["max_power_is_grid_minimum"].mean()),
        "interior_or_silent_optimum_fraction": float((~minima["max_power_is_grid_minimum"]).mean()),
    }
    return frame, summary


def constraint_checks(config: dict) -> list[dict]:
    checks: list[dict] = []
    center = np.asarray(config["scenario"]["nfz_center_m"], dtype=float)
    radius = config["scenario"]["nfz_radius_m"] + config["scenario"]["nfz_margin_m"]
    crossing_clear = segment_clear_of_circle(np.array([300.0, 500.0]), np.array([700.0, 500.0]), center, radius)
    checks.append({"case": "cross_nfz_segment_rejected", "passed": not crossing_clear})

    env = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    env.reset(seed=71)
    _, _, _, _, info = env.step(np.array([9.0, 9.0, 9.0], dtype=np.float32))
    first = env.step_records[0]
    checks.append(
        {
            "case": "power_speed_region_action_corrected",
            "passed": bool(
                info["action_corrected"]
                and first["speed_mps"] <= config["mobility"]["max_speed_mps"] + 1e-9
                and first["alice_power_w"] <= config["power"]["alice_peak_w"] + 1e-9
                and first["executed_hard_violation"] == 0
            ),
            "executed_speed_mps": first["speed_mps"],
            "executed_power_w": first["alice_power_w"],
        }
    )

    unreachable = copy.deepcopy(config)
    unreachable["time"]["slots"] = 40
    try:
        SecureLinkEnv(unreachable, method="B00")
        unreachable_passed = False
    except ScenarioConfigurationError:
        unreachable_passed = True
    checks.append({"case": "unreachable_configuration_rejected", "passed": unreachable_passed})

    reference = SecureLinkEnv(config, method="B00", attack_rule="uniform")
    reference_summary, _ = run_episode(reference, 73)
    checks.append(
        {
            "case": "reference_path_completes_without_hard_violation",
            "passed": bool(
                reference_summary["task_completed"] == 1
                and reference_summary["executed_hard_violations"] == 0
                and reference_summary["slots_recorded"] == config["time"]["slots"]
            ),
            "final_error_m": reference_summary["final_error_m"],
            "energy_j": reference_summary["alice_total_energy_j"],
        }
    )

    fallback_env = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    fallback_summary, fallback_steps = run_episode(fallback_env, 74)
    takeover_rates = [row["secrecy_rate_bpshz"] for row in fallback_steps if row["fallback_takeover"]]
    policy_rates = [row["secrecy_rate_bpshz"] for row in fallback_steps if not row["fallback_takeover"]]
    checks.append(
        {
            "case": "zero_motion_proposal_uses_safe_fallback_and_completes",
            "passed": bool(
                fallback_summary["task_completed"] == 1
                and fallback_summary["executed_hard_violations"] == 0
                and fallback_summary["reference_fallbacks"] > 0
            ),
            "final_error_m": fallback_summary["final_error_m"],
            "fallbacks": fallback_summary["reference_fallbacks"],
            "first_trigger_slot": fallback_summary["fallback_first_trigger_slot"],
            "independent_triggers": fallback_summary["fallback_independent_triggers"],
            "takeover_slots": fallback_summary["fallback_takeover_slots"],
            "exit_count": fallback_summary["fallback_exit_count"],
            "policy_control_rate_mean": float(np.mean(policy_rates)) if policy_rates else None,
            "takeover_rate_mean": float(np.mean(takeover_rates)) if takeover_rates else None,
        }
    )

    critical = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    critical.reset(seed=75)
    critical_nodes, critical_energy, critical_reason = critical._future_plan(
        critical.position, critical.n_slots
    )
    critical.force_battery_for_test(critical_energy + 1e-7)
    critical_summary, critical_steps = run_episode_with_existing_reset(critical)
    checks.append(
        {
            "case": "critical_energy_discrete_plan_replays",
            "passed": bool(
                critical_nodes is not None
                and critical_reason == "ok"
                and critical_summary["task_completed"] == 1
                and critical_summary["executed_hard_violations"] == 0
                and critical_summary["alice_total_energy_j"] <= critical_energy + 1e-6
            ),
            "critical_energy_j": critical_energy,
            "actual_energy_j": critical_summary["alice_total_energy_j"],
            "segments": len(critical_steps),
        }
    )

    edge = SecureLinkEnv(config, method="B11", attack_rule="none")
    edge.reset(seed=76)
    goal_nodes, goal_energy, goal_reason = edge._future_plan(edge.goal, 0)
    bad_nodes, _, bad_reason = edge._future_plan(edge.start, 0)
    near_nodes, near_energy, near_reason = edge._future_plan(edge.goal - np.array([0.5, 0.0]), 1)
    boundary_start = np.array([0.0, 10.0])
    boundary_nodes, boundary_energy, boundary_reason = edge._future_plan(boundary_start, edge.n_slots)
    boundary_valid = boundary_nodes is not None
    if boundary_nodes is not None:
        current = boundary_start.copy()
        for node in boundary_nodes:
            boundary_valid = boundary_valid and bool(
                np.linalg.norm(node - current) <= edge.max_step_m + 1e-8
                and segment_clear_of_circle(
                    current, node, edge.nfz_center, edge.planning_radius, 1e-9
                )
            )
            current = node
        boundary_valid = boundary_valid and np.linalg.norm(current - edge.goal) <= edge.terminal_tolerance
    checks.append(
        {
            "case": "k0_near_goal_boundary_and_turning_certificates",
            "passed": bool(
                goal_nodes == []
                and goal_energy == 0.0
                and goal_reason == "ok"
                and bad_nodes is None
                and bad_reason == "time_unreachable"
                and near_nodes is not None
                and near_reason == "ok"
                and near_energy > 0.0
                and boundary_valid
                and boundary_reason == "ok"
                and boundary_energy > 0.0
            ),
            "near_goal_energy_j": near_energy,
            "boundary_energy_j": boundary_energy,
        }
    )

    corrected = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    corrected.reset(seed=77)
    done = False
    while not done:
        _, _, done, _, corrected_info = corrected.step(
            np.array([9.0, -9.0, 9.0], dtype=np.float32)
        )
    corrected_summary = corrected_info["episode_summary"]
    checks.append(
        {
            "case": "multiple_action_corrections_preserve_hard_feasibility",
            "passed": bool(
                corrected_summary["action_corrections"] > 1
                and corrected_summary["task_completed"] == 1
                and corrected_summary["executed_hard_violations"] == 0
            ),
            "corrections": corrected_summary["action_corrections"],
            "takeover_slots": corrected_summary["fallback_takeover_slots"],
            "correction_reasons": corrected_summary["correction_reason_counts"],
        }
    )

    exhausted = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    exhausted.reset(seed=72)
    exhausted.force_battery_for_test(1.0)
    _, _, terminated, _, info = exhausted.step(np.zeros(3, dtype=np.float32))
    summary = info["episode_summary"]
    checks.append(
        {
            "case": "energy_exhaustion_terminates_and_pads",
            "passed": bool(
                terminated
                and summary["task_failure"] == 1
                and summary["slots_recorded"] == config["time"]["slots"]
                and summary["padded_failure_slots"] == config["time"]["slots"]
                and summary["slot_sop"] == 1.0
            ),
            "summary": summary,
        }
    )
    return checks


def run_episode_with_existing_reset(env: SecureLinkEnv) -> tuple[dict, list[dict]]:
    terminated = truncated = False
    info: dict = {}
    while not (terminated or truncated):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        _, _, terminated, truncated, info = env.step(action)
    return dict(info["episode_summary"]), [dict(row) for row in env.step_records]


def reproducibility_and_accounting(config: dict, output: Path) -> dict:
    episodes_a, steps_a = evaluate_policy(config, "B00", "uniform", [31415])
    episodes_b, steps_b = evaluate_policy(config, "B00", "uniform", [31415])
    episodes_c, steps_c = evaluate_policy(config, "B00", "uniform", [31416])
    digest_a = dataframe_digest(steps_a)
    digest_b = dataframe_digest(steps_b)
    digest_c = dataframe_digest(steps_c)
    row = episodes_a.iloc[0]
    secure_bits_from_steps = (
        config["channel"]["bandwidth_hz"]
        * config["time"]["slot_s"]
        * config["time"]["tx_duty"]
        * steps_a["secrecy_rate_bpshz"].sum()
    )
    energy_from_steps = steps_a["slot_energy_j"].sum()
    checks = {
        "same_seed_digest_equal": digest_a == digest_b,
        "different_seed_digest_differs": digest_a != digest_c,
        "slots_exact": int(row["slots_recorded"]) == int(config["time"]["slots"]),
        "energy_sum_consistent": abs(float(row["alice_total_energy_j"]) - energy_from_steps) <= 1e-9,
        "secure_bits_units_consistent": abs(float(row["secure_bits"]) - secure_bits_from_steps) <= 1e-6,
        "same_seed_digest": digest_a,
        "different_seed_digest": digest_c,
    }
    episodes_a.to_csv(output / "replay_episode.csv", index=False)
    steps_a.to_csv(output / "replay_steps.csv", index=False)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--output", default=str(ROOT / "results" / "d0"))
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    save_resolved_config(config, output / "resolved_config.yaml")

    formula = numerical_checks(config)
    _, scan_summary = power_scan(config, output)
    constraints = constraint_checks(config)
    replay = reproducibility_and_accounting(config, output)
    report = {
        "implementation_version": config.get("implementation", {}).get("version", "unknown"),
        "config_digest": config_digest(config),
        "formula_checks": formula,
        "constraint_checks": constraints,
        "reproducibility_and_accounting": replay,
        "power_scan_summary": scan_summary,
        "system": collect_system_info(),
    }
    report["all_required_checks_passed"] = bool(
        all(item["passed"] for item in formula)
        and all(item["passed"] for item in constraints)
        and all(value for key, value in replay.items() if key.endswith(("equal", "differs", "exact", "consistent")))
    )
    write_json(report, output / "d0_validation.json")
    print(
        json.dumps(
            {
                "all_required_checks_passed": report["all_required_checks_passed"],
                "config_digest": report["config_digest"],
                "power_scan_summary": report["power_scan_summary"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
            default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
        )
    )
    if not report["all_required_checks_passed"]:
        raise SystemExit("D0 validation failed")


if __name__ == "__main__":
    main()
