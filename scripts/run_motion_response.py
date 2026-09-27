"""Prespecified no-training response audit for the residual motion candidate."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import load_config, save_resolved_config
from securelink.environment import SecureLinkEnv


SEEDS = list(range(20001, 20006))
SLOTS = (0, 20, 40, 60, 80, 95)
PROBES = {
    "zero": np.array([0.0, 0.0], dtype=np.float32),
    "plus_x": np.array([1.0, 0.0], dtype=np.float32),
    "minus_x": np.array([-1.0, 0.0], dtype=np.float32),
    "plus_y": np.array([0.0, 1.0], dtype=np.float32),
    "minus_y": np.array([0.0, -1.0], dtype=np.float32),
    "plus_diag": np.array([0.70710677, 0.70710677], dtype=np.float32),
    "minus_diag": np.array([-0.70710677, -0.70710677], dtype=np.float32),
}


def source_action(source: str, slot: int, rng: np.random.Generator) -> np.ndarray:
    if source == "zero":
        return PROBES["zero"]
    if source == "offset":
        return np.array([0.5, -0.5], dtype=np.float32) if slot < 35 else PROBES["zero"]
    return rng.uniform(-1.0, 1.0, size=2).astype(np.float32)


def snapshots(config: dict) -> list[tuple[str, int, int, SecureLinkEnv]]:
    collected: list[tuple[str, int, int, SecureLinkEnv]] = []
    for seed in SEEDS:
        for source in ("zero", "offset", "random"):
            env = SecureLinkEnv(config, method="B10", attack_rule="uniform")
            env.reset(seed=seed)
            rng = np.random.default_rng(seed * 10 + {"zero": 0, "offset": 1, "random": 2}[source])
            history: list[SecureLinkEnv] = []
            for slot in range(env.n_slots):
                history.append(copy.deepcopy(env))
                _, _, done, _, _ = env.step(source_action(source, slot, rng))
                if done:
                    break
            near = min(
                range(len(history)),
                key=lambda n: abs(float(np.linalg.norm(history[n].position - env.nfz_center)) - env.planning_radius),
            )
            for slot in sorted(set(SLOTS + (near,))):
                if slot < len(history):
                    collected.append((source, seed, slot, history[slot]))
    return collected


def rollout_zero(env: SecureLinkEnv) -> tuple[int, int]:
    done = False
    info = {}
    while not done:
        _, _, done, _, info = env.step(PROBES["zero"])
    summary = info["episode_summary"]
    return int(summary["task_completed"]), int(summary["executed_hard_violations"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "motion_residual_v1.yaml"))
    parser.add_argument("--output", default=str(ROOT / "results" / "motion_response_v1"))
    args = parser.parse_args()
    config = load_config(args.config)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output / "resolved_config.yaml")
    states: list[dict] = []
    probes: list[dict] = []
    completions: list[dict] = []
    for source, seed, slot, original in snapshots(config):
        nodes, minimum_energy, reason = original._future_plan(original.position, original.n_slots - original.slot)
        if nodes is None:
            raise AssertionError(f"Collected state lacks a future route: {source} {seed} {slot}: {reason}")
        margins = {"native": original.battery_remaining_j - minimum_energy}
        for label, allowance in (("tight_50j", 50.0), ("medium_500j", 500.0)):
            if original.battery_remaining_j >= minimum_energy + allowance:
                margins[label] = allowance
        for margin_label, margin in margins.items():
            state_id = f"{seed}_{source}_{slot}_{margin_label}"
            states.append(
                {
                    "state_id": state_id,
                    "seed": seed,
                    "source": source,
                    "slot": slot,
                    "margin_label": margin_label,
                    "position_x_m": float(original.position[0]),
                    "position_y_m": float(original.position[1]),
                    "distance_to_nfz_boundary_m": float(np.linalg.norm(original.position - original.nfz_center) - original.planning_radius),
                    "minimum_future_energy_j": minimum_energy,
                    "native_energy_margin_j": original.battery_remaining_j - minimum_energy,
                    "applied_energy_margin_j": margin,
                }
            )
            state_results: list[tuple[str, dict, SecureLinkEnv]] = []
            source_id = {"zero": 0, "offset": 1, "random": 2}[source]
            margin_id = {"native": 0, "tight_50j": 1, "medium_500j": 2}[margin_label]
            random_xy = np.random.default_rng(seed * 10000 + slot * 100 + source_id * 10 + margin_id).uniform(
                -1.0, 1.0, size=2
            ).astype(np.float32)
            probe_actions = {**PROBES, "random_residual": random_xy}
            for name, xy in probe_actions.items():
                trial = copy.deepcopy(original)
                trial.battery_remaining_j = minimum_energy + margin
                _, _, done, _, info = trial.step(xy)
                row = trial.step_records[slot] if len(trial.step_records) > slot and not trial.step_records[slot]["padded_failure"] else None
                if row is None:
                    result = {"state_id": state_id, "probe": name, "decision": "no_feasible_motion", "done": int(done)}
                else:
                    future_length = trial._safe_path(trial.position)[1]
                    result = {
                        "state_id": state_id,
                        "probe": name,
                        "decision": row["motion_decision"],
                        "rejection_reason": row["motion_rejection_reason"],
                        "done": int(done),
                        "reference_dx_m": row["reference_dx_m"],
                        "reference_dy_m": row["reference_dy_m"],
                        "proposed_residual_dx_m": row["proposed_residual_dx_m"],
                        "proposed_residual_dy_m": row["proposed_residual_dy_m"],
                        "executed_residual_dx_m": row["executed_residual_dx_m"],
                        "executed_residual_dy_m": row["executed_residual_dy_m"],
                        "proposed_dx_m": row["proposed_dx_m"],
                        "proposed_dy_m": row["proposed_dy_m"],
                        "executed_dx_m": row["executed_dx_m"],
                        "executed_dy_m": row["executed_dy_m"],
                        "correction_dx_m": row["motion_correction_dx_m"],
                        "correction_dy_m": row["motion_correction_dy_m"],
                        "remaining_distance_m": row["goal_distance_after_m"],
                        "remaining_time_slack_m": (trial.n_slots - trial.slot) * trial.max_step_m - future_length,
                        "remaining_energy_slack_j": trial.battery_remaining_j - trial._guarantee_energy_from(trial.position),
                        "executed_hard_violation": row["executed_hard_violation"],
                    }
                probes.append(result)
                state_results.append((name, result, trial))
            if margin_label == "native" and slot <= 40:
                for name, result, trial in state_results:
                    if name == "zero" or result["decision"] not in {"policy_accepted", "scaled_policy_accepted"}:
                        continue
                    completed, violations = rollout_zero(trial)
                    completions.append(
                        {"state_id": state_id, "probe": name, "completed": completed, "executed_hard_violations": violations}
                    )
    state_frame = pd.DataFrame(states)
    probe_frame = pd.DataFrame(probes)
    completion_frame = pd.DataFrame(completions)
    state_frame.to_csv(output / "states.csv", index=False)
    probe_frame.to_csv(output / "probes.csv", index=False)
    completion_frame.to_csv(output / "off_path_completions.csv", index=False)
    valid = probe_frame[probe_frame["decision"] != "no_feasible_motion"].copy()
    choices = valid.groupby("state_id")[["executed_dx_m", "executed_dy_m"]].apply(
        lambda group: len({(round(x, 2), round(y, 2)) for x, y in zip(group["executed_dx_m"], group["executed_dy_m"])})
    )
    summary = {
        "seeds": SEEDS,
        "slots": list(SLOTS),
        "sources": ["zero", "offset", "random"],
        "probe_names": list(PROBES) + ["random_residual"],
        "state_count": len(state_frame),
        "unique_physical_states_rounded_1e_5": len(
            state_frame[["slot", "position_x_m", "position_y_m", "applied_energy_margin_j"]]
            .round(5)
            .drop_duplicates()
        ),
        "probe_count": len(probe_frame),
        "decision_counts": probe_frame["decision"].value_counts().to_dict(),
        "rejection_counts": probe_frame.get("rejection_reason", pd.Series(dtype=str)).value_counts().to_dict(),
        "states_with_two_or_more_distinct_executed_moves": int((choices >= 2).sum()),
        "states_with_single_executed_move": int((choices <= 1).sum()),
        "off_path_completion_count": len(completion_frame),
        "off_path_completed": int(completion_frame["completed"].sum()) if len(completion_frame) else 0,
        "off_path_hard_violations": int(completion_frame["executed_hard_violations"].sum()) if len(completion_frame) else 0,
        "probe_hard_violations": int(valid["executed_hard_violation"].sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
