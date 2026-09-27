"""Summarize the controlled direct-vs-residual comparison and same-state probes."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from securelink.config import load_config
from securelink.environment import SecureLinkEnv


def tagged(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    frame = frame.copy()
    frame["comparison_method"] = label
    return frame


def counterfactuals(config: dict, result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[dict] = []
    replays: list[dict] = []
    for method in ("B10", "B11"):
        for train_seed in (11001, 11002, 11003):
            model = PPO.load(result_dir / "models" / f"{method}_seed{train_seed}.zip", device="cpu")
            env = SecureLinkEnv(config, method=method, attack_rule="uniform")
            for scenario_seed in range(20001, 20031):
                observation, _ = env.reset(seed=scenario_seed)
                for slot in range(env.n_slots):
                    action = np.asarray(model.predict(observation, deterministic=True)[0], dtype=np.float32)
                    if slot in (10, 30, 50, 70, 90):
                        zero_action = action.copy()
                        zero_action[:2] = 0.0
                        zero_env = copy.deepcopy(env)
                        zero_env.step(zero_action)
                        zero = zero_env.step_records[-1]
                    observation, _, done, _, info = env.step(action)
                    if slot in (10, 30, 50, 70, 90):
                        actual = env.step_records[-1]
                        records.append(
                            {
                                "method": method,
                                "train_seed": train_seed,
                                "scenario_seed": scenario_seed,
                                "slot": slot,
                                "actual_decision": actual["motion_decision"],
                                "zero_decision": zero["motion_decision"],
                                "policy_residual_norm_m": float(np.hypot(actual["proposed_residual_dx_m"], actual["proposed_residual_dy_m"])),
                                "executed_difference_m": float(np.hypot(actual["executed_dx_m"] - zero["executed_dx_m"], actual["executed_dy_m"] - zero["executed_dy_m"])),
                                "actual_executed_residual_norm_m": float(np.hypot(actual["executed_residual_dx_m"], actual["executed_residual_dy_m"])),
                                "actual_hard_violation": actual["executed_hard_violation"],
                                "zero_hard_violation": zero["executed_hard_violation"],
                            }
                        )
                    if done:
                        summary = info["episode_summary"]
                        replays.append(
                            {
                                "method": method,
                                "train_seed": train_seed,
                                "scenario_seed": scenario_seed,
                                "asr_bpshz": summary["asr_bpshz"],
                                "fallback_takeover_slots": summary["fallback_takeover_slots"],
                                "executed_hard_violations": summary["executed_hard_violations"],
                            }
                        )
                        break
    return pd.DataFrame(records), pd.DataFrame(replays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", default=str(ROOT / "results" / "motion_residual_v1_32768"))
    parser.add_argument("--old", default=str(ROOT / "results" / "d1_v2_motion_joint"))
    parser.add_argument("--baselines", default=str(ROOT / "results" / "motion_baselines_v1"))
    parser.add_argument("--output", default=str(ROOT / "results" / "motion_control_summary_v1"))
    args = parser.parse_args()
    new_dir, old_dir, base_dir, output = map(lambda p: Path(p).resolve(), (args.new, args.old, args.baselines, args.output))
    output.mkdir(parents=True, exist_ok=True)
    new_e = pd.read_csv(new_dir / "raw_episodes.csv")
    new_s = pd.read_csv(new_dir / "raw_steps.csv")
    old_e = pd.read_csv(old_dir / "raw_episodes.csv")
    old_s = pd.read_csv(old_dir / "raw_steps.csv")
    base_e = pd.read_csv(base_dir / "raw_episodes.csv")
    base_s = pd.read_csv(base_dir / "raw_steps.csv")
    episodes = pd.concat(
        [
            tagged(old_e[old_e.method == "B10"], "direct_B10"),
            tagged(old_e[old_e.method == "B11"], "direct_B11"),
            tagged(new_e[new_e.method == "B10"], "residual_B10"),
            tagged(new_e[new_e.method == "B11"], "residual_B11"),
            base_e,
        ],
        ignore_index=True,
    )
    metrics = ["asr_bpshz", "slot_sop", "episode_sop", "task_completed", "propulsion_energy_j", "communication_energy_j", "alice_total_energy_j", "alice_radiated_energy_j", "fallback_independent_triggers", "fallback_takeover_slots", "fallback_exit_count", "executed_hard_violations"]
    overall = episodes.groupby("comparison_method")[metrics].mean().reset_index()
    overall["episodes"] = episodes.groupby("comparison_method").size().values
    by_seed = episodes[episodes.comparison_method.str.contains("B10|B11")].groupby(["comparison_method", "train_seed"])[metrics].mean().reset_index()
    overall.to_csv(output / "method_summary.csv", index=False)
    by_seed.to_csv(output / "seed_summary.csv", index=False)

    zero_path = base_s[base_s.comparison_method == "R0_dynamic_zero"][["scenario_seed", "slot", "x_m", "y_m"]].rename(columns={"x_m": "zero_x_m", "y_m": "zero_y_m"})
    learned = new_s.merge(zero_path, on=["scenario_seed", "slot"], how="left")
    learned["path_offset_m"] = np.hypot(learned.x_m - learned.zero_x_m, learned.y_m - learned.zero_y_m)
    learned["proposed_residual_norm_m"] = np.hypot(learned.proposed_residual_dx_m, learned.proposed_residual_dy_m)
    learned["executed_residual_norm_m"] = np.hypot(learned.executed_residual_dx_m, learned.executed_residual_dy_m)
    learned["phase"] = pd.cut(learned.slot, bins=[-1, 32, 66, 99], labels=["early", "middle", "late"])
    phase_rows = []
    for (method, phase), group in learned.groupby(["method", "phase"], observed=True):
        decisions = group.motion_decision.value_counts(normalize=True)
        accepted = group[group.motion_decision.isin(("policy_accepted", "scaled_policy_accepted"))]
        phase_rows.append(
            {
                "method": method,
                "phase": str(phase),
                "steps": len(group),
                "direct_accept_fraction": float(decisions.get("policy_accepted", 0.0)),
                "scaled_accept_fraction": float(decisions.get("scaled_policy_accepted", 0.0)),
                "takeover_fraction": float(decisions.get("fallback_enter", 0.0) + decisions.get("fallback_continue", 0.0)),
                "mean_proposed_residual_norm_m": float(group.proposed_residual_norm_m.mean()),
                "mean_accepted_executed_residual_norm_m": float(accepted.executed_residual_norm_m.mean()),
                "accepted_executed_residual_gt_0_5m_fraction": float((accepted.executed_residual_norm_m > 0.5).mean()),
                "mean_path_offset_m": float(group.path_offset_m.mean()),
            }
        )
    phase_frame = pd.DataFrame(phase_rows)
    phase_frame.to_csv(output / "phase_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    zero_trace = base_s[
        (base_s.comparison_method == "R0_dynamic_zero") & (base_s.scenario_seed == 20001)
    ]
    for axis, method in zip(axes, ("B10", "B11")):
        old_trace = old_s[
            (old_s.method == method) & (old_s.train_seed == 11001) & (old_s.scenario_seed == 20001)
        ]
        new_trace = new_s[
            (new_s.method == method) & (new_s.train_seed == 11001) & (new_s.scenario_seed == 20001)
        ]
        for trace, label, style in ((zero_trace, "R0 zero residual", "--"), (old_trace, "v2 direct", ":"), (new_trace, "learned residual", "-")):
            axis.plot(trace.x_m, trace.y_m, style, label=label)
        axis.add_patch(plt.Circle((500, 500), 141, fill=False, color="black", alpha=0.6))
        axis.scatter([100, 900], [100, 900], s=25, c=["green", "red"])
        axis.set(title=f"{method}, seed 11001, scenario 20001", xlabel="x (m)", ylabel="y (m)")
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "trajectory_comparison.png", dpi=180)
    plt.close(figure)

    paired = []
    for method in ("B10", "B11"):
        current = new_e[new_e.method == method].set_index(["train_seed", "scenario_seed"])
        direct = old_e[old_e.method == method].set_index(["train_seed", "scenario_seed"])
        assert current.index.equals(direct.index)
        for (train_seed, scenario_seed), row in current.iterrows():
            prior = direct.loc[(train_seed, scenario_seed)]
            paired.append(
                {
                    "method": method,
                    "train_seed": train_seed,
                    "scenario_seed": scenario_seed,
                    "asr_residual_minus_direct": row.asr_bpshz - prior.asr_bpshz,
                    "takeover_slots_residual_minus_direct": row.fallback_takeover_slots - prior.fallback_takeover_slots,
                    "propulsion_j_residual_minus_direct": row.propulsion_energy_j - prior.propulsion_energy_j,
                }
            )
    pd.DataFrame(paired).to_csv(output / "paired_episode_differences.csv", index=False)
    config = load_config(ROOT / "configs" / "motion_residual_v1.yaml")
    counter, replay = counterfactuals(config, new_dir)
    counter.to_csv(output / "same_state_counterfactuals.csv", index=False)
    replay.to_csv(output / "local_model_replay.csv", index=False)
    replay_keys = ["method", "train_seed", "scenario_seed"]
    replay_a = replay.set_index(replay_keys).sort_index()
    replay_b = new_e.set_index(replay_keys).sort_index()
    assert replay_a.index.equals(replay_b.index)
    replay_asr_max_difference = float(np.max(np.abs(replay_a.asr_bpshz - replay_b.asr_bpshz)))
    replay_takeover_max_difference = int(np.max(np.abs(replay_a.fallback_takeover_slots - replay_b.fallback_takeover_slots)))
    counter["accepted"] = counter.actual_decision.isin(("policy_accepted", "scaled_policy_accepted"))
    counter_summary = counter.groupby("method").apply(
        lambda group: pd.Series(
            {
                "probes": len(group),
                "accepted_probe_fraction": float(group.accepted.mean()),
                "mean_executed_difference_m": float(group.executed_difference_m.mean()),
                "accepted_changed_gt_0_5m_fraction": float((group[group.accepted].executed_difference_m > 0.5).mean()),
                "proposed_residual_norm_m": float(group.policy_residual_norm_m.mean()),
                "hard_violations": int(group.actual_hard_violation.sum()),
            }
        ),
        include_groups=False,
    ).reset_index()
    counter_summary.to_csv(output / "same_state_summary.csv", index=False)
    manifest = json.loads((new_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {
        "new_manifest": str(new_dir / "manifest.json"),
        "source_digest": manifest["source_digest"],
        "config_digest": manifest["config_digest"],
        "actual_training_transitions": manifest["budget"]["actual_training_transitions"],
        "development_transitions": manifest["budget"]["development_transitions"],
        "final_test_transitions": manifest["budget"]["final_test_transitions"],
        "all_episodes_have_100_slots": bool((new_e.slots_recorded == 100).all()),
        "all_completed": bool((new_e.task_completed == 1).all()),
        "executed_hard_violations": int(new_e.executed_hard_violations.sum()),
        "local_model_replay_asr_max_difference": replay_asr_max_difference,
        "local_model_replay_takeover_slots_max_difference": replay_takeover_max_difference,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
