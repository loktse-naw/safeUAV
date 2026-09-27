from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_result(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    episodes = pd.read_csv(root / "raw_episodes.csv")
    steps = pd.read_csv(root / "raw_steps.csv")
    runtime = pd.read_csv(root / "training_runtime.csv")
    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return episodes, steps, runtime, manifest


def summarize_methods(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in episodes.groupby("method"):
        rows.append(
            {
                "method": method,
                "episodes": len(group),
                "training_seeds": int(group["train_seed"].nunique(dropna=True)),
                "asr_mean": group["asr_bpshz"].mean(),
                "asr_std_episode": group["asr_bpshz"].std(ddof=1),
                "asr_q05": group["asr_bpshz"].quantile(0.05),
                "slot_sop": group["slot_sop"].mean(),
                "episode_sop": group["episode_sop"].mean(),
                "completion_rate": group["task_completed"].mean(),
                "total_energy_j": group["alice_total_energy_j"].mean(),
                "radiated_energy_j": group["alice_radiated_energy_j"].mean(),
                "hard_violations": int(group["executed_hard_violations"].sum()),
                "corrections_per_episode": group["action_corrections"].mean(),
                "fallback_triggers_per_episode": group["fallback_independent_triggers"].mean(),
                "takeover_slots_per_episode": group["fallback_takeover_slots"].mean(),
                "fallback_exits_per_episode": group["fallback_exit_count"].mean(),
            }
        )
    return pd.DataFrame(rows)


def summarize_control(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, takeover), group in steps.groupby(["method", "fallback_takeover"]):
        rows.append(
            {
                "method": method,
                "fallback_takeover": int(takeover),
                "slots": len(group),
                "slot_fraction_within_method": len(group) / len(steps[steps["method"] == method]),
                "mean_secrecy_rate_bpshz": group["secrecy_rate_bpshz"].mean(),
                "mean_proposed_move_m": (group["proposed_dx_m"] ** 2 + group["proposed_dy_m"] ** 2).pow(0.5).mean(),
                "mean_executed_move_m": (group["executed_dx_m"] ** 2 + group["executed_dy_m"] ** 2).pow(0.5).mean(),
                "mean_power_w": group["alice_power_w"].mean(),
            }
        )
    return pd.DataFrame(rows)


def power_budget_diagnostic(steps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, group in steps.groupby(["method", "train_seed", "episode_index"], dropna=False):
        group = group.sort_values("slot")
        exhausted = group[group["radiated_after_j"] <= 1e-9]
        rows.append(
            {
                "method": keys[0],
                "train_seed": keys[1],
                "episode_index": keys[2],
                "budget_exhausted": int(not exhausted.empty),
                "exhaustion_slot": exhausted["slot"].iloc[0] if not exhausted.empty else pd.NA,
                "front_power_w": group.iloc[:34]["alice_power_w"].mean(),
                "middle_power_w": group.iloc[34:67]["alice_power_w"].mean(),
                "back_power_w": group.iloc[67:]["alice_power_w"].mean(),
            }
        )
    return pd.DataFrame(rows)


def interaction_diagnostic(episodes: pd.DataFrame, bootstrap_samples: int = 5000) -> tuple[pd.DataFrame, dict]:
    baseline = (
        episodes[episodes["method"] == "B00"][["scenario_seed", "asr_bpshz", "alice_total_energy_j"]]
        .rename(columns={"asr_bpshz": "asr_B00", "alice_total_energy_j": "energy_B00"})
    )
    learned = episodes[episodes["method"] != "B00"]
    wide_asr = learned.pivot(index=["train_seed", "scenario_seed"], columns="method", values="asr_bpshz").reset_index()
    wide_energy = learned.pivot(
        index=["train_seed", "scenario_seed"], columns="method", values="alice_total_energy_j"
    ).reset_index()
    paired = wide_asr.merge(wide_energy, on=["train_seed", "scenario_seed"], suffixes=("_asr", "_energy"))
    paired = paired.merge(baseline, on="scenario_seed", how="left")
    paired["asr_interaction"] = (
        paired["B11_asr"] - paired["B10_asr"] - paired["B01_asr"] + paired["asr_B00"]
    )
    paired["energy_interaction_j"] = (
        paired["B11_energy"]
        - paired["B10_energy"]
        - paired["B01_energy"]
        + paired["energy_B00"]
    )
    paired["B11_minus_B00_asr"] = paired["B11_asr"] - paired["asr_B00"]
    paired["B10_minus_B00_asr"] = paired["B10_asr"] - paired["asr_B00"]
    paired["B01_minus_B00_asr"] = paired["B01_asr"] - paired["asr_B00"]

    rng = np.random.default_rng(88031)
    seeds = paired["train_seed"].unique()
    scenarios = paired["scenario_seed"].unique()
    draws = {column: [] for column in [
        "asr_interaction",
        "B11_minus_B00_asr",
        "B10_minus_B00_asr",
        "B01_minus_B00_asr",
    ]}
    indexed = paired.set_index(["train_seed", "scenario_seed"])
    for _ in range(bootstrap_samples):
        selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        values = {column: [] for column in draws}
        for seed in selected_seeds:
            selected_scenarios = rng.choice(scenarios, size=len(scenarios), replace=True)
            sample = indexed.loc[(seed, selected_scenarios), :]
            for column in draws:
                values[column].extend(np.asarray(sample[column], dtype=float).reshape(-1).tolist())
        for column in draws:
            draws[column].append(float(np.mean(values[column])))
    summary = {}
    for column, values in draws.items():
        summary[column] = {
            "mean": float(paired[column].mean()),
            "hierarchical_bootstrap_95pct": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
        }
    summary["energy_interaction_j_mean"] = float(paired["energy_interaction_j"].mean())
    summary["energy_ranges_j"] = {
        method: [
            float(episodes.loc[episodes["method"] == method, "alice_total_energy_j"].min()),
            float(episodes.loc[episodes["method"] == method, "alice_total_energy_j"].max()),
        ]
        for method in ["B00", "B01", "B10", "B11"]
    }
    summary["energy_matched_h3_available"] = False
    summary["reason"] = "B00/B01 energy ranges do not overlap B10/B11; no preplanned performance-energy curve was run."
    return paired.reset_index(drop=True), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-b01", required=True)
    parser.add_argument("--centered-b01", required=True)
    parser.add_argument("--motion-joint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    legacy_ep, legacy_steps, legacy_runtime, legacy_manifest = load_result(Path(args.legacy_b01))
    centered_ep, centered_steps, centered_runtime, centered_manifest = load_result(Path(args.centered_b01))
    motion_ep, motion_steps, motion_runtime, motion_manifest = load_result(Path(args.motion_joint))
    main_episodes = pd.concat([motion_ep, centered_ep], ignore_index=True)
    main_steps = pd.concat([motion_steps, centered_steps], ignore_index=True)
    main_runtime = pd.concat([motion_runtime, centered_runtime], ignore_index=True)

    method_summary = summarize_methods(main_episodes)
    method_summary.to_csv(output / "method_summary.csv", index=False)
    seed_summary = (
        main_episodes[main_episodes["method"] != "B00"]
        .groupby(["method", "train_seed"])
        .agg(
            episodes=("scenario_seed", "count"),
            asr_mean=("asr_bpshz", "mean"),
            slot_sop=("slot_sop", "mean"),
            episode_sop=("episode_sop", "mean"),
            takeover_slots=("fallback_takeover_slots", "mean"),
        )
        .reset_index()
    )
    seed_summary.to_csv(output / "seed_summary.csv", index=False)
    control_summary = summarize_control(main_steps)
    control_summary.to_csv(output / "control_takeover_summary.csv", index=False)
    motion_decisions = (
        main_steps.groupby(["method", "motion_decision"])
        .size()
        .rename("slots")
        .reset_index()
    )
    method_slot_counts = main_steps.groupby("method").size().rename("method_slots")
    motion_decisions = motion_decisions.merge(method_slot_counts, on="method")
    motion_decisions["fraction"] = motion_decisions["slots"] / motion_decisions["method_slots"]
    motion_decisions.to_csv(output / "motion_decision_summary.csv", index=False)
    rejection_reasons = (
        main_steps[main_steps["motion_rejection_reason"].fillna("") != ""]
        .groupby(["method", "motion_rejection_reason"])
        .size()
        .rename("slots")
        .reset_index()
    )
    rejection_reasons.to_csv(output / "motion_rejection_summary.csv", index=False)
    budget_detail = power_budget_diagnostic(main_steps)
    budget_summary = (
        budget_detail.groupby("method")
        .agg(
            episodes=("episode_index", "count"),
            exhausted_fraction=("budget_exhausted", "mean"),
            exhaustion_slot_mean=("exhaustion_slot", "mean"),
            front_power_w=("front_power_w", "mean"),
            middle_power_w=("middle_power_w", "mean"),
            back_power_w=("back_power_w", "mean"),
        )
        .reset_index()
    )
    budget_detail.to_csv(output / "power_budget_episode.csv", index=False)
    budget_summary.to_csv(output / "power_budget_summary.csv", index=False)
    paired_interaction, interaction_summary = interaction_diagnostic(main_episodes)
    paired_interaction.to_csv(output / "paired_interaction_rows.csv", index=False)
    with (output / "interaction_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(interaction_summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    mapping = pd.concat(
        [
            legacy_ep.assign(power_mapping="legacy_affine"),
            centered_ep.assign(power_mapping="remaining_budget_centered"),
        ],
        ignore_index=True,
    )
    mapping_summary = (
        mapping.groupby("power_mapping")
        .agg(
            episodes=("scenario_seed", "count"),
            asr_mean=("asr_bpshz", "mean"),
            asr_std_episode=("asr_bpshz", "std"),
            slot_sop=("slot_sop", "mean"),
            episode_sop=("episode_sop", "mean"),
            radiated_energy_j=("alice_radiated_energy_j", "mean"),
        )
        .reset_index()
    )
    mapping_summary.to_csv(output / "b01_mapping_summary.csv", index=False)

    report = {
        "implementation_version": centered_manifest.get("implementation_version"),
        "source_digests": sorted(
            {
                legacy_manifest.get("source_digest"),
                centered_manifest.get("source_digest"),
                motion_manifest.get("source_digest"),
            }
        ),
        "development_seeds": centered_manifest["development_seeds"],
        "final_test_status": centered_manifest["final_test_seeds"],
        "training_budget": {
            "requested": int(main_runtime["requested_timesteps"].sum()),
            "actual": int(main_runtime["actual_timesteps"].sum()),
            "training_seconds": float(main_runtime["seconds"].sum()),
            "development_transitions": int(len(main_steps)),
            "legacy_mapping_extra_training_actual": int(legacy_runtime["actual_timesteps"].sum()),
            "legacy_mapping_extra_training_seconds": float(legacy_runtime["seconds"].sum()),
        },
        "method_summary": method_summary.to_dict(orient="records"),
        "b01_mapping_summary": mapping_summary.to_dict(orient="records"),
        "interaction_diagnostic": interaction_summary,
        "all_hard_violations_zero": bool(main_episodes["executed_hard_violations"].sum() == 0),
        "all_slots_present": bool((main_episodes["slots_recorded"] == 100).all()),
    }
    with (output / "controlled_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(report["training_budget"], indent=2))


if __name__ == "__main__":
    main()
