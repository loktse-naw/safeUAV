from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    root = Path(args.input).resolve()
    episodes = pd.read_csv(root / "raw_episodes.csv")
    steps = pd.read_csv(root / "raw_steps.csv")
    runtimes = pd.read_csv(root / "training_runtime.csv")

    # B00 has no training randomness. New runs contain it once; old runs are
    # de-duplicated by scenario without altering their raw files.
    baseline = episodes[episodes["method"] == "B00"].drop_duplicates("scenario_seed", keep="first")
    fair_episodes = pd.concat(
        [
            baseline,
            episodes[episodes["method"] != "B00"],
        ],
        ignore_index=True,
    )
    metric_columns = [column for column in [
        "asr_bpshz",
        "slot_sop",
        "episode_sop",
        "task_completed",
        "task_failure",
        "alice_total_energy_j",
        "alice_radiated_energy_j",
        "executed_hard_violations",
        "action_corrections",
        "reference_fallbacks",
        "fallback_independent_triggers",
        "fallback_takeover_slots",
        "fallback_exit_count",
    ] if column in episodes.columns]
    rows = []
    for method, group in fair_episodes.groupby("method"):
        row = {"method": method, "episodes": len(group), "train_seed_count": group["train_seed"].nunique()}
        for column in metric_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_std"] = float(group[column].std(ddof=1)) if len(group) > 1 else 0.0
        row["asr_q05_bpshz"] = float(group["asr_bpshz"].quantile(0.05))
        rows.append(row)
    method_summary = pd.DataFrame(rows)
    method_summary.to_csv(root / "method_summary_fair.csv", index=False)

    progress_rows = []
    for progress_path in sorted((root / "training_logs").glob("*_seed*/progress.csv")):
        frame = pd.read_csv(progress_path)
        reward_column = "rollout/ep_rew_mean"
        progress_rows.append(
            {
                "run": progress_path.parent.name,
                "rows": len(frame),
                "first_ep_reward_mean": float(frame[reward_column].dropna().iloc[0])
                if reward_column in frame and not frame[reward_column].dropna().empty
                else np.nan,
                "last_ep_reward_mean": float(frame[reward_column].dropna().iloc[-1])
                if reward_column in frame and not frame[reward_column].dropna().empty
                else np.nan,
                "last_total_timesteps": float(frame["time/total_timesteps"].dropna().iloc[-1])
                if "time/total_timesteps" in frame and not frame["time/total_timesteps"].dropna().empty
                else np.nan,
            }
        )
    progress = pd.DataFrame(progress_rows)
    progress.to_csv(root / "training_progress_summary.csv", index=False)

    with (root / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    development_seeds = manifest.get("development_seeds", manifest.get("test_seeds", []))
    learning_methods = [method for method in manifest["methods"] if method != "B00"]
    expected_episodes = len(development_seeds) * (
        (1 if "B00" in manifest["methods"] else 0) + len(learning_methods) * len(manifest["train_seeds"])
    )
    expected_steps = expected_episodes * int(episodes["slots_expected"].iloc[0])
    actual_column = "actual_timesteps" if "actual_timesteps" in runtimes.columns else "timesteps"
    integrity = {
        "episode_rows": int(len(episodes)),
        "step_rows": int(len(steps)),
        "expected_episode_rows": expected_episodes,
        "expected_step_rows": expected_steps,
        "all_episode_rows_present": len(episodes) == expected_episodes,
        "all_step_rows_present": len(steps) == expected_steps,
        "all_slots_recorded_100": bool((episodes["slots_recorded"] == 100).all()),
        "all_tasks_completed": bool((episodes["task_completed"] == 1).all()),
        "executed_hard_violations_total": int(episodes["executed_hard_violations"].sum()),
        "padded_failure_slots_total": int(episodes["padded_failure_slots"].sum()),
        "test_seed_min": int(episodes["scenario_seed"].min()),
        "test_seed_max": int(episodes["scenario_seed"].max()),
        "training_seconds_total_learning_only": float(
            runtimes.loc[runtimes[actual_column] > 0, "seconds"].sum()
        ),
        "training_interactions_total_learning_only": int(
            runtimes.loc[runtimes[actual_column] > 0, actual_column].sum()
        ),
        "implementation_version": manifest.get("implementation_version", "legacy_unversioned"),
        "source_digest": manifest.get("source_digest"),
        "notes": [
            "B00 is counted once per development scenario; legacy duplicates are removed only in the fair summary.",
            "Three training seeds are prototype diagnostics, not sufficient publication statistics.",
        ],
    }
    with (root / "integrity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(integrity, handle, indent=2, ensure_ascii=False)
    print(json.dumps(integrity, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
