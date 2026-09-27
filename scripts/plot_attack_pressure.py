"""Create compact figures from the frozen attack-pressure results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", default=str(ROOT / "results" / "attack_pressure_scan_v1"))
    parser.add_argument("--episodes", default=str(ROOT / "results" / "attack_episode_rules_v1"))
    args = parser.parse_args()
    scan_dir = Path(args.scan)
    episode_dir = Path(args.episodes)
    points = pd.read_csv(scan_dir / "scan_points.csv")
    groups = pd.read_csv(scan_dir / "scan_groups.csv")

    figure, axis = plt.subplots(figsize=(9, 5))
    nominal = points[
        (points.phase == "confirm")
        & (points.alice_power_w == 0.2)
        & (points.self_interference_db == -110.0)
        & (points.willie_power_w <= 0.05)
    ]
    for name, group in nominal.groupby("position", sort=False):
        axis.plot(group.willie_power_w, group.mean_asr_bpshz, label=name)
    axis.axvline(0.002, color="black", linestyle="--", alpha=0.6, label="frozen 0.002 W rule")
    axis.set(xlabel="Willie power (W)", ylabel="Mean secrecy rate (bit/s/Hz)", title="Independent confirmation, nominal SI, Alice 0.2 W")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8, ncol=2)
    figure.tight_layout()
    figure.savefig(scan_dir / "nominal_low_power_scan.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sensitivity = points[
        (points.phase == "confirm")
        & (points.position == "path_mid")
        & (points.alice_power_w == 0.2)
    ]
    for si, group in sensitivity.groupby("self_interference_db"):
        axes[0].plot(group.willie_power_w, group.mean_asr_bpshz, label=f"SI {si:g} dB")
        zoom = group[group.willie_power_w <= 0.05]
        axes[1].plot(zoom.willie_power_w, zoom.mean_asr_bpshz, label=f"SI {si:g} dB")
    for axis, title in zip(axes, ("Full 0–1 W", "Low-power 0–0.05 W")):
        axis.set(xlabel="Willie power (W)", ylabel="Mean secrecy rate (bit/s/Hz)", title=title)
        axis.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(scan_dir / "self_interference_sensitivity.png", dpi=180)
    plt.close(figure)

    if (episode_dir / "rule_summary.csv").exists():
        rules = pd.read_csv(episode_dir / "rule_summary.csv")
        fine_path = ROOT / "results" / "attack_privileged_fine_v1" / "summary.csv"
        if fine_path.exists():
            fine = pd.read_csv(fine_path)
            fine["attack_label"] = "privileged_myopic_fine_189"
            rules = pd.concat([rules, fine], ignore_index=True)
        order = ["silent", "uniform_0p3", "frontload_peak", "random_bangbang", "low_constant", "privileged_myopic_fine_189"]
        labels = ["Silent", "Uniform", "Frontload", "Random", "Low 0.002 W", "True-channel\nmyopic (fine)"]
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        x = list(range(len(order)))
        for offset, (defense, color) in zip((-0.18, 0.18), (("B00", "steelblue"), ("residual_B10", "darkorange"))):
            subset = rules[rules.defense == defense].set_index("attack_label").loc[order]
            axes[0].bar([value + offset for value in x], subset.asr_bpshz, width=0.35, label=defense, color=color)
            axes[1].bar([value + offset for value in x], subset.willie_energy_j, width=0.35, label=defense, color=color)
        for axis in axes:
            axis.set_xticks(x, labels, rotation=20)
            axis.grid(axis="y", alpha=0.2)
        axes[0].set(ylabel="Defender ASR (bit/s/Hz; lower is stronger attack)", title="Attack pressure")
        axes[1].set(ylabel="Willie energy (J)", title="Actual energy used (6 J cap)")
        axes[0].legend()
        figure.suptitle("Frozen defenders, development scenarios 30001–30030")
        figure.tight_layout()
        figure.savefig(episode_dir / "rule_attack_pressure.png", dpi=180)
        plt.close(figure)
    print({"scan_groups": len(groups), "episode_plot": (episode_dir / "rule_attack_pressure.png").exists()})


if __name__ == "__main__":
    main()
