"""Presentation only: draw frozen Task 3 summaries without changing statistics."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "adaptive_rules_v1"


def main():
    summaries = pd.read_csv(OUT / "overall_summary.csv")
    lock = json.loads((OUT / "selection_lock.json").read_text(encoding="utf-8"))
    branch = json.loads((OUT / "branch_result.json").read_text(encoding="utf-8"))
    rules = ["F01_low_0p002", "F02_silent", "F03_uniform_0p3", "F04_frontload", "F05_random"]
    rules += [item["id"] for item in lock["winners"]]
    rules += ["C_posthoc_constant", "P189_true_channel"]
    labels = ["Low 0.002 W", "Silent", "Uniform", "Frontload", "Random"]
    labels += [item["id"] for item in lock["winners"]] + ["Posthoc constant\n(privileged)", "Myopic 189\n(privileged)"]
    data = summaries[summaries.defense == "residual_B10"].set_index("attack_label").loc[rules]
    figure, axes = plt.subplots(1,2,figsize=(14,5.5))
    colors = ["#3478a6" if rule != lock["overall_A_id"] else "#dd8833" for rule in rules]
    colors[-2:] = ["#8b8b8b", "#555555"]
    x = np.arange(len(rules))
    axes[0].bar(x,data.asr_bpshz,color=colors)
    axes[1].bar(x,data.willie_energy_j,color=colors)
    for axis in axes:
        axis.set_xticks(x,labels,rotation=35,ha="right")
        axis.axvline(7.5,color="black",linestyle="--",linewidth=.8)
        axis.grid(axis="y",alpha=.2)
    axes[0].set(ylabel="Defender ASR (bit/s/Hz; lower is stronger)",title=f"A frozen before confirmation: {lock['overall_A_id']}")
    axes[1].set(ylabel="Mean Willie energy (J)",title="Actual use under the 6 J cap")
    figure.suptitle(f"Fixed three B10 defenders; confirmation 32001–32100; q={branch['q']:.3f}" if branch["q"] is not None else "Fixed three B10 defenders; confirmation 32001–32100; q not interpreted")
    figure.tight_layout()
    figure.savefig(OUT / "confirmed_rule_comparison.png",dpi=180)
    plt.close(figure)
    scores = pd.read_csv(OUT / "candidate_scores.csv")
    figure,axes = plt.subplots(1,3,figsize=(13,4))
    for axis,family in zip(axes,("time_piecewise","geometry_threshold","late_burst")):
        subset = scores[scores.family==family].sort_values("id")
        values = subset.selection_mean_asr.to_numpy().reshape(3,4)
        heat = axis.imshow(values,aspect="auto",cmap="viridis")
        for i in range(3):
            for j in range(4):
                fraction = (values[i,j]-values.min())/(values.max()-values.min())
                axis.text(j,i,f"{subset.iloc[i*4+j]['id']}\n{values[i,j]:.4f}",ha="center",va="center",color="black" if fraction>.55 else "white",fontsize=8)
        axis.set(title=family,xticks=[],yticks=[])
        figure.colorbar(heat,ax=axis,fraction=.046,pad=.04)
    figure.suptitle("All 36 preregistered candidates; selection only (31001–31030)")
    figure.tight_layout()
    figure.savefig(OUT / "selection_all_candidates.png",dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
