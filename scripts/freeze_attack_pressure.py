"""Freeze Task C sources and checksum its raw artifacts after integrity checks."""

from pathlib import Path
import hashlib
import json
import zipfile

import pandas as pd
from run_attack_rule_episodes import bootstrap_paired

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "archive" / "attack_pressure_v1"
OUT.mkdir(parents=True, exist_ok=True)
result_dirs = [ROOT / "results" / name for name in (
    "attack_pressure_scan_v1", "attack_episode_rules_v1", "attack_privileged_fine_v1"
)]
checks = {}
for directory in result_dirs[1:]:
    episodes = pd.read_csv(directory / "raw_episodes.csv")
    steps = pd.read_csv(directory / "raw_steps.csv", low_memory=False)
    expected = 720 if directory.name == "attack_episode_rules_v1" else 120
    assert len(episodes) == expected and len(steps) == expected * 100
    assert not episodes.duplicated(["defense", "train_seed", "attack_label", "scenario_seed"]).any()
    assert set(episodes.scenario_seed) == set(range(30001, 30031))
    assert (episodes.slots_recorded == 100).all() and (episodes.task_completed == 1).all()
    assert episodes.executed_hard_violations.sum() == 0
    assert (steps.willie_power_w >= 0).all() and (steps.willie_power_w <= 1 + 1e-10).all()
    assert (episodes.willie_energy_j <= 6 + 1e-8).all()
    assert ((episodes.alice_radiated_energy_j - 4).abs() < 1e-8).all()
    if directory.name == "attack_episode_rules_v1":
        low = steps[steps.attack_label == "low_constant"]
        assert ((low.willie_power_w - 0.002).abs() < 1e-12).all()
    checks[directory.name] = {
        "episodes": len(episodes), "steps": len(steps), "all_completed": True,
        "hard_violations": 0, "max_willie_power_w": float(steps.willie_power_w.max()),
        "max_willie_energy_j": float(episodes.willie_energy_j.max()),
        "scenario_range": [30001, 30030], "final_test_used": False,
    }

main_episodes = pd.read_csv(result_dirs[1] / "raw_episodes.csv")
fine_episodes = pd.read_csv(result_dirs[2] / "raw_episodes.csv")
keys = ["defense", "train_seed", "scenario_seed"]
paired = main_episodes[main_episodes.attack_label == "low_constant"][keys + ["asr_bpshz"]].merge(
    fine_episodes[keys + ["asr_bpshz"]], on=keys, suffixes=("_low", "_fine"), validate="one_to_one"
)
paired["low_minus_fine_asr"] = paired.asr_bpshz_low - paired.asr_bpshz_fine
paired.to_csv(result_dirs[2] / "paired_low_minus_fine.csv", index=False)
intervals = []
for defense, group in paired.groupby("defense"):
    matrix = group.pivot(index="train_seed", columns="scenario_seed", values="low_minus_fine_asr").to_numpy()
    low, high = bootstrap_paired(matrix, 79901)
    intervals.append({"defense": defense, "low_minus_fine_asr": float(matrix.mean()),
                      "exploratory_crossed_bootstrap_low": low, "exploratory_crossed_bootstrap_high": high,
                      "bootstrap_replicates": 5000, "bootstrap_seed": 79901})
pd.DataFrame(intervals).to_csv(result_dirs[2] / "paired_low_minus_fine_intervals.csv", index=False)

script_names = (
    "run_attack_pressure_scan.py", "run_attack_rule_episodes.py",
    "run_attack_privileged_fine.py", "plot_attack_pressure.py", "freeze_attack_pressure.py",
)
sources = sorted((ROOT / "src" / "securelink").rglob("*.py"))
sources += sorted((ROOT / "configs").rglob("*.yaml"))
sources += [ROOT / "scripts" / name for name in script_names]
archive = OUT / "attack_pressure_v1_source.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
    for path in sources:
        handle.write(path, path.relative_to(ROOT).as_posix())
files = sources + [archive]
for directory in result_dirs:
    files += sorted(path for path in directory.iterdir() if path.is_file())
manifest = {
    "delivery_date": "2026-09-27", "integrity_checks": checks,
    "note": "Defense model hashes are in episode manifests; frozen models remain in the Task B result directory.",
    "files": [
        {"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in files
    ],
}
(OUT / "artifact_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"checks": checks, "checksum_files": len(files)}, indent=2))
