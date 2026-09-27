"""Execute the locked protocol in explicit stages; never train or run final seeds."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "protocols" / "adaptive_rules_v1"))
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from securelink.config import load_config, config_digest
from securelink.evaluation import run_episode
from securelink.seeding import SeedDescriptor
from securelink.system_info import collect_system_info
from adaptive_rule_adapter import RuleCallback, AuditedRuleEnv
from run_attack_privileged_fine import FinePrivilegedEnv
from statistics_v1 import candidate_choice, paired_difference, rule_branch

PROTO = ROOT / "protocols" / "adaptive_rules_v1"
OUT = ROOT / "results" / "adaptive_rules_v1"
MODELS = (11001, 11002, 11003)
PAYLOAD = json.loads((PROTO / "protocol_payload.json").read_text(encoding="utf-8"))
CATALOG = json.loads((PROTO / "candidate_catalog.json").read_text(encoding="utf-8"))
BASELINES = [
    {"id": "F01_low_0p002", "family": "constant", "power_w": 0.002},
    {"id": "F02_silent", "family": "constant", "power_w": 0.0},
    {"id": "F03_uniform_0p3", "family": "constant", "power_w": 0.3},
    {"id": "F04_frontload", "family": "frontload"},
    {"id": "F05_random", "family": "random"},
]
MYOPIC = {"id": "P189_true_channel", "family": "privileged_myopic"}
CONSTANTS = [{"id": f"C{index:02d}", "family": "constant", "power_w": power, "privileged_posthoc": True}
             for index, power in enumerate(PAYLOAD["posthoc_constant_grid_w"], 1)]
WORKER_MODELS = {}
CONFIG = None


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_paths():
    return [Path(__file__), ROOT / "scripts" / "adaptive_rule_adapter.py", ROOT / "scripts" / "check_adaptive_rules_v1.py"]


def verify_protocol():
    lock = json.loads((PROTO / "lock_manifest.json").read_text(encoding="utf-8"))
    for name, digest in lock["files"].items():
        assert sha(PROTO / name) == digest, name
    assert sha(ROOT.parent / lock["report_path"]) == lock["report_sha256"]
    for name, digest in PAYLOAD["base_source_hashes"].items():
        assert sha(ROOT / name) == digest, name
    for info in PAYLOAD["models"].values():
        assert sha(ROOT / info["path"]) == info["sha256"]
    assert config_digest(load_config(ROOT / "configs" / "motion_residual_v1.yaml")) == PAYLOAD["config_digest"]


def verify_execution():
    verify_protocol()
    seal = json.loads((OUT / "execution_seal.json").read_text(encoding="utf-8"))
    for name, digest in seal["execution_sources"].items():
        assert sha(ROOT / name) == digest, name
    assert sha(OUT / "interface_checks.json") == seal["interface_checks_sha256"]
    assert seal["interface_passed"]


def worker_init():
    global CONFIG, WORKER_MODELS
    torch.set_num_threads(1)
    CONFIG = load_config(PROTO / "resolved_config.yaml")
    WORKER_MODELS = {seed: PPO.load(ROOT / PAYLOAD["models"][str(seed)]["path"], device="cpu") for seed in MODELS}


def episode_path(stage, seed, spec, scenario):
    return OUT / stage / "checkpoints" / f"{seed}_{spec['id']}" / f"{scenario}.json.gz"


def execute_episode(stage, purpose, seed, spec, scenario):
    permitted = {
        "precheck": ("p1_rule_interface_check_v1", range(90001,90004)),
        "selection": ("p1_rule_select_v1", range(31001,31031)),
        "confirmation": ("p1_rule_confirm_v1", range(32001,32101)),
        "posthoc_constant": ("p1_rule_confirm_v1", range(32001,32101)),
    }
    allowed_purpose, allowed_scenarios = permitted[stage]
    assert purpose == allowed_purpose and scenario in allowed_scenarios
    method = "B00" if seed == 0 else "B10"
    cls = FinePrivilegedEnv if spec["family"] == "privileged_myopic" else AuditedRuleEnv
    env = cls(CONFIG, method=method, attack_rule=spec["id"],
              stream_context={"purpose": purpose, "run_seed": 0, "worker_id": 0})
    if cls is AuditedRuleEnv:
        child = SeedDescriptor(purpose + "_bangbang", 0, 0, 0, scenario).seed_sequence().spawn(1)[0]
        env.attack_policy = RuleCallback(spec, np.random.Generator(np.random.PCG64(child)))
    predictor = None if seed == 0 else lambda obs: WORKER_MODELS[seed].predict(obs, deterministic=True)[0]
    started = utc()
    clock = time.perf_counter()
    try:
        episode, steps = run_episode(env, scenario, predictor)
    except Exception:
        write_json(OUT / "errors" / f"partial_{stage}_{seed}_{spec['id']}_{scenario}.json",
                   {"started_utc": started, "wall_seconds": time.perf_counter()-clock,
                    "actual_recorded_slots": len(env.step_records), "steps": env.step_records})
        raise
    duration = time.perf_counter() - clock
    labels = {"defense": "B00" if seed == 0 else "residual_B10", "train_seed": seed,
              "attack_label": spec["id"], "attack_class": "privileged_reference" if
              spec["family"] == "privileged_myopic" or spec.get("privileged_posthoc") else "implementable",
              "stage": stage, "rule_family": spec["family"]}
    episode.update(labels)
    episode["wall_seconds"] = duration
    episode["start_utc"] = started
    episode["end_utc"] = utc()
    episode["willie_peak_executed_w"] = max(row["willie_power_w"] for row in steps)
    episode["budget_clipped_slots"] = sum(row.get("willie_budget_clipped", 0) for row in steps)
    episode["attack_observation_calls"] = env.attack_policy.calls if env.attack_policy else 0
    for row in steps:
        row.update(labels)
    path = episode_path(stage, seed, spec, scenario)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with gzip.open(temporary, "wt", encoding="utf-8") as handle:
        json.dump({"episode": episode, "steps": steps, "spec": spec}, handle, ensure_ascii=False)
    os.replace(temporary, path)
    assert len(steps) == 100 and episode["slots_recorded"] == 100
    assert episode["executed_hard_violations"] == 0
    assert episode["willie_peak_executed_w"] <= 1 + 1e-8 and episode["willie_energy_j"] <= 6 + 1e-8
    assert episode["alice_radiated_energy_j"] <= 4 + 1e-8
    assert abs(sum(row["willie_power_w"] * 0.2 for row in steps) - episode["willie_energy_j"]) < 1e-8
    assert episode["seed_key"] == SeedDescriptor(purpose, 0, 0, 0, scenario).key
    if env.attack_policy:
        assert env.attack_policy.calls == episode["real_slots"]
    return episode


def batch_worker(job):
    stage, purpose, seed, spec, scenarios = job
    summaries = []
    for scenario in scenarios:
        path = episode_path(stage, seed, spec, scenario)
        if path.exists():
            continue
        try:
            summaries.append(execute_episode(stage, purpose, seed, spec, scenario))
        except Exception as error:
            write_json(OUT / "errors" / f"{stage}_{seed}_{spec['id']}_{scenario}.json",
                       {"utc": utc(), "stage": stage, "seed": seed, "spec": spec, "scenario": scenario,
                        "error": repr(error), "status": "halt_integrity_gate"})
            raise
    return summaries


def run_stage(stage, specs, seeds, scenarios, purpose, workers):
    verify_execution()
    expected = [episode_path(stage, seed, spec, scenario) for seed in seeds for spec in specs for scenario in scenarios]
    jobs = []
    for seed in seeds:
        for spec in specs:
            remaining = [scenario for scenario in scenarios if not episode_path(stage, seed, spec, scenario).exists()]
            for start in range(0, len(remaining), 10):
                jobs.append((stage, purpose, seed, spec, remaining[start:start+10]))
    start_time = time.perf_counter()
    start_utc = utc()
    print(f"{stage}: {len(expected)} episodes, {len(jobs)} pending batches", flush=True)
    journal = OUT / "batch_ledger.jsonl"
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = [pool.submit(batch_worker, job) for job in jobs]
        for count, future in enumerate(as_completed(futures), 1):
            rows = future.result()
            with journal.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"stage": stage, "utc": utc(), "episodes": len(rows),
                    "real_slots": sum(row["real_slots"] for row in rows),
                    "padded_slots": sum(row["padded_failure_slots"] for row in rows),
                    "worker_episode_seconds": sum(row["wall_seconds"] for row in rows)}) + "\n")
            if count % 10 == 0 or count == len(futures):
                print(f"{stage}: batches {count}/{len(futures)}, elapsed {time.perf_counter()-start_time:.1f}s", flush=True)
    assert all(path.exists() for path in expected)
    elapsed = time.perf_counter() - start_time
    timing_path = OUT / "stage_sessions.jsonl"
    with timing_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "start_utc": start_utc, "end_utc": utc(),
                               "wall_seconds": elapsed, "workers": workers, "new_batches": len(jobs)}) + "\n")
    episodes = consolidate(stage, expected)
    verify_execution()
    return episodes


def read_checkpoint(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def consolidate(stage, paths):
    episode_fields, step_fields = set(), set()
    for path in paths:
        item = read_checkpoint(path)
        episode_fields.update(item["episode"])
        for row in item["steps"]:
            step_fields.update(row)
    folder = OUT / stage
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with (folder / "raw_episodes.csv").open("w", encoding="utf-8", newline="") as eh, \
         (folder / "raw_steps.csv").open("w", encoding="utf-8", newline="") as sh:
        ew, sw = csv.DictWriter(eh, sorted(episode_fields)), csv.DictWriter(sh, sorted(step_fields))
        ew.writeheader(); sw.writeheader()
        for path in paths:
            item = read_checkpoint(path)
            ew.writerow(item["episode"]); sw.writerows(item["steps"])
            rows.append(item["episode"])
    frame = pd.DataFrame(rows)
    assert not frame.duplicated(["train_seed", "attack_label", "scenario_seed"]).any()
    return frame


def lock_selection(episodes):
    assert len(episodes) == 3690 and set(episodes.scenario_seed) == set(range(31001, 31031))
    scores = episodes.groupby("attack_label").asr_bpshz.mean().to_dict()
    rows = []
    winners = []
    for family in ("time_piecewise", "geometry_threshold", "late_burst"):
        group = [spec for spec in CATALOG if spec["family"] == family]
        winner = candidate_choice({spec["id"]: scores[spec["id"]] for spec in group})
        winners.append(next(spec for spec in group if spec["id"] == winner))
    pool = winners + BASELINES
    overall = candidate_choice({spec["id"]: scores[spec["id"]] for spec in pool})
    for spec in CATALOG + BASELINES:
        rows.append({**spec, "selection_mean_asr": scores[spec["id"]]})
    pd.DataFrame(rows).to_csv(OUT / "candidate_scores.csv", index=False)
    manifest = {"locked_utc": utc(), "winners": winners, "overall_A_id": overall,
                "protocol_payload_sha256": sha(PROTO / "protocol_payload.json"),
                "execution_seal_sha256": sha(OUT / "execution_seal.json"),
                "selection_files": {name: sha(OUT / name) for name in
                    ("selection/raw_episodes.csv", "selection/raw_steps.csv", "candidate_scores.csv")}}
    path = OUT / "selection_lock.json"
    assert not path.exists(), "Do not overwrite the selection lock"
    write_json(path, manifest)
    (OUT / "selection_lock.sha256").write_text(sha(path) + "\n", encoding="utf-8")
    print(json.dumps({"winners": winners, "A": overall, "selection_lock_sha256": sha(path)}, indent=2), flush=True)


def verify_selection():
    path = OUT / "selection_lock.json"
    assert sha(path) == (OUT / "selection_lock.sha256").read_text().strip()
    lock = json.loads(path.read_text(encoding="utf-8"))
    assert lock["execution_seal_sha256"] == sha(OUT / "execution_seal.json")
    for name, digest in lock["selection_files"].items():
        assert sha(OUT / name) == digest
    return lock


def summary_tables():
    verify_execution()
    lock = verify_selection()
    e = pd.read_csv(OUT / "confirmation" / "raw_episodes.csv")
    c = pd.read_csv(OUT / "posthoc_constant" / "raw_episodes.csv")
    assert len(e) == 3600 and len(c) == 7600
    assert set(e.scenario_seed) == set(range(32001, 32101)) == set(c.scenario_seed)
    assert not e.duplicated(["train_seed", "attack_label", "scenario_seed"]).any()
    assert not c.duplicated(["train_seed", "attack_label", "scenario_seed"]).any()
    choices = []
    for (seed, scenario), group in c.groupby(["train_seed", "scenario_seed"]):
        assert len(group) == 19
        minimum = group.asr_bpshz.min()
        selected = group[group.asr_bpshz <= minimum + 1e-8].sort_values("attack_label").iloc[0].to_dict()
        selected["selected_constant_w"] = next(spec["power_w"] for spec in CONSTANTS if spec["id"] == selected["attack_label"])
        selected["original_attack_label"] = selected["attack_label"]
        selected["trajectory_checkpoint"] = str(episode_path("posthoc_constant", int(seed),
            next(spec for spec in CONSTANTS if spec["id"] == selected["attack_label"]), int(scenario)).relative_to(ROOT))
        selected["attack_label"] = "C_posthoc_constant"
        choices.append(selected)
    selected_c = pd.DataFrame(choices)
    selected_c.to_csv(OUT / "posthoc_constant_choices.csv", index=False)
    all_e = pd.concat([e, selected_c], ignore_index=True)
    all_e["takeover_rate"] = all_e.fallback_takeover_slots / 100
    measures = ["asr_bpshz", "slot_sop", "episode_sop", "willie_energy_j", "willie_peak_executed_w",
                "task_completed", "executed_hard_violations", "takeover_rate", "budget_clipped_slots",
                "alice_radiated_energy_j", "propulsion_energy_j", "alice_total_energy_j"]
    all_e.groupby(["defense", "train_seed", "attack_label", "attack_class"])[measures].mean().reset_index().to_csv(
        OUT / "per_model_summary.csv", index=False)
    all_e.groupby(["defense", "attack_label", "attack_class"])[measures].mean().reset_index().to_csv(
        OUT / "overall_summary.csv", index=False)
    primary = e[e.train_seed.isin(MODELS)].pivot(index=["train_seed", "scenario_seed"], columns="attack_label", values="asr_bpshz")
    primary = primary.reindex(pd.MultiIndex.from_product([MODELS, range(32001, 32101)], names=["train_seed", "scenario_seed"]))
    assert primary.notna().all().all()
    low = primary["F01_low_0p002"].to_numpy().reshape(3, 100)
    adaptive = primary[lock["overall_A_id"]].to_numpy().reshape(3, 100)
    privileged = primary["P189_true_channel"].to_numpy().reshape(3, 100)
    paired = pd.DataFrame({"L": low.ravel(), "A": adaptive.ravel(), "P": privileged.ravel()}, index=primary.index)
    paired["D"] = paired.L - paired.P
    paired["L_minus_A"] = paired.L - paired.A
    paired.to_csv(OUT / "paired_primary.csv")
    branch = rule_branch(low, adaptive, privileged, True)
    branch.update({"A_id": lock["overall_A_id"], "selection_lock_sha256": sha(OUT / "selection_lock.json"),
                   "protocol_payload_sha256": sha(PROTO / "protocol_payload.json"), "evaluation_scope": "fixed three B10, nominal -110 dB"})
    write_json(OUT / "branch_result.json", branch)
    intervals = []
    for rule in primary:
        interval = paired_difference(low, primary[rule].to_numpy().reshape(3, 100), 94201)
        intervals.append({"comparison": "low_minus_" + rule, "mean": interval["mean"], "ci_low": interval["ci95"][0], "ci_high": interval["ci95"][1]})
    constant_matrix = selected_c[selected_c.train_seed.isin(MODELS)].set_index(["train_seed", "scenario_seed"]).reindex(primary.index).asr_bpshz.to_numpy().reshape(3,100)
    for label, first, second in (("low_minus_C",low,constant_matrix), ("A_minus_C",adaptive,constant_matrix), ("C_minus_P",constant_matrix,privileged)):
        interval = paired_difference(first, second, 94201)
        intervals.append({"comparison": label, "mean": interval["mean"], "ci_low": interval["ci95"][0], "ci_high": interval["ci95"][1]})
    pd.DataFrame(intervals).to_csv(OUT / "paired_secondary_intervals.csv", index=False)
    phases = []
    for chunk in pd.read_csv(OUT / "confirmation" / "raw_steps.csv", chunksize=50000, low_memory=False):
        chunk["phase"] = pd.cut(chunk.slot, [-1,32,66,99], labels=["early", "middle", "late"])
        group = chunk.groupby(["defense", "train_seed", "attack_label", "phase"], observed=True)
        sums = group[["secrecy_rate_bpshz", "willie_power_w", "fallback_takeover"]].sum()
        sums["rows"] = group.size()
        phases.append(sums)
    total = pd.concat(phases).groupby(level=[0,1,2,3], observed=True).sum()
    for col in ("secrecy_rate_bpshz", "willie_power_w", "fallback_takeover"):
        total[col] /= total.rows
    total.to_csv(OUT / "phase_summary.csv")
    ledger = []
    for stage in ("precheck", "selection", "confirmation", "posthoc_constant"):
        path = OUT / stage / "raw_episodes.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        for (defense, privilege), group in frame.groupby(["defense", "attack_class"]):
            ledger.append({"stage": stage, "defense": defense, "class": privilege, "episodes": len(group),
                "real_transitions": int(group.real_slots.sum()), "padded_slots": int(group.padded_failure_slots.sum()),
                "record_slots": int(group.slots_recorded.sum()), "worker_episode_seconds": float(group.wall_seconds.sum()),
                "training_transitions": 0})
    pd.DataFrame(ledger).to_csv(OUT / "compute_ledger.csv", index=False)
    write_json(OUT / "result_manifest.json", {"completed_utc": utc(), "formal_episodes": 14890,
        "formal_real_transitions": int(sum(pd.read_csv(OUT / stage / "raw_episodes.csv").real_slots.sum() for stage in ("selection", "confirmation", "posthoc_constant"))),
        "executed_hard_violations": int(e.executed_hard_violations.sum() + c.executed_hard_violations.sum() + pd.read_csv(OUT / "selection" / "raw_episodes.csv").executed_hard_violations.sum()),
        "training_transitions": 0, "final_test_transitions": 0,
        "artifacts": {str(path.relative_to(OUT)): sha(path) for path in OUT.rglob("*")
                      if path.is_file() and path.suffix in (".csv", ".json", ".jsonl", ".sha256")}})
    print(json.dumps(branch, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("precheck", "select", "confirm", "summarize"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "precheck":
        verify_protocol()
        assert not (OUT / "execution_seal.json").exists()
        worker_init()
        from check_adaptive_rules_v1 import perform_checks
        checks = perform_checks(sys.modules[__name__])
        write_json(OUT / "interface_checks.json", checks)
        assert checks["passed"]
        with zipfile.ZipFile(OUT / "execution_sources.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in source_paths():
                archive.write(path, path.relative_to(ROOT).as_posix())
        seal = {"sealed_utc": utc(), "protocol_payload_sha256": sha(PROTO / "protocol_payload.json"),
            "execution_sources": {str(path.relative_to(ROOT)): sha(path) for path in source_paths()},
            "interface_checks_sha256": sha(OUT / "interface_checks.json"), "interface_passed": True,
            "execution_source_archive_sha256": sha(OUT / "execution_sources.zip"),
            "runtime": collect_system_info(), "workers": args.workers, "execution_location": "local Windows workspace",
            "commands": [f"{sys.executable} {Path(__file__)} --stage {stage} --workers {args.workers}" for stage in ("precheck", "select", "confirm", "summarize")]}
        write_json(OUT / "execution_seal.json", seal)
        print(json.dumps({"interface_passed": True, "seal_sha256": sha(OUT / "execution_seal.json")}, indent=2))
    elif args.stage == "select":
        assert not (OUT / "selection_lock.json").exists()
        episodes = run_stage("selection", CATALOG + BASELINES, MODELS, list(range(31001,31031)), "p1_rule_select_v1", args.workers)
        lock_selection(episodes)
    elif args.stage == "confirm":
        lock = verify_selection()
        specs = BASELINES + lock["winners"] + [MYOPIC]
        run_stage("confirmation", specs, MODELS + (0,), list(range(32001,32101)), "p1_rule_confirm_v1", args.workers)
        run_stage("posthoc_constant", CONSTANTS, MODELS + (0,), list(range(32001,32101)), "p1_rule_confirm_v1", args.workers)
    else:
        summary_tables()


if __name__ == "__main__":
    main()
