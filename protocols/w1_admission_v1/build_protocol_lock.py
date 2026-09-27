"""Task 4 metadata only: audit, register, hash and lock. Never instantiate an env/PPO."""
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from securelink.seeding import SeedDescriptor, _label_u32


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def scan(stream):
    reader = csv.reader(stream)
    header = next(reader, [])
    if "scenario_seed" not in header:
        return 0, []
    index = header.index("scenario_seed")
    count, seen = 0, set()
    for row in reader:
        if len(row) > index and row[index]:
            value = Decimal(row[index])
            if not value.is_finite() or value != value.to_integral_value():
                raise ValueError("Nonintegral or nonfinite scenario label")
            seen.add(int(value))
            count += 1
    return count, sorted(seen)


def main():
    if (OUT / "lock_manifest.json").exists():
        raise RuntimeError("Already locked: never overwrite; create a new version")
    start = datetime.now(timezone.utc).isoformat(timespec="seconds")
    old = ROOT / "protocols/adaptive_rules_v1"
    result = ROOT / "results/adaptive_rules_v1"
    prior = read(old / "protocol_payload.json")
    branch = read(result / "branch_result.json")
    manifest = read(result / "result_manifest.json")
    delivery = read(result / "delivery_manifest.json")
    report = ROOT.parent / "feedback/15_P1_W1准入决定与冻结协议.md"
    if branch["branch"] != "may_prepare_limited_W1_protocol" or manifest["executed_hard_violations"] != 0:
        raise RuntimeError("Task 14 admission branch/integrity gate failed")
    for name, digest in prior["base_source_hashes"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    for item in prior["models"].values():
        if sha(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Frozen defender changed")
    for name, digest in manifest["artifacts"].items():
        if sha(result / name) != digest:
            raise RuntimeError(f"Task 14 artifact changed: {name}")
    task14_path = ROOT.parent / "feedback/14_P1同信息自适应规则评估.md"
    if sha(task14_path) != delivery["artifacts"]["feedback/14_P1同信息自适应规则评估.md"]:
        raise RuntimeError("Task 14 report changed")
    proposed = set(range(33001,34001)) | set(range(34001,34031)) | set(range(35001,35101))
    proposed |= set(range(39001,39011)) | set(range(50001,50101))
    csv.field_size_limit(10_000_000)
    audit = []
    for path in sorted((ROOT / "results").rglob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            count, seen = scan(stream)
        if count:
            audit.append({"path":path.relative_to(ROOT).as_posix(),"sha256":sha(path),
                          "record_rows":count,"scenario_seeds":seen,
                          "proposed_block_hits":sorted(proposed.intersection(seen))})
    for path in sorted((ROOT / "archive").rglob("*.tar.gz")):
        archive_digest = sha(path)
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".csv") and "episode" in member.name:
                    raw = archive.extractfile(member)
                    with io.TextIOWrapper(raw,encoding="utf-8-sig",newline="") as stream:
                        count, seen = scan(stream)
                    if count:
                        audit.append({"path":path.relative_to(ROOT).as_posix()+"::"+member.name,
                                      "archive_sha256":archive_digest,"record_rows":count,
                                      "scenario_seeds":seen,
                                      "proposed_block_hits":sorted(proposed.intersection(seen))})
    if any(row["proposed_block_hits"] for row in audit):
        dump("failed_audit.json",audit)
        raise RuntimeError("Proposed W1/final label already used; stop registration")
    dump("scenario_usage_audit.json",{"started_utc":start,
         "completed_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "scope":"All local results CSVs with scenario_seed; archive tar.gz episode CSV members; no remote audit claimed",
         "files_or_members_with_records":len(audit),"proposed_block_record_hits":0,"records":audit,
         "remote_inventory_gate_if_remote_execution":True})
    records = []
    blocks = [("interface_check","p1_w1_interface_check_v1",39001,39010),
              ("development","p1_w1_development_v1",34001,34030),
              ("feasibility","p1_w1_feasibility_v1",35001,35100)]
    for block,purpose,first,last in blocks:
        for label in range(first,last+1):
            descriptor = SeedDescriptor(purpose,0,0,0,label)
            records.append({"block":block,"status":"reserved_not_used","scenario_id":label,
                "purpose":purpose,"run_seed":0,"worker_id":0,"episode_id":0,
                "key":descriptor.key,"fingerprint":descriptor.fingerprint(),
                "entropy":[2,_label_u32(purpose),0,0,0,1,label],
                "child_order":["channel","defender_observation","attack_location_error"]})
    for seed in (21001,21002,21003):
        for label in range(33001,34001):
            descriptor = SeedDescriptor("p1_w1_train_v1",seed,0,0,label)
            records.append({"block":"training","status":"reserved_not_used","scenario_id":label,
                "purpose":"p1_w1_train_v1","run_seed":seed,"worker_id":0,"episode_id":0,
                "key":descriptor.key,"fingerprint":descriptor.fingerprint(),
                "entropy":[2,_label_u32("p1_w1_train_v1"),seed,0,0,1,label],
                "child_order":["channel","defender_observation","attack_location_error"]})
    dump("scenario_registry.json",records)
    shutil.copyfile(old / "resolved_config.yaml",OUT / "resolved_config.yaml")
    shutil.copyfile(old / "statistics_v1.py",OUT / "fixed_defender_statistics_v1.py")
    config = {"protocol_id":"p1-w1-admission-v1","status":"protocol_locked_no_training_authorization",
      "admission":"limited_feasibility_protocol_prepared","config_digest":prior["config_digest"],
      "white_list":prior["white_list"],"position_contract":prior["position_contract"],
      "models":prior["models"],"frozen_rule":read(result / "selection_lock.json")["winners"][0],
      "attack_run_seeds":[21001,21002,21003],"config_candidates":1,"performance_selection":False,
      "observation":{"order":["slot/100","remaining_slots/100","willie_remaining_j/6","willie_peak_w/1",
                              "alice_x_est_m/1000","alice_y_est_m/1000"],
                     "dtype":"float32","coordinate_clip":False,"history":False,"vec_normalize":False},
      "action":{"shape":[1],"box":[-1,1],"z":"(clip(u,-1,1)+1)/2",
                "requested_power":"0.001 * expm1(z * log1p(peak_w/0.001))",
                "peak_w":1.0,"energy_cap_j":6.0,"tx_seconds":0.2,
                "executed_power":"min(requested_power,peak_w,remaining_j/0.2)"},
      "reward":{"train":"-max(0,C_B-C_W)","source":"simulator_privileged_reward",
                "failure_padding":0.0,"shaping":False,"online_reward_history":False},
      "ppo":{"implementation":"stable_baselines3.PPO","version":"2.7.0","policy":"MlpPolicy",
             "policy_layers":[64,64],"activation":"Tanh","ortho_init":True,
             "optimizer":"Adam","optimizer_eps":1e-5,"log_std_init":0.0,
             "n_envs":1,"n_steps":512,"batch_size":128,"n_epochs":5,"learning_rate":0.0003,
             "gamma":1.0,"gae_lambda":0.95,"ent_coef":0.01,"clip_range":0.2,"clip_range_vf":None,
             "normalize_advantage":True,"vf_coef":0.5,"max_grad_norm":0.5,"use_sde":False,
             "target_kl":None,"device":"cpu","torch_threads":1},
      "training":{"per_run_sampler_calls_cap":32768,"all_runs_sampler_calls_cap":98304,
                  "rollouts_per_run":64,"episodes_per_run_cap":1000,
                  "scenario_schedule":"33001+zero_based_episode_index; no wrap or repeat",
                  "defender_schedule":"[11001,11002,11003][(episode_index+run_index)%3]",
                  "defender_deterministic":True,"attack_stochastic":True,
                  "checkpoint":"after last update at exactly 32768 sampler calls; no performance checkpoint selection",
                  "partial_episode":"retain prefix without extra stepping; exclude from episode averages; count real steps",
                  "failure":"preserve zero-physical-step sampler calls separately; never count padded slots as transitions"},
      "evaluation":{"attack_deterministic":True,"defender_deterministic":True,
                    "rules":["T10","F01_low_0p002"],"development_episodes":450,"development_slots_cap":45000,
                    "feasibility_episodes":1500,"feasibility_slots_cap":150000,
                    "no_selection_of_attack_runs":True,"no_optional_privileged_replays":True},
      "precheck":{"episode_cap":30,"sampler_calls_cap":3000,"labels":[39001,39010],
                  "purpose":"p1_w1_interface_check_v1","no_optimizer_steps":True},
      "statistics":{"script":"statistics_v1.py::w1_primary","delta":0.005,"replicates":10000,
                    "seed":94301,"shared_scenario_weights":True,"resample_defenders":False,
                    "resample_attack_runs":False,"primary":"mean T10 ASR - mean all three W1 runs ASR",
                    "success":"95% CI lower > delta; integrity passed"},
      "future_total_sampler_calls_cap":296304,"future_pool_generation":0,
      "current_actual_interactions":{"train":0,"precheck":0,"development":0,"feasibility":0,"pool":0,"final":0},
      "final_test":{"status":"reserved_not_used","labels":[50001,50100],
                    "purpose":"p1_final_v1","actual_transitions":0,"inherited_task13_protocol":True},
      "execution_gate":"separate execution instruction; adapter/driver/ledger/statistic verification and source seal before first training reset",
      "result_paths":{"root":"results/w1_feasibility_v1","training":"training/raw_steps.csv + training/partial_episodes.csv",
                      "development":"development/raw_episodes.csv + development/raw_steps.csv",
                      "feasibility":"feasibility/raw_episodes.csv + feasibility/raw_steps.csv"}}
    dump("w1_config.json",config)
    examples = []
    for power in (0,0.001,0.002,0.004,0.01,0.02,0.1,0.3,1):
        u = 2*math.log1p(power/0.001)/math.log1p(1/0.001)-1
        slope = 0.001*math.exp((u+1)/2*math.log1p(1000))*math.log1p(1000)/2
        examples.append({"requested_w":power,"action_u":u,"dp_du":slope,
                         "approx_dp_for_du_0p001":slope*0.001})
    dump("action_mapping_reference.json",{"metadata_only":True,"examples":examples})
    inputs = [ROOT.parent / "guide/005_P1自适应规则基线与W1准入判定.md",task14_path,
              old / "protocol_payload.json",old / "lock_manifest.json",result / "branch_result.json",
              result / "result_manifest.json",result / "delivery_manifest.json",result / "selection_lock.json",
              ROOT / "scripts/run_d1.py",ROOT / "scripts/adaptive_rule_adapter.py",
              ROOT / "scripts/run_adaptive_rules_v1.py"]
    source_hashes = {"pytorch_code/"+name:digest for name,digest in prior["base_source_hashes"].items()}
    for item in inputs:
        name = item.relative_to(ROOT.parent).as_posix()
        source_hashes[name] = sha(item)
    for item in (Path(sys.prefix)/"Lib/site-packages/stable_baselines3/ppo/ppo.py",
                 Path(sys.prefix)/"Lib/site-packages/stable_baselines3/common/policies.py"):
        if not item.exists():
            raise RuntimeError("Pinned local PPO implementation unavailable")
        source_hashes["installed/"+item.name] = sha(item)
    locked = datetime.now(timezone.utc).isoformat(timespec="seconds")
    names = ["w1_config.json","scenario_registry.json","scenario_usage_audit.json","resolved_config.yaml",
             "action_mapping_reference.json","statistics_v1.py","fixed_defender_statistics_v1.py","build_protocol_lock.py"]
    payload = {"protocol_id":config["protocol_id"],"locked_utc":locked,"admission_branch":branch,
               "input_hashes":source_hashes,"models":prior["models"],
               "artifacts":{name:sha(OUT/name) for name in names},"current_training_transitions":0,
               "current_evaluation_transitions":0,"final_test_transitions":0,
               "task14_prior_cost":{"episodes":manifest["formal_episodes"],"real_transitions":manifest["formal_real_transitions"],
                                    "hard_violations":manifest["executed_hard_violations"]}}
    dump("protocol_payload.json",payload)
    text = report.read_text(encoding="utf-8")
    if text.count("{{LOCK_TIME}}") != 1 or text.count("{{PAYLOAD_SHA}}") != 1:
        raise RuntimeError("Report lock placeholders missing or duplicated")
    report.write_text(text.replace("{{LOCK_TIME}}",locked).replace("{{PAYLOAD_SHA}}",sha(OUT/"protocol_payload.json")),encoding="utf-8")
    names += ["protocol_payload.json"]
    dump("lock_manifest.json",{"locked_utc":locked,"protocol_payload_sha256":sha(OUT/"protocol_payload.json"),
         "report_sha256":sha(report),"files":{name:sha(OUT/name) for name in names},
         "status":"protocol_locked_no_experiment_executed"})
    print(json.dumps({"locked_utc":locked,"payload_sha256":sha(OUT/"protocol_payload.json"),
          "report":str(report),"audit_records":len(audit),"registered_keys":len(records),
          "new_training_transitions":0,"new_evaluation_transitions":0},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
