"""Explicit W1 stages under 15/006. Precheck never performs W1 training."""
import argparse
import csv
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import zipfile
from datetime import datetime,timezone

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "protocols/w1_admission_v1"
OUT = ROOT / "results/w1_feasibility_v1"
sys.path.insert(0,str(ROOT / "src"))
sys.path.insert(0,str(PROTO))
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from securelink.config import load_config,config_digest
from securelink.seeding import SeedDescriptor
from securelink.system_info import collect_system_info
from adaptive_rule_adapter import RuleCallback,AuditedRuleEnv,requested_power
from w1_adapter import W1Env,DEFENDERS,inverse_power
from w1_ledger import CallLedger
from build_protocol_lock import scan

CFG = json.loads((PROTO / "w1_config.json").read_text(encoding="utf-8"))


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            d.update(chunk)
    return d.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".partial")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    os.replace(temporary,path)


def sources():
    return [Path(__file__),ROOT/"scripts/w1_adapter.py",ROOT/"scripts/w1_ledger.py",ROOT/"scripts/check_w1_execution.py"]


def verify_contract():
    lock=read(PROTO/"lock_manifest.json")
    for name,digest in lock["files"].items():
        assert sha(PROTO/name)==digest,name
    revision=read(PROTO/"report_revision_lock.json")
    assert sha(PROTO/"report_locked_original.md")==lock["report_sha256"]
    assert sha(PROTO/"lock_manifest.json")==revision["original_lock_sha256"]
    assert sha(ROOT.parent/"feedback/15_P1_W1准入决定与冻结协议.md")==revision["delivered_report_sha256"]
    payload=read(PROTO/"protocol_payload.json")
    assert sha(PROTO/"protocol_payload.json")==lock["protocol_payload_sha256"]
    for name,digest in payload["input_hashes"].items():
        if name.startswith("installed/"):
            file=Path(sys.prefix)/"Lib/site-packages/stable_baselines3"/("ppo/ppo.py" if name.endswith("/ppo.py") else "common/policies.py")
        else:
            file=ROOT.parent/name
        assert sha(file)==digest,name
    for model in CFG["models"].values():
        assert sha(ROOT/model["path"])==model["sha256"]
    runtime=collect_system_info()
    expected={"numpy":"2.3.3","gymnasium":"1.2.1","stable_baselines3":"2.7.0","torch":"2.5.1"}
    for key,value in expected.items():
        assert runtime[key]==value,(key,runtime[key],value)
    assert sys.version.split()[0]=="3.12.12"
    assert config_digest(load_config(PROTO/"resolved_config.yaml"))==CFG["config_digest"]
    torch.set_num_threads(1)
    return runtime


def active_seal_path():
    pointer=OUT/"execution_seal_current.json"
    if not pointer.exists():
        return OUT/"execution_seal.json"
    current=read(pointer)
    assert current["filename"]=="execution_seal_revision1.json"
    path=OUT/current["filename"]
    assert sha(path)==current["sha256"]
    assert sha(OUT/"execution_seal.json")==current["original_seal_sha256"]
    return path


def verify_seal():
    verify_contract()
    seal=read(active_seal_path())
    assert seal["precheck_passed"]
    for name,digest in seal["execution_sources"].items():
        assert sha(ROOT/name)==digest,name
    assert sha(OUT/"precheck_checks.json")==seal["precheck_checks_sha256"]
    assert sha(ROOT.parent/"guide/006_P1有限预算W1可行性验证.md")==seal["execution_instruction_sha256"]


def inventory_audit():
    import tarfile,io
    proposed=set(range(33001,34031))|set(range(35001,35101))|set(range(39001,39011))|set(range(50001,50101))
    records=[]
    csv.field_size_limit(10_000_000)
    for path in sorted((ROOT/"results").rglob("*.csv")):
        with path.open(encoding="utf-8-sig",newline="") as stream:
            count,seen=scan(stream)
        if count:
            records.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=sha(path),rows=count,
                                labels=seen,hits=sorted(proposed.intersection(seen))))
    for path in sorted((ROOT/"archive").rglob("*.tar.gz")):
        with tarfile.open(path,"r:gz") as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".csv") and "episode" in member.name:
                    with io.TextIOWrapper(archive.extractfile(member),encoding="utf-8-sig",newline="") as stream:
                        count,seen=scan(stream)
                    if count:
                        records.append(dict(path=path.relative_to(ROOT).as_posix()+"::"+member.name,
                                            archive_sha256=sha(path),rows=count,labels=seen,hits=sorted(proposed.intersection(seen))))
    result=dict(utc=utc(),scope="All local results CSVs with scenario_seed; archive episode CSV members; local execution only",
                entries=len(records),hits=sum(len(row["hits"]) for row in records),records=records)
    write(OUT/"preexecution_inventory_audit.json",result)
    assert result["hits"]==0,"Proposed scene already consumed"
    return result


def defenders():
    return {seed:PPO.load(ROOT/CFG["models"][str(seed)]["path"],device="cpu") for seed in DEFENDERS}


def checkpoint(path,item):
    path=Path(path)
    if path.exists():
        raise RuntimeError("Never overwrite a completed episode")
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(".partial")
    with gzip.open(temporary,"wt",encoding="utf-8") as stream:
        json.dump(item,stream,ensure_ascii=False)
    os.replace(temporary,path)


def consolidate(folder,items):
    folder.mkdir(parents=True,exist_ok=True)
    for filename,key in (("raw_episodes.csv","episode"),("raw_steps.csv","steps")):
        rows=[]
        for item in items:
            labels={"defender_seed":item["defender_seed"],"attack_label":item["attack_label"]}
            rows.extend([{**row,**labels} for row in item[key]] if key=="steps" else [{**item[key],**labels}])
        fields=sorted(set().union(*(row.keys() for row in rows)))
        with (folder/filename).open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


class AuditCallback(BaseCallback):
    def __init__(self, env, folder):
        super().__init__()
        self.attack_env,self.folder=env,folder
        self.rollouts=0
        self.complete_items=[]
        self.start=time.perf_counter()

    def _on_step(self):
        raw=float(self.locals["actions"][0,0])
        clipped=float(self.locals["clipped_actions"][0,0])
        logprob=float(self.locals["log_probs"][0].detach().cpu())
        if not np.isfinite([raw,clipped,logprob]).all():
            raise RuntimeError("Nonfinite policy action/log probability")
        transition=self.attack_env.last_transition
        for row in transition["rows"]:
            row["ppo_raw_action"]=raw if not row["padded_failure"] else None
            row["ppo_clipped_action"]=clipped if not row["padded_failure"] else None
            row["ppo_old_log_prob_raw_action"]=logprob if not row["padded_failure"] else None
        while self.attack_env.completed:
            item=self.attack_env.completed.pop(0)
            item["attack_label"]=f"W1_seed{self.attack_env.run_seed}"
            item["episode"]["wall_seconds_cumulative"]=time.perf_counter()-self.start
            checkpoint(self.folder/"checkpoints"/f"{item['episode']['scenario_seed']}.json.gz",item)
            self.complete_items.append(item)
        return True

    def _on_rollout_end(self):
        self.rollouts+=1


class FinitePPO(PPO):
    """Same native update; stop immediately if an update produces invalid losses/weights."""
    def train(self):
        super().train()
        for key in ("train/loss","train/value_loss","train/policy_gradient_loss","train/entropy_loss"):
            value=self.logger.name_to_value.get(key)
            if value is not None and not np.isfinite(value):
                raise RuntimeError(f"Nonfinite PPO loss: {key}")
        if not all(torch.isfinite(parameter).all() for parameter in self.policy.parameters()):
            raise RuntimeError("Nonfinite PPO weights after update")


def create_ppo(env,seed):
    p=CFG["ppo"]
    return FinitePPO("MlpPolicy",env,seed=seed,device="cpu",n_steps=p["n_steps"],batch_size=p["batch_size"],
        n_epochs=p["n_epochs"],learning_rate=p["learning_rate"],gamma=p["gamma"],gae_lambda=p["gae_lambda"],
        ent_coef=p["ent_coef"],clip_range=p["clip_range"],clip_range_vf=p["clip_range_vf"],
        normalize_advantage=p["normalize_advantage"],vf_coef=p["vf_coef"],max_grad_norm=p["max_grad_norm"],
        use_sde=False,target_kl=None,policy_kwargs=dict(net_arch=p["policy_layers"],activation_fn=torch.nn.Tanh,
                    ortho_init=True,log_std_init=0.,optimizer_class=torch.optim.Adam,
                    optimizer_kwargs=dict(eps=1e-5)),verbose=0)


def train():
    verify_seal()
    frozen=defenders()
    model_records={}
    for seed in CFG["attack_run_seeds"]:
        folder=OUT/"training"/f"seed{seed}"
        ledger=CallLedger(folder/"call_ledger.jsonl",32768,1000)
        if ledger.calls or ledger.episodes:
            raise RuntimeError("Existing training cannot restart without an exact saved live-state continuation; mark incomplete")
        env=W1Env(load_config(PROTO/"resolved_config.yaml"),frozen,"training",ledger,run_seed=seed)
        callback=AuditCallback(env,folder)
        started=utc();clock=time.perf_counter()
        try:
            model=create_ppo(env,seed)
            model.learn(total_timesteps=32768,callback=callback,progress_bar=False)
            assert model.num_timesteps==ledger.calls==32768 and callback.rollouts==64
            assert model._n_updates==64*5
            assert all(torch.isfinite(parameter).all() for parameter in model.policy.parameters())
            partial=env.partial()
            if partial is not None:
                write(folder/"partial_episode.json",partial)
            path=OUT/"models"/f"W1_seed{seed}.zip"
            path.parent.mkdir(parents=True,exist_ok=True)
            model.save(path)
            consolidate(folder,callback.complete_items)
            record=dict(seed=seed,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),
                        started_utc=started,ended_utc=utc(),wall_seconds=time.perf_counter()-clock,
                        rollouts=callback.rollouts,ppo_updates=model._n_updates,**ledger.summary())
            write(folder/"runtime.json",record)
            model_records[str(seed)]=record
        except Exception as error:
            write(folder/"incomplete.json",dict(error=repr(error),utc=utc(),ledger=ledger.summary(),
                  partial=env.partial(),wall_seconds=time.perf_counter()-clock))
            raise
        finally:
            env.close()
        verify_seal()
    write(OUT/"attack_models_lock.json",dict(utc=utc(),models=model_records,execution_seal_sha256=sha(active_seal_path())))


def evaluate(stage):
    verify_seal()
    attack_lock=read(OUT/"attack_models_lock.json")
    assert set(attack_lock["models"])=={"21001","21002","21003"}
    for item in attack_lock["models"].values():
        assert item["sampler_calls"]==32768 and item["rollouts"]==64
        assert sha(ROOT/item["path"])==item["sha256"]
    if stage=="feasibility":
        scheme=read(OUT/"feasibility_scheme_lock.json")
        assert scheme["attack_models_lock_sha256"]==sha(OUT/"attack_models_lock.json")
        assert scheme["execution_seal_sha256"]==sha(active_seal_path())
    folder=OUT/stage
    if (folder/"call_ledger.jsonl").exists():
        raise RuntimeError("Fixed evaluation cannot silently replay an existing batch")
    scenes=range(34001,34031) if stage=="development" else range(35001,35101)
    cap=45000 if stage=="development" else 150000
    ledger=CallLedger(folder/"call_ledger.jsonl",cap,450 if stage=="development" else 1500)
    fixed=defenders()
    attacks={int(seed):PPO.load(ROOT/item["path"],device="cpu") for seed,item in attack_lock["models"].items()}
    items=[];clock=time.perf_counter()
    specs=[CFG["frozen_rule"],dict(id="F01_low_0p002",family="constant",power_w=.002)]
    arms=[(spec["id"],spec,None) for spec in specs]+[(f"W1_seed{seed}",None,model) for seed,model in attacks.items()]
    for label,spec,actor in arms:
        for defender in DEFENDERS:
            env=W1Env(load_config(PROTO/"resolved_config.yaml"),fixed,stage,ledger,defender_seed=defender)
            try:
                for scenario in scenes:
                    observation,_=env.reset(seed=scenario)
                    done=False
                    while not done:
                        action=actor.predict(observation,deterministic=True)[0] if actor else inverse_power(requested_power(env.attack_observation,spec))
                        observation,_,done,_,_=env.step(action)
                    item=env.completed.pop()
                    item["attack_label"]=label
                    checkpoint(folder/"checkpoints"/f"{label}_{defender}_{scenario}.json.gz",item)
                    items.append(item)
            except Exception as error:
                write(folder/"incomplete.json",dict(error=repr(error),ledger=ledger.summary(),partial=env.partial(),utc=utc()))
                raise
            finally:
                env.close()
    consolidate(folder,items)
    write(folder/"runtime.json",dict(utc=utc(),wall_seconds=time.perf_counter()-clock,**ledger.summary()))
    verify_seal()
    if stage=="development":
        registry=read(PROTO/"scenario_registry.json")
        write(OUT/"feasibility_scheme_lock.json",dict(utc=utc(),
            attack_models_lock_sha256=sha(OUT/"attack_models_lock.json"),
            execution_seal_sha256=sha(active_seal_path()),models=CFG["models"],
            frozen_rule=CFG["frozen_rule"],auxiliary_low_w=.002,
            scenes=[row for row in registry if row["block"]=="feasibility"],
            statistics_sha256=sha(PROTO/"statistics_v1.py"),primary=CFG["statistics"]))


def precheck():
    if (OUT/"precheck/call_ledger.jsonl").exists() or (OUT/"execution_seal.json").exists():
        raise RuntimeError("Precheck allowance is cumulative; no silent repeated batch")
    runtime=verify_contract()
    from check_w1_execution import pure_checks,paired_checks
    pure=pure_checks()
    audit=inventory_audit()
    fixed=defenders()
    results=paired_checks(sys.modules[__name__],fixed)
    results.update(pure=pure,inventory_audit_sha256=sha(OUT/"preexecution_inventory_audit.json"),runtime=runtime)
    write(OUT/"precheck_checks.json",results)
    verify_contract()
    archive=OUT/"execution_sources.zip"
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as z:
        for path in sources():
            z.write(path,path.relative_to(ROOT).as_posix())
    seal=dict(sealed_utc=utc(),precheck_passed=True,execution_sources={path.relative_to(ROOT).as_posix():sha(path) for path in sources()},
        precheck_checks_sha256=sha(OUT/"precheck_checks.json"),source_archive_sha256=sha(archive),
        protocol_payload_sha256=sha(PROTO/"protocol_payload.json"),original_protocol_lock_sha256=sha(PROTO/"lock_manifest.json"),
        report_revision_lock_sha256=sha(PROTO/"report_revision_lock.json"),config_sha256=sha(PROTO/"w1_config.json"),
        resolved_config_sha256=sha(PROTO/"resolved_config.yaml"),models=CFG["models"],runtime=runtime,
        execution_instruction="guide/006_P1有限预算W1可行性验证.md",
        execution_instruction_sha256=sha(ROOT.parent/"guide/006_P1有限预算W1可行性验证.md"),
        scene_audit_sha256=sha(OUT/"preexecution_inventory_audit.json"),device="cpu",torch_threads=torch.get_num_threads(),n_envs=1,
        accounting="write-ahead reserve calls; commit zero/one physical transition and padding separately; pending calls cannot replay",
        precheck_command=[sys.executable,str(Path(__file__)),"--stage","precheck"],
        future_commands={stage:[sys.executable,str(Path(__file__)),"--stage",stage] for stage in ("train","development","feasibility")},
        new_training_transitions=0,final_test_transitions=0)
    write(OUT/"execution_seal.json",seal)
    print(json.dumps(dict(passed=True,seal_sha256=sha(OUT/"execution_seal.json"),**results["ledger"]),ensure_ascii=False,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",choices=["precheck","train","development","feasibility"],required=True)
    stage=parser.parse_args().stage
    try:
        if stage=="precheck": precheck()
        elif stage=="train": train()
        else: evaluate(stage)
    except Exception as error:
        write(OUT/"errors"/f"{stage}_{int(time.time())}.json",dict(utc=utc(),stage=stage,error=repr(error),status="technical_precheck_or_execution_failure"))
        raise
