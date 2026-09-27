"""Preserve first precheck seal; version training-only finite-loss guard, without replay."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import zipfile
import torch
import run_w1_feasibility_v1 as d


def main():
    if (d.OUT/"execution_seal_revision1.json").exists():
        raise RuntimeError("Revision already sealed; never overwrite")
    original=d.read(d.OUT/"execution_seal.json")
    assert original["precheck_passed"] and original["new_training_transitions"]==0
    assert d.sha(d.OUT/"execution_sources.zip")==original["source_archive_sha256"]
    d.verify_contract()
    for name,digest in original["execution_sources"].items():
        if name!="scripts/run_w1_feasibility_v1.py":
            assert d.sha(d.ROOT/name)==digest,name
    assert d.sha(d.OUT/"precheck_checks.json")==original["precheck_checks_sha256"]
    with zipfile.ZipFile(d.OUT/"execution_sources.zip") as archive:
        old=archive.read("scripts/run_w1_feasibility_v1.py")
        import hashlib
        assert hashlib.sha256(old).hexdigest()==original["execution_sources"]["scripts/run_w1_feasibility_v1.py"]
    guard=d.FinitePPO.__new__(d.FinitePPO)
    guard._logger=SimpleNamespace(name_to_value={"train/loss":.1})
    guard.policy=torch.nn.Linear(6,1)
    cases=0
    with patch.object(d.PPO,"train",return_value=None):
        d.FinitePPO.train(guard);cases+=1
        for key in ("train/loss","train/value_loss","train/policy_gradient_loss","train/entropy_loss"):
            guard._logger.name_to_value={key:float("nan")}
            try: d.FinitePPO.train(guard)
            except RuntimeError: cases+=1
            else: raise AssertionError("Nonfinite loss not rejected")
        guard._logger.name_to_value={"train/loss":.1}
        with torch.no_grad(): guard.policy.weight[0,0]=float("inf")
        try: d.FinitePPO.train(guard)
        except RuntimeError: cases+=1
        else: raise AssertionError("Nonfinite weights not rejected")
    changed=[name for name,digest in original["execution_sources"].items() if d.sha(d.ROOT/name)!=digest]
    assert changed==["scripts/run_w1_feasibility_v1.py"]
    notes=dict(utc=d.utc(),reason="After first precheck seal, add finite-loss/weight stopping guard for future PPO updates and explicit versioned seal selection",
        changed_sources=changed,physical_adapter_ledger_checker_unchanged=True,
        synthetic_guard_cases=cases,real_optimizer_steps=0,new_environment_calls=0,
        precheck_not_replayed=True,original_seal_sha256=d.sha(d.OUT/"execution_seal.json"))
    d.write(d.OUT/"execution_revision_checks.json",notes)
    files=d.sources()+[Path(__file__)]
    archive_path=d.OUT/"execution_sources_revision1.zip"
    with zipfile.ZipFile(archive_path,"w",zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,path.relative_to(d.ROOT).as_posix())
    current=copy.deepcopy(original)
    current.update(sealed_utc=d.utc(),revision="execution-only-r1",predecessor="execution_seal.json",
        predecessor_sha256=d.sha(d.OUT/"execution_seal.json"),
        execution_sources={path.relative_to(d.ROOT).as_posix():d.sha(path) for path in files},
        source_archive="execution_sources_revision1.zip",source_archive_sha256=d.sha(archive_path),
        revision_checks_sha256=d.sha(d.OUT/"execution_revision_checks.json"),
        revision_command=[sys.executable,str(Path(__file__))],new_environment_calls_since_precheck=0,
        stop_guard="Native PPO update followed by finite loss/weight check; no altered hyperparameters or gradients")
    path=d.OUT/"execution_seal_revision1.json"
    d.write(path,current)
    d.write(d.OUT/"execution_seal_current.json",dict(filename=path.name,sha256=d.sha(path),
            original_seal_sha256=d.sha(d.OUT/"execution_seal.json")))
    d.verify_seal()
    print(dict(passed=True,current_seal_sha256=d.sha(path),synthetic_guard_cases=cases,new_environment_calls=0))


if __name__=="__main__":
    main()
