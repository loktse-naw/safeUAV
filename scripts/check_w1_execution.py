"""Frozen-interface precheck: pure/synthetic checks, then exactly 15 paired replays."""
import ast
import inspect
import math
from pathlib import Path
import tempfile
import textwrap
import time
from types import SimpleNamespace
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from gymnasium import spaces
from w1_adapter import encode,power_request,inverse_power,PendingPower,W1Env,KEYS
from w1_ledger import CallLedger
from adaptive_rule_adapter import RuleCallback,AuditedRuleEnv,requested_power
from securelink.seeding import SeedDescriptor


def pure_checks():
    cases=0
    for power in (0,.0001,.001,.002,.004,.01,.02,.1,.3,1):
        actual,_=power_request(inverse_power(power),1.)
        assert abs(actual-power)<1e-12
        cases+=1
    for u in (-4.,-1.,-.8,0.,.8,1.,4.):
        p,clipped=power_request([u],1.)
        expected=.001*((1001.)**((min(1.,max(-1.,u))+1)/2)-1)
        assert abs(p-expected)<1e-12 and -1<=clipped<=1
        for remaining in (0.,.0001,.01,6.):
            actual=min(p,1.,remaining/.2)
            assert 0<=actual<=1+1e-12 and actual*.2<=remaining+1e-12
            cases+=1
    for bad in ([float("nan")],[float("inf")],[],[0,1]):
        try: power_request(bad,1.)
        except ValueError: pass
        else: raise AssertionError("Invalid action accepted")
    obs=dict(slot=33,remaining_slots=67,willie_remaining_j=6.,willie_peak_w=1.,alice_x_est_m=-25.,alice_y_est_m=1100.)
    vector=encode(obs)
    assert vector.shape==(6,) and vector.dtype==np.float32 and vector[4]<0 and vector[5]>1
    hidden={"g_ab":1.,"current_defender_action":[1,1],"defender_seed":11001,
            "reward_history":[-100],"logs":object(),"scenario_id":50001}
    for key,value in hidden.items():
        try: encode({**obs,key:value})
        except ValueError: pass
        else: raise AssertionError("Forbidden observation field accepted")
    hidden.update(g_ab=1e20,current_defender_action=[-1,-1],defender_seed=11003)
    assert np.array_equal(vector,encode(obs))
    for function in (encode,power_request,inverse_power):
        assert not inspect.getclosurevars(function).nonlocals
        names={node.id for node in ast.walk(ast.parse(inspect.getsource(function))) if isinstance(node,ast.Name)}
        assert not names.intersection({"env","defenders","reward","logs","eval","getattr"})
    callback=PendingPower();callback.value=.02
    assert set(vars(callback))=={"value","calls"} and callback(obs)==.02
    # Native policy probability path: no optimizer.step and no physical environment.
    torch.manual_seed(21001)
    policy=ActorCriticPolicy(spaces.Box(-np.inf,np.inf,shape=(6,),dtype=np.float32),
            spaces.Box(-1.,1.,shape=(1,),dtype=np.float32),lambda _:0.0003,net_arch=[64,64])
    before=[parameter.detach().clone() for parameter in policy.parameters()]
    tensor=torch.as_tensor(np.tile(vector,(64,1)))
    with torch.no_grad():
        raw,values,old_logprob=policy(tensor,deterministic=False)
        _,recomputed,_=policy.evaluate_actions(tensor,raw)
        _,clipped_logprob,_=policy.evaluate_actions(tensor,torch.clamp(raw,-1,1))
    assert torch.allclose(old_logprob,recomputed,atol=1e-6)
    mask=(raw.abs()>1).flatten()
    assert mask.any() and not torch.allclose(old_logprob[mask],clipped_logprob[mask])
    buffer=RolloutBuffer(512,policy.observation_space,policy.action_space,device="cpu",gae_lambda=.95,gamma=1.,n_envs=1)
    selected=int(torch.where(mask)[0][0])
    buffer.add(vector.reshape(1,6),raw[selected:selected+1].numpy(),np.array([-.5]),np.array([True]),
               values[selected:selected+1],old_logprob[selected:selected+1])
    assert buffer.actions[0,0,0]==float(raw[selected,0])
    assert all(torch.equal(old,new) for old,new in zip(before,policy.parameters()))
    assert not policy.optimizer.state
    tree=ast.parse(textwrap.dedent(inspect.getsource(OnPolicyAlgorithm.collect_rollouts)))
    calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=="add"]
    assert any(len(call.args)>2 and isinstance(call.args[1],ast.Name) and call.args[1].id=="actions" for call in calls)
    # Synthetic durable accounting and wrapper failure: no SecureLinkEnv/reset.
    with tempfile.TemporaryDirectory(prefix="w1_synthetic_") as folder:
        ledger=CallLedger(Path(folder)/"counts.jsonl",5,3)
        ledger.begin_episode(synthetic=True)
        ledger.reserve(synthetic=True);ledger.commit(1,0)
        ledger.reserve(synthetic=True);ledger.commit(0,99)
        reopened=CallLedger(ledger.path,5,3)
        assert reopened.summary()["sampler_calls"]==2 and reopened.real==1 and reopened.padded==99
        reopened.reserve(synthetic=True)
        pending=CallLedger(ledger.path,5,3)
        try: pending.reserve(synthetic=True)
        except RuntimeError: pass
        else: raise AssertionError("Interrupted call silently replayed")
        pending.commit(1,0)
        pending.reserve(synthetic=True);pending.commit(1,0)
        pending.reserve(synthetic=True);pending.commit(1,0)
        try: pending.reserve(synthetic=True)
        except RuntimeError: pass
        else: raise AssertionError("Budget overshoot")
        for physical in (0,1):
            trial=CallLedger(Path(folder)/f"fake{physical}.jsonl",1,1)
            trial.begin_episode(synthetic=True)
            env=W1Env({}, {},"precheck",trial)
            class Spy:
                def predict(self, observation, deterministic):
                    assert env.decision_events==["attack_committed"] and deterministic
                    assert observation.shape==(18,)
                    return np.zeros(2,dtype=np.float32),None
            env.defenders={11001:Spy()}
            row=dict(padded_failure=0 if physical else 1,secrecy_rate_bpshz=.4 if physical else 0.,willie_power_w=.02 if physical else 0.)
            class Fake:
                def __init__(self):
                    self.episode_summary=None;self.step_records=[];self.slot=0;self.seed_value=39001
                    self.attack_policy=PendingPower();self.cached=obs;self.executed_hard_violations=0;self.willie_remaining_j=6.
                def step(self, action):
                    self.step_records.append(dict(row))
                    self.step_records.extend(dict(padded_failure=1,secrecy_rate_bpshz=0.,willie_power_w=0.) for _ in range(99))
                    self.episode_summary={"real_slots":physical,"padded_failure_slots":100-physical}
                    return np.zeros(18,dtype=np.float32),.4 if physical else 0.,True,False,{"hidden":object()}
            env.base=Fake();env.attack_observation=obs;env.defender_observation=np.zeros(18,dtype=np.float32)
            next_observation,reward,done,_,info=env.step(inverse_power(.02))
            assert done and info=={} and np.array_equal(next_observation,np.zeros(6,dtype=np.float32))
            assert reward==(-.4 if physical else 0.) and trial.real==physical and trial.padded==100-physical
            try: env.step(np.zeros(1))
            except RuntimeError: pass
            else: raise AssertionError("Post-terminal extra step")
            assert trial.calls==1 and env.partial() is None
        unfinished=W1Env({}, {},"precheck",ledger)
        unfinished.base=SimpleNamespace(episode_summary=None,seed_value=33001,step_records=[dict(row)])
        partial=unfinished.partial()
        assert partial["partial"] and len(partial["steps"])==1 and ledger.calls==2
    return dict(passed=True,mapping_cases=cases,extra_fields_rejected=len(hidden),
        policy_actor_critic_input_shape=[6],raw_log_prob_recomputed=True,
        clipped_log_prob_differs_outside_box=True,native_rollout_keeps_raw_action=True,
        policy_optimizer_steps=0,policy_parameters_unchanged=True,
        synthetic_failure_terminal_partial_budget_recovery_passed=True,
        ambiguous_call_replay_rejected=True,hidden_info_not_returned_to_policy=True,
        synthetic_counts_not_in_physical_budget=True)


def paired_checks(driver,models):
    ledger=driver.CallLedger(driver.OUT/"precheck/call_ledger.jsonl",3000,30)
    items=[];pairs=[];clock=time.perf_counter()
    config=driver.load_config(driver.PROTO/"resolved_config.yaml")
    spec=driver.CFG["frozen_rule"]
    try:
        for defender in driver.DEFENDERS:
            for scenario in range(39001,39006):
                ledger.begin_episode(arm="original_T10",scenario=scenario,defender=defender)
                base=AuditedRuleEnv(config,method="B10",attack_rule="T10",
                    stream_context=dict(purpose="p1_w1_interface_check_v1",run_seed=0,worker_id=0))
                base.attack_policy=RuleCallback(spec,None)
                observation,_=base.reset(seed=scenario)
                done=False
                while not done:
                    action=models[defender].predict(observation,deterministic=True)[0]
                    before=len(base.step_records)
                    ledger.reserve(arm="original_T10",scenario=scenario,slot=base.slot,defender=defender)
                    observation,_,done,_,_=base.step(action)
                    appended=base.step_records[before:]
                    ledger.commit(sum(1-row["padded_failure"] for row in appended),sum(row["padded_failure"] for row in appended))
                original=dict(episode=dict(base.episode_summary),steps=[dict(row) for row in base.step_records],
                              defender_seed=defender,attack_label="original_T10")
                driver.checkpoint(driver.OUT/"precheck/checkpoints"/f"original_{defender}_{scenario}.json.gz",original)
                items.append(original)
                base.close()
                adapter=driver.W1Env(config,models,"precheck",ledger,defender_seed=defender)
                observation,_=adapter.reset(seed=scenario)
                done=False
                while not done:
                    n=int(round(float(observation[0])*100))
                    request=.02 if 33<=n<67 else .004
                    action=driver.inverse_power(request)
                    observation,_,done,_,info=adapter.step(action)
                    assert info=={} and adapter.decision_events==["attack_committed","defender_predicted","physical_step"]
                adapted=adapter.completed.pop()
                adapted["attack_label"]="adapter_T10_inverse"
                driver.checkpoint(driver.OUT/"precheck/checkpoints"/f"adapter_{defender}_{scenario}.json.gz",adapted)
                items.append(adapted)
                rows0,rows1=original["steps"],adapted["steps"]
                assert len(rows0)==len(rows1)==100
                assert original["episode"]["executed_hard_violations"]==adapted["episode"]["executed_hard_violations"]==0
                assert original["episode"]["real_slots"]==adapted["episode"]["real_slots"]==100
                assert adapter.base.noise_draw_calls==adapter.base.attack_policy.calls==100
                numeric=["x_m","y_m","next_x_m","next_y_m","alice_power_w","willie_power_w","g_ab","g_aw","g_wb",
                         "secrecy_rate_bpshz","battery_after_j","radiated_after_j","slot_energy_j",
                         "policy_action_0","policy_action_1","executed_dx_m","executed_dy_m"]
                max_differences={key:max(abs(float(a[key])-float(b[key])) for a,b in zip(rows0,rows1)) for key in numeric}
                for key,error in max_differences.items():
                    assert error<= (1e-7 if key in {"x_m","y_m","next_x_m","next_y_m","executed_dx_m","executed_dy_m"} else 1e-8),(key,error)
                for a,b in zip(rows0,rows1):
                    assert a["padded_failure"]==b["padded_failure"] and a["fallback_takeover"]==b["fallback_takeover"]
                    for key in ("alice_x_est_m","alice_y_est_m"):
                        assert abs(a[key]-b["attack_observation"][key])<=1e-10
                errors=np.array([[row["attack_observation"]["alice_x_est_m"]-row["x_m"],
                                  row["attack_observation"]["alice_y_est_m"]-row["y_m"]] for row in rows1])
                child=SeedDescriptor("p1_w1_interface_check_v1",0,0,0,scenario).seed_sequence().spawn(3)[2]
                expected=np.random.Generator(np.random.PCG64(child)).normal(size=(100,2))*25
                noise_error=float(np.max(abs(errors-expected)))
                assert noise_error<=1e-10
                for row in rows1:
                    assert len(row["attack_input"])==6 and set(row["attack_observation"])==set(KEYS)
                pairs.append(dict(defender=defender,scenario=scenario,passed=True,
                    numeric_max_errors=max_differences,noise_max_error_m=noise_error,
                    observation_calls=adapter.base.noise_draw_calls,callback_calls=adapter.base.attack_policy.calls,
                    asr_original=original["episode"]["asr_bpshz"],asr_adapter=adapted["episode"]["asr_bpshz"]))
                adapter.close()
                print(f"precheck: pair {len(pairs)}/15, calls={ledger.calls}",flush=True)
        assert ledger.calls==ledger.real==3000 and ledger.episodes==30 and ledger.padded==0
        driver.consolidate(driver.OUT/"precheck",items)
        driver.write(driver.OUT/"precheck/pair_comparisons.json",pairs)
        return dict(passed=True,utc=driver.utc(),pairs=15,episodes=30,ledger=ledger.summary(),
                    hard_violations=0,wall_seconds=time.perf_counter()-clock,
                    once_per_slot=True,noise_sequence_matches=True,attack_precedes_defender=True,
                    whitelist=driver.CFG["white_list"],scenarios=list(range(39001,39006)),
                    reserved_precheck_labels_not_used=list(range(39006,39011)),training_optimizer_steps=0)
    except Exception as error:
        if items:
            driver.consolidate(driver.OUT/"precheck",items)
        driver.write(driver.OUT/"precheck/failed_checks.json",dict(error=repr(error),ledger=ledger.summary(),
                     complete_pairs=pairs,wall_seconds=time.perf_counter()-clock))
        raise
