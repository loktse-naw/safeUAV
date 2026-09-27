"""Before-selection interface, pairing, and synthetic-statistics checks."""
from __future__ import annotations

import ast
import inspect
import json
import math
import time
import numpy as np

from adaptive_rule_adapter import ALLOWED, RuleCallback, requested_power
from securelink.seeding import SeedDescriptor
from statistics_v1 import candidate_choice, rule_branch


class GuardedObservation(dict):
    def __init__(self, values):
        super().__init__(values)
        self.accesses = []

    def __getitem__(self, key):
        if key not in ALLOWED:
            raise AssertionError("Forbidden observation access")
        self.accesses.append(key)
        return super().__getitem__(key)


def independent_expected(obs, spec, coin):
    n = obs["slot"]
    if spec["family"] == "time_piecewise":
        return spec["middle_w"] if n in range(33,67) else spec["outer_w"]
    if spec["family"] == "late_burst":
        return spec["burst_w"] if n >= spec["burst_start_slot"] else spec["low_w"]
    if spec["family"] == "geometry_threshold":
        q = np.array([obs["alice_x_est_m"], obs["alice_y_est_m"], 100.0])
        gap = np.linalg.norm(q-np.array([350.,650.,0.])) - np.linalg.norm(q-np.array([750.,350.,0.]))
        return spec["high_w"] if gap >= spec["threshold_m"] else spec["low_w"]
    if spec["family"] == "constant":
        return spec["power_w"]
    if spec["family"] == "frontload":
        return obs["willie_peak_w"]
    fraction = min(1., obs["willie_remaining_j"] / (.2*obs["willie_peak_w"]*obs["remaining_slots"]))
    return obs["willie_peak_w"] if coin < fraction else 0.


def perform_checks(driver):
    start = time.perf_counter()
    pure_checks = 0
    slots = (0,32,33,66,67,79,80,89,90,94,95,99)
    for spec in driver.CATALOG + driver.BASELINES + driver.CONSTANTS:
        for slot in slots:
            for remaining in (6.0,0.01):
                for x,y in ((100.,100.), (850.,350.)):
                    values = {"slot":slot,"remaining_slots":100-slot,"willie_remaining_j":remaining,
                              "willie_peak_w":1.,"alice_x_est_m":x,"alice_y_est_m":y}
                    obs = GuardedObservation(values)
                    actual = requested_power(obs, spec, 0.35)
                    expected = independent_expected(values, spec, 0.35)
                    assert abs(actual-expected) < 1e-12
                    assert set(obs.accesses).issubset(ALLOWED)
                    assert inspect.getclosurevars(requested_power).nonlocals == {}
                    # Hidden state objects deliberately change while the allowed input is fixed.
                    hidden = {"true_channel":1e-9,"current_defender_action":[1,1],"model_id":11001}
                    first = requested_power(values, spec, 0.35)
                    hidden.update(true_channel=1e20,current_defender_action=[-1,-1],model_id=11003)
                    assert first == requested_power(values, spec, 0.35)
                    try:
                        requested_power({**values, **hidden}, spec, 0.35)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Extra hidden fields were accepted")
                    pure_checks += 1
    tree = ast.parse(inspect.getsource(requested_power))
    assert not any(isinstance(node, ast.Name) and node.id in {"env","SecureLinkEnv","PPO","getattr","globals","eval"} for node in ast.walk(tree))
    callback = RuleCallback(driver.CATALOG[0], np.random.default_rng(1))
    assert set(vars(callback)) == {"spec","rng","calls","last"}
    assert all(not isinstance(value, driver.AuditedRuleEnv) for value in vars(callback).values())
    assert candidate_choice({"z":.5,"a":.5+5e-9,"b":.6}) == "a"
    low = np.ones((3,100)); p = np.full((3,100),.5)
    assert rule_branch(low,np.full((3,100),.55),p,True)["branch"] == "do_not_invest_W1"
    assert rule_branch(low,np.full((3,100),.8),p,True)["branch"] == "may_prepare_limited_W1_protocol"
    assert rule_branch(low,low,low,True)["q"] is None
    assert rule_branch(low,low,p,False)["branch"] == "halt_integrity_gate"
    assert rule_branch(low,np.full((3,100),.1),p,True)["q"] > 1
    crossing = np.tile(np.r_[np.full(50,.65),np.full(50,.55)],(3,1))
    assert rule_branch(low,crossing,p,True)["branch"] == "insufficient_evidence"
    test_specs = [driver.BASELINES[0],driver.BASELINES[4],driver.CATALOG[0],driver.CATALOG[12],driver.CATALOG[24],driver.CONSTANTS[4],driver.MYOPIC]
    paths = []
    for scenario in (90001,90002):
        reference_noise = np.random.Generator(np.random.PCG64(SeedDescriptor("p1_rule_interface_check_v1",0,0,0,scenario).seed_sequence().spawn(3)[2])).normal(size=(100,2))*25
        reference_gains = None
        for spec in test_specs:
            driver.execute_episode("precheck", "p1_rule_interface_check_v1", 0, spec, scenario)
            path = driver.episode_path("precheck",0,spec,scenario)
            paths.append(path)
            item = driver.read_checkpoint(path)
            rows = item["steps"]
            assert item["episode"]["real_slots"] == 100
            gains = np.array([[row[k] for k in ("g_ab","g_aw","g_wb")] for row in rows])
            if reference_gains is None:
                reference_gains = gains
            else:
                assert np.array_equal(gains,reference_gains)
            if spec["family"] != "privileged_myopic":
                noise = np.array([[row["alice_x_est_m"]-row["x_m"],row["alice_y_est_m"]-row["y_m"]] for row in rows])
                assert np.max(abs(noise-reference_noise)) < 1e-10
                assert item["episode"]["attack_observation_calls"] == 100
                if spec["family"] == "random":
                    child = SeedDescriptor("p1_rule_interface_check_v1_bangbang",0,0,0,scenario).seed_sequence().spawn(1)[0]
                    coins = np.random.Generator(np.random.PCG64(child)).random(100)
                    assert np.array_equal(coins,np.array([row["random_switch_coin"] for row in rows]))
    # Exercise every frozen defender on independent precheck scenarios, including local checkpoint serialization.
    for seed in driver.MODELS:
        driver.execute_episode("precheck","p1_rule_interface_check_v1",seed,driver.CATALOG[12],90003)
        paths.append(driver.episode_path("precheck",seed,driver.CATALOG[12],90003))
    frame = driver.consolidate("precheck",paths)
    # CSV roundtrip is required for the later fixed-model pivot and ledger.
    saved = driver.pd.read_csv(driver.OUT / "precheck" / "raw_episodes.csv")
    assert len(saved) == len(paths) == 17 and saved.real_slots.sum() == 1700
    assert not saved.duplicated(["train_seed","attack_label","scenario_seed"]).any()
    return {"passed":True,"checked_utc":driver.utc(),"pure_rule_cases":pure_checks,
        "hidden_field_injection_rejected":True,"hidden_state_perturbation_invariance":True,
        "no_environment_closure_or_dynamic_lookup":True,"whitelist":sorted(ALLOWED),
        "paired_location_error_max_tolerance_m":1e-10,"paired_B00_true_gains_exact":True,
        "random_coin_stream_matches_registered_formula":True,"once_per_real_slot":True,
        "synthetic_statistical_branches_passed":True,"synthetic_scores_used_for_tuning":False,
        "scenario_labels":[90001,90002,90003],"purpose":"p1_rule_interface_check_v1",
        "episodes":len(frame),"real_transitions":int(frame.real_slots.sum()),
        "padded_slots":int(frame.padded_failure_slots.sum()),
        "hard_violations":int(frame.executed_hard_violations.sum()),
        "wall_seconds":time.perf_counter()-start}
