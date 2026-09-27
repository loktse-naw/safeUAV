from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from securelink.config import load_config
from securelink.environment import ScenarioConfigurationError, SecureLinkEnv
from securelink.evaluation import dataframe_digest, evaluate_policy
from securelink.geometry import segment_clear_of_circle
from securelink.physics import channel_gains, secrecy_metrics, slot_energy_j
from securelink.seeding import SeedDescriptor


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def config() -> dict:
    return load_config(ROOT / "configs" / "base.yaml")


def test_zero_power_and_capacity_clipping(config: dict) -> None:
    near_willie = np.asarray(config["scenario"]["willie_m"], dtype=float)
    gains = channel_gains(near_willie, config, fading=False)
    zero = secrecy_metrics(0.0, 0.3, gains, config)
    assert zero["secrecy_rate_bpshz"] == 0.0
    negative = secrecy_metrics(0.2, 0.0, gains, config)
    assert negative["capacity_difference_bpshz"] < 0.0
    assert negative["secrecy_rate_bpshz"] == 0.0


def test_eavesdropper_off(config: dict) -> None:
    gains = channel_gains(np.array([650.0, 350.0]), config, fading=False)
    metrics = secrecy_metrics(0.2, 0.3, gains, config, disable_eavesdropper=True)
    assert metrics["gamma_w"] == 0.0
    assert metrics["capacity_w_bpshz"] == 0.0
    assert metrics["secrecy_rate_bpshz"] == pytest.approx(metrics["capacity_b_bpshz"])


def test_nfz_segment_not_only_endpoint(config: dict) -> None:
    center = np.asarray(config["scenario"]["nfz_center_m"], dtype=float)
    radius = config["scenario"]["nfz_radius_m"] + config["scenario"]["nfz_margin_m"]
    assert not segment_clear_of_circle(np.array([300.0, 500.0]), np.array([700.0, 500.0]), center, radius)


def test_reference_episode_units_and_constraints(config: dict) -> None:
    env = SecureLinkEnv(config, method="B00", attack_rule="uniform")
    summary, steps = _episode(env, 123)
    assert len(steps) == config["time"]["slots"]
    assert summary["task_completed"] == 1
    assert summary["executed_hard_violations"] == 0
    assert summary["alice_total_energy_j"] == pytest.approx(sum(row["slot_energy_j"] for row in steps))
    expected_bits = (
        config["channel"]["bandwidth_hz"]
        * config["time"]["slot_s"]
        * config["time"]["tx_duty"]
        * sum(row["secrecy_rate_bpshz"] for row in steps)
    )
    assert summary["secure_bits"] == pytest.approx(expected_bits)


def _episode(env: SecureLinkEnv, seed: int):
    obs, _ = env.reset(seed=seed)
    done = False
    info = {}
    while not done:
        obs, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
        done = terminated or truncated
    return info["episode_summary"], env.step_records


def test_failure_is_padded_not_filtered(config: dict) -> None:
    env = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    env.reset(seed=88)
    env.force_battery_for_test(1.0)
    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    summary = info["episode_summary"]
    assert terminated
    assert summary["task_failure"] == 1
    assert summary["slots_recorded"] == config["time"]["slots"]
    assert summary["padded_failure_slots"] == config["time"]["slots"]
    assert summary["asr_bpshz"] == 0.0
    assert summary["slot_sop"] == 1.0


def test_unreachable_configuration_rejected(config: dict) -> None:
    invalid = copy.deepcopy(config)
    invalid["time"]["slots"] = 40
    with pytest.raises(ScenarioConfigurationError):
        SecureLinkEnv(invalid, method="B00")


def test_reproducible_replay(config: dict) -> None:
    _, steps_a = evaluate_policy(config, "B00", "uniform", [777])
    _, steps_b = evaluate_policy(config, "B00", "uniform", [777])
    _, steps_c = evaluate_policy(config, "B00", "uniform", [778])
    assert dataframe_digest(steps_a) == dataframe_digest(steps_b)
    assert dataframe_digest(steps_a) != dataframe_digest(steps_c)


def test_automatic_training_seed_increments(config: dict) -> None:
    env = SecureLinkEnv(
        config,
        method="B11",
        attack_rule="uniform",
        stream_context={"purpose": "train", "run_seed": 900, "worker_id": 0},
    )
    env.reset(seed=900)
    assert env.seed_value == 900
    _, info_a = env.reset()
    _, info_b = env.reset()
    assert info_a["seed_key"] != info_b["seed_key"]
    assert info_a["seed_fingerprint"] != info_b["seed_fingerprint"]


def test_seed_streams_do_not_overlap_across_worker_or_purpose(config: dict) -> None:
    descriptors = [
        SeedDescriptor(purpose, run_seed, worker, episode)
        for purpose in ("train", "development", "final_test")
        for run_seed in (11, 23)
        for worker in range(4)
        for episode in range(20)
    ]
    fingerprints = [descriptor.fingerprint() for descriptor in descriptors]
    assert len(fingerprints) == len(set(fingerprints))


def test_observation_contract_and_privileged_information_isolation(config: dict) -> None:
    env = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    observation, _ = env.reset(seed=321)
    contract = env.observation_contract()
    assert contract["dimension"] == observation.shape[0] == 18
    assert contract["feedback_mode"] == "exact_one_slot_delayed"
    assert observation[10] == pytest.approx(0.0)
    before = observation.copy()
    assert env._current_gains is not None
    env._current_gains["g_aw"] *= 1e9
    env._current_gains["g_wb"] *= 1e9
    after = env._observation()
    assert np.array_equal(before, after)
    forbidden = set(contract["forbidden"])
    field_names = {field["name"] for field in contract["fields"]}
    assert forbidden.isdisjoint(field_names)


def test_attack_callback_receives_only_allowlisted_observation(config: dict) -> None:
    captured: dict = {}

    def attack_policy(observation: dict) -> float:
        captured.update(observation)
        return 0.0

    env = SecureLinkEnv(config, method="B00", attack_policy=attack_policy)
    env.reset(seed=322)
    env.step(np.zeros(1, dtype=np.float32))
    assert set(captured) == {
        "slot",
        "remaining_slots",
        "willie_remaining_j",
        "willie_peak_w",
        "alice_x_est_m",
        "alice_y_est_m",
    }
    assert not {"g_ab", "g_aw", "g_wb", "gamma_b", "gamma_w"}.intersection(captured)


def test_future_plan_certificate_edge_cases(config: dict) -> None:
    env = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    env.reset(seed=400)
    nodes, energy, reason = env._future_plan(env.goal, 0)
    assert nodes == [] and energy == 0.0 and reason == "ok"
    nodes, _, reason = env._future_plan(env.start, 0)
    assert nodes is None and reason == "time_unreachable"

    near_goal = env.goal - np.array([0.5, 0.0])
    nodes, energy, reason = env._future_plan(near_goal, 1)
    assert reason == "ok" and nodes is not None and len(nodes) == 1 and energy > 0.0

    boundary = np.array([0.0, 10.0])
    nodes, _, reason = env._future_plan(boundary, env.n_slots)
    assert nodes is not None and reason == "ok"
    replay = boundary.copy()
    for node in nodes:
        assert np.linalg.norm(node - replay) <= env.max_step_m + 1e-8
        assert segment_clear_of_circle(replay, node, env.nfz_center, env.planning_radius, 1e-9)
        replay = node
    assert np.linalg.norm(replay - env.goal) <= env.terminal_tolerance


def test_critical_energy_certificate_replays_to_goal(config: dict) -> None:
    env = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    env.reset(seed=401)
    nodes, exact_energy, reason = env._future_plan(env.position, env.n_slots)
    assert nodes is not None and reason == "ok"
    env.force_battery_for_test(exact_energy + 1e-7)
    terminated = False
    while not terminated:
        _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    summary = info["episode_summary"]
    assert summary["task_completed"] == 1
    assert summary["executed_hard_violations"] == 0
    assert summary["alice_total_energy_j"] <= exact_energy + 1e-6


def test_fallback_trigger_takeover_and_safe_exit_are_separate(config: dict) -> None:
    env = SecureLinkEnv(config, method="B11", attack_rule="uniform")
    env.reset(seed=402)
    trigger_seen = False
    exit_seen = False
    for _ in range(env.n_slots):
        if env.forced_fallback_mode and env.guarantee_nodes:
            displacement = env.guarantee_nodes[0] - env.position
            action = np.array(
                [displacement[0] / env.max_step_m, displacement[1] / env.max_step_m, -1.0],
                dtype=np.float32,
            )
        else:
            action = np.zeros(3, dtype=np.float32)
        _, _, terminated, _, _ = env.step(action)
        row = env.step_records[-1]
        trigger_seen = trigger_seen or bool(row["fallback_trigger"])
        exit_seen = exit_seen or bool(row["fallback_exit"])
        if terminated:
            break
    assert trigger_seen
    assert exit_seen
    assert env.fallback_independent_triggers >= 1
    assert env.fallback_takeover_slots >= env.fallback_independent_triggers
    assert env.step_records[-1]["executed_hard_violation"] == 0


def test_centered_power_mapping_zero_means_uniform_remaining_budget(config: dict) -> None:
    centered = copy.deepcopy(config)
    centered["power"]["action_mapping"] = "remaining_budget_centered"
    env = SecureLinkEnv(centered, method="B01", attack_rule="uniform")
    env.reset(seed=403)
    _, power, _, _ = env._parse_action(np.zeros(1, dtype=np.float32))
    expected = centered["power"]["alice_radiated_budget_j"] / (
        centered["time"]["slots"] * centered["time"]["slot_s"] * centered["time"]["tx_duty"]
    )
    assert power == pytest.approx(expected)


def test_propulsion_energy_positive(config: dict) -> None:
    stationary = slot_energy_j(0.0, 0.0, config)
    moving = slot_energy_j(10.0, 0.2, config)
    assert stationary["total_j"] > 0.0
    assert moving["total_j"] > 0.0
    assert moving["speed_mps"] <= config["mobility"]["max_speed_mps"]
