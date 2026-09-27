from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from securelink.config import load_config
from securelink.environment import SecureLinkEnv


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def config() -> dict:
    return load_config(ROOT / "configs" / "motion_residual_v1.yaml")


def test_reference_uses_current_position_and_ignores_privileged_channel(config: dict) -> None:
    env = SecureLinkEnv(config, method="B10")
    env.reset(seed=20001)
    action = np.array([0.5, -0.5], dtype=np.float32)
    proposed, _, _, audit = env._parse_action(action)
    assert np.linalg.norm(audit["proposed_residual"]) == pytest.approx(
        env.max_step_m * config["mobility"]["residual_step_fraction"] * np.linalg.norm(action)
    )
    assert np.allclose(proposed, np.asarray(audit["reference_displacement"]) + np.asarray(audit["proposed_residual"]))
    assert env._current_gains is not None
    env._current_gains["g_aw"] *= 1e10
    env._current_gains["g_wb"] *= 1e10
    same, _, _, _ = env._parse_action(action)
    assert np.array_equal(proposed, same)

    _, _, _, _, _ = env.step(action)
    reference, status = env._reference_displacement()
    assert status == "current_state_safe_path"
    route, _, reason = env._future_plan(env.position, env.n_slots - env.slot)
    assert reason == "ok" and route is not None
    assert np.allclose(env.position + reference, route[0])


def test_zero_residual_baseline_completes_and_records_reference(config: dict) -> None:
    env = SecureLinkEnv(config, method="B10")
    env.reset(seed=20002)
    done = False
    info = {}
    while not done:
        _, _, done, _, info = env.step(np.zeros(2, dtype=np.float32))
    summary = info["episode_summary"]
    assert summary["task_completed"] == 1
    assert summary["executed_hard_violations"] == 0
    assert len(env.step_records) == env.n_slots
    assert all(row["motion_action_mode"] == "reference_residual" for row in env.step_records)
    assert all(abs(row["proposed_residual_dx_m"]) < 1e-12 for row in env.step_records)
    assert all(abs(row["proposed_residual_dy_m"]) < 1e-12 for row in env.step_records)


def test_direct_action_remains_v2_mapping(config: dict) -> None:
    config["mobility"]["motion_action_mode"] = "direct"
    env = SecureLinkEnv(config, method="B10")
    env.reset(seed=20003)
    displacement, _, _, audit = env._parse_action(np.array([0.25, -0.5], dtype=np.float32))
    assert np.allclose(displacement, np.array([0.25, -0.5]) * env.max_step_m)
    assert audit["reference_source"] == "direct_origin"
