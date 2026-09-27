"""Observation-only rules and audit-only environment wrapper for protocol 13."""
from __future__ import annotations

import math
import numpy as np
from securelink.environment import SecureLinkEnv

ALLOWED = frozenset(("slot", "remaining_slots", "willie_remaining_j", "willie_peak_w", "alice_x_est_m", "alice_y_est_m"))
PUBLIC_MAP = ((750.0, 350.0), (350.0, 650.0), 100.0)


def requested_power(observation, spec, coin=None, public_map=PUBLIC_MAP):
    if set(observation) != ALLOWED:
        raise ValueError("Exactly six allowed observation fields required")
    family = spec["family"]
    n = int(observation["slot"])
    if family == "time_piecewise":
        return float(spec["middle_w"] if 33 <= n < 67 else spec["outer_w"])
    if family == "geometry_threshold":
        bob, willie, altitude = public_map
        x, y = float(observation["alice_x_est_m"]), float(observation["alice_y_est_m"])
        d_b = math.sqrt((x - bob[0])**2 + (y - bob[1])**2 + altitude**2)
        d_w = math.sqrt((x - willie[0])**2 + (y - willie[1])**2 + altitude**2)
        return float(spec["high_w"] if d_w - d_b >= spec["threshold_m"] else spec["low_w"])
    if family == "late_burst":
        return float(spec["low_w"] if n < spec["burst_start_slot"] else spec["burst_w"])
    if family == "constant":
        return float(spec["power_w"])
    if family == "frontload":
        return float(observation["willie_peak_w"])
    if family == "random":
        peak = float(observation["willie_peak_w"])
        fraction = min(1.0, float(observation["willie_remaining_j"]) /
                       max(0.2 * peak * int(observation["remaining_slots"]), 1e-12))
        if coin is None:
            raise ValueError("Random rule needs its independent internal coin")
        return peak if coin < fraction else 0.0
    raise ValueError(f"Unknown rule family {family}")


class RuleCallback:
    """Holds static parameters and an independent RNG, never an environment."""
    def __init__(self, spec, rng):
        self.spec = dict(spec)
        self.rng = rng
        self.calls = 0
        self.last = None

    def __call__(self, observation):
        coin = float(self.rng.random()) if self.spec["family"] == "random" else None
        power = requested_power(observation, self.spec, coin)
        self.calls += 1
        self.last = {**observation, "willie_requested_w": power, "random_switch_coin": coin}
        return power


class AuditedRuleEnv(SecureLinkEnv):
    """Adds offline logging only; uses the unchanged base transition and clipping."""
    def step(self, action):
        before = len(self.step_records)
        result = super().step(action)
        callback = self.attack_policy
        for row in self.step_records[before:]:
            if row["padded_failure"]:
                row.update({name: None for name in ("alice_x_est_m", "alice_y_est_m", "willie_requested_w", "random_switch_coin")})
            else:
                assert callback is not None and callback.last is not None
                row.update({name: callback.last[name] for name in ("alice_x_est_m", "alice_y_est_m", "willie_requested_w", "random_switch_coin")})
            request = row.get("willie_requested_w")
            row["willie_budget_clipped"] = int(request is not None and request > row["willie_power_w"] + 1e-12)
        return result
