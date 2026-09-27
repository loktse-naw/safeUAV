from __future__ import annotations

import copy
from collections import Counter
import math
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import config_digest
from .geometry import (
    point_in_bounds,
    polyline_length,
    resample_polyline,
    segment_clear_of_circle,
    segment_in_bounds,
    shortest_safe_polyline,
)
from .physics import (
    channel_gains,
    db_to_linear,
    secrecy_metrics,
    slot_energy_j,
)
from .seeding import SeedDescriptor, substream_fingerprints


AttackObservation = dict[str, float | int]
AttackPolicy = Callable[[AttackObservation], float]


class ScenarioConfigurationError(ValueError):
    pass


class SecureLinkEnv(gym.Env[np.ndarray, np.ndarray]):
    """Single-UAV SecureLink environment with hard feasibility enforcement.

    Transmission occurs at q[n], followed by movement to q[n+1]. An episode
    always contributes exactly N metric slots: early failures are padded with
    zero secrecy rates and outage indicators.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: dict[str, Any],
        method: str = "B11",
        attack_rule: str = "uniform",
        attack_policy: AttackPolicy | None = None,
        disable_eavesdropper: bool = False,
        stream_context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if method not in {"B00", "B01", "B10", "B11"}:
            raise ValueError(f"Unknown method: {method}")
        self.config = copy.deepcopy(config)
        self.method = method
        self.attack_rule = attack_rule
        self.attack_policy = attack_policy
        self.disable_eavesdropper = disable_eavesdropper
        context = stream_context or {}
        self.stream_purpose = str(context.get("purpose", "development"))
        self.stream_run_seed = int(context.get("run_seed", 0))
        self.stream_worker_id = int(context.get("worker_id", 0))
        self.digest = config_digest(self.config)

        self.n_slots = int(self.config["time"]["slots"])
        self.slot_s = float(self.config["time"]["slot_s"])
        self.tx_s = self.slot_s * float(self.config["time"]["tx_duty"])
        self.move_s = self.slot_s - self.tx_s
        self.max_step_m = float(self.config["mobility"]["max_speed_mps"]) * self.move_s
        self.motion_action_mode = str(self.config["mobility"].get("motion_action_mode", "direct"))
        if self.motion_action_mode not in {"direct", "reference_residual"}:
            raise ValueError(f"Unknown mobility.motion_action_mode: {self.motion_action_mode}")
        self.residual_step_fraction = float(
            self.config["mobility"].get("residual_step_fraction", 0.25)
        )
        if not 0.0 < self.residual_step_fraction <= 1.0:
            raise ValueError("mobility.residual_step_fraction must be in (0, 1]")
        self.area = np.asarray(self.config["scenario"]["area_m"], dtype=np.float64)
        self.start = np.asarray(self.config["scenario"]["start_m"], dtype=np.float64)
        self.goal = np.asarray(self.config["scenario"]["goal_m"], dtype=np.float64)
        self.bob = np.asarray(self.config["scenario"]["bob_m"], dtype=np.float64)
        self.willie = np.asarray(self.config["scenario"]["willie_m"], dtype=np.float64)
        self.nfz_center = np.asarray(self.config["scenario"]["nfz_center_m"], dtype=np.float64)
        self.nfz_radius = float(self.config["scenario"]["nfz_radius_m"]) + float(
            self.config["scenario"]["nfz_margin_m"]
        )
        self.geometry_epsilon = float(self.config["environment"]["geometry_epsilon_m"])
        self.path_clearance = float(self.config["environment"]["path_discretization_clearance_m"])
        self.planning_radius = self.nfz_radius + self.path_clearance
        self.terminal_tolerance = float(self.config["scenario"]["terminal_tolerance_m"])
        self.reference_polyline, self.reference_length_m = shortest_safe_polyline(
            self.start,
            self.goal,
            self.nfz_center,
            self.planning_radius,
            epsilon_m=self.path_clearance,
        )
        self.reference_nodes = resample_polyline(self.reference_polyline, self.n_slots + 1)
        if np.max(np.linalg.norm(np.diff(self.reference_nodes, axis=0), axis=1)) > self.max_step_m + 1e-8:
            raise ScenarioConfigurationError("Reference path cannot reach the goal within N slots")
        reference_energy = sum(
            slot_energy_j(float(np.linalg.norm(delta)), float(self.config["power"]["alice_fixed_w"]), self.config)[
                "total_j"
            ]
            for delta in np.diff(self.reference_nodes, axis=0)
        )
        if reference_energy > float(self.config["energy"]["alice_battery_j"]):
            raise ScenarioConfigurationError("Reference path exceeds Alice battery budget")

        if method == "B01":
            action_shape = (1,)
        elif method == "B10":
            action_shape = (2,)
        elif method == "B11":
            action_shape = (3,)
        else:
            action_shape = (1,)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=action_shape, dtype=np.float32)
        self.observation_space = spaces.Box(low=-20.0, high=20.0, shape=(18,), dtype=np.float32)

        self.seed_value = 0
        self.auto_episode_id = 0
        self.seed_key = ""
        self.seed_fingerprint = ""
        self.seed_substreams: dict[str, str] = {}
        self.channel_rng = np.random.default_rng(0)
        self.observation_rng = np.random.default_rng(1)
        self.attack_rng = np.random.default_rng(2)
        self.step_records: list[dict[str, Any]] = []
        self.episode_summary: dict[str, Any] | None = None
        self._current_gains: dict[str, float] | None = None
        self._current_h_ab_estimate = 0.0 + 0.0j
        self._reset_state()

    def _reset_state(self) -> None:
        self.slot = 0
        self.position = self.start.copy()
        self.battery_remaining_j = float(self.config["energy"]["alice_battery_j"])
        self.radiated_remaining_j = float(self.config["power"]["alice_radiated_budget_j"])
        self.willie_remaining_j = float(self.config["power"]["willie_budget_j"])
        self.prev_gamma_b = 0.0
        self.prev_interference_w = 0.0
        self.prev_displacement = np.zeros(2, dtype=np.float64)
        self.prev_alice_power_w = 0.0
        self.willie_location_estimate = self.willie.copy()
        self.action_corrections = 0
        self.reference_fallbacks = 0
        self.fallback_first_trigger_slot: int | None = None
        self.fallback_independent_triggers = 0
        self.fallback_takeover_slots = 0
        self.fallback_exit_count = 0
        self.fallback_reason_counts: Counter[str] = Counter()
        self.correction_reason_counts: Counter[str] = Counter()
        self.raw_power_violations = 0
        self.raw_speed_violations = 0
        self.raw_region_violations = 0
        self.executed_hard_violations = 0
        self.failure_reason = ""
        self.forced_fallback_mode = False
        self.guarantee_nodes: list[np.ndarray] = [node.copy() for node in self.reference_nodes[1:]]
        self.step_records = []
        self.episode_summary = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            episode_id = self.auto_episode_id
            self.auto_episode_id += 1
            scenario_id = None
        else:
            episode_id = 0
            scenario_id = int(seed)
        descriptor = SeedDescriptor(
            purpose=self.stream_purpose,
            run_seed=self.stream_run_seed,
            worker_id=self.stream_worker_id,
            episode_id=episode_id,
            scenario_id=scenario_id,
        )
        root_state = descriptor.seed_sequence().generate_state(2, dtype=np.uint32)
        derived_seed = (int(root_state[0]) << 32) | int(root_state[1])
        super().reset(seed=int(root_state[0]))
        self.seed_value = int(scenario_id) if scenario_id is not None else derived_seed
        self.seed_key = descriptor.key
        self.seed_fingerprint = descriptor.fingerprint()
        self.seed_substreams = substream_fingerprints(descriptor, ("channel", "observation", "attack"))
        children = descriptor.seed_sequence().spawn(3)
        self.channel_rng = np.random.default_rng(children[0])
        self.observation_rng = np.random.default_rng(children[1])
        self.attack_rng = np.random.default_rng(children[2])
        self._reset_state()
        location_sigma = float(self.config["channel"]["willie_location_error_m"])
        self.willie_location_estimate = self.willie + self.observation_rng.normal(0.0, location_sigma, size=2)
        self._prepare_current_channel()
        return self._observation(), {
            "config_digest": self.digest,
            "scenario_seed": self.seed_value,
            "seed_key": self.seed_key,
            "seed_fingerprint": self.seed_fingerprint,
            "seed_substreams": dict(self.seed_substreams),
            "reference_length_m": self.reference_length_m,
        }

    def _prepare_current_channel(self) -> None:
        self._current_gains = channel_gains(self.position, self.config, self.channel_rng)
        amplitude = math.sqrt(self._current_gains["g_ab"])
        phase = self.observation_rng.uniform(-math.pi, math.pi)
        true_h = amplitude * complex(math.cos(phase), math.sin(phase))
        nmse = db_to_linear(float(self.config["channel"]["channel_nmse_db"]))
        error_scale = amplitude * math.sqrt(nmse / 2.0)
        error = error_scale * (self.observation_rng.normal() + 1j * self.observation_rng.normal())
        self._current_h_ab_estimate = true_h + error

    def _observation(self) -> np.ndarray:
        peak_power = max(float(self.config["power"]["alice_peak_w"]), 1e-12)
        battery = max(float(self.config["energy"]["alice_battery_j"]), 1e-12)
        radiated = max(float(self.config["power"]["alice_radiated_budget_j"]), 1e-12)
        h_scale = 1e5
        interference_dbm = 10.0 * math.log10(max(self.prev_interference_w, 1e-18)) + 30.0
        obs = np.concatenate(
            [
                self.position / 1000.0,
                (self.goal - self.position) / 1000.0,
                self.bob / 1000.0,
                self.willie_location_estimate / 1000.0,
                np.array(
                    [self._current_h_ab_estimate.real * h_scale, self._current_h_ab_estimate.imag * h_scale]
                ),
                np.array([math.log10(1.0 + self.prev_gamma_b)]),
                np.array([np.clip((interference_dbm + 150.0) / 100.0, -2.0, 2.0)]),
                np.array([self.battery_remaining_j / battery]),
                np.array([self.radiated_remaining_j / radiated]),
                self.prev_displacement / max(self.max_step_m, 1e-12),
                np.array([self.prev_alice_power_w / peak_power]),
                np.array([(self.n_slots - self.slot) / self.n_slots]),
            ]
        ).astype(np.float32)
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    @staticmethod
    def observation_contract() -> dict[str, Any]:
        """Public defender-observation contract; no privileged channel is listed."""

        return {
            "version": "defender-observation-v2",
            "dimension": 18,
            "feedback_mode": "exact_one_slot_delayed",
            "initial_feedback": "zero_sentinel_disambiguated_by_remaining_time",
            "fields": [
                {"slice": [0, 2], "name": "alice_xy", "source": "navigation", "delay_slots": 0},
                {"slice": [2, 4], "name": "goal_vector", "source": "mission_map", "delay_slots": 0},
                {"slice": [4, 6], "name": "bob_xy", "source": "mission_map", "delay_slots": 0},
                {
                    "slice": [6, 8],
                    "name": "willie_xy_estimate",
                    "source": "external_noisy_episode_estimate",
                    "delay_slots": 0,
                },
                {
                    "slice": [8, 10],
                    "name": "h_ab_estimate_complex",
                    "source": "current_pilot",
                    "delay_slots": 0,
                    "error": "complex Gaussian, configured NMSE",
                },
                {
                    "slice": [10, 11],
                    "name": "previous_gamma_b",
                    "source": "exact Bob feedback",
                    "delay_slots": 1,
                },
                {
                    "slice": [11, 12],
                    "name": "previous_interference",
                    "source": "exact Bob feedback",
                    "delay_slots": 1,
                    "floor_w": 1e-18,
                },
                {"slice": [12, 14], "name": "alice_energy_fractions", "source": "onboard_state", "delay_slots": 0},
                {"slice": [14, 16], "name": "previous_displacement", "source": "onboard_state", "delay_slots": 1},
                {"slice": [16, 17], "name": "previous_alice_power", "source": "onboard_state", "delay_slots": 1},
                {"slice": [17, 18], "name": "remaining_time_fraction", "source": "mission_clock", "delay_slots": 0},
            ],
            "forbidden": [
                "true_willie_position",
                "g_aw",
                "g_wb",
                "gamma_w",
                "current_willie_power",
                "current_bob_interference",
                "secrecy_rate",
            ],
        }

    def _reference_displacement(self) -> tuple[np.ndarray, str]:
        """Current-state route using only navigation, time, and battery state."""

        nodes, minimum_energy, reason = self._future_plan(self.position, self.n_slots - self.slot)
        if nodes is None or minimum_energy > self.battery_remaining_j + 1e-8:
            return np.zeros(2, dtype=np.float64), reason if nodes is None else "energy_unavailable"
        return nodes[0] - self.position, "current_state_safe_path"

    def _parse_action(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        received = np.asarray(action, dtype=np.float64).reshape(-1)
        finite = np.nan_to_num(received, nan=0.0, posinf=1.0, neginf=-1.0)
        clipped = np.clip(finite, -1.0, 1.0)
        corrected = not np.array_equal(received, clipped)
        reasons: list[str] = []
        if not np.array_equal(received, finite):
            reasons.append("nonfinite_action_replaced")
        if not np.array_equal(finite, clipped):
            reasons.append("normalized_action_clipped")
        reference = np.zeros(2, dtype=np.float64)
        residual = np.zeros(2, dtype=np.float64)
        reference_source = "none"
        if self.method in {"B00", "B01"}:
            target = self.reference_nodes[self.slot + 1]
            displacement = target - self.position
            reference = displacement.copy()
            reference_source = "fixed_initial_path"
        else:
            raw_xy = clipped[:2]
            norm = float(np.linalg.norm(raw_xy))
            if norm > 1.0:
                raw_xy = raw_xy / norm
                corrected = True
                reasons.append("motion_vector_norm_clipped")
            if self.motion_action_mode == "reference_residual":
                reference, reference_source = self._reference_displacement()
                residual = raw_xy * self.max_step_m * self.residual_step_fraction
                displacement = reference + residual
            else:
                residual = raw_xy * self.max_step_m
                displacement = residual.copy()
                reference_source = "direct_origin"
        if self.method in {"B00", "B10"}:
            power = float(self.config["power"]["alice_fixed_w"])
        elif self.method == "B01":
            power = self._map_power_action(float(clipped[0]))
        else:
            power = self._map_power_action(float(clipped[2]))
        if float(np.linalg.norm(displacement)) > self.max_step_m + 1e-9:
            self.raw_speed_violations += 1
        candidate = self.position + displacement
        if not segment_in_bounds(self.position, candidate, self.area) or not segment_clear_of_circle(
            self.position, candidate, self.nfz_center, self.planning_radius
        ):
            self.raw_region_violations += 1
        if power < 0.0 or power > float(self.config["power"]["alice_peak_w"]):
            self.raw_power_violations += 1
        audit = {
            "policy_action": received.tolist(),
            "env_action": clipped.tolist(),
            "parse_reasons": reasons,
            "motion_action_mode": self.motion_action_mode if self.method in {"B10", "B11"} else "fixed_initial_path",
            "reference_source": reference_source,
            "reference_displacement": reference.tolist(),
            "proposed_residual": residual.tolist(),
        }
        return displacement, power, corrected, audit

    def _map_power_action(self, normalized: float) -> float:
        peak = float(self.config["power"]["alice_peak_w"])
        mapping = str(self.config["power"].get("action_mapping", "legacy_affine"))
        value = float(np.clip(normalized, -1.0, 1.0))
        if mapping == "legacy_affine":
            return 0.5 * (value + 1.0) * peak
        if mapping == "remaining_budget_centered":
            remaining_slots = max(1, self.n_slots - self.slot)
            center = min(peak, self.radiated_remaining_j / (self.tx_s * remaining_slots))
            if value >= 0.0:
                return center + value * (peak - center)
            return center * (1.0 + value)
        raise ValueError(f"Unknown power.action_mapping: {mapping}")

    def _safe_path(self, start: np.ndarray) -> tuple[np.ndarray, float]:
        start = np.asarray(start, dtype=np.float64)
        curve_radius = self.planning_radius + self.path_clearance
        radial = start - self.nfz_center
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm <= self.planning_radius:
            raise ValueError("Point is inside the conservative planning NFZ")
        if radial_norm < curve_radius + self.geometry_epsilon:
            outward = self.nfz_center + radial / radial_norm * (curve_radius + self.geometry_epsilon)
            tail, _ = shortest_safe_polyline(
                outward,
                self.goal,
                self.nfz_center,
                self.planning_radius,
                epsilon_m=self.path_clearance,
            )
            points = np.vstack([start, outward, tail[1:]])
            return points, polyline_length(points)
        return shortest_safe_polyline(
            start,
            self.goal,
            self.nfz_center,
            self.planning_radius,
            epsilon_m=self.path_clearance,
        )

    def _future_plan(
        self, start: np.ndarray, moves: int
    ) -> tuple[list[np.ndarray] | None, float, str]:
        start = np.asarray(start, dtype=np.float64)
        if moves == 0:
            if float(np.linalg.norm(start - self.goal)) <= self.terminal_tolerance:
                return [], 0.0, "ok"
            return None, math.inf, "time_unreachable"
        try:
            path, safe_length = self._safe_path(start)
        except ValueError:
            return None, math.inf, "future_path_unavailable"
        if safe_length > moves * self.max_step_m + self.terminal_tolerance:
            return None, math.inf, "time_unreachable"
        nodes = resample_polyline(path, moves + 1)
        segments = np.diff(nodes, axis=0)
        if len(segments) != moves:
            return None, math.inf, "future_plan_length_mismatch"
        total_energy = 0.0
        for segment_start, segment_end, segment in zip(nodes[:-1], nodes[1:], segments):
            length = float(np.linalg.norm(segment))
            if length > self.max_step_m + 1e-8:
                return None, math.inf, "future_speed"
            if not segment_in_bounds(segment_start, segment_end, self.area, 1e-9):
                return None, math.inf, "future_boundary"
            if not segment_clear_of_circle(
                segment_start, segment_end, self.nfz_center, self.planning_radius, 1e-9
            ):
                return None, math.inf, "future_nfz"
            total_energy += slot_energy_j(length, 0.0, self.config)["total_j"]
        if float(np.linalg.norm(nodes[-1] - self.goal)) > self.terminal_tolerance:
            return None, math.inf, "future_terminal_error"
        return [node.copy() for node in nodes[1:]], total_energy, "ok"

    def _candidate_certificate(
        self, displacement: np.ndarray, remaining_moves: int
    ) -> tuple[bool, str, list[np.ndarray], float]:
        length = float(np.linalg.norm(displacement))
        if length > self.max_step_m + 1e-8:
            return False, "speed", [], math.inf
        next_position = self.position + displacement
        if not segment_in_bounds(self.position, next_position, self.area, 1e-9):
            return False, "boundary", [], math.inf
        if not segment_clear_of_circle(
            self.position, next_position, self.nfz_center, self.planning_radius, 1e-9
        ):
            return False, "nfz", [], math.inf
        future_nodes, future_energy, reason = self._future_plan(next_position, remaining_moves)
        if future_nodes is None:
            return False, reason, [], math.inf
        current_zero_power = slot_energy_j(length, 0.0, self.config)["total_j"]
        if current_zero_power + future_energy > self.battery_remaining_j + 1e-8:
            return False, "energy", [], future_energy
        return True, "ok", future_nodes, future_energy

    def _motion_feasible(self, displacement: np.ndarray, remaining_moves: int) -> bool:
        feasible, _, _, _ = self._candidate_certificate(displacement, remaining_moves)
        return feasible

    def _guarantee_step(self, moves_including_current: int) -> np.ndarray | None:
        if len(self.guarantee_nodes) != moves_including_current:
            nodes, energy, _ = self._future_plan(self.position, moves_including_current)
            if nodes is None or energy > self.battery_remaining_j + 1e-8:
                return None
            self.guarantee_nodes = nodes
        elif self._guarantee_energy_from(self.position) > self.battery_remaining_j + 1e-8:
            return None
        if not self.guarantee_nodes:
            return None
        next_node = self.guarantee_nodes.pop(0)
        return next_node - self.position

    def _initialize_fallback(self, moves_including_current: int) -> np.ndarray | None:
        nodes, minimum_energy, _ = self._future_plan(self.position, moves_including_current)
        if nodes is None or minimum_energy > self.battery_remaining_j + 1e-8:
            return None
        self.guarantee_nodes = nodes
        self.forced_fallback_mode = True
        return self._guarantee_step(moves_including_current)

    def _enforce_motion(self, proposed: np.ndarray) -> tuple[np.ndarray | None, bool, bool, dict[str, Any]]:
        remaining_moves = self.n_slots - self.slot - 1
        feasible, reason, future_nodes, future_energy = self._candidate_certificate(proposed, remaining_moves)
        if feasible:
            exited = self.forced_fallback_mode
            if exited:
                self.fallback_exit_count += 1
            self.forced_fallback_mode = False
            self.guarantee_nodes = future_nodes
            return proposed, False, False, {
                "motion_decision": "policy_accepted",
                "motion_rejection_reason": "",
                "fallback_exit": int(exited),
                "guaranteed_future_energy_j": future_energy,
            }
        for scale in self.config["environment"]["action_search_scales"]:
            if abs(float(scale) - 1.0) <= 1e-12:
                continue
            scaled = proposed * float(scale)
            scaled_feasible, _, scaled_nodes, scaled_energy = self._candidate_certificate(scaled, remaining_moves)
            if scaled_feasible:
                exited = self.forced_fallback_mode
                if exited:
                    self.fallback_exit_count += 1
                self.forced_fallback_mode = False
                self.guarantee_nodes = scaled_nodes
                return scaled, True, False, {
                    "motion_decision": "scaled_policy_accepted",
                    "motion_rejection_reason": reason,
                    "motion_scale": float(scale),
                    "fallback_exit": int(exited),
                    "guaranteed_future_energy_j": scaled_energy,
                }
        entering = not self.forced_fallback_mode
        fallback = self._guarantee_step(remaining_moves + 1)
        if fallback is None:
            fallback = self._initialize_fallback(remaining_moves + 1)
        if fallback is not None:
            if entering:
                self.fallback_independent_triggers += 1
                if self.fallback_first_trigger_slot is None:
                    self.fallback_first_trigger_slot = self.slot
                self.fallback_reason_counts[reason] += 1
            self.forced_fallback_mode = True
            self.fallback_takeover_slots += 1
            return fallback, True, True, {
                "motion_decision": "fallback_enter" if entering else "fallback_continue",
                "motion_rejection_reason": reason,
                "motion_scale": None,
                "fallback_exit": 0,
                "guaranteed_future_energy_j": self._guarantee_energy_from(self.position + fallback),
            }
        return None, True, True, {
            "motion_decision": "no_feasible_motion",
            "motion_rejection_reason": reason,
            "motion_scale": None,
            "fallback_exit": 0,
            "guaranteed_future_energy_j": math.inf,
        }

    def _guarantee_energy_from(self, start: np.ndarray) -> float:
        current = np.asarray(start, dtype=np.float64)
        total = 0.0
        for node in self.guarantee_nodes:
            total += slot_energy_j(float(np.linalg.norm(node - current)), 0.0, self.config)["total_j"]
            current = node
        return total

    def _cap_alice_power(
        self, requested_w: float, displacement: np.ndarray
    ) -> tuple[float, bool, list[str]]:
        peak = float(self.config["power"]["alice_peak_w"])
        radiated_cap = self.radiated_remaining_j / self.tx_s
        requested = float(np.clip(requested_w, 0.0, min(peak, radiated_cap)))
        corrected = abs(requested - requested_w) > 1e-12
        reasons: list[str] = []
        if requested_w < 0.0 or requested_w > peak:
            reasons.append("power_peak_clip")
        if requested_w * self.tx_s > self.radiated_remaining_j + 1e-12:
            reasons.append("radiated_budget_clip")
        next_position = self.position + displacement
        future_min = self._guarantee_energy_from(next_position)
        zero_energy = slot_energy_j(float(np.linalg.norm(displacement)), 0.0, self.config)["total_j"]
        energy_available_for_rf = self.battery_remaining_j - future_min - zero_energy
        battery_cap = max(0.0, energy_available_for_rf) * float(self.config["energy"]["pa_efficiency"]) / self.tx_s
        capped = min(requested, battery_cap)
        if capped < requested - 1e-12:
            reasons.append("battery_reserve_clip")
        return max(0.0, capped), corrected or capped < requested - 1e-12, reasons

    def _attack_observation(self) -> AttackObservation:
        return {
            "slot": int(self.slot),
            "remaining_slots": int(self.n_slots - self.slot),
            "willie_remaining_j": float(self.willie_remaining_j),
            "willie_peak_w": float(self.config["power"]["willie_peak_w"]),
            "alice_x_est_m": float(
                self.position[0]
                + self.attack_rng.normal(0.0, float(self.config["channel"]["alice_location_error_at_willie_m"]))
            ),
            "alice_y_est_m": float(
                self.position[1]
                + self.attack_rng.normal(0.0, float(self.config["channel"]["alice_location_error_at_willie_m"]))
            ),
        }

    def _willie_power(self) -> float:
        peak = float(self.config["power"]["willie_peak_w"])
        budget_cap = self.willie_remaining_j / self.tx_s
        if self.attack_policy is not None:
            requested = float(self.attack_policy(self._attack_observation()))
        elif self.attack_rule == "none":
            requested = 0.0
        elif self.attack_rule == "uniform":
            requested = float(self.config["power"]["willie_uniform_w"])
        elif self.attack_rule in {"max_budgeted", "frontload"}:
            requested = peak
        elif self.attack_rule == "random_bangbang":
            remaining = self.n_slots - self.slot
            needed_fraction = min(1.0, self.willie_remaining_j / max(self.tx_s * peak * remaining, 1e-12))
            requested = peak if self.attack_rng.random() < needed_fraction else 0.0
        elif self.attack_rule == "privileged_myopic_reference":
            assert self._current_gains is not None
            grid = np.linspace(0.0, min(peak, budget_cap), 51)
            requested = min(
                grid,
                key=lambda value: secrecy_metrics(
                    float(self.prev_alice_power_w or self.config["power"]["alice_fixed_w"]),
                    float(value),
                    self._current_gains,
                    self.config,
                    self.disable_eavesdropper,
                )["secrecy_rate_bpshz"],
            )
        else:
            raise ValueError(f"Unknown attack rule: {self.attack_rule}")
        return float(np.clip(requested, 0.0, min(peak, budget_cap)))

    def _record_failure_padding(self) -> None:
        while len(self.step_records) < self.n_slots:
            n = len(self.step_records)
            self.step_records.append(
                {
                    "slot": n,
                    "scenario_seed": self.seed_value,
                    "seed_key": self.seed_key,
                    "seed_fingerprint": self.seed_fingerprint,
                    "method": self.method,
                    "attack_rule": self.attack_rule,
                    "padded_failure": 1,
                    "failure_reason": self.failure_reason,
                    "x_m": float(self.position[0]),
                    "y_m": float(self.position[1]),
                    "next_x_m": float(self.position[0]),
                    "next_y_m": float(self.position[1]),
                    "alice_power_w": 0.0,
                    "proposed_alice_power_w": 0.0,
                    "executed_dx_m": 0.0,
                    "executed_dy_m": 0.0,
                    "proposed_dx_m": 0.0,
                    "proposed_dy_m": 0.0,
                    "willie_power_w": 0.0,
                    "secrecy_rate_bpshz": 0.0,
                    "slot_outage": 1,
                    "propulsion_j": 0.0,
                    "communication_j": 0.0,
                    "slot_energy_j": 0.0,
                    "action_corrected": 0,
                    "reference_fallback": 0,
                    "fallback_takeover": 0,
                    "fallback_trigger": 0,
                    "motion_decision": "failure_padding",
                    "motion_rejection_reason": "",
                    "executed_hard_violation": 0,
                }
            )

    def _finalize_episode(self) -> dict[str, Any]:
        if self.failure_reason:
            self._record_failure_padding()
        if len(self.step_records) != self.n_slots:
            raise RuntimeError(f"Metric record must contain exactly {self.n_slots} slots")
        secrecy_rates = np.asarray([row["secrecy_rate_bpshz"] for row in self.step_records], dtype=np.float64)
        total_energy = float(sum(row["slot_energy_j"] for row in self.step_records))
        propulsion = float(sum(row["propulsion_j"] for row in self.step_records))
        communication = float(sum(row["communication_j"] for row in self.step_records))
        asr = float(secrecy_rates.mean())
        slot_sop = float(np.mean(secrecy_rates < float(self.config["metric"]["secrecy_target_bpshz"])))
        completion = bool(np.linalg.norm(self.position - self.goal) <= self.terminal_tolerance and not self.failure_reason)
        task_failure = not completion or self.executed_hard_violations > 0
        secure_bits = float(self.config["channel"]["bandwidth_hz"]) * self.tx_s * float(secrecy_rates.sum())
        self.episode_summary = {
            "config_digest": self.digest,
            "scenario_seed": self.seed_value,
            "seed_key": self.seed_key,
            "seed_fingerprint": self.seed_fingerprint,
            "method": self.method,
            "attack_rule": self.attack_rule,
            "slots_expected": self.n_slots,
            "slots_recorded": len(self.step_records),
            "real_slots": int(sum(1 - int(row["padded_failure"]) for row in self.step_records)),
            "padded_failure_slots": int(sum(int(row["padded_failure"]) for row in self.step_records)),
            "asr_bpshz": asr,
            "slot_sop": slot_sop,
            "episode_sop": int(asr < float(self.config["metric"]["episode_target_bpshz"])),
            "secure_bits": secure_bits,
            "secure_throughput_bps": float(self.config["channel"]["bandwidth_hz"])
            * float(self.config["time"]["tx_duty"])
            * asr,
            "task_completed": int(completion),
            "task_failure": int(task_failure),
            "failure_reason": self.failure_reason,
            "final_error_m": float(np.linalg.norm(self.position - self.goal)),
            "propulsion_energy_j": propulsion,
            "communication_energy_j": communication,
            "alice_total_energy_j": total_energy,
            "alice_radiated_energy_j": float(
                sum(row["alice_power_w"] * self.tx_s for row in self.step_records)
            ),
            "willie_energy_j": float(sum(row["willie_power_w"] * self.tx_s for row in self.step_records)),
            "executed_hard_violations": self.executed_hard_violations,
            "raw_speed_violations": self.raw_speed_violations,
            "raw_region_violations": self.raw_region_violations,
            "raw_power_violations": self.raw_power_violations,
            "action_corrections": self.action_corrections,
            "reference_fallbacks": self.reference_fallbacks,
            "fallback_first_trigger_slot": self.fallback_first_trigger_slot,
            "fallback_independent_triggers": self.fallback_independent_triggers,
            "fallback_takeover_slots": self.fallback_takeover_slots,
            "fallback_exit_count": self.fallback_exit_count,
            "fallback_reason_counts": dict(self.fallback_reason_counts),
            "correction_reason_counts": dict(self.correction_reason_counts),
            "see_bit_per_j": secure_bits / total_energy if total_energy > 0 else 0.0,
        }
        return self.episode_summary

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.slot >= self.n_slots or self.episode_summary is not None:
            raise RuntimeError("step called after episode termination; call reset")
        assert self._current_gains is not None
        proposed_displacement, requested_power, corrected, action_audit = self._parse_action(action)
        displacement, motion_corrected, fallback, motion_audit = self._enforce_motion(proposed_displacement)
        if displacement is None:
            self.failure_reason = "no_feasible_motion_or_reachability"
            summary = self._finalize_episode()
            return self._observation(), 0.0, True, False, {"episode_summary": summary, "failure": True}
        alice_power, power_corrected, power_reasons = self._cap_alice_power(requested_power, displacement)
        corrected = corrected or motion_corrected or power_corrected
        correction_reasons = list(action_audit["parse_reasons"])
        if motion_corrected:
            correction_reasons.append(str(motion_audit["motion_decision"]))
            rejection = str(motion_audit["motion_rejection_reason"])
            if rejection:
                correction_reasons.append(f"motion_{rejection}")
        correction_reasons.extend(power_reasons)
        self.correction_reason_counts.update(correction_reasons)
        if corrected:
            self.action_corrections += 1
        if fallback:
            self.reference_fallbacks += 1

        willie_power = self._willie_power()
        metrics = secrecy_metrics(
            alice_power,
            willie_power,
            self._current_gains,
            self.config,
            self.disable_eavesdropper,
        )
        energy = slot_energy_j(float(np.linalg.norm(displacement)), alice_power, self.config)
        next_position = self.position + displacement
        reference_displacement = np.asarray(action_audit["reference_displacement"], dtype=np.float64)
        proposed_residual = np.asarray(action_audit["proposed_residual"], dtype=np.float64)
        executed_residual = displacement - reference_displacement
        executed_violation = int(
            float(np.linalg.norm(displacement)) > self.max_step_m + 1e-7
            or not point_in_bounds(next_position, self.area, 1e-7)
            or not segment_clear_of_circle(
                self.position, next_position, self.nfz_center, self.nfz_radius, 1e-7
            )
            or alice_power > float(self.config["power"]["alice_peak_w"]) + 1e-8
            or alice_power * self.tx_s > self.radiated_remaining_j + 1e-8
            or energy["total_j"] > self.battery_remaining_j + 1e-8
        )
        self.executed_hard_violations += executed_violation
        row = {
            "slot": self.slot,
            "scenario_seed": self.seed_value,
            "seed_key": self.seed_key,
            "seed_fingerprint": self.seed_fingerprint,
            "method": self.method,
            "attack_rule": self.attack_rule,
            "padded_failure": 0,
            "failure_reason": "",
            "x_m": float(self.position[0]),
            "y_m": float(self.position[1]),
            "next_x_m": float(next_position[0]),
            "next_y_m": float(next_position[1]),
            "policy_action": list(action_audit["policy_action"]),
            "env_action": list(action_audit["env_action"]),
            "policy_action_0": float(action_audit["policy_action"][0]) if action_audit["policy_action"] else math.nan,
            "policy_action_1": float(action_audit["policy_action"][1]) if len(action_audit["policy_action"]) > 1 else math.nan,
            "policy_action_2": float(action_audit["policy_action"][2]) if len(action_audit["policy_action"]) > 2 else math.nan,
            "env_action_0": float(action_audit["env_action"][0]) if action_audit["env_action"] else math.nan,
            "env_action_1": float(action_audit["env_action"][1]) if len(action_audit["env_action"]) > 1 else math.nan,
            "env_action_2": float(action_audit["env_action"][2]) if len(action_audit["env_action"]) > 2 else math.nan,
            "proposed_dx_m": float(proposed_displacement[0]),
            "proposed_dy_m": float(proposed_displacement[1]),
            "motion_action_mode": action_audit["motion_action_mode"],
            "reference_source": action_audit["reference_source"],
            "reference_dx_m": float(reference_displacement[0]),
            "reference_dy_m": float(reference_displacement[1]),
            "proposed_residual_dx_m": float(proposed_residual[0]),
            "proposed_residual_dy_m": float(proposed_residual[1]),
            "executed_residual_dx_m": float(executed_residual[0]),
            "executed_residual_dy_m": float(executed_residual[1]),
            "motion_correction_dx_m": float(displacement[0] - proposed_displacement[0]),
            "motion_correction_dy_m": float(displacement[1] - proposed_displacement[1]),
            "executed_dx_m": float(displacement[0]),
            "executed_dy_m": float(displacement[1]),
            "proposed_alice_power_w": float(requested_power),
            "alice_power_w": alice_power,
            "willie_power_w": willie_power,
            **metrics,
            "slot_outage": int(
                metrics["secrecy_rate_bpshz"] < float(self.config["metric"]["secrecy_target_bpshz"])
            ),
            "g_ab": self._current_gains["g_ab"],
            "g_aw": self._current_gains["g_aw"],
            "g_wb": self._current_gains["g_wb"],
            "move_distance_m": float(np.linalg.norm(displacement)),
            "speed_mps": energy["speed_mps"],
            "propulsion_j": energy["propulsion_j"],
            "communication_j": energy["communication_j"],
            "slot_energy_j": energy["total_j"],
            "battery_before_j": float(self.battery_remaining_j),
            "battery_after_j": float(self.battery_remaining_j - energy["total_j"]),
            "radiated_before_j": float(self.radiated_remaining_j),
            "radiated_after_j": float(self.radiated_remaining_j - alice_power * self.tx_s),
            "remaining_slots_before": int(self.n_slots - self.slot),
            "goal_distance_before_m": float(np.linalg.norm(self.position - self.goal)),
            "goal_distance_after_m": float(np.linalg.norm(next_position - self.goal)),
            "feedback_valid": int(self.slot > 0),
            "motion_decision": motion_audit["motion_decision"],
            "motion_rejection_reason": motion_audit["motion_rejection_reason"],
            "motion_scale": motion_audit.get("motion_scale"),
            "guaranteed_future_energy_j": motion_audit["guaranteed_future_energy_j"],
            "correction_reasons": correction_reasons,
            "power_correction_reasons": power_reasons,
            "action_corrected": int(corrected),
            "reference_fallback": int(fallback),
            "fallback_takeover": int(fallback),
            "fallback_trigger": int(
                fallback and motion_audit["motion_decision"] == "fallback_enter"
            ),
            "fallback_exit": int(motion_audit.get("fallback_exit", 0)),
            "executed_hard_violation": executed_violation,
        }
        self.step_records.append(row)

        self.position = next_position
        self.battery_remaining_j -= energy["total_j"]
        self.radiated_remaining_j -= alice_power * self.tx_s
        self.willie_remaining_j -= willie_power * self.tx_s
        self.prev_gamma_b = metrics["gamma_b"]
        self.prev_interference_w = willie_power * self._current_gains["g_wb"]
        self.prev_displacement = displacement.copy()
        self.prev_alice_power_w = alice_power
        self.slot += 1
        terminated = self.slot >= self.n_slots
        if terminated and np.linalg.norm(self.position - self.goal) > self.terminal_tolerance:
            self.failure_reason = "terminal_not_reached"
        if executed_violation:
            self.failure_reason = "executed_hard_constraint_violation"
            terminated = True
        info: dict[str, Any] = {
            "secrecy_rate_bpshz": metrics["secrecy_rate_bpshz"],
            "action_corrected": corrected,
            "reference_fallback": fallback,
        }
        if terminated:
            info["episode_summary"] = self._finalize_episode()
            observation = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            self._prepare_current_channel()
            observation = self._observation()
        reward = float(metrics["secrecy_rate_bpshz"])
        return observation, reward, terminated, False, info

    def force_battery_for_test(self, battery_j: float) -> None:
        """Testing hook used only by D0 energy-exhaustion validation."""

        self.battery_remaining_j = float(battery_j)
