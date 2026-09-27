from __future__ import annotations

import math
from typing import Any

import numpy as np


def db_to_linear(value_db: float) -> float:
    return 10.0 ** (float(value_db) / 10.0)


def dbm_to_watt(value_dbm: float) -> float:
    return 10.0 ** ((float(value_dbm) - 30.0) / 10.0)


def receiver_noise_w(config: dict[str, Any]) -> float:
    channel = config["channel"]
    noise_dbm = (
        float(channel["noise_psd_dbm_hz"])
        + 10.0 * math.log10(float(channel["bandwidth_hz"]))
        + float(channel["noise_figure_db"])
    )
    return dbm_to_watt(noise_dbm)


def propulsion_power_w(speed_mps: float, config: dict[str, Any]) -> float:
    energy = config["energy"]
    speed = max(0.0, float(speed_mps))
    p0 = float(energy["p0_w"])
    pi = float(energy["pi_w"])
    tip = float(energy["rotor_tip_speed_mps"])
    v0 = float(energy["induced_velocity_mps"])
    drag = float(energy["fuselage_drag_ratio"])
    rho = float(energy["air_density_kg_m3"])
    solidity = float(energy["rotor_solidity"])
    area = float(energy["rotor_disc_area_m2"])
    profile = p0 * (1.0 + 3.0 * speed**2 / tip**2)
    induced_inner = math.sqrt(1.0 + speed**4 / (4.0 * v0**4)) - speed**2 / (2.0 * v0**2)
    induced = pi * math.sqrt(max(0.0, induced_inner))
    parasite = 0.5 * drag * rho * solidity * area * speed**3
    return profile + induced + parasite


def slot_energy_j(displacement_m: float, alice_power_w: float, config: dict[str, Any]) -> dict[str, float]:
    slot_s = float(config["time"]["slot_s"])
    tx_s = slot_s * float(config["time"]["tx_duty"])
    move_s = slot_s - tx_s
    speed = float(displacement_m) / move_s
    hover_w = propulsion_power_w(0.0, config)
    propulsion_j = hover_w * tx_s + propulsion_power_w(speed, config) * move_s
    communication_j = (
        float(alice_power_w) / float(config["energy"]["pa_efficiency"])
        + float(config["energy"]["circuit_power_w"])
    ) * tx_s
    return {
        "speed_mps": speed,
        "propulsion_j": propulsion_j,
        "communication_j": communication_j,
        "total_j": propulsion_j + communication_j,
    }


def minimum_future_energy_j(path_length_m: float, moves: int, config: dict[str, Any]) -> float:
    if moves <= 0:
        return 0.0 if path_length_m <= float(config["scenario"]["terminal_tolerance_m"]) else math.inf
    move_s = float(config["time"]["slot_s"]) * (1.0 - float(config["time"]["tx_duty"]))
    speed = float(path_length_m) / (moves * move_s)
    if speed > float(config["mobility"]["max_speed_mps"]) + 1e-9:
        return math.inf
    return moves * slot_energy_j(path_length_m / moves, 0.0, config)["total_j"]


def rician_power_gain(rng: np.random.Generator, k_linear: float) -> float:
    los = math.sqrt(k_linear / (k_linear + 1.0))
    scatter_scale = math.sqrt(1.0 / (2.0 * (k_linear + 1.0)))
    sample = los + scatter_scale * (rng.normal() + 1j * rng.normal())
    return float(abs(sample) ** 2)


def rayleigh_power_gain(rng: np.random.Generator) -> float:
    sample = (rng.normal() + 1j * rng.normal()) / math.sqrt(2.0)
    return float(abs(sample) ** 2)


def channel_gains(
    alice_xy: np.ndarray,
    config: dict[str, Any],
    rng: np.random.Generator | None = None,
    fading: bool | None = None,
) -> dict[str, float]:
    scenario = config["scenario"]
    channel = config["channel"]
    bob = np.asarray(scenario["bob_m"], dtype=np.float64)
    willie = np.asarray(scenario["willie_m"], dtype=np.float64)
    altitude = float(scenario["altitude_m"])
    d_ab = math.sqrt(float(np.dot(alice_xy - bob, alice_xy - bob)) + altitude**2)
    d_aw = math.sqrt(float(np.dot(alice_xy - willie, alice_xy - willie)) + altitude**2)
    d_wb = float(np.linalg.norm(willie - bob))
    beta0 = db_to_linear(float(channel["reference_gain_db"]))
    use_fading = bool(config["environment"]["fading"] if fading is None else fading)
    if use_fading:
        if rng is None:
            raise ValueError("rng is required when fading is enabled")
        k_linear = db_to_linear(float(channel["rician_k_db"]))
        f_ab = rician_power_gain(rng, k_linear)
        f_aw = rician_power_gain(rng, k_linear)
        f_wb = rayleigh_power_gain(rng)
    else:
        f_ab = f_aw = f_wb = 1.0
    return {
        "g_ab": beta0 * d_ab ** (-float(channel["pathloss_ab"])) * f_ab,
        "g_aw": beta0 * d_aw ** (-float(channel["pathloss_aw"])) * f_aw,
        "g_wb": beta0 * d_wb ** (-float(channel["pathloss_wb"])) * f_wb,
        "d_ab_m": d_ab,
        "d_aw_m": d_aw,
        "d_wb_m": d_wb,
    }


def secrecy_metrics(
    alice_power_w: float,
    willie_power_w: float,
    gains: dict[str, float],
    config: dict[str, Any],
    disable_eavesdropper: bool = False,
) -> dict[str, float]:
    noise = receiver_noise_w(config)
    p_a = max(0.0, float(alice_power_w))
    p_w = max(0.0, float(willie_power_w))
    gamma_b = p_a * gains["g_ab"] / (p_w * gains["g_wb"] + noise)
    if disable_eavesdropper:
        gamma_w = 0.0
    else:
        beta_si = db_to_linear(float(config["power"]["self_interference_db"]))
        gamma_w = p_a * gains["g_aw"] / (beta_si * p_w + noise)
    c_b = math.log2(1.0 + gamma_b)
    c_w = math.log2(1.0 + gamma_w)
    raw_difference = c_b - c_w
    secrecy = max(0.0, raw_difference)
    return {
        "gamma_b": gamma_b,
        "gamma_w": gamma_w,
        "capacity_b_bpshz": c_b,
        "capacity_w_bpshz": c_w,
        "capacity_difference_bpshz": raw_difference,
        "secrecy_rate_bpshz": secrecy,
    }
