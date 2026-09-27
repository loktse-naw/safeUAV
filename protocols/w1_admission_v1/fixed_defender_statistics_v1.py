"""Frozen statistics for Task 005; importing this module runs no evaluation."""
from __future__ import annotations

import numpy as np

DELTA = 0.005
Q_THRESHOLD = 0.80
REPLICATES = 10_000
CONFIRM_BOOTSTRAP_SEED = 94201
FINAL_BOOTSTRAP_SEED = 94202


def candidate_choice(scores: dict[str, float]) -> str:
    """Equal-model/equal-scenario means supplied by the selection-only caller."""
    if not scores or not all(np.isfinite(value) for value in scores.values()):
        raise ValueError("Empty or nonfinite candidate scores")
    minimum = min(scores.values())
    return min(name for name, score in scores.items() if score <= minimum + 1e-8)


def paired_difference(first: np.ndarray, second: np.ndarray, seed: int) -> dict:
    """Fixed three defenders; one shared scenario resample for every model."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if first.shape != second.shape or first.ndim != 2 or first.shape[0] != 3:
        raise ValueError("Expected paired arrays [3 fixed models, n scenarios]")
    if first.shape[1] != 100 or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Expected 100 complete finite paired scenarios")
    per_scenario = (first - second).mean(axis=0)
    draws = np.random.Generator(np.random.PCG64(seed)).integers(0, 100, (REPLICATES, 100))
    distribution = per_scenario[draws].mean(axis=1)
    return {"mean": float(per_scenario.mean()),
            "ci95": np.quantile(distribution, [0.025, 0.975], method="linear").tolist()}


def rule_branch(low: np.ndarray, adaptive: np.ndarray, privileged: np.ndarray,
                interface_and_constraints_pass: bool) -> dict:
    """Primary preregistered D and q; no model resampling or ratio clipping."""
    arrays = [np.asarray(value, float) for value in (low, adaptive, privileged)]
    if any(value.shape != (3, 100) or not np.isfinite(value).all() for value in arrays):
        raise ValueError("Expected three matched [3,100] arrays; never drop missing/failed scenarios")
    low, adaptive, privileged = arrays
    l, a, p = (value.mean(axis=0) for value in arrays)
    d, numerator = l - p, l - a
    rng = np.random.Generator(np.random.PCG64(CONFIRM_BOOTSTRAP_SEED))
    indices = rng.integers(0, 100, (REPLICATES, 100))
    d_draws = d[indices].mean(axis=1)
    n_draws = numerator[indices].mean(axis=1)
    d_interval = np.quantile(d_draws, [0.025, 0.975], method="linear")
    result = {"L": float(l.mean()), "A": float(a.mean()), "P": float(p.mean()),
              "D": float(d.mean()), "D_ci95": d_interval.tolist(),
              "delta": DELTA, "q_threshold": Q_THRESHOLD,
              "bootstrap_replicates": REPLICATES,
              "bootstrap_seed": CONFIRM_BOOTSTRAP_SEED,
              "q": None, "q_ci95": None, "branch": "insufficient_evidence"}
    if not interface_and_constraints_pass:
        result["branch"] = "halt_integrity_gate"
        return result
    if d_interval[0] <= DELTA:
        result["reason"] = "D lower confidence limit does not exceed delta; q not interpreted"
        return result
    if np.any(d_draws <= 0):
        result["reason"] = "Nonpositive bootstrap denominator; do not discard or clip draws"
        return result
    q_draws = n_draws / d_draws
    q_interval = np.quantile(q_draws, [0.025, 0.975], method="linear")
    result["q"] = float(numerator.mean() / d.mean())
    result["q_ci95"] = q_interval.tolist()
    if q_interval[0] >= Q_THRESHOLD:
        result["branch"] = "do_not_invest_W1"
    elif q_interval[1] < Q_THRESHOLD:
        result["branch"] = "may_prepare_limited_W1_protocol"
    return result
