"""Prospective W1 statistic; no environment, training, or automatic evaluation."""
import numpy as np
from fixed_defender_statistics_v1 import paired_difference

DELTA = 0.005
FEASIBILITY_BOOTSTRAP_SEED = 94301


def w1_primary(rule_asr, w1_asr):
    """Axes: rule [defender, scenario]; W1 [attack run, defender, scenario]."""
    rule = np.asarray(rule_asr, dtype=float)
    learned = np.asarray(w1_asr, dtype=float)
    if rule.shape != (3, 100) or learned.shape != (3, 3, 100):
        raise ValueError("All three fixed attack runs, defenders, and 100 scenarios required")
    if not np.isfinite(rule).all() or not np.isfinite(learned).all():
        raise ValueError("Keep failed episodes under the frozen padding contract; no missing data")
    result = paired_difference(rule, learned.mean(axis=0), FEASIBILITY_BOOTSTRAP_SEED)
    result.update(delta=DELTA, bootstrap_seed=FEASIBILITY_BOOTSTRAP_SEED,
                  estimand="equal fixed three W1 runs and three B10 defenders, new scenarios",
                  branch=("additional_pressure_detected" if result["ci95"][0] > DELTA
                          else "additional_pressure_not_detected"))
    return result
