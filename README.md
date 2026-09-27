# SecureLink P1 prototype

Current implementation: `p1-prototype-v2.0.0-acceptance`.

This directory contains the D0 deterministic simulator checks and the D1 PPO
prototype defined by `feedback/04_P1主模型冻结稿.md` and
`feedback/05_P1假设与实验矩阵.md`.

The simulator is synthetic and configuration-driven. It does not use an
external dataset. All episode records include the configuration digest,
scenario seed, attack rule, executed actions, failures, padded failure slots,
energy use, ASR and both SOP definitions.

The v2 acceptance environment additionally records the policy/environment
action, proposed and executed physical actions, classified correction reasons,
the first and independent fallback triggers, takeover slots, safe exits, and a
replayable remaining-path energy certificate. Random streams are namespaced by
purpose, training run, worker, episode and channel/observation/attack substream.

## Quick start

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python scripts/run_d0.py --config configs/base.yaml --output results/d0
pytest -q
python scripts/run_d1.py --config configs/prototype.yaml --output results/d1
python scripts/run_acceptance.py --config configs/acceptance.yaml --output results/acceptance_v2 --old-results results/d1_remote
python scripts/run_d1.py --config configs/controlled_rerun.yaml --output results/d1_v2 --n-envs 4 --device cpu
```

On Windows PowerShell, replace the `export` line with:

```powershell
$env:PYTHONPATH="$PWD\src"
```

`configs/controlled_rerun.yaml` uses exactly 32768 actual transitions per
learning run, three training seeds and development scenarios 20001--20030.
Final scenarios starting at 50001 are reserved and are not run by these
commands. This is a diagnostic budget, not publication-scale evidence.
