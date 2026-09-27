# P1 motion residual v1 source snapshot

`motion_residual_v1_source.tar.gz` is the exact source snapshot uploaded for the
32,768-transition residual B10/B11 training run. SHA-256:

`48EAC6692B447A18A3DF031ADBB4A498D1098FBE6070DE0DBA039A650F3B55FB`

The archive contains `src/`, `scripts/`, `configs/`, `tests/`, and
`requirements.txt`. It was extracted into
`/root/SafeUAV/pytorch_code_motion_v1` on the remote server. The remote copy
had the same SHA-256. The command used for the first training budget was:

```bash
PYTHONPATH=src python scripts/run_d1.py \
  --config configs/motion_residual_v1.yaml \
  --output results/motion_residual_v1_32768 \
  --methods B10 B11 --n-envs 4 --device cpu
```

Post-run analysis scripts were added locally after this snapshot; they do not
change the environment, training code, or frozen configuration used above.

`motion_residual_v1_32768_results.tar.gz` is the untouched remote results
bundle. Its SHA-256 is
`410C0BE048DA22FCE9EA61B8E1596D132AFCD84767C60AE03D08BD1EBF799249`.
The extracted local result directory is
`pytorch_code/results/motion_residual_v1_32768/`.
