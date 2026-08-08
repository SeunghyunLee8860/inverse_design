# Run 003: converged constrained PTE inverse design

Run 003 restarts the signed-current optimization from the original beta=2
initial state.  It reuses the already certified Maxwell/material-Jacobian and
CUDA thermal/PTE operators from Run 002, but it does **not** reuse Run 002's
optimization trajectory or its stopped one-step-per-beta supervisor.

The authoritative method is in `OPTIMIZATION_CONTRACT.md`.  Raw FSP/NPZ and
solver logs live below `/home/seunghyun/tairte4/raw_artifacts/run003_*` and are
never committed.  This directory contains the restartable driver, tests,
per-iteration figures, compact checkpoints, reports, and SHA provenance.

Run Maxwell, CUDA thermal/PTE, and the differentiable disk constraint on one
licensed GPU with (example physical GPU 2):

```bash
CUDA_VISIBLE_DEVICES=2 /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python run_optimization.py --gpu 2 --constraint-device cuda:0
```

CPU FDTD and CPU thermal fallbacks are prohibited.
