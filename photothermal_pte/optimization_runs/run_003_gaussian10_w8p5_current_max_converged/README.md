# Run 003: halted constrained PTE inverse-design diagnostic

Status: `HALTED_PREMATURE_BETA2_SATURATION_AND_LATE_CONSTRAINT_REPAIR`

The accepted trajectory ends at `g095`. Partial `g096` was interrupted and is
not accepted. Run 003 is neither converged nor exact-500-nm feasible; see
[`results/RUN003_CONTINUATION_PATHOLOGY_AUDIT.md`](results/RUN003_CONTINUATION_PATHOLOGY_AUDIT.md)
before attempting any restart.

Run 003 restarts the signed-current optimization from the original beta=2
initial state.  It reuses the already certified Maxwell/material-Jacobian and
CUDA thermal/PTE operators from Run 002, but it does **not** reuse Run 002's
optimization trajectory or its stopped one-step-per-beta supervisor.

The authoritative method is in `OPTIMIZATION_CONTRACT.md`.  Raw FSP/NPZ and
solver logs live below `/home/seunghyun/tairte4/raw_artifacts/run003_*` and are
never committed.  This directory contains the restartable driver, tests,
per-iteration figures, compact checkpoints, reports, and SHA provenance.

The historical invocation below is retained for provenance. Do **not** resume it
without an approved replacement continuation contract:

```bash
CUDA_VISIBLE_DEVICES=2 /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python run_optimization.py --gpu 2 --constraint-device cuda:0
```

CPU FDTD and CPU thermal fallbacks are prohibited.
