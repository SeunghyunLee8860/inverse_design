# Run 005: low-beta topology-exploration pilot

Run 005 restarts from the immutable original beta=2 density. It is deliberately
limited to one fresh GPU Maxwell/CUDA thermal-PTE update with `move=0.01`.
Run 003 and Run 004 remain unchanged checkpoints.

This pilot tests whether useful topology/FOM motion is possible without using
the discontinuous exact bad-cell count as a low-beta optimization veto:

- differentiable 500 nm solid/void disk constraints remain active;
- both smooth caps must be satisfied;
- a smooth-feasible step may lose at most 0.2% actual FOM;
- exact DRC is diagnostic at beta=2;
- only a greater-than-50% and greater-than-25-cell exact-count increase halts
  the pilot as a catastrophic guard;
- no smaller-move line search is allowed in this one-point experiment;
- beta promotion and the full continuation are prohibited.

The next 3--5 accepted-update pilot requires a separate review of this point.
Every later beta cap must first be calibrated by reprojecting the accepted
checkpoint; Run 005 cannot silently continue to beta=4.

The one-point run passed and then paused: FOM increased by 19.4826%, smooth
solid/void constraints remained feasible, and diagnostic exact bad cells fell
from 158 to 46. This is evidence for a healthy first topology step, not evidence
that beta=2 or the complete constrained optimization has converged.

```bash
CUDA_VISIBLE_DEVICES=2 /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  run_optimization.py --gpu 2 --constraint-device cuda:0 \
  --pilot-accepted-updates 1
```
