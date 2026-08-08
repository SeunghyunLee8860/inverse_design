# Run 004: joint FOM and 500 nm constrained PTE inverse design

Run 004 restarts from the immutable original beta=2 latent density. It does not
continue from halted Run 003 checkpoint g095. The verified Run 002 GPU Maxwell,
complex Yee-material Jacobian, and CUDA thermal/PTE operators are unchanged.

The correction is confined to the optimization policy:

- the differentiable 500 nm disk-opening solid and void constraints are active
  from the first MMA update;
- the initial MMA move is 0.01 instead of 0.02;
- each stage is limited to 20 accepted updates;
- an automatic six-update no-progress gate stops the run if neither FOM nor
  exact morphology improves meaningfully;
- exact thresholded morphology is audited throughout, but its discontinuous
  per-step veto starts only at beta=32 so early topology changes are not frozen;
- no post-hoc repair, empirical gradient scaling, CPU FDTD, or CPU thermal
  fallback is permitted.

See `OPTIMIZATION_CONTRACT.md` for the complete contract.

```bash
CUDA_VISIBLE_DEVICES=2 /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  run_optimization.py --gpu 2 --constraint-device cuda:0 \
  --pilot-accepted-updates 3
```
