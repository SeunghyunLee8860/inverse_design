# Au dual-polarization PTE inverse-design code handoff

## Scope

This directory contains the code path for a 4 um Au topology on a fixed
TaIrTe4 flake.  The target is the signed dual-polarization objective

\[
\max_\rho \min\left(I_{E\parallel a},-I_{E\parallel b}\right).
\]

The coordinate contract is **Lumerical/FDTDX x = crystal b** and
**y = crystal a**.  Do not swap the polarization labels or coordinate axes.

## Read these files first

1. `contract.py` -- immutable geometry, source, axes, design pitch, and
   reporting power.
2. `fdtdx_4um_model.py` -- six-PML FDTDX Maxwell model and material layout.
3. `multiphysics_4um.py` -- conservative optical-Q remap, explicit 3-D
   thermal solve, electrical weighting solve, and PTE current.
4. `combined_4um.py` -- two-solve Maxwell adjoint and the complete optical,
   thermal, and electrical density gradient.
5. `dfm.py` -- 500 nm filter, differentiable solid/void constraints, and
   exact binary audit.
6. `objective.py` -- signed current utilities and epigraph objective.
7. `10_optimize_4um_dualpol_au_ld_mma.py` -- nominal NLopt LD_MMA path.
8. `13_optimize_robust_binary_au_ld_mma.py` -- eroded/dilated robust
   continuation path.
9. `14_diagnose_gray_law_mismatch.py` -- current gray-material blocker.
10. `15_validate_4um_z_mesh_convergence.py` -- fail-closed z-mesh gate that
    must be closed before another production optimization.

The scripts `00` through `09` contain the runsetup, source calibration,
forward, thermal/electrical, and AD-FD certificates used by the code above.

## Current immutable physical contract

- wavelength: 4 um
- scalar Gaussian waist: 4 um
- optical domain: 20 x 20 um laterally, six PML boundaries
- source aperture: 16 x 16 um
- fixed TaIrTe4 flake: 16 x 16 x 0.1 um
- Au design region: 8 x 8 x 0.05 um
- design variables: 80 x 80 at 100 nm pitch
- reporting incident power: 285 uW
- minimum solid and void feature audit: 500 nm
- no symmetry, volume-fraction, or connectivity constraint
- no Q clipping, smoothing, gain, polarization matching, or closure rescaling

## Important blockers -- do not silently bypass

1. The existing optimization used inconsistent gray laws: optical Au
   oscillator strength uses `rho**3`, while thermal/electrical Au uses
   `rho**1`.  The endpoints are exact, but gray cells are not a single
   physical material.  See `14_diagnose_gray_law_mismatch.py`.
2. AD-FD validates the derivative of a chosen discrete mesh; it does not
   certify mesh convergence.
3. The original optical z mesh used only 2 Au cells and 5 TaIrTe4 cells.
   The z-convergence script therefore checks Au/TaIrTe4/SiO2 refinement
   factors 1, 2, 4, and 8 with a separate all-air source calibration for
   every mesh.
4. A new optimization must not be promoted until z convergence, then x/y
   convergence and combined-gradient convergence, pass fail-closed gates.

## Raw checkpoint dependency

Raw NPZ files are intentionally not committed.  The z-mesh diagnostic uses:

```text
/home/seunghyun/tairte4/raw/au_dualpol_4um_current_switch/
robust_projection_ld_mma/evaluation_0112.npz
SHA-256 ef8b99bec0029588b89f56edc68bd9c747fa9ed0897933def138c787509332e3
```

Fail closed if this file is absent or its SHA differs.  A clean checkout must
receive the checkpoint explicitly rather than inventing or rescaling it.

## Reproduction commands

From the repository root:

```bash
python -m pytest -q \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/test_preflight.py

CUDA_VISIBLE_DEVICES=<free_gpu> photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_z_mesh_convergence_gpu.sh --audit-only

CUDA_VISIBLE_DEVICES=<free_gpu> photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_z_mesh_convergence_gpu.sh
```

`run_combined_gpu_python.sh` selects the checked Python/JAX/PyTorch environment
used by the project.

## Next correct sequence

1. Finish and report z-mesh convergence without changing the density or
   material laws during the sweep.
2. If z converges, check x/y optical convergence and downstream PTE current.
3. Certify the combined gradient on the selected production mesh.
4. Replace or justify the inconsistent gray law and repeat endpoint/AD-FD
   checks.
5. Only then restart LD_MMA continuation and finish with an exact binary
   500 nm solid/void audit.
