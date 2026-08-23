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
6. `material_fraction.py` and `MATERIAL_FRACTION_AUDIT.md` -- the shared
   linear Au fraction and its fail-closed migration from historical O3/TE1.
7. `objective.py` -- signed current utilities and epigraph objective.
8. `10_optimize_4um_dualpol_au_ld_mma.py` -- nominal NLopt LD_MMA path.
9. `13_optimize_robust_binary_au_ld_mma.py` -- eroded/dilated robust
   continuation path.  Read `ROBUST_OBJECTIVE_AUDIT.md` before using it.
10. `14_diagnose_gray_law_mismatch.py` -- historical gray-material blocker.
11. `15_validate_4um_z_mesh_convergence.py` -- fail-closed z-mesh gate that
    must be closed before another production optimization.
12. `production_readiness.py` and `PRODUCTION_READINESS.md` -- executable
    certificate chain that currently blocks all production entry points.
13. `CODE_PHYSICS_AUDIT.md` and `physical_device_contract.json` -- full
    physics audit and the deliberately blocked target-device contract.

The scripts `00` through `09` contain the runsetup, source calibration,
forward, thermal/electrical, and AD-FD certificates used by the code above.

## Current code assumptions -- not yet a confirmed physical device

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

1. The existing optimization used inconsistent O3/TE1 gray laws.  Production
   code now uses one shared linear fraction (`f_Au=rho`) in all three physics,
   but it has not yet received new AD-FD or mesh-convergence certificates.
   Historical O3/TE1 outputs must not be presented as certificates for the
   corrected law.  See `MATERIAL_FRACTION_AUDIT.md`.
2. AD-FD validates the derivative of a chosen discrete mesh; it does not
   certify mesh convergence.
3. The original optical z mesh used only 2 Au cells and 5 TaIrTe4 cells.
   A historical partial z sweep checked Au/TaIrTe4/SiO2 factors 1, 2, 4, and
   8, but its tables are now explicitly stale: they use O3/TE1, the old
   Shockley-Ramo sign, and a cache key bound only to the checkpoint. Do not use
   its reported Q/current changes for the current shared-linear code. Rerun
   only after time closure, then perform full-domain z convergence.
4. A new optimization must not be promoted until z convergence, then x/y
   convergence and combined-gradient convergence, pass fail-closed gates.
5. The completed sweep refined only Au, TaIrTe4, and SiO2.  It did not refine
   the Si substrate, surrounding air, or z PML, and it did not quantify
   previous-window versus late-window stationarity.  Diagnose temporal/Q
   closure first, then define a full-domain z sweep; do not call the current
   result a full z-mesh certificate.
   The replacement time/closure diagnostic is now complete and blocked:
   spatial Q NRMSE grows from 1.63% (24 periods) to 45.61% (32) and 97.13%
   (40), with negative late closed flux. Isolate the long-time FDTD instability
   before running any partial or full-domain mesh sweep. See
   `results_4um_time_absorption_closure/TIME_ABSORPTION_CLOSURE_FINDINGS.md`.
   It also found a 1.11--1.14% continuous-target versus then-realized float32
   discrete-ADE Q mismatch. Forward heat and direct-loss gradients use the
   realized discrete loss, and the current material builder refits the actual
   float32 ADE carrier response to <1e-5 complex-permittivity error. Old
   gradient/phase tables are explicitly stale.
6. Electrical void cells retain tiny sheet/contact floors to regularize the
   floating Au block.  Quantify floor sensitivity; do not describe the
   electrical `rho=0` endpoint as exactly disconnected until that passes.
7. The historical robust optimizer omitted nominal `eta=0.50` from its
   signed-current epigraph and constrained grayness only at nominal.  The code
   now includes eta=0.35/0.50/0.65 in both current and grayness constraints,
   but the corrected robust path has not been run.  Historical robust results
   remain invalid for promotion.  See `ROBUST_OBJECTIVE_AUDIT.md`.
8. Production optimization and exact-binary search are code-blocked until a
   confirmed device contract, full shared-linear mesh certificate, and a
   hash-linked shared-linear multidirection combined AD-FD certificate all
   pass. There is no runtime bypass. The certificate must name the selected
   full-domain-z grid, Courant factor, time windows, and same-grid Ea/Eb source
   calibration; all three production entry points now consume those exact
   values and reject a grid-hash mismatch. See `PRODUCTION_READINESS.md`.
9. The present square flake, full-edge terminals, unrotated x=b/y=a axes,
   100 nm thickness, 285 nm oxide, centered beam, and floating direct-contact
   Au are assumptions. The local 2026 paper explicitly makes the weighting
   field device-geometry dependent and its transverse example uses a 45 degree
   crystal/electrode angle. Confirm the target flake/electrodes/axes/stack and
   illumination in `physical_device_contract.json` before mesh certification.
10. The implemented current sign is `I=integral(J_local.grad(psi)) dA` with
    `psi(x_min)=0`, `psi(x_max)=1`; positive current is internal conventional
    current along solver `+x` (`x_min -> x_max`). The target remains `Ia>0`,
    `Ib<0`. Earlier prose saying positive current was right-to-left was wrong.
    The current network also does not yet support a rotated crystal/electrode
    geometry or off-diagonal in-plane transport tensors.

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
used by the project.  Launchers derive the repository root from their own
location.  `AU_DUALPOL_PYTHON`, `FDTDX_SOURCE_DIR`, and
`AU_DUALPOL_RAW_ROOT` override the host defaults when reproducing elsewhere.

## Next correct sequence

1. Confirm the target geometry, contacts, crystal-axis angle, layer stack, and
   illumination in `physical_device_contract.json`.
2. Treat the existing factor 1/2/4/8 tables as stale historical output, not as
   evidence for the current code or as a converged production mesh.
3. Finish `20_validate_4um_stable_time_contract.py`. Isolation already traced
   the factor-8/Courant-0.5 failure to Au and showed the same grid stable in an
   Au-only 32-period case at Courant 0.25; full-dispersion 32/40-period
   stationarity and absorption closure remain the gate. Only after that gate
   passes, expand z refinement to Si, air, and PML.
4. Revalidate the new shared linear material-fraction contract with endpoint,
   floor-sensitivity, and AD-FD checks on the selected mesh.
5. Check x/y optical convergence, thermal-mesh convergence, electrical-mesh
   convergence, and downstream PTE current.
6. Certify the combined gradient on the selected production mesh.
7. Only then restart LD_MMA continuation and finish with an exact binary
   500 nm solid/void audit.
