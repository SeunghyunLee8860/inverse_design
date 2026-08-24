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
   `lumerical_maxwell_contract.py` defines both the shared projected-density
   identity used during optimization and the stronger physical-geometry hash
   used at exact endpoints/final promotion. `lumerical_4um_exact_au.py` builds
   sampled-dispersive 4-um materials and deterministic exact-mask Au prisms
   for those endpoint/final controls; it is not the continuous optimizer
   carrier and remains audit/build code until B200 fit readback passes.
2. `fdtdx_4um_model.py` -- six-PML FDTDX Maxwell model and material layout.
3. `multiphysics_4um.py` -- conservative optical-Q remap, explicit 3-D
   thermal solve, electrical weighting solve, and PTE current.
4. `combined_4um.py` -- two-solve Maxwell adjoint and the complete optical,
   thermal, and electrical density gradient.
5. `dfm.py` -- 500 nm filter, differentiable solid/void constraints, and
   exact binary audit.
6. `au_density_relaxation.py`, `material_fraction.py`, and
   `MATERIAL_FRACTION_AUDIT.md` -- the selected nonlinear `n-k` Lumerical
   relaxation, the historical shared-linear FDTDX baseline, and the reason
   optical `rho**3` is removed.
   `lumerical_4um_density.py` defines the canonical 81x81 nodal projected
   state, Lumerical `importnk2` map, exact 80x80 four-node PDE cell average,
   transpose, and physical-coordinate-bound state hash.
7. `objective.py` -- signed current utilities and epigraph objective.
8. `10_optimize_4um_dualpol_au_ld_mma.py` -- nominal NLopt LD_MMA path.
9. `13_optimize_robust_binary_au_ld_mma.py` -- eroded/dilated robust
   continuation path.  Read `ROBUST_OBJECTIVE_AUDIT.md` before using it.
10. `14_diagnose_gray_law_mismatch.py` -- historical gray-material blocker.
11. `15_validate_4um_z_mesh_convergence.py` -- fail-closed full-domain-z gate
    that refines every Si/material/air/z-PML segment and must be closed before
    optical x/y convergence or another production optimization.
12. `production_readiness.py` and `PRODUCTION_READINESS.md` -- executable
    certificate chain that currently blocks all production entry points.
13. `CODE_PHYSICS_AUDIT.md` and `physical_device_contract.json` -- full
    physics audit and the deliberately blocked target-device contract.
14. `lumerical_4um_mesh_contract.py` and
    `22_audit_lumerical_4um_exact_au_runsetup.py` -- the sequential
    source/time/full-domain-z/x-y/PML/domain convergence matrix. The z axis
    links thin-stack refinement to explicit Si-bulk/air/PML refinement, so it
    does not repeat the old partial-layer sweep. Audit-only output is not a
    mesh certificate and the actual Maxwell runner remains B200-blocked.
15. `23_audit_4um_au_density_relaxation.py` -- solver-free exact endpoint,
    passivity, no-rho-cubed, and analytic complex-derivative gate for the new
    optical law. It does not replace any B200 field or AD-FD gate.
16. `24_audit_4um_density_state_map.py` -- solver-free certificate for the
    canonical 81x81 nodal state, 80x80 PDE cell map, state hash, exact
    transpose identity, and centered directional FD. It does not certify the
    Lumerical component-Yee Jacobian.
17. `lumerical_4um_forward.py` and
    `25_run_lumerical_4um_exact_au_control.py` -- the actual fail-closed B200
    source/exact-empty/full/simple-L/imported-density runner. It records actual
    mesh coordinates, fitted and finite-dt material readback, native-Yee Q,
    six-face flux, a fixed air-side endpoint field, raw engine-log GPU
    evidence, and either the canonical exact-Au geometry or canonical nodal
    density identity. Exact controls also retain the disjoint coordinate-based
    Au/TaIrTe4/SiO2/Si/air Q partition; gray density correctly marks that
    partition not applicable. Raw epsilon remains saved because conformal
    interface cells cannot be reduced to one physical label. It contains no
    HEAT/CHARGE or alternative Maxwell solver. No Maxwell result exists here
    because this Codex host has no B200.
18. `run_lumerical_4um_endpoint_b200.sh` -- sequential Ea/Eb exact-empty,
    source-only, imported-rho0, imported-rho1, and exact-full batch. A passed,
    hash-matching source-only JSON is mandatory before every material case.
    FSP/NPZ/log outputs belong in the supplied external/local output root and
    must not be added to Git.

The scripts `00` through `09` contain the runsetup, source calibration,
forward, thermal/electrical, and AD-FD certificates used by the code above.

## Current code assumptions -- not yet a confirmed physical device

- wavelength: 4 um
- scalar Gaussian waist: 4 um
- optical domain: 20 x 20 um laterally, six PML boundaries
- source aperture: 16 x 16 um
- fixed TaIrTe4 flake: 16 x 16 x 0.1 um
- Au design region: 8 x 8 x 0.05 um
- design topology: 81 x 81 projected nodes spanning 80 x 80 physical cells at
  100 nm pitch; custom thermal/electrical maps use the exact four-node cell
  average and its transpose
- reporting incident power: 285 uW
- minimum solid and void feature audit: 500 nm
- no symmetry, volume-fraction, or connectivity constraint
- no Q clipping, smoothing, gain, polarization matching, or closure rescaling

## Important blockers -- do not silently bypass

0. The user-selected Maxwell solver is Lumerical FDTD, while thermal and
   electrical remain the repository custom CUDA PDE solvers; no Lumerical HEAT
   or CHARGE license is assumed. Read `LUMERICAL_MAXWELL_GPU_PDE_ROUTE.md` and
   run `21_audit_lumerical_maxwell_preflight.py` first. The selected method is
   density topology, not shape/level-set: latent rho -> 500-nm filter -> tanh
   projection -> one shared 81x81 nodal projected occupancy. Lumerical optical uses the
   published nonlinear `n-k` interpolation in `au_density_relaxation.py`; it
   does not use optical `rho**3`. Custom thermal/electrical 80x80 cell fields
   must be derived from that hash-bound nodal state by the committed average
   operator; independently optimized/resampled rho fields are prohibited. Final
   promotion still requires an independent exact-binary ordinary
   sampled-data dispersive-Au reevaluation. FDTDX/JAX results are historical
   diagnostics only. The current host has RTX 6000 Ada GPUs, not B200, so it
   cannot issue a B200 run certificate.

   The older material readback froze only the central 4-um n,k values. That is
   insufficient for a time-domain claim of dispersive Au. The new Lumerical
   builder samples Ordal Au, anisotropic TaIrTe4, and Kitamura SiO2 over a
   3.2--4.8 um guard band around the 3.6--4.4 um source pulse. Its status is
   deliberately `NOT_FIT_READBACK` until the actual Lumerical fitted material
   is read back and compared on the B200 run.

   The new runner enforces this readback at 81 points over 3.6--4.4 um and
   also checks Lumerical's finite-time-step numerical permittivity. A material
   run additionally requires a passed all-air source-only JSON with an exact
   hash match on polarization, source waist input, time, x/y/z meshes, PML,
   and domain. The 285-uW normalization is applied only to reported scalar
   absorbed power; field and Q arrays remain raw.

   The earlier `np density` carrier claim is retracted. `np density` is a
   semiconductor electron/hole-density attribute, not an Au topology field.
   Its code and tests were removed; read `NP_DENSITY_ROUTE_REJECTED.md`.
   Lumerical 2026 R1.3 is not required on that basis. Decide version
   compatibility only from an ordinary exact-Au control on the actual B200.

1. The existing optimization used inconsistent O3/TE1 gray laws. The later
   shared-linear fraction was a consistency diagnostic, not the selected
   Lumerical optical law. The replacement `n-k`-then-square relaxation is now
   implemented and solver-free tested, but it is still blocked pending B200
   4-um endpoint parity, quantified source-band error, uniform-density
   resonance sweep, component-Yee
   Jacobian FD/transpose tests, and full combined AD-FD. See
   `MATERIAL_FRACTION_AUDIT.md`.
2. AD-FD validates the derivative of a chosen discrete mesh; it does not
   certify mesh convergence.
3. The original optical z mesh used only 2 Au cells and 5 TaIrTe4 cells.
   A historical partial z sweep checked Au/TaIrTe4/SiO2 factors 1, 2, 4, and
   8, but its tables are now explicitly stale: they use O3/TE1, the old
   Shockley-Ramo sign, and a cache key bound only to the checkpoint. Do not use
   its reported Q/current changes for the exact-Au route. Rebuild the
   convergence path around ordinary dispersive-Au Lumerical geometry.
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
   The replacement full-domain script uses factors 1/2/4, Courant 0.25,
   40 total periods, and a 4-period late window. It recalibrates Ea and Eb
   separately on every grid, rechecks time stationarity and Q/TD/phasor flux
   in every material case, and persists hash-verified per-case progress.
   That replacement sweep is now complete and **blocked**, not converged. All
   18 individual runs passed their time/closure/remap/linear-solver physics
   gates, but all six robust-density/polarization comparisons failed the final
   factor-2 to factor-4 spatial gate. The worst changes were 3.314% in total Q,
   34.072% in the remapped Q field, 3.634% in the TaIrTe4 temperature field,
   30.150% in Tmax, and 37.664% in PTE current. See
   `results_4um_shared_linear_full_z_convergence/FULL_Z_CONVERGENCE_REPORT.md`.
   This is decisive evidence that the legacy shared-gray FDTDX grid is not
   converged, but it does not select a production mesh for the superseding
   exact-Au Lumerical route. Do not spend more production effort on mixed-gray
   FDTDX refinement; repeat the convergence hierarchy with the validated
   Lumerical density carrier and finish with ordinary exact-Au binary geometry
   on the actual target GPU.
6. Electrical void cells retain tiny sheet/contact floors to regularize the
   floating Au block.  Quantify floor sensitivity; do not describe the
   electrical `rho=0` endpoint as exactly disconnected until that passes.
7. The historical robust optimizer omitted nominal `eta=0.50` from its
   signed-current epigraph and constrained grayness only at nominal.  The code
   now includes eta=0.35/0.50/0.65 in both current and grayness constraints,
   but the corrected robust path has not been run.  Historical robust results
   remain invalid for promotion.  See `ROBUST_OBJECTIVE_AUDIT.md`.
8. Production optimization is now unconditionally code-blocked because the
   existing entry points still implement the historical gray/FDTDX path.
   Legacy shared-linear certificates cannot clear this gate. Connect the new
   Lumerical `n-k` density carrier to the component-Yee discrete adjoint, then
   issue new certificates naming the selected
   full-domain-z grid, Courant factor, time windows, and same-grid Ea/Eb source
   calibration. The combined adjoint also derives
   its Au/TaIrTe4 material offsets from the realized placed slices; do not
   reintroduce baseline `LAYOUT` offsets. See `PRODUCTION_READINESS.md`.
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
photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh \
  -m pytest -q \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/test_preflight.py

photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/23_audit_4um_au_density_relaxation.py

photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_combined_gpu_python.sh \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/24_audit_4um_density_state_map.py

LUMERICAL_B200_GPU_INDEX=<physical_b200_index> \
  AU_LUMERICAL_ROOT=<absolute_v261_install_root> \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_lumerical_4um_endpoint_b200.sh \
  /absolute/local/output/root

CUDA_VISIBLE_DEVICES=<free_gpu> photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_z_mesh_convergence_gpu.sh --audit-only

CUDA_VISIBLE_DEVICES=<free_gpu> photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_z_mesh_convergence_gpu.sh
```

`run_combined_gpu_python.sh` selects the checked Python/JAX/PyTorch environment
used by the project.  Launchers derive the repository root from their own
location.  `AU_DUALPOL_PYTHON`, `FDTDX_SOURCE_DIR`, and
`AU_DUALPOL_RAW_ROOT` override the historical host defaults.
`AU_LUMERICAL_PYTHON` and `AU_LUMERICAL_ROOT` select the B200 host's Python
environment and v261 installation without assuming the login user's home.

## Next correct sequence

1. Confirm the target geometry, contacts, crystal-axis angle, layer stack, and
   illumination in `physical_device_contract.json`.
2. Treat all existing FDTDX factor 1/2/4/8, reduced-Courant, and shared-linear
   full-domain-z tables as historical diagnostics, not evidence for Lumerical
   or a production mesh. The completed shared-linear factor-1/2/4 sweep is
   useful negative evidence: its stable final pair failed in all 6/6 cases.
3. On the actual B200, use `25_run_lumerical_4um_exact_au_control.py` to pass
   source-only Ea/Eb first, then matching ordinary empty/full/simple-L exact
   Au time, Q/flux, linked stack+bulk/air/PML-z, x/y, PML-layer, and domain
   controls. Run imported-rho0/1 parity against the matching ordinary
   empty/full baselines,
   quantify the single-frequency carrier's source-band error, and sweep
   uniform projected density for artificial field/Q resonances. Raw output
   must remain outside the Git worktree. The unified runner and endpoint batch
   now exist, but no B200 result is committed and this Codex host fails the
   B200 preflight.
4. Build and validate the nonuniform density-to-component-Yee material
   Jacobian and its discrete adjoint; do not substitute bundled LumOpt's
   real/lossless metal path.
5. Check x/y optical convergence, thermal-mesh convergence, electrical-mesh
   convergence, and downstream PTE current.
6. Certify the combined gradient on the selected production mesh.
7. Only then start LD_MMA filter/projection continuation and finish with an
   independent 500-nm solid/void audit plus ordinary dispersive-Au binary
   reevaluation.
