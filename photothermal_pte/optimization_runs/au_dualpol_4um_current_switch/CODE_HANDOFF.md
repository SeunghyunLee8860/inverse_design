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
    `25_run_lumerical_4um_exact_au_control.py` -- the actual fail-closed GPU
    source/exact-empty/full/simple-L/imported-density runner. It records actual
    mesh coordinates, fitted and finite-dt material readback, native-Yee Q,
    six-face flux, a fixed air-side endpoint field, raw engine-log GPU
    evidence, and either the canonical exact-Au geometry or canonical nodal
    density identity. Exact controls also retain the disjoint coordinate-based
    Au/TaIrTe4/SiO2/Si/air Q partition; gray density correctly marks that
    partition not applicable. Raw epsilon remains saved because conformal
    interface cells cannot be reduced to one physical label. It contains no
    HEAT/CHARGE or alternative Maxwell solver. Its default accelerator policy
   remains B200. The explicit `development` policy permits RTX debugging but
   marks every result non-promotable.
   `import_density` accepts either a uniform scalar `--rho` control or an
   explicit nonuniform 81x81 NPY/NPZ `--rho-file`. The file input is
   validated, SHA-pinned, labeled by the canonical density-state hash, and
   copied into the external raw NPZ with its physical x/y coordinates. This
   is the first runner path that can carry an actual optimizer topology into
   Lumerical; never reconstruct it from a scalar or an 80x80 PDE cell grid.
18. `run_lumerical_4um_endpoint_b200.sh` -- sequential Ea/Eb exact-empty,
    source-only, imported-rho0, imported-rho1, and exact-full batch. A passed,
    hash-matching source-only JSON is mandatory before every material case.
    FSP/NPZ/log outputs belong in the supplied external/local output root and
    must not be added to Git.
19. `run_lumerical_development_gpu.sh` -- explicit non-B200 development
    launcher. It never issues a B200 certificate and keeps the B200 launcher
    unchanged.
20. `lumerical_4um_yee_jacobian.py` -- Au-specific, nonperiodic sparse
    `J_c=d epsilon_Yee,c/d rho_bar` construction. It measures Lumerical's
    complete `importnk2 -> index_detail` map with 25 layout-only colors,
    centered interior differences, feasible one-sided 0/1 endpoint
    differences, locality/roundtrip gates, independent mapping FD, and exact
    real-design JVP/VJP transpose tests. It performs zero Maxwell solves. The
    solver-free synthetic tests pass; an actual hash-linked Lumerical density
    FSP certificate is still required before adjoint use.
21. `26_build_lumerical_4um_yee_jacobian.py` and
    `run_lumerical_layout_python.sh` -- consume a SHA-pinned completed
    nonuniform `import_density` FSP/result/density triplet, verify the same
    solver version and native field/index Yee coordinates, build the sparse
    matrices without `fdtd.run`, run independent FD/transpose gates, and save
    raw matrices/coordinates outside Git. The launcher is layout-only and
    deliberately performs no GPU/B200 Maxwell certification.
22. `FDTDX_FRESH_DEPENDENCY.md`, `FDTDX_FRESH_ANCHOR_PLACEMENT.md`,
    `FDTDX_FRESH_SOURCE_ONLY.md`, `FDTDX_FRESH_EXACT_BINARY_PILOT.md`, and
    `fdtdx_fresh_exact_binary_matrix.py`, and
    `fdtdx_fresh_time_settling_certificate.py`,
    `fdtdx_fresh_courant_certificate.py`,
    `fdtdx_fresh_full_z_certificate.py`, and
    `run_fdtdx_fresh_full_z_campaign.sh` -- the separate pinned-FDTDX
    forensic/rebuild track. Read `FDTDX_INCREMENT_STATE_CANDIDATE.md`
    and `FDTDX_INCREMENT_STATE_INTEGRATION.md` for the cancellation-resistant
    ADE, reproducible two-patch fork, checkpointed AD-FD evidence, runtime
    boundary, and the remaining continuous-device mixing blocker. Its runtime,
    mesh, six
    PML faces, source pair, placements, exact endpoint material readback, and
    component-Yee-volume energy balance are fail-closed. This track must not
    edit, launch, or reinterpret the concurrent Lumerical work.

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
   cannot issue a B200 run certificate. RTX development runs are allowed only
   through the explicit development policy and must be repeated on B200.

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
   and domain. It also requires the same accelerator policy, physical GPU UUID,
   and Lumerical solver version. The 285-uW normalization is applied only to
   reported scalar absorbed power; field and Q arrays remain raw.

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

## Parallel fresh FDTDX forensic track

This branch also contains an independent reconstruction of the historical
FDTDX path so its old failures can be diagnosed from first principles. It is
not the selected Lumerical production path and it does not authorize changes
to the concurrent Lumerical session.

The fresh track pins FDTDX commit
`f26f84b70a8cceec9b889553955a868624736bf1`, a locked Python/JAX/CUDA
runtime on B200 GPU 7, explicit six-face PML and source contracts, and the
`196 x 196 x 160` anchor placement. The promoted all-air Ea/Eb source-pair
certificate SHA-256 is
`cc86457678ba50becff8ec44408f7f519a8fd3ae44abedc248082eefeee28ee6`;
it forbids polarization-specific normalization.

The exact-empty/full x Ea/Eb four-case optical control matrix is complete. All
four cases passed material readback, passivity, finite-value, previous/late
stationarity, nonnegative absorbed-power, TD/phasor agreement, and
Q/closed-surface gates. Both empty Au heat values are exactly zero. Both full
cases contain 6,400/6,400 pure-Au design pixels with no gray density or rho
exponent. The machine-readable matrix certificate re-hashed the four reports
and NPZ files and recomputed component-Yee-volume Au/TaIrTe4 powers; all 32
top-level gates passed. Its SHA-256 is
`06e69f15e292ef29b6515282332b01d1e88c8348cfa965a951d3c1c3e98a431b`.
Read `FDTDX_FRESH_EXACT_BINARY_PILOT.md` for all external hashes and metrics.
Those endpoint controls did not establish time or mesh convergence. The L500
time-settling and Courant results below close only the time-duration and
time-step axes; no spatial-mesh, adjoint, thermal/electrical, PTE-current, or
optimizer claim is authorized.

The v2 campaign is specified by `FDTDX_FRESH_CONVERGENCE_DESIGN.md`; every
source-only report, Ea/Eb pair, and material report binds one canonical
`MeshSpec + time + CPML` request and external case-file SHA-256. The first GPU
stage is now complete for the exact-binary 375-pixel
`l_shape_4um_with_500nm_arms` reference. Runs were made at clean commit
`01a8ad8a`; the independent verifier and its 12 focused tests were committed
and pushed as `5e376ce1`. That checkpoint had 133 explicit FDTDX `unittest`
tests. The current suite has **147 passing tests** after the candidate material
pair verifier, 32-period extension coverage, and long-time ADE precision gate
were added. The separate pytest forensic file is not runnable in
the locked fresh venv because pytest is intentionally absent.

External raw root:
`/home/seunghyun200/fdtdx_results/l500_time_settling_01a8ad8a_20260824`.
The clean-commit certificate is
`time_settling_certificate_5e376ce1/FDTDX_FRESH_TIME_SETTLING_CERTIFICATE.json`,
SHA-256
`20ab99b8488606475d2ed8d604d1810c9f3953176b68f42ed0689685ed505ab0`.
It re-hashed all three numerical cases, three source pairs, six material reports
and six NPZs, reconstructed the exact L500 mask and component-Yee volumes, and
recomputed raw Au/TaIrTe4 Q and field/Q stationarity. All 21 top-level and five
selection gates passed.

The 16-period cases remain rejected settling controls: Ea field stationarity was
`1.1229036e-2`; Eb field stationarity was `1.7813813e-2` and spatial Q change
was `5.5404220e-3`. Both 24- and 32-period Ea/Eb cases passed their internal
gates. Both 16-to-24 and 24-to-32 comparisons passed every optical limit. The
selected duration is **24 periods** and the confirmation duration is **32
periods**. This is only a time-settling certificate: `is_mesh_certificate=false`
and `optimizer_start_allowed=false`.

The exact same L500 reference was then run at 24 periods on Courant levels
`[0.5, 0.375, 0.25, 0.1875]`, with a distinct hashed case, Ea/Eb source-only
run, source-pair certificate, and Ea/Eb material run at every level. Raw root:
`/home/seunghyun200/fdtdx_results/l500_courant_4d79a439_20260824`. The
clean-commit certificate is
`courant_certificate_876cfff3/FDTDX_FRESH_COURANT_CERTIFICATE.json`, SHA-256
`7fd86bc8582d27002c226b6395a7d803f29ba98deda4abff00e60def9560a869`.
It was generated at clean commit `876cfff3`; all gates passed.

The 0.5-to-0.375 pair is preserved as a real coarse failure: worst
material/Cartesian Q-component change was 2.339%, above the 2% gate. The
0.375-to-0.25 and 0.25-to-0.1875 comparisons both passed, so the selected
Courant factor is **0.25**, with **0.1875** as confirmation. The first three
levels came from clean commit `4d79a439`; the 0.1875 extension came from
`14624869`. The certificate explicitly audits that only certificate/test files
changed between them and verifies identical runner, source-pair generator,
material-contract, pinned-FDTDX, and runtime provenance. It does not hide the
cross-commit origin.

The time and Courant certificates remain `is_mesh_certificate=false` and
`optimizer_start_allowed=false`. The first fresh full-domain-z ladder was then
run at clean commit `150a7592` for z factors 2, 4, and 8, giving grids
`196 x 196 x 80`, `196 x 196 x 160`, and `196 x 196 x 320`. Raw root:
`/home/seunghyun200/fdtdx_results/l500_full_z_150a7592_20260824`. The corrected
clean-commit certificate is
`full_z_certificate_7b687684/FDTDX_FRESH_FULL_Z_CERTIFICATE.json`, SHA-256
`319743a29b8dd4869c5d1feedf564850ff10e4c30fb1888fd28eb7ed8764036c`.
Its raw generator commit is `150a7592`; the corrected verifier commit is
`7b687684`.

All nonselection gates pass: canonical contracts, distinct source pairs, source
and material NPZ hash/schema/finite-value checks, exact raw grid coordinates,
solver-independent placement, repository/runtime/FDTDX provenance, CFL, physical
end time, binary masks, and conservative common-volume remap. The ladder itself
is rejected. z2-to-z4 and z4-to-z8 both fail. For z4-to-z8 the worst metrics are
1.8458 percent total-Q change, 6.8822 percent fixed-probe complex-E NRMSE,
13.9248 percent conservative spatial-Q NRMSE, and 93.8685 percent exact-Au
material-field NRMSE, versus limits of 1, 2, 5, and 5 percent. The source change,
closed-flux consistency, and within-case stationarity pass. No z factor is
selected.

The canonical z16 extension case was then created at commit `8766b3c6` (file
SHA-256
`74fca414c3c82ce1031f0f688cab0c3a3d252de6ea66e2fceb22ee40c0493e3a`).
It has a `196 x 196 x 640` grid, 1.5625-nm Au cells, and 1.25-nm TaIrTe4 cells.
The first source-only Ea preflight failed before any FDTD field solve: the
single-Drude realized float32 ADE error was `1.17579e-4`, above the frozen
`1e-5` gate. Preserve that partial directory; under that rejected single-pole
law, no z16 source pair or material result was created.

The clean-commit diagnostic added and pushed as `a4cf66d5` binds the z16 case,
failure JSON, material contract, optical model, and pinned FDTDX recurrence.
Its external JSON is
`/home/seunghyun200/fdtdx_results/l500_full_z_150a7592_20260824/ade_precision_a4cf66d5/FDTDX_FRESH_ADE_PRECISION_DIAGNOSTIC.json`,
SHA-256
`bfa98e74b81eae816b888bfbe1b460f94d5cf407f4be4954742c91e2b540911c`.
The current z16 Au search gives `1.1757867e-4`; a wide 0.01-to-10
damping scan still bottoms out at `2.2144332e-5`. Do not loosen the material
gate or rerun the unchanged model. The full-tensor follow-up committed at
`ecc33c22` is external at
`ade_precision_ecc33c22/FDTDX_FRESH_FULL_MATERIAL_ADE_PRECISION_DIAGNOSTIC.json`,
SHA-256
`cb15e83073887fc0b7bd328f81b1b5463087024d98277bd740027bd82a412741`.
It additionally finds that z16 TaIrTe4 a fails at `2.7593129e-5`; at z32 Au
and TaIrTe4 a/b/c all fail the current single-pole gate. Stable positive
two-pole candidates with recurrence roots no larger than one reach
`7.2571180e-9` for Au, `2.1516201e-8` for TaIrTe4 a, and `2.1030075e-8` for
TaIrTe4 b/c, but are not promoted. The algorithm must receive a separate
material-law hash, exact two-pole readback on every axis, time/stationarity
validation, and same-law z8/z16/z32 reruns. Old single-pole and new two-pole
levels are not comparable.

The candidate-law generator was committed at `f959a9ef`. Clean external z8,
z16, and z32 law-file SHA-256 values are respectively
`6352e58e0b3b2449f5316948adb3247bfc9c71547cbb2252a8beba69571d67bc`,
`558eae569446993096081320c1f6e9439ee78ef799c8aeb0b0af8810a72e6fb2`,
and `302ab4e8991b55d0fb17c2ff5332b156fb29401ac04e026e0394f7e6c1fbcd1d`.
The new z32 canonical case-file SHA-256 is
`33398486f542fa0f1c7b063011e61992f7830b7cd36c25c8d6863c553aa3fbf4`;
its grid is `196 x 196 x 1280`. These law files remain candidate-only. They are now
applied only by the zero-time-step placed-array preflight below; no field run uses them yet.

The pinned coefficient preflight was committed at `7504045c`. Its clean z8,
z16, and z32 external JSON SHA-256 values are
`1b892395e5d989dcb12a679d0d0c19389d2017cff026326670fd9a074cf0aeb2`,
`aa91d260271982f2bf3c4aba523cda6127a18ab6ddac50514628fbe8f5f59a9f`, and
`4f5e3da15bcbc571fd8c9d98bc30ca4be4f5b5ac8b3266efe78a01d93d9202b6`.
All four axes at all three levels reproduce c1/c2/c3 bit-exactly through
pinned FDTDX and c4 is exactly zero.

The opt-in model adapter and placed solver-array preflight were committed and
pushed as `011e0d36`. The historical single-pole path remains the default; the
candidate path requires the exact canonical law self-hash and rejects adjoint
placement. Clean GPU-7 outputs are external under
`/home/seunghyun200/fdtdx_results/l500_full_z_150a7592_20260824/two_pole_solver_array_011e0d36`:

- z8 JSON SHA-256: `126ac0aa053cc31c576700f1527e8a6f9a9d1dbdbda433bf7b38df13f272ec5c`
- z16 JSON SHA-256: `28887e54bf29b51f818b962e0072fb774cb496e9bf19cfcf4c0bf858c9d0465c`
- z32 JSON SHA-256: `adfa0e0332bc487df31296b7116ea757f501a408f73e77939cf989ec68c74266`

All three reports pass. Their actual c1/c2/c3 arrays have shapes
`(2,3,196,196,320)`, `(2,3,196,196,640)`, and
`(2,3,196,196,1280)`; c4 is absent. Every Au pole is read back at exact air/Au
binary endpoints, every TaIrTe4 pole preserves `Ex->b`, `Ey->a`, `Ez->c`, and
Si/SiO2 inverse permittivity, dt, PML, mesh, case, and law hashes match. The
worst realized-epsilon relative error is below `3.1e-8` and every axis remains
passive. These runs execute zero FDTD time steps.

This closes only the placed solver-array blocker. It is not a material, field,
time, source-pair, or mesh certificate.

A distinct candidate-bound all-air source/pair path was committed and pushed as
`e722ba73`; the historical source/pair status and CLI remain separate. The clean
z8 Ea/Eb runs and pair are external under
`two_pole_forward_e722ba73/z8`. Artifact SHA-256 values are:

- Ea report: `30cbc8b18c5aaaa289994b5bafe2c7b8821983aff31b7f97e7c9647d4b113901`
- Ea NPZ: `7e586000eb4a5681011062f9fe78e972120a8b0bb9b05c3eda55fd24f326d133`
- Eb report: `49788884d2ce62660cfba923a123940426211f8394fe98c6103a7058782bf459`
- Eb NPZ: `93cff39dbeaf200f0db43987c077b700ae1713774c71fc480bc7c982f8e393e1`
- original `e722ba73` source-pair certificate:
  `d5196cc7c715260e5c0436ccc59ca258c22173ae9adc69340405dc5e1e05a582`
- current source-pair certificate regenerated after the `7c527b6d` generator
  provenance hardening:
  `28c84ac1b2c21cb6d6db537248f90101ca2add6b37aef49c002da0b7b214fa64`

Both source cases pass every finite-value, stationarity, polarization, beam,
and closed-flux gate. Ea/Eb incident powers are `1.8834239723e-12` and
`1.8834237555e-12 W`; their relative mismatch is `1.1513098e-7` versus the
`5e-3` gate. The pair reconstructs the exact canonical material law, case-file
byte hash, model audits, and raw NPZ hashes, and creates only one common scale.

The current pair, not the original certificate, is the input bound into the z8
material runs. The candidate-only forward material runner was committed at
`7c527b6d`; the fail-closed pair verifier and six focused tests were committed
at `b489bbdc`. External material artifacts are:

- Ea report:
  `215e8eaf37788623dd4c0f98f9e6661f462b49618867cfd572d8ccf10fb978c8`
- Ea NPZ:
  `61ffeec5dd24d2bcc85f5bf83e8bd692af90ec7789a062797b21b6ea216d9cea`
- Eb report:
  `b29f9832f601079737f59a2953bcf7332f4eb23948e6a7ddaaceee55a6615c02`
- Eb NPZ:
  `54a586db06fa9deaa058bae48d3510e381db3b3824140c571dee56e92fc2b8c4`
- material-pair certificate:
  `ac71dd8e786ee3ca59d86f233f3d20b980bf3796abbc3a2b30167d9bae78b3d5`

The certificate lives at
`two_pole_forward_e722ba73/z8/material_pair_eefae409/`
`FDTDX_FRESH_TWO_POLE_EXACT_BINARY_PAIR.json`. It was generated from a clean
`eefae409` tree and re-hashes both reports and both 101 MiB NPZs. It
reconstructs the canonical 375-cell L mask, verifies exact binary integer
readback with `rho_power=null`, recomputes all component-resolved Q integrals
from Q density and Yee dual volume, revalidates the current source pair, and
passes all top-level gates.

At the common 285 uW reporting power, Ea total/Au/TaIrTe4 absorption is
`67.3793/0.8049/66.5744 uW`; Eb is
`116.4644/1.7476/114.7167 uW`. Eb/Ea total absorption is `1.72849`. Maximum
complex-field stationarity NRMSE is `1.3853e-3` for Ea and `1.5225e-3` for Eb;
Q versus closed-flux mismatch remains below `0.52%` for both. These are optical
heat-source results, **not** a PTE-current magnitude or sign result. The
certificate records `pte_current_claim_allowed=false` and
`optimizer_start_allowed=false`.

The z8 candidate two-pole source and material pair is complete. The z16
all-air source pair was then completed at clean commit `bcbe5ecd` under
`two_pole_forward_e722ba73/z16`. Artifact SHA-256 values are:

- Ea source report:
  `9b6ced4e90c912a4ce00b99803b63645d4e85ec1a10a8f7c90ca6d9747298695`
- Ea source NPZ:
  `40a2f60caa2462f71c91a59a699f1e212204a49e654b0f97a6e2391b4bd28632`
- Eb source report:
  `3e9d1d07f68c53908e52678fd42ae4068286cece7f3ba4542c7bff6b1465045d`
- Eb source NPZ:
  `836a425b661c3aee8ec949cf7569ce4f55a1b2b389766fcb7eac5a86d734dd72`
- source-pair certificate:
  `7cfcd8280cf63194aa53f328661613dd942e2b0da0f4045f2b4d2b8f881c7d35`

Both source solves use the independent z16 case (`196 x 196 x 640`,
24,586,240 Yee cells, 307,249 steps) and took `862.79 s` and `864.00 s`.
Both pass every gate with maximum field-stationarity NRMSE below `2.98e-6`.
Ea and Eb incident powers are exactly equal at the recorded precision,
`1.8837208269e-12 W`, so the pair mismatch is `0.0`.

The matching 24-period z16 exact-binary material runs were then completed from
clean commit `502893da`; both are **blocked before mesh comparison**. Artifact
SHA-256 values are:

- Ea material report:
  `62f30a998513636a6f04b47bc01a04fe03d0350e5bb785660b08d665a45a4bc6`
- Ea material NPZ:
  `5c9091fab42e58dd226b010a9cea7a783da4f7e12344cd41f10e24261374eb1e`
- Eb material report:
  `403837ce5de0db120820013a5bbb09ff0146782b61a392c2b62966d8f8da939b`
- Eb material NPZ:
  `033247f55ceafb50530c2dad5bdb09751f02a477ed034c59d3d2f7f92fa6566e`
- current blocked-pair certificate generated at clean `eefae409`:
  `b590872c263b70d10e3355d936d41a8a74ef4e502a5957c5716054fa8c4f7b0c`

Ea fails complex-field stationarity: `1.8074e-2` versus the `5e-3` limit. Eb
fails the same gate at `1.9032e-2` and spatial Q stationarity at `5.2720e-3`
versus `5e-3`. Material readback, nonnegative Q, total-Q stationarity, and both
closed-flux closures pass; closure mismatch stays below `0.54%`. Raw component
analysis shows the driven in-plane component is stable (`0.167%` Ea Ey,
`0.163%` Eb Ex), while the z-normal field is not (`2.143%` Ea Ez, `2.361%` Eb
Ez), even after best complex-scale removal. Solid and air portions of the Au
window both show about `1.8-2.1%` change. This is a real 24-period settling
blocker, not an ADE/material/energy-conservation failure.

The current blocked certificate re-hashes and recomputes both raw NPZs; only
`case_status_scope_and_ready` and `case_evaluation_gates` fail. Its corrected
next step is a separately hashed longer-time z16 numerical case with its own
Ea/Eb source pair. The existing 24-period z16 source pair must not be reused
when time changes. No z8-to-z16 mesh comparison is accepted, and z32 still has
neither a source nor material pair. Do not compare an old single-pole field
result to a two-pole result.

The isolated z16 settling extension is now contracted at clean code commit
`84461793`. It holds the complete `196 x 196 x 640` spatial grid, Courant
`0.25`, source startup, analysis window, CPML, exact 375-cell binary Au mask,
and candidate two-pole material axes fixed, changing only total duration from
24 to 32 periods (`409,666` time steps). Its external bindings are:

- t32 case JSON SHA-256:
  `6476b57bd577bcba0106e42c85ceb1707256384ff2d6a41824e3a2a3de47ba2f`
- internal case-contract SHA-256:
  `0c30a5c68efb3b4a79fbd248db104919a439a539e5eff84e5b10f8bfbd6ab07f`
- t32 material-law JSON SHA-256:
  `717f5ed3d24c33ebd4f870b108a4b0c618e87aabc7144991207c18db9e0ced31`
- internal material-law SHA-256:
  `d4d140b09e624c5140f72778865fc9df60f8a79c2c0690de2b4d01ebf008cd70`
- placed solver-array preflight SHA-256:
  `e0e992e1fdaf4edfcb9f96842759ed1b2410b4f293516bd5067dc15021ab2a1b`

The sorted `material_axes` payload hash is identical for t24 and t32
(`445b5bf65eae93c5778edc8ee98b7abae4117bebec40937abdcaa494d08bb7aa`).
The zero-step preflight is `ready=true` with no failed checks and payload hash
`be6a3c2112ec84b7f4ffa274fa68731b62d2159782ebe6816b4cddcde04f982b`.
It reads back the full two-pole/three-component solver arrays, TaIrTe4 a/b/c
axis assignment, ordinary-Au endpoints, exact binary mask, Si/SiO2, PML, mesh,
and realized time step. At that checkpoint it authorized only a fresh t32
source pair; it was not a source, material, mesh, PTE-current, or optimizer
certificate.

The fresh t32 all-air source pair was subsequently completed at clean commit
`1c7cd8ee` under external root
`two_pole_forward_settling_1c7cd8ee/z16_t32`. Artifact SHA-256 values are:

- Ea source report:
  `beefc073c1f0403010858502ea42de452308e554f7847d3283173e495e5eef66`
- Ea source NPZ:
  `20b54b16d9f9a634a3a97a0a938e8cadc3e1d4f343496500b8160a228d481573`
- Eb source report:
  `0d565cfee7c435f46f8891c26dbd296bae04531b88dbe4170bb5af11e0026e9e`
- Eb source NPZ:
  `50862803d45551c9d508d939ab77427557aaab5147b1576b5588f6a50911eaa4`
- source-pair certificate:
  `278dff85e307042d1b7d004316ac74be010fb40593179b65948bbdc878c4b7bc`

Ea/Eb took `1144.41 s` / `1144.54 s`, passed every source gate, and
reported maximum complex-field stationarity NRMSE `4.0193e-6` / `4.0165e-6`.
Both unscaled incident powers are exactly `1.883720176371062e-12 W` at the
recorded precision, so pair mismatch is `0.0`; common-285-uW scaling is exact.
This source pair is valid only for this t32 case.

The matching t32 material Ea/Eb runs were completed from clean commit
`b662b07b` and are both **blocked**. The independent pair verifier was rerun at
clean `2edb38d8`. Artifact SHA-256 values are:

- Ea material report / NPZ:
  `337c7d6e07b8fa7da9cd8394a89c524ebbccef0a4bb00d0b0d1d69aecde965c0` /
  `2184cdb2a263f59ce96a2acbe3f4a654461482ef938756def5d30f8c66e42275`
- Eb material report / NPZ:
  `cdc4f1b1baa55c9153b314d8bfff7f35c851a235be7976f0efcc3bd12ae22317` /
  `3ad02aee9af1214fdf27fc3aa519160945e7bc7d42f387245d7c477047c5be89`
- blocked material-pair certificate:
  `999a28f273c15ef86d43e77112ca877a1c449ea14710a90f561389c94abc757e`

Ea still fails field stationarity at `1.6197%`. Eb is worse than t24: field
stationarity is `2.6502%` and spatial-Q stationarity is `0.6583%`, versus the
fixed `0.5%` limits. Every raw/canonical/source/material/readback gate passes;
only case readiness/evaluation fails. Q/closed-flux mismatch stays below
`0.52%`, and total Q is stable. The consecutive 4-period Au-window changes
over 16--32 periods are non-monotonic: Ea is
`1.807% -> 1.910% -> 1.620%`, and Eb is
`1.903% -> 2.958% -> 2.650%`. The dominant unsettled component remains Au
`Ez`; best complex-scale removal does not fix it. Therefore a longer t40/t48
run is not justified, and z32 FDTD remains forbidden.

Commit `2edb38d8` adds `fdtdx_fresh_ade_transient_precision.py` and four tests.
This CPU-only gate integrates the locked two-pole recurrence under the actual
four-period linear source ramp in float32 and float64 before any FDTD run. Its
external certificates are:

- z8/t24: file SHA-256
  `48e6780c39f4256eac0ba116460bc937dc62c34069bcf5f7be86e2122e70c4ce`,
  payload `63f491c00084bf56dad0e9a829b9cfa7d4085f3a45d9614318089203a6a8c734`,
  **validated**
- z16/t32: file SHA-256
  `426c067f4971edddd2435134d714efe2e20e6b78492c15207d3c8a83e4b3b191`,
  payload `3c847d98ee6e9493bb97845418aab7b394c11f81aed28a659df759e37a49daf0`,
  **blocked**
- z32/t24: file SHA-256
  `3397023337a48bc843eb28de38d82860a8567e9581b3c05a16eaae5c367176b4`,
  payload `77fb8033b54fd15144635e0720da6d28f209a5fab204180b803589257d641f0a`,
  **blocked**

For z16 Au the scalar float32 late-window drift is `1.713%`, while the same
locked coefficients with float64 state settle to `4.93e-10`; their late
responses differ by `3.069%`. The carrier denominator condition estimate is
`1.66e7`, or about `1.98` times the reciprocal float32 precision budget. z8 Au
passes (`0.0716%` drift, `0.2269%` float32/float64 difference), while z32 Au
is catastrophically blocked (`99.78%` late-response difference). This closes
the root cause as fine-dt float32 ADE recurrence conditioning missed by the
old one-frequency carrier-fit gate, not insufficient simulation duration.

Measured one-polarization runtimes are `260 s` for z8/t24, `861--863 s` for
z16/t24, and `1137--1145 s` for z16/t32. A dual-polarization forward pair can
run concurrently on two verified-idle GPUs, but one FDTDX solve is single-GPU.
Even with Ea/Eb parallel, z16/t32 forward-plus-adjoint is at least about
`38 min/iteration`; 100 Maxwell iterations would exceed `63 h` before
thermal/electrical work. This grid is validation-only and must not be used for
optimization. Future independent cases must use distinct GPUs only after
checking compute-process ownership; occupied GPUs are never touched.

Do not proceed to x/y, domain, PML, thermal/electrical, or optimization until
z convergence closes. Preserve the failed z2/z4/z8 comparisons and the z16
ADE failure. This FDTDX track must not edit, launch, or reinterpret the
concurrent Lumerical work.

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

LUMERICAL_GPU_INDEX=<free_rtx_index> \
  AU_LUMERICAL_ROOT=/opt/lumerical/v261 \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_lumerical_development_gpu.sh \
  photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/25_run_lumerical_4um_exact_au_control.py \
  --case source_only --polarization Ea --gpu-index <free_rtx_index> \
  --output-dir /absolute/local/output/root

CUDA_VISIBLE_DEVICES=<free_gpu> photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_z_mesh_convergence_gpu.sh --audit-only

CUDA_VISIBLE_DEVICES=<free_gpu> photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/run_z_mesh_convergence_gpu.sh
```

`run_combined_gpu_python.sh` selects the checked Python/JAX/PyTorch environment
used by the project.  Launchers derive the repository root from their own
location.  `AU_DUALPOL_PYTHON`, `FDTDX_SOURCE_DIR`, and
`AU_DUALPOL_RAW_ROOT` override the historical host defaults.
`AU_LUMERICAL_PYTHON` and `AU_LUMERICAL_ROOT` select the B200 host's Python
environment and v261 installation without assuming the login user's home.

## 2026-08-24 local RTX development evidence

Raw artifacts are outside Git under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.
Do not copy them into this worktree.

1. Lumerical v261 solver `8.35.4413` opened successfully. GPU logs identify
   physical RTX 6000 Ada GPU 5 and its UUID, contain the `fdtd-engine -gpu`
   command, GPU time-stepping timing, and successful completion.
2. The original 4.000-um source-object waist produced about 4.044 um at the
   flake plane and failed the 0.5% waist gate. One-step calibration selected
   `3.956143303046143 um`. Both Ea and Eb source-only baseline runs then passed;
   their realized effective waists were about 4.00077 um.
3. Baseline source mesh readback was `183 x 183 x 63`, with flake x/y maximum
   steps 100 nm and thin-stack maximum z step about 19.79 nm.
4. Ordinary sampled-data dispersive-Au `full/Ea` completed on that baseline.
   Every fitted and finite-dt material gate passed, native Yee Q was finite and
   nonnegative, and native Q agreed with `pabs_adv` to floating-point precision.
   It is nevertheless **failed**, because six-face flux exceeded native Q by
   30.43%. Moving the closed surface outside all estimated PML cells left the
   discrepancy unchanged, so PML-face placement is not the sole cause.
5. The linked 5-nm stack-z / 50-nm bulk-z Ea source-only run passed. After two
   license-blocked attempts, retry 2 acquired all nine tasks and completed the
   exact full-Au solve on GPU 5. Its realized grid was `183 x 183 x 212`; all
   material, mesh, GPU, decay, nonnegative-Q, and native-Q/`pabs_adv` gates
   passed. Q/flux still failed by 29.239%, versus 30.428% on the 20-nm/200-nm
   baseline. Fine z therefore does not explain the discrepancy. The next
   exact-empty control on this identical mesh/source then passed with only
   0.01636% Q/flux error. The blocker is therefore specific to the Au
   metal-interface discretization/absorption path, not the closed box or
   TaIrTe4 background. Do not change Q or flux formulas to force agreement.
   Run CV0 source/empty/full on the same spatial grid, then staircase if
   needed; Ansys explicitly warns about CV1 artifacts at extreme metal/
   dielectric permittivity contrast.
6. The default source-object waist in the unified runner and B200 endpoint
   batch is now the calibrated value. Every non-baseline mesh still reruns and
   revalidates source-only rather than assuming the calibration transfers.

## Next correct sequence

1. Confirm the target geometry, contacts, crystal-axis angle, layer stack, and
   illumination in `physical_device_contract.json`.
2. Treat all existing FDTDX factor 1/2/4/8, reduced-Courant, and shared-linear
   full-domain-z tables as historical diagnostics, not evidence for Lumerical
   or a production mesh. The completed shared-linear factor-1/2/4 sweep is
   useful negative evidence: its stable final pair failed in all 6/6 cases.
3. On the current RTX host, run CV0 source-only, exact-empty, and exact-full Ea
   on the same 5-nm/50-nm spatial grid. If full still fails, run the staircase
   trio before refining to 2.5-nm/25-nm z. The passed CV1 empty control and
   failed CV1 full control already exclude the closed box, TaIrTe4 background,
   and simple thin-stack z roughness as the sole root. These runs remain
   development evidence only.
4. On the actual B200, use `25_run_lumerical_4um_exact_au_control.py` to pass
   source-only Ea/Eb first, then matching ordinary empty/full/simple-L exact
   Au time, Q/flux, linked stack+bulk/air/PML-z, x/y, PML-layer, and domain
   controls. Run imported-rho0/1 parity against the matching ordinary
   empty/full baselines,
   quantify the single-frequency carrier's source-band error, and sweep
   uniform projected density for artificial field/Q resonances. Raw output
   must remain outside the Git worktree. The unified runner and endpoint batch
   now exist, but no B200 result is committed and this Codex host fails the
   B200 preflight.
5. Run the new component-Yee builder on a completed hash-identical nonuniform
   density FSP, then connect its validated sparse transpose to the discrete
   adjoint; do not substitute bundled LumOpt's real/lossless metal path.
6. Check x/y optical convergence, thermal-mesh convergence, electrical-mesh
   convergence, and downstream PTE current.
7. Certify the combined gradient on the selected production mesh.
8. Only then start LD_MMA filter/projection continuation and finish with an
   independent 500-nm solid/void audit plus ordinary dispersive-Au binary
   reevaluation.


## 2026-08-25 FDTDX increment-state update

Commit `05d8e9ba` added the CPU-only cancellation-resistant `(P, delta-P)` ADE candidate. At that checkpoint all z8/z16/z32 material-axis scalar gates passed and the FDTDX-related suite was `152 passed`; optimizer permission remained false. The old one-point CCPR fallback was rejected because passive candidates retained fine-dt cancellation while the better-conditioned candidate was non-passive. See `FDTDX_INCREMENT_STATE_CANDIDATE.md` for equations, hashes, runtime, exact promotion boundaries, and the required small forward/checkpointed-AD-FD sequence. Do not launch another long FDTDX pair yet.


## Fork-bound increment-state integration status

The clean isolated fork is now at
`6cc0e97252ee0b95de5016e8db1a5b414177efa4`. Patch `0001` adds the isolated
kernel; patch `0002` integrates the opt-in state through config, coefficient
placement, diagonal `update_E`, sources, mode detectors, and broadband
spectrum reconstruction. CCPR, oriented poles, and dispersive full tensors
fail closed. The actual-JIT z8/z16/z32 state gates still pass. One small driven
Lorentz `B` checkpointed full-FDTD AD-FD control passes with symmetric relative
error `2.5563e-4`. Patch `0003` adds the scoped-float64 Drude `C=0`
full-FDTD AD-FD gate, which passes at `4.6816e-7`; the full fork unit suite is `2605 passed, 2 skipped, 1 xfailed`; the project-side FDTDX audit suite is `164 passed`. See
`FDTDX_INCREMENT_STATE_INTEGRATION.md` for hashes and runtimes.

No GPU was used for this integration. Do not launch a long pair or any
optimizer yet. The next allowed solve is a short coarse exact-binary timing and
closure control after defining a newly hashed runner. Recheck live
compute-process ownership immediately before launch and run Ea/Eb concurrently only on two distinct idle GPUs. The generic continuous
`Device` path still interpolates `A/C` with density, so the gray optical law and
material-placement Jacobian remain blockers.

## Increment-state exact-binary pre-GPU checkpoint

The newly isolated project builder now selects the patched fork only through
`dispersive_state_representation="increment"`. It uses one mesh-independent
passive pole per Au/TaIrTe4 axis, records `A/C/B` semantics explicitly, and
leaves the historical builder default untouched. CPU placement on the anchor
grid (`196 x 196 x 160`, `25,664` steps) passes exact Au and complete material
stack readback. Realized 4-um epsilon errors are at most `1.00931e-5`, all
passive. The generic gray `Device` interpolation is still unvalidated and no
optimizer is authorized.

Use `fdtdx_increment_state_exact_binary_control.py` only for the next cold
forward timing/energy-closure control. The wrapper
`run_fdtdx_increment_state_control_gpu.sh GPU_INDEX Ea|Eb OUTPUT_DIRECTORY`
rejects a GPU with any existing compute process. Run Ea and Eb on distinct
verified-idle devices and keep their JSON output outside Git. The runner does
not reuse the old source pair and does not claim absolute absorption or an
Ea/Eb comparison. At this checkpoint no GPU run has occurred; commit/push this
state before launch, then record measured time and closure in a separate
commit. The concurrent Lumerical session remains out of scope.

## Increment-state B200 timing result and immediate blocker

At project commit `c843276d1265a4652355b73ceecda2ce5be6230f`, exact-L Ea/Eb
controls ran concurrently on verified-idle physical B200 GPUs 6/7. Cold
compile+forward was `24.652 s` (Ea) and `24.766 s` (Eb); total process time was
`48.076 s` and `47.631 s`, so the parallel pair took about 48 s. Peak JAX
bytes-in-use were about 3.71 GB per case. This makes the anchor forward
practical, but it is not an adjoint/full-iteration timing. External report
hashes are `36b48d9870b4cf46f2b5cc8159d9712e857af7ed6ec2f04a1d84c7fca45d485f`
(Ea) and `707bddd2fba704d9a4409139d54e4574b3f5a8c3c8d207e2c182f6c99221ec85`
(Eb), under `/home/seunghyun200/fdtdx_results/increment_state_control_c843276d/`.

Energy closure passed strongly (`9.0497e-5` Ea, `5.3532e-5` Eb Q/phasor
relative difference), but both reports are blocked by Au complex-field
stationarity (`1.1339e-2` Ea, `1.7934e-2` Eb versus `5e-3`). Eb spatial-Q
change is also `5.5919e-3` versus `5e-3`. Do not loosen gates or start mesh
convergence. Extend the identical control to 24/32 periods first. Lumerical
remains out of scope and no gray law or optimizer is authorized.

The time-control runner is now v2 and accepts canonical
`--total-periods/--window-periods`; defaults remain 16/4. For the immediate
settling check, invoke the safe wrapper with trailing
`--total-periods 24 --window-periods 4` on each polarization. This changes no
mesh, material, source, PML, mask, or gate.

## Passed 24-period increment-state control

The canonical 24/4 extension at commit `a7f2d6b9` passed every gate for both
polarizations. Shared case hash:
`4a1b16092a693953c075b9848bba3342951233b712e397005dc34312f6e30532`.
External report hashes are
`858f8d5b7ba42be29e18e0e1276a6da157d2cc21947c947a5a316f1f6baff309`
(Ea) and `ce7138c66301d7b16ba4f472a53a5c3e95e2aa9d89251f0714b724ec8e323d41`
(Eb), under
`/home/seunghyun200/fdtdx_results/increment_state_control_a7f2d6b9_t24/`.
Cold compile+forward was `36.429/36.693 s`; parallel pair total was about 60 s.
Au field NRMSE is now `1.858e-4/2.489e-4`, and Q/phasor closure is
`1.151e-4/7.047e-5`. Do not run 32 periods. Next generate a newly hashed
patched-fork 24-period source-only Ea/Eb pair, then begin spatial convergence.
Historical source pairs remain forbidden; gray law, adjoint timing, and
optimizer are still open blockers.
