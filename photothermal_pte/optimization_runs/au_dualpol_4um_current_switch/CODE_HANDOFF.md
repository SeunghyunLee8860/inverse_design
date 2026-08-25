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
   For an engine-completed exact control whose Python API client was
   interrupted after submitting the solve, `--recover-completed-fsp` loads
   the saved FSP and regenerates only NPZ/JSON postprocessing. It fails closed
   unless the completed engine log, requested GPU UUID, source hash, solver
   version, and mesh readback all match. That path never calls `runsetup`,
   `run`, `strict_gpu_run`, or `save`, and it does not accept gray-density or
   source-only cases.
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
    solver-free synthetic tests pass. A hash-linked nonuniform Lumerical RTX
    development certificate now also passes on the 5/50-nm staircase mesh:
    worst mapping FD error `3.54e-11`, worst transpose error `3.96e-16`, and
    zero Maxwell solves. This closes the local material-map derivative only;
    it is not a full Maxwell or combined AD-FD certificate.
21. `26_build_lumerical_4um_yee_jacobian.py` and
    `run_lumerical_layout_python.sh` -- consume a SHA-pinned completed
    nonuniform `import_density` FSP/result/density triplet, verify the same
    solver version and native field/index Yee coordinates, build the sparse
    matrices without `fdtd.run`, run independent FD/transpose gates, and save
    raw matrices/coordinates outside Git. The launcher is layout-only and
    deliberately performs no GPU/B200 Maxwell certification. New results also
    record the completed forward solver time and layout-session wall time.
22. `lumerical_4um_control_comparison.py` and
    `27_compare_lumerical_4um_control_pair.py` -- hash-verify two exact-control
    JSON/NPZ bundles, fail closed on any case/polarization/geometry/GPU/solver
    or fixed non-z-axis mismatch, apply each run's measured source-only power,
    and compare normalized Q, flux, complex endpoint field, and E2 with the
    finer result as denominator. This is explicitly a Maxwell sub-gate, not a
    volumetric-Q/thermal/current or production certificate.
23. `lumerical_4um_multiphysics_comparison.py` and
    `28_validate_lumerical_4um_z_multiphysics_pair.py` -- apply Ansys'
    official `pabs_adv` common-grid `Pabs * exact(index_x material mask)`
    definition, conservatively remap only identified material power without
    closure rescaling, and run the custom CUDA thermal/electrical solvers.
24. `29_extract_lumerical_4um_official_pabs.py` -- SHA-verify a completed FSP
    and run only its saved `pabs_adv` analysis to create a Pabs/index_x
    companion NPZ/JSON. It never reruns Maxwell. New exact-control runs save
    these arrays directly. Read `LUMERICAL_Z_MULTIPHYSICS_FINDINGS.md`: the
    finest Ea empty/full official-filter downstream pair still fails.
25. `lumerical_4um_official_downstream.py`,
    `lumerical_4um_interface_comparison.py`, and
    `30_validate_lumerical_4um_interface_methods.py` -- reuse the same
    official material filter and custom CUDA PDE path to hash-compare MCM6
    CV0, CV1, and staircase at one fixed mesh. Read
    `LUMERICAL_INTERFACE_METHOD_FINDINGS.md`: staircase was the historical
    linked-z diagnostic choice, not the current bounded-cost optimizer mesh
    and not a final mesh certificate.
26. `32_validate_lumerical_4um_component_yee_z_multiphysics_pair.py` --
    replaces the axis-biased common-grid `index_x` material partition for
    symmetric exact controls with collocated `Qx/epsilon_x`, `Qy/epsilon_y`,
    and `Qz/epsilon_z`; it restores the one-ppm zero-current gate without Q
    rescaling. Read `LUMERICAL_Z_MULTIPHYSICS_FINDINGS.md`.
27. `lumerical_4um_gray_q_coupling.py` and
    `33_validate_lumerical_4um_gray_q_cuda_pde.py` -- the relaxed-density
    optical-to-PDE route. It does not compare component epsilon with exact
    material values, because that would discard absorption in intermediate
    design samples. It maps every native `Qx/Qy/Qz` array to thermal-cell
    power by literal Cartesian overlap and applies the exact transpose back
    to native Q. The completed nonuniform Ea development forward passed this
    full custom-CUDA downstream chain in 20.79 s with zero new Maxwell solves:
    power and transpose errors were zero at reported precision and the
    native-Q/thermal-adjoint contraction error was `1.59e-16`. Its current was
    `-5.213 nA`; this is an unoptimized test state, not a switching result.
28. `lumerical_4um_adjoint.py` and
    `34_run_lumerical_4um_gray_maxwell_adjoint.py` -- the first complete
    relaxed-density Maxwell-gradient preparation on the Lumerical-only route.
    Script 34 hash-binds the R1.2 forward, native-Q custom-CUDA pullback, and
    sparse component-Yee material Jacobian; imports one official FieldRegion
    vector source; preserves the exact forward mesh with a zero-amplitude
    Gaussian mesh anchor; reconstructs the FieldRegion-only CW field from
    `cwnorm(1)`/`cwnorm(2)`; and forms indirect plus explicit-loss optical
    gradients and the direct PDE material gradient. The R1.2 Ea development
    run passed every preparation gate in 101.32 s, including exact source
    round trip, zero forward/adjoint grid difference, and a `1.41e-16` CW
    reconstruction residual. Its own artifact correctly retains
    `AD_FD_claimed=false`, because script 34 alone does not run finite
    differences.
29. `lumerical_4um_adfd.py`,
    `35_prepare_lumerical_4um_ea_combined_adfd.py`, and
    `36_compare_lumerical_4um_ea_combined_adfd.py` -- the first independent
    centered-forward gate for the complete Lumerical-Maxwell/custom-CUDA-PDE
    derivative. The direction is a deterministic low-frequency function of
    coordinates selected without reading the gradient. At `h=0.0025`, the
    R1.2 RTX Ea pair gave AD `-1.363032899e-8 A` and FD
    `-1.363002816e-8 A` per unit projected occupancy: same sign and relative
    error `2.207e-5` (0.00221%), with no fit or empirical rescaling. This
    certifies one projected-density direction on the current development
    mesh only; it does not certify Eb, latent filter/projection derivatives,
    mesh convergence, or B200 production.
30. `lumerical_4um_design_mapping.py` and
    `37_audit_lumerical_4um_latent_design_map.py` -- the Lumerical optimizer's
    actual design-variable chain. Both latent and projected occupancy are
    81x81 nodal arrays. The finite 500-nm conic filter, tanh projection, exact
    81x81-to-80x80 cell average, and every transpose are explicit. The old
    80x80-cell `dfm.MAPPING` remains historical FDTDX code and is not the new
    optimizer carrier. DFM opening residuals now operate on the derived
    physical cells and use a softplus positive part instead of nondifferentiable
    ReLU; its pointwise approximation excess is bounded by
    `positive_tau*log(2)`. Script 37 passed filter/projection and cell-chain
    transpose errors `2.99e-16`/`2.58e-15`, directional-FD errors
    `2.10e-9`/`1.18e-9`, and maximum DFM directional-FD error `5.87e-7`, with
    zero solver calls. Final promotion thresholds the four-node cell average
    and still requires ordinary dispersive-Au binary reevaluation.
31. `38_prepare_lumerical_4um_ea_latent_adfd.py` and the latent mode of script
    36 -- the complete optimizer-coordinate gate through the same nodal
    filter/projection used by the Lumerical carrier. A deterministic analytic
    81x81 latent state and independent direction were selected without fields
    or gradients. At beta 4 and `h=0.0025`, the R1.2 RTX Ea chain gave AD
    `-2.766595495e-8 A` and centered FD `-2.766380278e-8 A`: same sign and
    relative error `7.779e-5` (0.00778%). The projected-JVP and latent-VJP
    contractions agree to `1.20e-16`. The four required Maxwell solves
    (baseline, adjoint, plus, minus) used about 237 s of solver time in total;
    no Lumerical HEAT/CHARGE, FDTDX, empirical rescaling, or optimizer
    iteration was used. This closes one Ea latent direction on the current
    development mesh only.
32. Scripts 34, 36, and 38 now bind either `Ea` or `Eb` explicitly. The
    component-Yee material Jacobian may be reused across polarization only
    when the target forward has the same projected-density state, exact
    component epsilon hashes and shapes, sub-attometre Yee coordinates, and
    frequency. The complete raw NPZ is deliberately not an equality gate
    because its E and Q arrays are polarization-dependent. A solver-free Ea
    artifact audit and backward-compatible Ea latent comparison pass.
33. One complete beta-4 latent Eb direction now passes on the same R1.2 RTX
    development route. At `h=0.0025`, AD was `-5.529878050e-8 A`, centered FD
    was `-5.529062519e-8 A`, the sign agreed, and relative error was
    `1.4748e-4` (0.01475%). The mapping-chain transpose error was `1.20e-16`.
    The cross-polarization material-Jacobian audit passed before reuse. Four
    Eb Maxwell solves totaled about 269 s and three custom-CUDA evaluations
    about 55 s; no Lumerical HEAT/CHARGE, FDTDX Maxwell, or optimizer run was
    used. This closes one Eb derivative direction, not signed switching: the
    unoptimized Ea/Eb baseline currents are both negative.
34. `lumerical_4um_signed_objective.py` and
    `39_validate_lumerical_4um_signed_dual_objective.py` -- the first
    Lumerical-carrier objective-level gate. It hash-loads both passed latent
    certificates, requires the exact same latent/projected state, beta, step,
    and direction, pulls both gradients to latent rho, and forms
    `t-I_Ea<=0`, `t+I_Eb<=0`. The combined balanced-objective AD-FD error is
    `7.779e-5`; the two constraint errors are `7.779e-5` and `1.4748e-4`.
    It used no solver and leaves the optimizer disabled.
35. `lumerical_4um_adfd.py` and script 38 now expose four deterministic,
    low-frequency directions. Direction 0 retains semantic SHA
    `44f111...`; direction 1 has normalized overlap `-0.00720` with it. The
    second direction passes Ea AD-FD at `7.011e-5` relative error and Eb at
    `4.093e-5`; the signed objective and both constraints pass too. Its four
    forward solves totaled 225.7 s and four custom-CUDA evaluations 71.7 s.
    Ea/Eb then had two independent directions each.
36. Direction 2 also passes: Ea AD/FD
    `-2.440938241e-8`/`-2.440694620e-8 A` (error `9.981e-5`) and Eb AD/FD
    `-5.517740350e-8`/`-5.516697265e-8 A` (error `1.890e-4`). The signed
    objective and constraint gate passes. Its four forwards totaled 229.2 s
    and four custom-CUDA evaluations 72.7 s.
37. Direction 3 completes the planned four-direction development-mesh family.
    Ea AD/FD was `-7.435606117e-9`/`-7.434900185e-9 A` (error
    `9.494e-5`), and Eb AD/FD was
    `-1.759945920e-8`/`-1.759713187e-8 A` (error `1.322e-4`). The signed
    balanced objective and both epigraph constraints pass. Four Lumerical
    forwards totaled 232.3 s and four custom-CUDA evaluations 75.4 s. No
    optimizer, Lumerical HEAT/CHARGE, FDTDX Maxwell, mesh sweep, or B200 claim
    was made. The baseline remains non-switching because both currents are
    negative.
38. At the user's direction, the bounded-cost development/optimization mesh
    is CV0 `2.5/50 nm`, not staircase: 100-nm in-plane flake mesh, 2.5-nm
    thin-stack z mesh, 50-nm bulk/air/PML z mesh, 200-nm outer in-plane mesh,
    mesh accuracy 3, eight PML layers, 20-um lateral span, z = +/-3 um, and a
    1-ps window. Fresh R1.2 build-4522 source/empty/full controls passed for
    both Ea and Eb on `183x183x303`. Ea solver times were
    `37.60/125.00/148.44 s`; Eb times were `34.33/123.40/178.30 s`, totaling
    10.78 minutes. This is an engineering selection, not a formal convergence
    certificate.
39. The complete same-mesh CV0 beta-4 latent AD-FD chain now passes one common
    direction at `h=0.0025`. The component-Yee material-map FD error is
    `4.14e-11` worst case and its transpose error is `1.05e-15`. Ea AD/FD are
    `-2.795034298e-8`/`-2.794796295e-8 A` (error `8.515e-5`); Eb AD/FD are
    `-5.532360856e-8`/`-5.531639071e-8 A` (error `1.3047e-4`). The exact
    signed epigraph and both constraints pass. The unoptimized baseline is
    still non-switching at `I_Ea=-8.70019 nA`, `I_Eb=-16.8637 nA`; no
    optimizer or B200 promotion was claimed. The older four-direction family
    remains additional `5/50-nm` staircase evidence. Raw artifacts remain
    outside Git. The final certificates are in raw directories
    `r12_ea_latent_beta4_dir0_combined_adfd_result_z2p5_bulk50_cv0_v1`,
    `r12_eb_latent_beta4_dir0_combined_adfd_result_z2p5_bulk50_cv0_v1`, and
    `r12_ea_eb_latent_beta4_dir0_signed_objective_z2p5_bulk50_cv0_v1`. Use
    Jacobian directory `r12_ea_latent_beta4_yee_jacobian_z2p5_bulk50_cv0_v2`;
    its `v1` predecessor is an intentionally retained blocked license-session
    record, not a valid operator.
40. `FDTDX_PARITY_HANDOFF.md` defines the temporary license-free FDTDX
    candidate-generation route. It does not authorize the legacy 80x80
    optimizers or the shared-linear/c3-only optical carrier. The new route must
    preserve the canonical 81x81 nodal filter/projection state, implement the
    same n-k-then-square target in a differentiable discrete ADE carrier, use
    the 2.5/50-nm rectilinear resolution contract, pass complete Ea/Eb and
    signed-objective AD-FD, and run only a two-iteration smoke optimization
    before review. FDTDX remains non-promotable; final binary CV0/finer
    Lumerical reevaluation is mandatory.

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
   resonance sweep, multi-direction latent-variable AD-FD, and
   both-polarization validation. The component-Yee mapping FD/transpose and
   four complete latent-variable directional AD-FD checks for each of Ea and Eb
   now pass on the RTX development mesh. See `MATERIAL_FRACTION_AUDIT.md`.
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
   remain invalid for promotion. Scripts 10 and 13 also remain 80x80-cell
   FDTDX optimizers and must not be pointed at the 81x81 Lumerical carrier.
   The new nodal design map exists, but no Lumerical LD_MMA entry point is
   enabled yet. See `ROBUST_OBJECTIVE_AUDIT.md`.
8. Production optimization is now unconditionally code-blocked because the
   existing entry points still implement the historical gray/FDTDX path.
   Legacy shared-linear certificates cannot clear this gate. The new
   Lumerical `n-k` density carrier is now connected through component-Yee
   discrete adjoints for Ea and Eb. All four planned latent-variable directions
   for each polarization and their signed objective now pass centered AD-FD.
   Production remains blocked until the Lumerical evaluation driver and the
   selected converged mesh pass. Then issue
   certificates naming the selected full-domain-z grid, Courant factor, time
   windows, and same-grid Ea/Eb source calibration. The combined adjoint also derives
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

LUMERICAL_GPU_INDEX=<free_rtx_index> \
  AU_LUMERICAL_ROOT=/home/seunghyun/lumerical_r12/opt/lumerical/v261 \
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
The launcher now fails closed unless `VERSION` reports 2026 R1.2 build 4522.
Do not use `/opt/lumerical/v261`: that tree is R1.0 build 4413 and its
FieldRegion `importdataset` path reproduced the known `Failed to evaluate
code` error before an adjoint solve. Python API, CAD, and `fdtd-engine` must
come from the same R1.2 installation root.

## 2026-08-24 local RTX development evidence

Raw artifacts are outside Git under
`/home/seunghyun/tairte4_raw_artifacts/au_dualpol_4um_lumerical_development/`.
Do not copy them into this worktree.

1. Historical Lumerical v261 R1.0 solver `8.35.4413` opened successfully for
   forward-only controls, but is now prohibited for FieldRegion adjoints. GPU logs identify
   physical RTX 6000 Ada GPU 5 and its UUID, contain the `fdtd-engine -gpu`
   command, GPU time-stepping timing, and successful completion.
2. The original 4.000-um source-object waist produced about 4.044 um at the
   flake plane and failed the 0.5% waist gate. One-step calibration selected
   `3.956143303046142 um`. Both Ea and Eb source-only baseline runs then passed;
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
   CV0, CV1, and staircase later all reproduced about 29.1% error, and 2.5 nm
   z plus a 2 ps/1e-9 decay run did not change it.
6. The actual blocker was Au sampled-data overfitting. Holding every physical
   and mesh input fixed while reducing only the Au MCM maximum from 20 to 6
   changed closure from 29.159% to 0.08935%. MCM4/6/8/12/16 form one stable
   field/Q plateau; MCM20 selects a different failed branch despite its smaller
   pointwise n-k readback error. Au now defaults to MCM6. Full evidence is in
   `AU_MCM_FIT_FINDINGS.md`.
7. Imported rho=1 closes Q/flux to 0.04374%, but is not yet exact-endpoint
   parity: its realized Yee epsilon differs from MCM6 and the complex endpoint
   field NRMSE is 1.849%. Do not promote the imported carrier until this is
   calibrated and independently checked against exact MCM6.
8. The default source-object waist in the unified runner and B200 endpoint
   batch is now the calibrated value. Every non-baseline mesh still reruns and
   revalidates source-only rather than assuming the calibration transfers.
9. MCM6 z refinement is now measured rather than assumed. The isolated
   bulk/air/PML 50-to-25-nm refinement passes the 0.5% gate, but the isolated
   thin-stack 5-to-2.5-nm refinement fails: source-normalized Q changes
   1.3298%, complex endpoint field changes 0.9850%, and E2 changes 1.1618%.
   The linked 5/50-to-2.5/25-nm pair also fails. Extending the linked pair to
   1.25/12.5 nm still fails narrowly. The next linked
   1.25/12.5-to-0.625/6.25-nm pair passes the exact-full Maxwell sub-gate:
   normalized Q, flux, complex field, and E2 change by 0.3550%, 0.3551%,
   0.2669%, and 0.3176%; exact-empty also passes all four metrics. See
   `LUMERICAL_Z_MESH_FINDINGS.md`. The Ansys-official multi-material
   `pabs_adv`/`index_x` filter was then extracted from the saved FSPs without
   rerunning Maxwell. Unassigned conformal-interface absorption decreases
   from 3.0869% to 1.5552% for empty and from 2.1728% to 1.1939% for full.
   Empty/full remapped-Q NRMSE is 2.4932%/2.3285% and TaIrTe4 temperature
   NRMSE is 1.7909%/1.3931%; both symmetry-current controls also fail the
   one-ppm cancellation gate. An earlier effective-epsilon/physical-overlap
   diagnostic gave smaller but still failed errors and is not the selected
   definition. See `LUMERICAL_Z_MULTIPHYSICS_FINDINGS.md`. Eb, simple-L,
   final-topology, and B200 z gates remain open, so this is not a production
   mesh certificate.
10. The prior 2-ps/1e-9 run used rejected MCM20, so the MCM6 duration/decay
    pair was rerun correctly. Exact-full 1 ps versus 2 ps changes were Q
    0.00456%, flux 0.01086%, complex field 0.00184%, and E2 0.00135%; exact
    empty also passed. The MCM6 time axis is closed for this RTX Ea control.
    See `LUMERICAL_TIME_CONVERGENCE_FINDINGS.md`.
11. The bounded 5/50-nm interface triage is complete. CV0 and staircase
    agree below 0.15% in all tested source-normalized Maxwell metrics for both
    empty and full controls, but the official exact-index material filter
    leaves 11.8313%/7.7844% of CV0 empty/full absorption unassigned. CV1
    differs from staircase by 1.74--3.25% in normalized Q and leaves
    6.2624%/11.2663% unassigned. Staircase leaves only 0.001012%/0.195399%
    unassigned. It is therefore selected for the next linked-z development
    pair, not promoted as converged. That staircase 5/50-to-2.5/25-nm pair
    has now been run and fails all four Maxwell metrics for both controls:
    empty changes are 0.9522--1.1752%, and full changes are
    1.0105--1.3954%. The next 2.5/25-to-1.25/12.5-nm pair also fails:
    empty misses only E2 at 0.6013%, while full changes are
    0.5268--0.6884%. Fixed-mesh symmetry-current controls and all remaining
    z/polarization/geometry gates stay open. The next
    1.25/12.5-to-0.625/6.25-nm staircase pair now passes the Maxwell sub-gate:
    empty changes are 0.1689--0.2988%, and full changes are
    0.2668--0.3436%. Its official-Pabs custom-CUDA downstream comparison
    still fails: empty/full remapped-source L2 NRMSE is 1.5580%/1.6799%, while
    temperature NRMSE and Tmax changes pass below 0.5%. Material omission is
    only 0.001019%/0.198992%, so the persistent 53.4/602-ppm zero-current
    residual is not explained by mixed-index omission alone. The Pabs L2
   error improves almost exactly first order over three staircase meshes.
    The subsequent axis audit showed that the official common-grid
    `pabs_adv/index_x` filter itself creates an x-staggered material-mask bias.
    A new component-Yee filter now pairs `Qx/Qy/Qz` with collocated fitted
    `epsilon_x/epsilon_y/epsilon_z`. On the same pair it leaves effectively
    zero Q unassigned, conserves Q below `3e-15`, and passes empty/full
    zero-current at `1.12e-10`/`1.48e-8`. Temperature/Tmax also pass; only
    empty/full volumetric-Q L2 remains failed at 1.5580%/1.2458%. Script 32
    reproduces this result. A 0.3125-nm stack-only source control passed, but
    the matching material run projected nine hours and was stopped at 3.63%;
    do not restart that brute-force sequence by default. See
   `LUMERICAL_INTERFACE_METHOD_FINDINGS.md`.
12. The system R1.0 build 4413 failed FieldRegion `importdataset` before any
    adjoint solve. After pinning the API/CAD/engine pair to R1.2 build 4522,
    the matching Ea source-only and nonuniform forward passed in 20.77 s and
    54.53 s of solver time. The R1.2 forward raw NPZ is byte-identical to the
    earlier state (`6c528b...a6bb`), so the audited Jacobian and custom-CUDA
    pullback remain bound to the exact same fields, epsilon, Q, and grid. One
    R1.2 FieldRegion adjoint then passed in 101.32 s total/80.25 s solver time;
    mesh and monitor-grid differences were exactly zero, source-profile
    round-trip error was zero, and the CW reconstruction residual was
    `1.41e-16`. The resulting total projected-density gradient L2 norm is
    `9.09e-10 A`. The layout-only launcher is now also fail-closed on R1.2
    build 4522; it can no longer silently fall back to the incompatible R1.0
    `/opt` tree. The calibrated source-waist decimal was corrected by one ULP
    so its default metre value exactly reproduces the hash-bound source record.
13. One independently chosen smooth projected-density direction then passed
    complete centered AD-FD at `h=0.0025`. The `rho+` and `rho-` Lumerical
    forward solves took 54.33 s and 56.00 s; their custom-CUDA
    thermal/electrical evaluations took 17.60 s and 17.87 s. AD was
    `-1.363032899e-8 A` and FD was `-1.363002816e-8 A` per unit rho, a
    `2.207e-5` relative error with equal sign. The plus/minus signal was 1.30%
    of the current magnitude, and pair reconstruction was within the
    step-scaled float64 roundoff bound. No empirical gradient rescaling,
    finite-difference fit, Lumerical HEAT/CHARGE solve, or optimizer iteration
    occurred. This closes only one Ea projected-state direction on the RTX
    5/50-nm staircase development mesh.
14. The optimizer-carrier audit found that the historical DFM mapping was
    80x80 cell-centered while the Lumerical shared state is 81x81 nodal. A
    separate nodal latent/filter/projection implementation now maps to the
    exact Lumerical nodes and derives the 80x80 PDE/DFM cells only through the
    tested four-node average. It also replaced the DFM residual's ReLU kink
    with a bounded softplus positive part. The solver-free script-37 audit
    passed all JVP/VJP, centered-FD, state-hash, no-rho3, and no-`np density`
    gates in about four seconds.
15. The complete beta-4 latent Ea chain then passed centered AD-FD at
    `h=0.0025`. AD was `-2.766595495e-8 A`; FD was
    `-2.766380278e-8 A`; relative error was `7.779e-5`, with equal sign. The
    plus/minus signal was 1.646% of the current magnitude, midpoint curvature
    ratio was `8.807e-4`, and the mapping-chain transpose error was
    `1.20e-16`. Baseline/adjoint/plus/minus Maxwell solver time totaled about
    237 s; the three custom-CUDA evaluations totaled about 60 s. This is still
    only one Ea direction on the RTX 5/50-nm staircase development mesh. The
    current solver-free comparison is stored outside Git at
    `r12_ea_latent_beta4_combined_adfd_result_v2/ea_combined_adfd_result.json`.
16. The complete beta-4 latent Eb chain passed centered AD-FD at the same
    `h=0.0025`. AD was `-5.529878050e-8 A`; FD was
    `-5.529062519e-8 A`; relative error was `1.4748e-4`, with equal sign. The
    plus/minus signal was 1.758% of current magnitude, midpoint curvature
    ratio was `6.277e-4`, and the mapping-chain transpose error was
    `1.20e-16`. The source-only solve took 20.69 s; the four
    baseline/adjoint/plus/minus Maxwell solves totaled 268.68 s and the three
    custom-CUDA evaluations totaled 54.72 s. The comparison is outside Git at
    `r12_eb_latent_beta4_combined_adfd_result_v1/eb_combined_adfd_result.json`.
    The unoptimized baseline currents remain same-sign (`Ea=-8.334 nA`,
    `Eb=-15.591 nA`), so this is a gradient certificate, not the requested
    switching device.
17. Script 39 combined those exact Ea/Eb artifacts into the signed epigraph
    without a new solve. The common baseline has utilities
    `I_Ea=-8.334 nA` and `-I_Eb=+15.591 nA`, so Ea is active and balanced
    utility is still negative. Balanced-objective AD-FD error was
    `7.779e-5`; the two signed constraint errors were `7.779e-5` and
    `1.4748e-4`. The raw result is outside Git at
    `r12_ea_eb_latent_beta4_signed_objective_v2/signed_dual_objective_result.json`.
18. Direction index 1 passed the same full chain. Ea AD/FD were
    `1.966482804e-8`/`1.966344926e-8 A` (error `7.011e-5`); Eb AD/FD were
    `3.871015880e-8`/`3.870857434e-8 A` (error `4.093e-5`). The signed
    objective artifact is outside Git at
    `r12_ea_eb_latent_beta4_dir1_signed_objective_v1/`.
19. Direction index 2 passed the same chain. Ea/Eb errors were
    `9.981e-5`/`1.890e-4`. The signed objective artifact is outside Git at
    `r12_ea_eb_latent_beta4_dir2_signed_objective_v1/`.

## Next correct sequence

1. Confirm the target geometry, contacts, crystal-axis angle, layer stack, and
   illumination in `physical_device_contract.json`.
2. Treat all existing FDTDX factor 1/2/4/8, reduced-Courant, and shared-linear
   full-domain-z tables as historical diagnostics, not evidence for Lumerical
   or a production mesh. The completed shared-linear factor-1/2/4 sweep is
   useful negative evidence: its stable final pair failed in all 6/6 cases.
3. Treat the CV0/CV1/staircase, z, time, and MCM sweeps as RTX development
   evidence only. Use Au MCM6, not 20. The MCM6 duration/decay pair now passes.
   Reproduce the passed Ea empty/full Maxwell sub-gate with script 27 and the
   historical downstream gate with scripts 28/29. The official common-grid
   `pabs_adv/index_x` result is retained as a diagnostic but is not the
   selected material partition because its x-staggered classifier creates a
   false current. The selected development map is native component-Yee Q with
   collocated fitted epsilon; reproduce it with script 32. The bounded MCM6
   CV0/CV1/staircase axis is now complete. Staircase was used for the
   historical linked-z diagnostic because its material omission is below
   0.5%, while its Maxwell observables agree with CV0 below 0.5%; the user
   later selected CV0 `2.5/50 nm` for bounded-cost optimization development.
   The staircase 5/50-to-2.5/25-nm linked refinement is complete and
   fails the Maxwell prerequisite. The matching staircase 1.25/12.5-nm set
   is also complete; its pair with 2.5/25 nm still fails narrowly. The
   staircase 0.625/6.25-nm source/empty/MCM6-full set and comparison are now
   complete: Maxwell, temperature, and symmetry-current sub-gates pass with
   the component-Yee map, but volumetric-Q L2 remains above 0.5%. Do not spend
   nine hours per control on the aborted 0.3125-nm brute-force extension.
   Bound or reformulate that interface-sensitive L2 certificate before the
   remaining Eb/simple-L and x/y gates. Do not hide any gap by rescaling.
4. On the actual B200, use `25_run_lumerical_4um_exact_au_control.py` to pass
   source-only Ea/Eb first, then matching ordinary empty/full/simple-L exact
   Au time, Q/flux, linked stack+bulk/air/PML-z, x/y, PML-layer, and domain
   controls with Au MCM6. Run imported-rho0/1 parity against the matching ordinary
   empty/full baselines,
   quantify the single-frequency carrier's source-band error, and sweep
   uniform projected density for artificial field/Q resonances. Raw output
   must remain outside the Git worktree. The unified runner and endpoint batch
   now exist, but no B200 result is committed and this Codex host fails the
   B200 preflight.
5. The selected bounded-cost development mesh is CV0 `2.5/50 nm`. Its R1.2
   source/empty/full controls and one complete common Ea/Eb beta-4 latent
   AD-FD direction pass. Build the fail-closed evaluation driver without
   rerunning these hash-bound controls. The component-Yee builder,
   hash-bound R1.2 distributed-source adjoints,
   and all four planned beta-4 latent centered AD-FD directions for each of Ea
   and Eb now pass on the 5/50-nm staircase mesh. Their exact signed epigraph
   also passes in all four common directions. The selected CV0 mesh adds one
   common direction with Ea/Eb errors `8.515e-5`/`1.3047e-4`. The
   polarization-general runner now permits reuse of the Ea material Jacobian
   for Eb only after the exact epsilon/grid/frequency binding passes; it does
   not require the physically different Ea/Eb field and Q arrays to match.
   Repeat the material Jacobian and AD-FD on the B200; do not substitute
   bundled LumOpt's real/lossless metal path.
6. Check x/y optical convergence, thermal-mesh convergence, electrical-mesh
   convergence, and downstream PTE current.
7. Certify the combined gradient on the selected production mesh.
8. Only then start LD_MMA filter/projection continuation and finish with an
   independent 500-nm solid/void audit plus ordinary dispersive-Au binary
   reevaluation.
