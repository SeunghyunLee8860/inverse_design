# FDTDX handoff for the 4-um dual-polarization Au topology

## Purpose and authority

Use FDTDX as a temporary, license-free candidate generator while Lumerical
FDTD licenses are unavailable.  Reproduce the present 4-um physical problem,
design coordinates, constitutive laws, custom CUDA thermal/electrical path,
and signed-current objective as closely as FDTDX permits.

This is **not** permission to restart the historical FDTDX optimizer.  It is a
new cross-solver parity path.  FDTDX has no CV0 conformal meshing, so its
rectilinear result is not a replacement for the selected Lumerical CV0 result.
Every promoted binary candidate must later be reevaluated with ordinary
dispersive Au in Lumerical on the selected CV0 mesh and on a finer mesh.

Do not call Lumerical, HEAT, or CHARGE in this temporary route.  Do not change
the production contract's `fdtdx_allowed=False`; that flag describes final
authority, not whether FDTDX may be used for this explicitly labelled
diagnostic/candidate-generation task.

## Start here

1. Checkout branch `agent/optimize-au-dualpol-4um-pte` and use the newest
   commit containing this file.
2. Read, in order:
   - `contract.py`
   - `au_density_relaxation.py`
   - `lumerical_4um_design_mapping.py`
   - `lumerical_4um_density.py`
   - `objective.py`
   - `multiphysics_4um.py`
   - `fdtdx_4um_model.py` and `combined_4um.py` as historical implementation
     references only
   - `LUMERICAL_MAXWELL_GPU_PDE_ROUTE.md`
   - `CODE_HANDOFF.md`
3. Pin FDTDX to `/home/seunghyun200/dependencies/fdtdx-f26f84b70a8cceec9b889553955a868624736bf1`, currently commit
   `f26f84b70a8cceec9b889553955a868624736bf1`.  Fail if a different import is
   resolved unless the change is explicitly audited and recorded.
4. Put raw arrays, checkpoints, logs, images, and iteration results outside
   Git, under a new directory such as
   `/home/seunghyun200/tairte4_raw_artifacts/au_dualpol_4um_fdtdx_parity/`.
   Commit only code, tests, small handoff/audit documents, and small manifests
   that do not contain machine-specific raw paths as authority.

## Hard prohibitions

- Do not run `10_optimize_4um_dualpol_au_ld_mma.py`, `12`, or `13` as the new
  optimizer.  They are legacy 80x80-cell FDTDX entry points.
- Do not use the historical optical `rho**3` law.
- Do not use `material_fraction.py`'s `c3 <- rho*c3_Au` as the optical law and
  call it Lumerical parity.  It produces a linear-epsilon susceptibility, not
  the selected nonlinear n-k law.
- Do not create independent optical, thermal, and electrical density arrays.
- Do not use `np density`; it is a carrier-density attribute, not the Au
  topology variable.
- Do not clip Q, delete negative cells, smooth Q, fit AD to FD, rescale a
  gradient, match the two polarizations empirically, or rescale Q to close a
  flux balance.
- Do not claim CV0, Lumerical equivalence, mesh convergence, binary success,
  or production readiness from an FDTDX run.
- Do not start MMA until the new FDTDX material carrier and the complete
  latent-to-current chain pass independent centered AD-FD for both
  polarizations.

## Frozen physical problem

### Coordinates and signs

- Solver `x = TaIrTe4 b`, solver `y = TaIrTe4 a`.
- `Ea` means `E || a`, so the source electric polarization vector is
  `(Ex,Ey,Ez)=(0,1,0)`.
- `Eb` means `E || b`, so the source electric polarization vector is
  `(1,0,0)`.
- Light propagates along `-z` from air onto Au/TaIrTe4.
- The weighting-potential terminals are the full TaIrTe4 flake edges:
  `psi=0` at `x_min`, `psi=1` at `x_max`.
- Positive current is conventional internal current along `+x`, from `x_min`
  to `x_max`.
- Target signs are fixed: `I_Ea > 0` and `I_Eb < 0`.

### Geometry

All objects are laterally centered at `(x,y)=(0,0)`.

| object | x/y footprint | z bounds |
|---|---:|---:|
| Si substrate | complete optical domain | `-3 um` to `-385 nm` in the provisional domain |
| SiO2 | complete optical domain | `-385 nm` to `-100 nm` |
| TaIrTe4 flake | `16 um x 16 um` | `-100 nm` to `0` |
| Au design region | `8 um x 8 um` | `0` to `+50 nm` |
| air | remaining domain | through `z=+3 um` |

- The patterned Au is floating and directly contacts TaIrTe4.  It is an
  optical absorber/scatterer and thermal/electrical shunt, not an optical
  representation of the measurement terminals.
- The physical device assumptions remain provisional; do not reinterpret the
  flake, contacts, axes, or terminal shapes during this task.

### Illumination

- Vacuum wavelength: exactly `4.0 um`.
- Target Gaussian intensity `1/e^2` radius at the flake plane: exactly
  `4.0 um`.
- Source aperture: `16 um x 16 um`.
- Source plane for the FDTDX construction: `z=+0.75 um`.
- Waist/source-profile target plane: `z=0`; source-only calibration must
  measure the profile there.
- Incident-power monitor: all-air plane at `z=+0.50 um`, below the source.
- Air-side endpoint-field plane: `z=+0.10 um`, 50 nm above the Au top.
- The inward Q/flux box top is `z=+0.50 um`.
- Reporting incident power: `285 uW`.
- Calibrate the FDTDX source parameter on the new grid; do not copy the
  Lumerical source-object value `3.956143303046142 um`, because the source
  implementations differ.  Measure the realized flake-plane Gaussian profile
  and waist.
- Run source-only all-air calibration for Ea and Eb.  Scale each physical
  result only by `285 uW / P_incident,pol`.  This is incident-power
  normalization, not polarization matching.  Record the unscaled incident
  powers and require their relative mismatch to be below `0.5%`.

### Single-frequency material targets at 4 um

Use the passive `n+ik` convention.  The authoritative readback is
`results_materials_4um/4um_material_contract.json`.

| material/axis | relative permittivity at 4 um |
|---|---:|
| Au, Ordal | `-830.37 + 127.16 i`, from `n+ik=2.2+28.9i` |
| TaIrTe4 a | `-30.713256371885343 + 50.848086107787424 i` |
| TaIrTe4 b | `15.900726644538812 + 9.289194887622557 i` |
| TaIrTe4 c | use the documented closure `epsilon_c=epsilon_b` |
| SiO2, Kitamura | `1.8685272521070964 + 0 i` at reported precision |
| Si, Palik readback | `11.76078436 + 0 i` |

FDTDX is time-domain.  Its realized **float32 discrete ADE response**, not an
ideal continuous formula, must match the target carrier-frequency complex
permittivity.  Require relative complex-permittivity error `<1e-5` for every
fitted axis/material used in the run.  Record ADE coefficients, time step,
carrier response, target response, and error.

## Canonical topology variable

The topology `rho` is an occupancy variable, not electron/hole density and
not a physical gray alloy.

```text
81x81 latent rho in [0,1]
  -> finite nonperiodic 500-nm conic filter
  -> tanh projection, eta=0.5, beta continuation
  -> one shared 81x81 projected nodal occupancy rho_bar
  -> exact four-node average
  -> one shared 80x80 cell occupancy for FDTDX and custom PDEs
```

- Design window: `8 um x 8 um`.
- Physical cell pitch: `100 nm`.
- Canonical latent/projected nodal shape: `81x81`.
- Derived physical-cell shape: `80x80`.
- Outside the finite design window is void.
- Filter radius: `500 nm`, finite/nonperiodic, zero padding with truncated-row
  normalization exactly as implemented by `LumericalNodalDesignMapping`.
- Projection threshold: `eta=0.5`.
- Use `nodal_to_cell_average` and its committed transpose.  Do not invent a
  second 80x80 optimizer variable.
- Reuse `independent_latent_baseline()` from script 38 for the first parity
  certificate:

  `0.5 + 0.16*sin(0.8*pi*x)*cos(0.6*pi*y)` on normalized nodal coordinates.

- First AD-FD point: `beta=4`, centered step `h=0.0025`, direction index 0
  from `lumerical_4um_adfd.py`.  The direction must be selected without
  reading a field or gradient.

## Optical density law that FDTDX must implement

For the cell occupancy derived from the canonical nodes, use the same target
law as Lumerical:

```text
n(rho_bar) = 1 + rho_bar*(2.2 - 1)
k(rho_bar) = rho_bar*28.9
epsilon_target(rho_bar) = [n(rho_bar) + i*k(rho_bar)]^2
```

The target derivative is

```text
d epsilon_target / d rho_bar
  = 2*[n+i*k]*[(2.2-1) + i*28.9].
```

Do not implement this by holding an Au pole fixed and scaling only its `c3`
coefficient linearly.  Build a stable, differentiable FDTDX ADE carrier whose
realized discrete response follows `epsilon_target(rho_bar)`.  The complete
derivatives of every density-dependent coefficient must enter the Maxwell
gradient.  Acceptable implementations include a smooth analytic coefficient
map or an independently validated differentiable interpolation table.  A
non-differentiable root search inside an optimizer evaluation is not
acceptable.

Before any field AD-FD or optimizer run, test at least 101 uniform densities
from 0 to 1:

- finite and passive carrier response;
- exact air endpoint at `rho=0`;
- Au target endpoint at `rho=1`;
- realized discrete epsilon error `<1e-5` over the sweep;
- analytic/JAX coefficient JVP versus centered FD;
- no unstable or anomalous field/Q resonance in explicit forward controls at
  representative `rho=0,0.25,0.5,0.75,1`.

If FDTDX cannot represent this law stably and differentiably, stop and report
the mismatch.  Do not silently fall back to the old shared-linear law.

## Hash-bound Lumerical reference point; never fit to it

The same independent beta-4 latent baseline and direction index 0 already have
selected-mesh Lumerical evidence on CV0 `2.5/50 nm`:

- baseline `I_Ea=-8.700192221 nA`;
- baseline `I_Eb=-16.863681721 nA`;
- Ea directional AD/FD
  `-2.795034298e-8/-2.794796295e-8 A`, relative error `8.515e-5`;
- Eb directional AD/FD
  `-5.532360856e-8/-5.531639071e-8 A`, relative error `1.3047e-4`.

Use these only as a cross-solver report: compare baseline currents, Q,
temperature, direction signs, and normalized spatial fields where meaningful.
Do not multiply, rotate, phase-fit, or otherwise calibrate FDTDX currents or
gradients to reproduce them.  FDTDX staircase discretization and its causal
ADE surrogate are expected to create a measurable solver discrepancy.  The
comparison must expose that discrepancy rather than hide it.

## FDTDX optical grid and time contract

FDTDX is rectilinear/staircase and cannot reproduce Lumerical CV0.  Use a new
grid that matches the selected engineering resolution limits while keeping
every physical interface on an exact grid face.

### Lateral grid

- Domain: `x,y in [-10,+10] um`.
- TaIrTe4 region `[-8,+8] um`: `100 nm` cells.
- One-micrometre air margin between flake and PML: `200 nm` cells.
- One-micrometre PML on each lateral face: 8 cells of `125 nm`.
- This preserves the established FDTDX lateral edge layout and the exact
  100-nm design/flake cells.

### Vertical grid

- Domain: `z in [-3,+3] um`.
- Bottom PML: `[-3,-2.6] um`, 8 cells of `50 nm`.
- Bulk Si: `[-2.6,-0.385] um`, choose monotone cells no larger than `50 nm`
  while ending exactly at `-385 nm`.
- SiO2 `[-385,-100] nm`: exactly 114 cells of `2.5 nm`.
- TaIrTe4 `[-100,0] nm`: exactly 40 cells of `2.5 nm`.
- Au design `[0,+50] nm`: exactly 20 cells of `2.5 nm`.
- Air `[+50 nm,+2.6 um]`: 51 cells of `50 nm`.
- Top PML `[+2.6,+3.0] um`: 8 cells of `50 nm`.
- Expected order of magnitude: about `186 x 186 x 286` cells before any
  component staggering.  Save and hash the exact edge arrays.

Never reuse baseline z-cell offsets such as `LAYOUT.sio2_cells` to place an
adjoint source on this grid.  Derive all material and source slices from the
realized placed objects and assert face adjacency.

### Time stepping and monitors

- GPU backend, `float32` time stepping.
- Start from the validated FDTDX stability contract: Courant factor `0.25`,
  40 carrier periods, and a 4-period late phasor window.
- Also save the immediately preceding 4-period window and require stationarity
  before accepting a material run.
- A shorter run may be used only as a compile/memory smoke test and must not
  supply optimization physics or gradients.
- Six PML boundaries, 8 cells each face.
- Closed inward-flux surface must lie outside all absorbing materials and
  inside every PML.  Record its exact physical faces.
- Save native component fields/Q and conservative remap metadata.  Q must be
  calculated from the realized discrete ADE loss used by the time stepper.

## Custom thermal/electrical contract

Reuse `multiphysics_4um.py`; do not use Lumerical HEAT/CHARGE.

### Thermal

- 3-D steady finite-volume CUDA solve.
- TaIrTe4 conductivity `(kx,ky,kz)=(3.8,14.4,1.0) W/(m K)` for solver
  `(x=b,y=a,z=c)`.
- Au `317 W/(m K)`, SiO2 `1.38`, Si `145`, air `0.026`.
- SiO2/Si interface conductance `1.1e9 W/(m^2 K)`.
- TaIrTe4/SiO2 `7.37e6 W/(m^2 K)`.
- TaIrTe4/air `1 W/(m^2 K)`.
- Au/TaIrTe4 scenario `R''=5.8e-8 m^2 K/W`, hence
  `G=1/(5.8e-8) W/(m^2 K)`.
- Top-air convection `10 W/(m^2 K)`.
- Zero temperature-rise Dirichlet boundaries at thermal `x/y min/max` and
  `z_min`; top uses the documented Robin boundary.
- Use the existing explicit thermal edge arrays and CUDA solver tolerances;
  do not coarsen them for convenience.

### Electrical and PTE current

- TaIrTe4 sheet grid: `160x160`, 100-nm lateral cells, 100-nm thickness.
- `sigma_(x=b,y=a)=(1.10e5,4.91e5) S/m`.
- `S_(x=b,y=a)=(+27,-6) uV/K`.
- Au bulk conductivity `1/2.43e-8 = 4.115226337e7 S/m`.
- Au/TaIrTe4 electrical contact scenario `1e10 S/m^2`.
- Preserve the documented electrical sheet and contact floors
  (`1e-8` and `1e-10` fractions) and report their use.  They are numerical
  regularizers, not a claim that void conducts physically.
- Weighting potential: `psi=0` on the full `x_min` flake edge and `psi=1` on
  the full `x_max` edge.
- Current definition:

  `I = integral[(-sigma*S*grad(T)) dot grad(psi)] dA`.

- Positive I is conventional current along `+x`.

The one projected occupancy is mapped by exact four-node averaging into both
thermal and electrical property maps.  Property-specific constitutive laws
are allowed; independent occupancies are not.

## Maxwell-to-PDE coupling and complete gradient

- Deposit FDTDX native `Qx/Qy/Qz` power into explicit thermal cells by literal
  Cartesian overlap, using the existing conservative map.
- Apply the exact discrete transpose to return the thermal-adjoint cotangent
  to native component Q.
- The complete density gradient must contain all of:
  1. Maxwell field redistribution;
  2. direct density dependence of dispersive Au loss;
  3. thermal conductivity and Au/TaIrTe4 thermal-contact dependence;
  4. Au sheet conductivity and Au/TaIrTe4 electrical-contact dependence;
  5. the 80x80-cell to 81x81-node transpose;
  6. projection derivative and finite-filter transpose.
- Use the discrete ADE coefficients actually advanced by FDTDX.  Do not
  differentiate an ideal continuous epsilon while forwarding a different
  float32 recurrence.

## Figure of merit and optimizer

Optical absorption is an intermediate source, not the FoM.  The exact
opposite-current problem is

```text
maximize t
subject to
    t - I_Ea <= 0
    t + I_Eb <= 0
```

Equivalently, maximize `min(I_Ea,-I_Eb)`.

- Optimize the exact epigraph with NLopt `LD_MMA`; do not optimize a smooth
  minimum except in plots/smoke diagnostics.
- Scale current and epigraph by `1 nA` for optimizer conditioning, but report
  physical amperes and nA.
- Do not add symmetry, volume-fraction, connectivity, or hand-written sign
  updates.
- Include differentiable 500-nm solid and void opening constraints through
  `smooth_lumerical_500nm_constraints`, and use `exact_500nm_audit` on every
  thresholded candidate.
- Use the canonical nodal latent bounds `[0,1]`.  MMA owns continuous updates;
  do not clip after an update.
- First run only a two-iteration smoke optimization at `beta=4` after all
  derivative gates pass.  Require finite checkpoints, improving or at least
  correctly predicted directional behavior, and no physics-gate failure.
- A reasonable continuation after the smoke test is
  `beta=4,8,16,32,64`.  Do not increase beta merely because a stage exhausted
  its evaluation count; record current feasibility, DFM residuals, grayness,
  and exact binary audit at each transition.
- Run the opposite orientation `min(-I_Ea,+I_Eb)` only as a separately named
  experiment if requested.  Do not mix its history with the frozen target
  `I_Ea>0, I_Eb<0`.

## Mandatory gates before MMA

1. Solver-free nodal filter/projection/cell-average JVP, VJP, transpose, DFM,
   and centered-FD tests pass.
2. FDTDX density-to-discrete-ADE material map passes uniform sweep, endpoint,
   passivity, JVP, VJP, and centered-FD tests.
3. Source-only Ea/Eb runs pass profile, incident-power, time-stationarity,
   finite-field, PML, and source-mismatch gates on the exact optimizer grid.
4. Empty, full, and the common nonuniform beta-4 baseline pass finite Q,
   nonnegative Q, previous/late stationarity, and Q/six-face closure.
5. Conservative Q-remap power error `<1e-12` and its transpose/contraction
   errors `<1e-12`.
6. Thermal energy balance `<1%`; thermal forward/adjoint explicit residuals
   `<1e-8`.
7. Electrical forward/adjoint residuals `<1e-8` and terminal balance is finite.
8. Complete centered latent AD-FD at `h=0.0025` passes for the same independent
   direction for Ea and Eb: same nonzero sign and relative error `<1%`.
9. The exact signed epigraph objective and both constraint directional
   derivatives pass `<1%` from those same hash-bound results.
10. Repeat at least four deterministic smooth directions before a long run;
    near-null directions must use an absolute-error/conditioning audit rather
    than hiding noise with a relative-error denominator.

Recommended per-run numerical gates are previous/late field NRMSE `<0.5%`,
previous/late source-normalized Q change `<0.5%`, and volume-Q versus inward
six-face flux `<0.5%`.  If the new n-k carrier cannot meet these limits, stop
and diagnose it before optimization.

## Checkpoints, provenance, and fail-closed behavior

Save after every unique physics evaluation and every completed MMA stage:

- latent 81x81 array and SHA-256;
- filtered and projected nodal arrays and SHA-256;
- derived 80x80 cell occupancy and SHA-256;
- beta, eta, filter kernel/normalization hashes, DFM values, grayness;
- exact x/y/z grid-edge SHA-256;
- FDTDX source-tree commit and imported module path;
- GPU model, logical device, physical UUID, JAX/FDTDX versions;
- material targets, realized ADE coefficients/responses/errors;
- source calibration hashes and physical normalization;
- Ea/Eb currents, utilities, active constraint, epigraph value/slacks;
- Q totals/components/materials, flux closure, temperature/current summaries;
- forward/adjoint residuals, wall times, and gradient component norms;
- optimizer state needed for exact restart.

Refuse restart when any code, grid, material, source, density, axis, sign,
filter, beta-stage, or checkpoint hash is incompatible.  Never silently load
an old 80x80 FDTDX checkpoint into the new 81x81 path.

## Required implementation order

1. Add a new FDTDX-parity contract/audit and tests; do not mutate legacy
   results into passing status.
2. Add the 2.5/50-nm rectilinear grid builder and source/monitor placement by
   physical coordinates; run layout and memory checks.
3. Implement and certify the nonlinear n-k-to-discrete-ADE density carrier.
4. Recalibrate Ea/Eb sources on that exact grid/time contract.
5. Run empty/full/nonuniform forward and optical-Q/flux controls.
6. Connect the existing custom CUDA thermal/electrical forward and adjoint
   through the canonical nodal-to-cell map.
7. Issue one-direction and then four-direction complete Ea/Eb latent AD-FD
   certificates plus signed-objective certificates.
8. Build a new fail-closed 81x81 LD_MMA driver.
9. Run two beta-4 smoke iterations and checkpoint them.
10. Only after the smoke report is reviewed, continue the staged optimization.
11. Export several exact 80x80 binary masks using the committed thresholded
    cell-average rule.  Keep FDTDX binary results as candidate ranking only.
12. When Lumerical licenses return, reevaluate the candidates with ordinary
    sampled-data dispersive Au on CV0 `2.5/50 nm`, then on a finer final mesh.

## Current implementation status (2026-08-25)

Steps 1 and 2 now have a solver-free, fail-closed implementation in
`fdtdx_parity_contract.py`, with non-editable-install provenance checking in
`fdtdx_runtime_provenance.py` and seven focused regression tests.  The audited
rectilinear grid is exactly `186 x 186 x 286 = 9,894,456` cells.  Its complete
x/y/z edge hash is
`15e2ce87ec5485de2712718b0f12a289e64233a69b98f4cae23b3cb5349e7805`.
Source, incident-power, endpoint-field, and flake planes are exact z edges
241, 236, 228, and 207, respectively.

On the current host, the runtime gate passes at FDTDX commit
`f26f84b70a8cceec9b889553955a868624736bf1`.  The normal site-packages import
is accepted only because `direct_url.json` points to the pinned clean source
and the installed and source package trees have the identical SHA-256
`c66b34671750258ff71478f9e9530f3abcb07a937591775236b1f7bdea739d58`.
The full solver-free suite is `140 passed`.

The requested 2.5-nm z cells dominate CFL: `dt=2.0834738305187266 as`,
`6,403.9988` steps per 4-um period, and `256,160` steps for 40 periods.
That is `2,534,563,848,960` cell-steps per forward solve.  For one
axis-aligned ADE pole, pinned FDTDX allocations give only a persistent-array
lower bound of `0.918 GiB` and a one-dynamic-checkpoint lower bound of
`0.476 GiB`; these explicitly exclude XLA temporaries, cotangents, checkpoint
scheduling, detectors, and CUDA workspace.  Therefore no wall-time or GPU
peak-memory feasibility claim is made yet.  After selecting the ADE carrier,
a dry allocation and short timed microbenchmark on a verified-idle permitted
GPU are mandatory before any 40-period field solve.

No Maxwell field, CUDA PDE, GPU, optimizer, Lumerical, HEAT, or CHARGE run is
claimed by this status.  `optimizer_enabled` remains false.  The next gate is
the nonlinear n-k-to-discrete-ADE carrier and its 101-density/JVP certificate.

## What completion means

The temporary FDTDX task is complete only when it produces a restartable,
hash-bound candidate optimization whose material law and complete gradients
pass the gates above.  It does not complete the project.  The scientific
claim remains blocked until an exact-binary candidate has opposite current
signs in Lumerical and survives the finer-mesh/material/contact sensitivity
reevaluation.

## Copy-paste prompt for a new Codex session

> Checkout the latest `agent/optimize-au-dualpol-4um-pte` commit and read
> `photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/FDTDX_PARITY_HANDOFF.md`
> completely, followed by `CODE_HANDOFF.md`.  Lumerical licenses are currently
> unavailable: do not call Lumerical, HEAT, or CHARGE.  Build a new FDTDX GPU
> parity path for the 4-um Ea/Eb Au topology; do not run legacy scripts 10/12/13
> and do not use optical rho^3 or c3-only linear scaling.  Preserve the canonical
> 81x81 latent -> 500-nm finite conic filter -> tanh projection -> shared nodal
> occupancy, derive the 80x80 FDTDX/PDE cells only by the exact four-node map,
> implement the same n-k-then-square target through a stable differentiable
> discrete ADE carrier, use the 2.5/50-nm rectilinear grid contract, custom CUDA
> thermal/electrical solvers, and exact epigraph `max min(I_Ea,-I_Eb)`.  Pass
> uniform material, source, Q/flux, remap-transpose, complete four-direction
> Ea/Eb latent AD-FD, and signed-objective gates before building a new 81x81
> LD_MMA driver.  Then run only a two-iteration beta-4 smoke optimization,
> checkpoint every evaluation outside Git, document exact timings/results, and
> push each coherent code/document slice to the same branch.  Treat all FDTDX
> outputs as candidate-generation evidence; final binary Lumerical CV0/finer
> reevaluation remains mandatory.
