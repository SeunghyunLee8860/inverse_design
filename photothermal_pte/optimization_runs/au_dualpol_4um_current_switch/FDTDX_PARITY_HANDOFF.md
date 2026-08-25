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

Steps 1 through 3 now have a fresh fail-closed implementation in
`fdtdx_parity_contract.py`, `fdtdx_parity_ade.py`,
`fdtdx_parity_fixed_materials.py`, and `fdtdx_parity_model.py`.  The model
builder imports no historical integer layout or density carrier.  Physical
coordinates are constrained first, and the resolved FDTDX slices and float32
edges must match the certified arrays exactly before allocation can pass.  The
audited rectilinear grid is exactly
`186 x 186 x 286 = 9,894,456` cells.  Its planned float64 x/y/z edge hash is
`15e2ce87ec5485de2712718b0f12a289e64233a69b98f4cae23b3cb5349e7805`.
Pinned FDTDX stores those coordinates as float32; the realized solver-edge hash
is `1aa397f7313f05e0b47d741d58686f076dcc7d8bf04355606fa1e7e993d6464c`.
The maximum coordinate roundoff is `4.452886051152429e-13 m`, and the
realized minimum pitches are `99.99985195463523 nm` laterally and
`2.4999735614983365 nm` vertically.  Both planned and solver hashes are
mandatory provenance; neither may substitute silently for the other.  Source,
incident-power, endpoint-field, and flake planes are exact z edges 241, 236,
228, and 207, respectively.

On the current host, the runtime gate passes at FDTDX commit
`f26f84b70a8cceec9b889553955a868624736bf1`.  The normal site-packages import
is accepted only because `direct_url.json` points to the pinned clean source
and the installed and source package trees have the identical SHA-256
`c66b34671750258ff71478f9e9530f3abcb07a937591775236b1f7bdea739d58`.
The parity contract/ADE/fixed-material/model/timing/source/optical-control
suite is `50 passed`.
The full
solver-free suite for this work folder is `186 passed`.

The requested 2.5-nm z cells dominate CFL.  The realized FDTDX float32-edge
CFL is `dt=2.083451820604655 as`, `6,404.0664` steps per 4-um period, and
`256,163` steps for 40 periods.  That is `2,534,593,532,328` cell-steps per
forward solve.  After correcting the analytic accounting to the actual
one-component broadcast `inv_permittivity`, the selected three-pole carrier has
a persistent-array lower bound of `1.9503207207 GiB` and a one-dynamic-state
checkpoint lower bound of `0.9182474613 GiB`.  These exclude detector buffers,
XLA temporaries, cotangents, checkpoint scheduling, allocator overhead, and
CUDA workspace.

A no-field Ea dry allocation passed on verified-idle B200 UUID
`GPU-48bf3705-8160-2de5-531b-dc480c83eabe`: object placement plus device-ready
allocation took `33.5272 s`; the ArrayContainer leaf total was
`2,160,207,296 bytes = 2.0118498206 GiB`; all shape, PML, dt, material-state,
and placement gates passed; zero FDTD steps ran.  A second no-field audit on
verified-idle UUID `GPU-0e94c58d-ebdd-2b12-ce98-28159e8dd756` applied
`rho=0,0.25,0.5,0.75,1`.  For every density, the maximum coefficient error
through the complete `80 x 80 x 20` Au volume, adjacent-void leakage, and
TaIrTe4 coefficient change were all exactly zero.  Setup plus all five
readbacks took `33.6238 s`.

The GPU builder now fails closed below loaded cuBLAS runtime 13.2.  The isolated
FDTDX environment passed with `130601` (13.6.1).  The thermal PyTorch
environment supplies `130100` (13.1) and must not be placed ahead of the FDTDX
environment for optical GPU runs; doing so triggers JAX's documented
silent-corruption warning.

The committed bounded runner `fdtdx_parity_microbenchmark.py` was then executed
from clean commit `0b71db40198e9cfd31fe0eb3f569c6f7b9fc77d8` on verified-idle
B200 UUID `GPU-b288c55e-827d-e6b4-d05a-4b27eb65477f`.  It ran only 8,065
partial steps with detector recording: inactive, previous-window, and
late-window slopes were `1.75227`, `1.78465`, and `1.85233 ms/step`.  Their
frozen-schedule extrapolation is `452.2596 s = 7.5377 min` for one 256,163-step
forward, so the explicit 30-minute single-forward timing gate passes.  The
full forward was not run, all 14 detector leaves and E/H/ADE-P were finite,
and this remains timing evidence rather than a physics result or an optimizer
iteration estimate.  The raw JSON is outside Git at
`/home/seunghyun200/fdtdx_parity_raw/microbenchmark_0b71db40_gpu7.json`; its
file SHA-256 is
`1b0c821d09ece0205b4cbec90ab1b648aff559002aeaef803e6a0914f9e5ed70`.
No GPU peak-memory claim is made.

Step 4, the independent all-air Ea/Eb source calibration, now passes through
`fdtdx_parity_source_calibration.py`.  Pinned FDTDX stores
`E_stored=E_SI/eta0` and `H_stored=H_SI`, so SI power is the detector
`E x H` area integral multiplied by `eta0`, not divided by it.  Each case
requires previous-versus-late phasor stability, late time-domain-versus-phasor
agreement, vacuum normalized impedance, exact source injection polarization,
finite detector state, a positive incident power, and a clean hash-bound run.

Two fail-closed attempts were deliberately preserved rather than relabeled.
Commit `f9730057` incorrectly combined transverse and longitudinal endpoint
fields; its Ea/Eb file hashes are
`e369ddc687906e386069612ce1f999d64803d43aecbd847c67e00d74da694985` and
`056907a9db5dd1ad68d7dcfde9908d69afff8acabf4e963ba676f626ac637778`.
Commit `2872bbda` separated those fields but still gated the propagated endpoint
instead of the source injection plane; its hashes are
`06f17e99ca4fbbb1339bf015ee715caf95117dd119a769a35de55befc1501f05` and
`a87dbd41094d11219bb1399981ffa2fc6780b026003cd788c9b0594fd9a718f6`.
These BLOCKED files remain outside Git as negative audit evidence.

The corrected clean commit `a7fc44051c0eefd471b27469248162c81269129c`
ran Ea and Eb in parallel on separately verified-idle B200 UUIDs.  Full all-air
forward times were `172.0787 s` and `172.3871 s`.  Both source injection arrays
were exactly `[Ex,Ey,Ez]=[0,1,0]` for Ea and `[1,0,0]` for Eb.  Their late
incident powers were bitwise equal at `1.882400012336031e-12 W`; Ea/Eb mismatch
was zero.  Previous-to-late mismatch was at most `6.781e-7`, time-domain-to-
phasor mismatch at most `8.383e-6`, and vacuum impedance error `0.143502%`.
The propagated endpoint diagnostics were retained, not used as source-purity
gates: transverse cross ratio was `1.40093%` and longitudinal ratio `16.4410%`
for both finite-width beams.

At the requested `285 uW`, each polarization independently uses
`power_or_Q_scale=151402463.9461828` and
`field_amplitude_scale=12304.570855831696`; no Ea/Eb matching is performed.
The final Ea, Eb, and aggregate files are outside Git at
`/home/seunghyun200/fdtdx_parity_raw/source_Ea_a7fc4405_gpu6.json`,
`source_Eb_a7fc4405_gpu7.json`, and `source_calibration_a7fc4405.json`.  Their
file SHA-256 values are respectively
`b3f5599225e56f1b26055c7a1755b85909fb482a031252417eae80bdb8e93a69`,
`e10bc6f66998f7647cecd094c305203e78e7c8bda0caf5ad2aad887c69558405`, and
`4c37e497a5b6b89a02e554b04b316716faee5c87d7e0d3058bcfeb12b977a905`.
The aggregate status is `PASS_SOURCE_CALIBRATION`; this validates source power
only, not any Au/TaIrTe4 device response.

The first physical-device gate now passes through
`fdtdx_parity_optical_controls.py` at clean commit
`ae896797afa557f621b745a3bb3d64686ee6f424`.  This is deliberately a full-Au
control, not an optimizer result: constant `81 x 81` nodal rho=1 is mapped by
the exact four-node average to constant `80 x 80` cells.  The runner imports no
legacy FDTDX model, `material_fraction`, combined path, or optimizer checkpoint.
It independently integrates target n-k-square Q and realized three-pole ADE Q
with component-wise Yee dual volumes, and compares them against both late
time-domain and phasor inward flux through the closed material surface.

Ea and Eb ran in parallel on separately verified-idle B200 UUIDs
`GPU-0e94c58d-ebdd-2b12-ce98-28159e8dd756` and
`GPU-b288c55e-827d-e6b4-d05a-4b27eb65477f`.  Their physical full-forward times
were `458.9125 s = 7.6485 min` and `457.4265 s = 7.6238 min`; build times were
`36.7544/37.1888 s`, and compile times were `4.4123/4.3708 s`.  Both executed
all `256,163` steps on the exact `186 x 186 x 286` grid with cuBLAS `130601`.
The actual forward times agree with the prior `7.5377 min` bounded estimate and
remain below the frozen 30-minute single-forward limit.

All precommitted gates passed.  For Ea/Eb respectively, previous-to-late Q
power changes were `5.455e-6/1.577e-5`, spatial Q NRMSE values were
`5.557e-4/4.915e-4`, and target-versus-discrete-ADE Q mismatches were
`1.847e-6/1.937e-7`.  Time-domain-versus-phasor closed-flux mismatches were
`4.615e-4/2.983e-4`.  The conservative energy-closure errors were larger but
still below the frozen 2% gate: discrete Q versus time-domain flux was
`0.74286%/0.12317%`, and versus phasor flux was `0.78866%/0.09337%`.
After the independently hash-bound all-air normalization to `285 uW`, target
full-Au absorption Q is `9.805608 uW` for Ea and `14.865924 uW` for Eb.  These
different values are allowed because TaIrTe4 is anisotropic; no Ea/Eb
absorption-equality gate was imposed.

The Ea/Eb JSON and raw NPZ artifacts remain outside Git at
`/home/seunghyun200/fdtdx_parity_raw/optical_full_Ea_ae896797_gpu6.*` and
`optical_full_Eb_ae896797_gpu7.*`.  JSON file hashes are
`16332bda10abe4dcff9885c3d7198acff49f423ca1c01c795a311bdffe2d25b7` and
`a80fa5db5dbe300bec5a6a3699a65e2375869b67d16ade32482dbf38c0183f47`;
NPZ hashes are
`db5f0d862c37316ac4b610990bc1df294f78f963cbf50e2156a3711e28290d7e` and
`e587c77aed07aec9e8946eb8218040237b9a55e4bbca09020bde5611d3c8d7e8`.
The aggregate certificate is
`/home/seunghyun200/fdtdx_parity_raw/optical_full_aggregate_ae896797.json`,
file hash
`750d3f437b0c3d98c6d6a111135fa10ec9d0a6052315a14e752483ced3d8ac86`,
report hash
`619f40d040e67b34d3eb324db7a5ae8a6e9e06cf8121bdd22d8327a2dd317fdc`,
and status `PASS_FULL_AU_EA_EB_OPTICAL_CONTROLS`.  Q and time-domain flux were
also recomputed directly from the saved E fields, dual volumes, and raw flux
series and exactly matched the JSON values.

The shared design map is now implemented independently in
`fdtdx_parity_design_mapping.py` at commit
`61dd2031bfe88d266f2e36fd43100e5db4ca97f0`.  Its 81x81 finite nonperiodic
500-nm conic kernel, truncated-row normalization, eta=0.5 tanh projection, and
four-node 80x80 cell average match the current Lumerical parity mapping arrays
exactly.  Constants 0, 0.5, and 1 are preserved with zero error.  The complete
NumPy chain passes exact JVP/VJP transpose and centered FD tests; the JAX
float32 realization passes both output and autodiff comparisons.  The frozen
mapping coefficient hash is
`2c2f8cc8a613777f0bf08c7985efe3960992b8028ef46de399c5bce6b25f41d5`.
The deterministic beta-4 gray control is not an optimizer candidate; its
latent, projected-node, and cell ranges are `0.333672..0.681498`,
`0.201106..0.818642`, and `0.201315..0.818298`.

The empty-Au physical control then passed through
`fdtdx_parity_density_controls.py` at clean commit
`2cac2cdddb43a293cf7c2dd6b2e9eb4bcf11d4ab`.  It accepts no arbitrary 80x80
input: constant latent zero must pass through the same filter, projection, and
four-node map.  Ea/Eb ran in parallel on the same separately verified-idle
GPU6/GPU7 UUIDs.  Forward times were `458.9150 s` and `457.6403 s`.  All three
density states were exactly zero, and both target and realized-ADE Au Q were
exactly `0.0 W`; no residual Au loss survived.  TaIrTe4-only target Q at the
hash-bound `285 uW` source normalization was `74.314952 uW` for Ea and
`123.962901 uW` for Eb.

For Ea/Eb respectively, target-versus-discrete total-Q mismatch was
`2.723e-6/2.240e-7`, time-domain-versus-phasor flux mismatch was
`4.236e-5/1.698e-5`, discrete Q versus time-domain flux mismatch was
`0.53564%/0.37598%`, and versus phasor flux was `0.53986%/0.37429%`.
Every precommitted temporal, pointwise epsilon, zero-Au, positive-TaIrTe4, and
2% energy-closure gate passed.

The empty Ea/Eb JSON file hashes are
`7a0e36d29a99d018cb67dc2de96e14b75626781f19d7d6f10df041f81b621317` and
`f6bf560adf26a5c260e1e7bfbd4076415a5f9f518421dc6980c0c6d3e2d8277b`;
their raw NPZ hashes are
`7eb2b176197c02813d4a5d515c00d3866ad963c8267fafc53f1a367532b52e2d` and
`406df8113b1d0791e535374a9b0395de4a63d36ca2cfadf70323c8355bbeaed7`.
The external aggregate
`/home/seunghyun200/fdtdx_parity_raw/density_empty_aggregate_2cac2cdd.json`
has file hash
`774dbf9cc81a146d0eba9cb244b2d86e5961c76bceab7942da8ccde5410864cb`,
report hash
`be60aa2aa31e4c0f1f343187d6f542625479cbe507641b96b576528c19fc5d05`,
and status `PASS_EMPTY_EA_EB_OPTICAL_CONTROLS`.  Direct recomputation from both
NPZ files again produced Au Q exactly zero and reproduced TaIrTe4 Q and the
time-domain closed flux exactly.

The deterministic nonuniform-gray physical control also passed at clean commit
`0962163352bf42f724aa1473160e3569c92a7bdc`.  The beta-4 mapping produced the
predeclared cell range `0.2013149064..0.8182984845`, so all 6,400 Au cells were
strictly interior and both rho and rho-squared carriers were active with
spatially varying weights.  The pointwise maximum target-versus-realized Au
complex-epsilon mismatch was `5.555620238485794e-7`, below the frozen `1e-5`
gate.  Ea/Eb forward times were `458.9394 s` and `457.4337 s`.

At the `285 uW` source normalization, target total Q was `24.385373 uW` for Ea
and `41.557943 uW` for Eb.  The corresponding Au contributions were positive
at `9.365369 uW` and `18.318443 uW`; TaIrTe4 contributions were
`15.020004 uW` and `23.239500 uW`.  For Ea/Eb respectively,
target-versus-discrete total-Q mismatch was `1.688e-6/1.606e-7`,
time-domain-versus-phasor closed-flux mismatch was `1.717e-4/9.442e-5`,
discrete Q versus time-domain flux mismatch was `0.63219%/0.03436%`, and
versus phasor flux was `0.64925%/0.04380%`.  Every strictly-gray,
nonuniform-density, positive-Au-Q, temporal, material, and energy-closure gate
passed.

The nonuniform Ea/Eb JSON file hashes are
`7b8bbd8a6c5568aa9d8a67a252b76836e1666d77602126125a2a739add88ce9f` and
`74b910b6d6fa151e6fd58cc2f7e0b0e2866a6383b86e38ba496484c846bfc360`;
raw NPZ hashes are
`44573c8f8056de1ec4e3e18a7186e21875378d182f14ed5243ad90546105497d` and
`6efb2844b4c6aa720abeeb7551436621de56a46371286f433b1ddb2c991d8dbb`.
The external aggregate
`/home/seunghyun200/fdtdx_parity_raw/density_gray_aggregate_09621633.json`
has file hash
`f51fff179141accc50aec8b02988fb8e6240d4098dbad3cbefb20ae5a9e22315`,
report hash
`b597bb7915a6ecb79bf6d7c0fef80f0c4780900a42c99fe759633b39b571e3b1`,
and status `PASS_NONUNIFORM_GRAY_EA_EB_OPTICAL_CONTROLS`.  The saved latent,
projected-node, and cell arrays exactly equal a fresh mapping regeneration.
Independent raw-field integration exactly reproduced material-resolved Q,
total Q, and time-domain closed flux for both polarizations.

Together, the source-only, empty-Au, spatially varying gray-Au, and full-Au
controls validate the fresh optical forward path at rho=0, throughout a
strictly interior nonuniform density, and at rho=1.  The bounded exact-grid
Ea/Eb optical reverse-mode connection is also validated by
`FDTDX_AD_PROBE_REPORT.md`, but the 40-period production gradient, PDE, current,
and optimizer remain unvalidated.

The nonlinear carrier is now implemented in `fdtdx_parity_ade.py` as three
positive, damped Lorentz bases: one weighted by `rho` and two weighted by
`rho^2`.  This exactly follows the linear-plus-quadratic decomposition of the
selected n-k-square susceptibility and is not the old `rho*c3_Au` endpoint
scaling.  All `c4` values are zero, every recurrence is strictly stable, and
the pinned FDTDX API reproduces every frozen float32 coefficient exactly.  On
101 uniform densities the maximum relative complex-epsilon error is
`1.0813927774623183e-6`; the Au endpoint error is
`1.3956872401013224e-7`.  The coefficient hash is
`71f6738a4c587387c334c3a31edcf8df1ff9415b8fdf2d66537b7a65b6b07b0f`.
JAX coefficient JVP equals the analytic JVP, and its largest relative L2 error
against centered FD is `2.3878186766523868e-5`.

The fixed TaIrTe4 carrier is independently frozen in
`fdtdx_parity_fixed_materials.py`, with solver-axis order `x=b`, `y=a`,
`z=c=b`.  Its coefficient hash is
`fa9a435d79a7d01db22ec695940ebe993e6234b62fe5567fbd55a1664d08ede5`.
The realized float32 relative complex-epsilon errors are
`4.306472432831919e-6` for a and `1.1435456653834023e-7` for b/c;
both positive-Lorentz recurrences are strictly stable and reproduce exactly
through the pinned FDTDX API.  SiO2 and Si remain the lossless real readbacks
from the material JSON.

The empty-Au, nonuniform-gray, and full-Au Maxwell field/Q/closed-flux
controls are validated for both polarizations.  Exact-grid bounded Ea/Eb
field-only AD-FD also passes.  The checkpointed production-profile route remains
runtime-blocked at an optimistic 87.351 minutes for two-polarization optical AD
per iteration, so checkpoint-count tuning is closed and `optimizer_enabled`
remains false.

The alternative reverse audit has now proved the pinned dispersive ADE, six-face
CPML, and late three-component PhasorDetector `c3` gradient against direct
unrolled FDTDX on a 24-step real scene.  Exact-primal sliced resets then
match direct gradients on 24- and 70-step scenes, including a partial final
slice; the complete CPU suite passes 226 tests.  Read
`FDTDX_REVERSIBLE_ADJOINT_REPORT.md` and
`FDTDX_REVERSIBLE_SLICE_REPORT.md` for the exact scope.  This is not yet a
production gradient: full-domain P makes the current reset payload too large.
The next task is certified sparse regional-P slice storage, followed by a short
exact-grid GPU timing probe.

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
> completely, then read `FDTDX_AD_PROBE_REPORT.md`,
> `FDTDX_DYNAMIC_CHECKPOINT_REPORT.md`,
> `FDTDX_SPARSE_ADE_CHECKPOINT_REPORT.md`,
> `FDTDX_GRADIENT_DETECTOR_REPORT.md`, and `CODE_HANDOFF.md`.
> Lumerical licenses are currently unavailable: do not call Lumerical, HEAT,
> or CHARGE.  The current checkpointed 40-period gradient is runtime-blocked;
> do not run it, the 16-forward certificate, or an optimizer.  Immutable arrays,
> sparse regional ADE P-state, and gradient-detector pruning are complete.  Even
> the detector-free 65,536-step lower bound projects above 30 minutes per
> polarization, so do not repeat checkpoint-count tuning.
> The dispersive ADE, six-face CPML, and late PhasorDetector small-scene VJP
> audits are complete; read `FDTDX_REVERSIBLE_ADJOINT_REPORT.md` and do not
> repeat them.  Sliced exact-primal resets also pass 24/70-step direct-gradient
> parity; read `FDTDX_REVERSIBLE_SLICE_REPORT.md`.  Do not launch its full-P
> prototype on the exact grid.  Next integrate the certified disjoint
> TaIrTe4/Au regional-P checkpoints, prove sparse-versus-full slice gradient
> parity, and only then run a bounded exact-grid GPU timing and Ea/Eb latent
> AD-FD probe on freshly verified-idle GPU UUIDs.
> Continue only the fresh FDTDX GPU parity path for the 4-um Ea/Eb Au topology; do not run legacy scripts 10/12/13
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
