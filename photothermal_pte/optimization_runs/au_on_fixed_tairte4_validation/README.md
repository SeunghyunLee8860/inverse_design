# Au topology validation on a fixed TaIrTe4 flake

## Current promoted route (2026-08-21)

Au is the **nanoantenna/nanocube design material**, not an electrode in this
workflow.  The v261 moving/conformal Au boundary derivative is retained as a
failed diagnostic; it is not the production gradient.

The first working free-form-metal checkpoint is now a fixed-grid causal
dispersive route:

```bash
env PYTHONPATH=/home/seunghyun/.local/au_fdtdx \\
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \\
  /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \\
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/39_validate_3d_drude_nanostructure_adfd.py

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \\
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/40_summarize_3d_drude_nanostructure_adfd.py
```

The 2-D design density is extruded through a fixed Au thickness and scales a
passive Drude pole as `s(rho)=rho^3`.  At 10 um the Au endpoint is exactly
`n+ik=12.1+69.2i`.  This is a causal numerical relaxation, not a claim that a
gray pixel is a fabricated effective medium.  The 3-D GPU Maxwell AD--FD
control passes five directional tests.  See
`results/AU_3D_CAUSAL_DRUDE_NANOSTRUCTURE_ADFD_REPORT.md`.

This checkpoint does not yet promote a coupled TaIrTe4 thermal/PTE objective
or an Au optimization.  The next gate adds fixed anisotropic TaIrTe4 and
separates Au and TaIrTe4 optical absorption before thermal coupling.

That fixed-TaIrTe4 optical gate and its independent exact-binary Lumerical
endpoint cross-check now pass.  The differentiable FDTDX/JAX control keeps Au
and TaIrTe4 absorption separate and passes five-direction AD--FD.  The v261
cross-check uses the same finite 10-um Gaussian and mesh for Au-absent and
exact-Au cases, reads back all three anisotropic TaIrTe4 components and exact
Au epsilon, proves GPU time stepping from the engine log, and closes native
Yee `P_Q` against the six-face flux balance.  Raw FSP/NPZ files remain outside
Git.  See:

- `results/AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_REPORT.md`
- `results/LUMERICAL_AU_ON_TAIRTE4_BINARY_ENDPOINTS_REPORT.md`

The exact-binary v261 result does not validate its failed moving/conformal
metal derivative or gray `importnk2` path.  The promoted differentiable route
is still the causal fixed-grid dispersive solver.

The same route now also passes a production-width nonuniform-Au optical
gradient smoke test.  The calculation uses the 48 um x 48 um rectilinear
domain, the 8.5-um-waist Gaussian, physical 100-nm TaIrTe4 and 50-nm Au, and a
20x20 density field at 500-nm pitch mapped to component-native Yee samples.
The strong smooth directional AD--FD error is 0.095844% at `h=0.005`; the
fixed-seed near-null direction has only 0.023846% error when normalized by the
full gradient L2 norm.  Local material Q closes against the empty-subtracted
six-face flux to 0.001937%.  No clipping, smoothing, gain, endpoint matching,
or gradient rescaling is used.  See:

- `results_fdtdx_production_gradient_smoke/FDTDX_PRODUCTION_WIDTH_NONUNIFORM_AU_GRADIENT_REPORT.md`

The first four-checkpoint production attempt is retained as a performance
diagnostic: it did not finish in 60 minutes.  Sixteen checkpoints used 35.8 GB
on a 49-GB GPU and completed one value+gradient in 506.6 s.  This changes only
the time--memory tradeoff, not the physical or differentiation contract.

Isolated Au/TaIrTe4 thermal/contact and floating-Au electrical/weighting
AD--FD controls now pass.  The 10-um SiO2/Si optical substrate endpoint also
passes after correcting component-specific Yee dual-volume integration.
The same 32-period substrate-bearing contract now passes a nonuniform-Au
total-optical-Q gradient smoke on its stable central-FD step plateau.  For one
strong smooth direction, the AD--FD errors are `0.098122%` at `h=0.02` and
`0.085184%` at `h=0.01`; the two FD values differ by only `0.183150%`.
The `h=0.005` error of `1.004673%` is retained as a fail-closed float32
subtraction/cancellation diagnostic rather than deleted or rescaled.  The
baseline loss/flux closure is `0.137738%` and the late-window change is
`0.014438%`.  Status:
`VALIDATED_FDTDX_DIAGNOSTIC_SUBSTRATE_NONUNIFORM_AU_GRADIENT_STABLE_STEP_PLATEAU`.

The strict value+gradient required `5773.955 s` and about 36.2 GB with 16
checkpoints.  It is therefore the accuracy reference, not yet an approved
per-iteration optimization contract.  A shorter-period candidate must first
match this objective and gradient direction.  Coupled PTE,
combined-gradient, and optimization validation remain pending.  See
`results_fdtdx_substrate_gradient_plateau_gpu3/`.

That shorter-duration screening now passes for 16 total periods with a
4-period observation window.  Relative to 32/4, total Q, gradient L2 norm,
and the identical smooth directional derivative change by `0.000503%`,
`0.002810%`, and `0.000870%`; its internal AD--FD error is `0.007423%`.
Substrate-only and material-bearing closures are `0.475776%` and `0.123109%`.
AD execution falls from `5773.955 s` to `2919.009 s` (`1.978x`). Status:
`VALIDATED_FDTDX_DIAGNOSTIC_16PERIOD4WINDOW_OBJECTIVE_DIRECTIONAL_GRADIENT_EQUIVALENCE`.

The immutable 32-period run did not store the 20x20 gradient vector, so this
does not certify a full-vector gradient angle.  It promotes the 16/4 contract
only for the tested objective/norm/direction screening, not for combined PTE
or optimization. See `results_fdtdx_fast_contract_equivalence_gpu3/`.

The thermal/contact and floating-Au weighting controls are now also joined in
one fixed-Q coupled PTE operator.  The same 20x20 density changes Au lateral
thermal conductivity, Au/TaIrTe4 thermal contact area, floating-Au sheet
conductivity, finite vertical electrical contact, and therefore the weighting
solution.  At `h=0.0025`, the worst strong-direction AD--FD error is
`0.000012%` and the worst gradient-L2-normalized error is `0.000004%` over
five directions.  Maximum linear residual is `7.10e-12`; thermal and terminal
balances are at roundoff. Status:
`VALIDATED_COUPLED_AU_THERMAL_WEIGHTING_PTE_FIXED_Q_CONTROL`.

This is deliberately a fixed-Q operator control, not a Maxwell-coupled PTE
prediction.  `G_Au/Ta=1.724138e7 W/(m2 K)` is an Au/MoS2 calculated analogue,
not TaIrTe4 data; the electrical contact is also a numerical scenario and
`S_Au=0`. See `results_au_coupled_thermal_weighting_pte_fixed_q/`.

The 16-period/4-window forward now also exports the complete spatial heat
source on the component-native Yee grids.  `Qx`, `Qy`, and `Qz` are stored
separately for Au, TaIrTe4, and lossy SiO2 together with their staggered
physical coordinates, axis-wise dual widths, and dual volumes.  Independent
offline reintegration reproduces total `P_Q=2.4779538432e-13 W` with a
`3.62e-8` relative error; the worst individual component error is `1.19e-7`.
Matched-volume loss/flux closure is `0.122366%` and the late-window change is
`0.019832%`. Status:
`VALIDATED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_ARTIFACT`.

The 18,009,578-byte raw NPZ remains outside Git and is pinned by SHA-256
`f513473ecd38425bbbefe01ff026ee40ba2484f8c0edd5ab687306b458ddc7ff`.
This checkpoint certifies the spatial optical artifact only. Conservative
material-overlap remapping, an explicit thermal solve, combined PTE AD--FD,
and optimization remain separate fail-closed gates. See
`results_fdtdx_substrate_spatial_q_export_16period_4window_gpu3/`.

The subsequent material-overlap remap gate now passes. Each native Yee
component first forms `p=Q*V_dual`; that power is distributed by exact
separable intersection with the absorbing material's primal thermal cells.
The map does not use nearest-cell projection and does not delete a boundary
sample or apply a global gain. Source and target totals are both
`2.4779539328655175e-13 W`; the worst component conservation error is
`1.96e-16` and the worst transpose dot-test error is `5.73e-15`. Status:
`VALIDATED_FDTDX_SPATIAL_Q_CONSERVATIVE_MATERIAL_OVERLAP_REMAP`.

The remapped thermal-Q NPZ is also outside Git: 23,624,266 bytes, SHA-256
`6ab62c06174cd1a0a2b2b1cd8778dcc30e09a3832c2fa09d9b56397bba278d61`.
This validates the mapping operator and its transpose, not a temperature or
PTE result. See
`results_fdtdx_material_overlap_thermal_remap_16period_4window/`.

This folder is intentionally separate from the completed TaIrTe4/void
optimization runs.  It validates a new physical contract:

- fixed TaIrTe4 flake;
- fixed terminal/electrode locations;
- a separate Au/air topology layer above the flake;
- direct Au/TaIrTe4 electrical and thermal coupling will be added only after
  the optical metal endpoint and adjoint are certified.

No previous Run 040–058 artifact is modified by this work.

## Current checkpoint

The first checkpoint freezes the Au endpoint at 10 um and audits two density
paths.  The production candidate is the nonlinear plasmonic interpolation

```text
n(rho) = (1-rho) n_air + rho n_Au
epsilon(rho) = n(rho)^2
```

with `n_Au = 12.1 + 69.2i` from the exact 10-um row of Ordal et al.  The
linear-complex-epsilon law is retained only as a failure/diagnostic control.
Gray density is not interpreted as a physical Au/air effective medium.

Run the offline checkpoint with the environment that contains NumPy and
Matplotlib:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/01_audit_au_material_and_density_path.py
```

Run tests:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python -m pytest -q \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/tests
```

The next checkpoint opens a v261 design session but performs no Maxwell solve
and acquires no GPU engine:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/02_probe_lumerical_au_readback.py
```

The initial sandboxed attempt failed before material import, but the normal
host session subsequently passed as `VALIDATED_LUMERICAL_AU_MATERIAL_READBACK`.
The exact 10-um `(n,k)` material passed the complex-epsilon fit gate; the
global full-table Ordal fit did not and remains diagnostic only. No GPU solve
was launched. Regenerate the consolidated report and manifest with:

```bash
/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/au_on_fixed_tairte4_validation/03_summarize_checkpoint.py
```

## Fail-closed sequence

1. Au material source and nonlinear-density-path audit.
2. Lumerical requested/fitted material readback at 10 um.
3. Binary air/uniform-Au/stripe/island optical controls and mesh convergence.
4. Nonuniform Au density-to-component-Yee Jacobian and optical AD-FD.
5. Explicit Au/TaIrTe4 thermal contact controls and thermal-only AD-FD.
6. Two-layer TaIrTe4/Au weighting-potential controls and electrical AD-FD.
7. Combined physical-density and latent/filter/projection AD-FD.
8. Optimization followed by exact-binary Au/air reevaluation.

If the density route fails material readback, binary equivalence, or AD-FD,
the approved fallback is sharp-interface level-set/shape optimization.

The first binary representation control is deliberately smaller than the
device. It compares a 50-nm finite Au film represented by an exact scalar
`(n,k)` material and by uniform `importnk2` under the same finite Gaussian,
six-PML, native-Yee-Q contract. Runsetup is audited before any GPU forward:

```bash
python 04_run_au_binary_representation_control.py \
  --output-dir /path/to/raw_case --rho 1 --representation scalar \
  --gpu-device 'GPU 6' --contract-only
```

The completed binary checkpoint found that exact scalar Au is stable and
closes its 20-um control volume, while the identical uniform `rho=1`
`importnk2` representation diverges. Therefore no gray-density or density
AD-FD test is promoted. The workflow now follows the approved fallback:
sharp-interface binary Au with level-set/shape derivatives.

The first sharp-interface control moves the two x-normal faces of the exact
scalar-Au film while keeping the source, mesh, monitors, material and all
other faces fixed. It is a forward central-FD geometry control, not yet an
adjoint certificate:

```bash
python 06_run_au_sharp_interface_width_control.py \
  --au-half-x-um 10.0 --output-dir /path/to/raw_case \
  --gpu-device 'GPU 6'
```

No gray Au/air cell is introduced by this route. The next gate compares a
mesh-aware central-FD plateau with the bundled v261 polygon boundary
perturbation adjoint before this representation is allowed into the coupled
thermal/electrical model.

Audit the exact bundled v261 source path without opening Lumerical:

```bash
python 07_audit_v261_sharp_interface_adjoint_path.py
```

This audit only establishes which boundary formula and polygon contract are
installed. It intentionally leaves the numerical AD--FD status pending.

Summarize the sharp-interface forward-FD controls:

```bash
python 08_summarize_au_sharp_interface_width_controls.py
```

At the current 100 nm lateral edge mesh, the `h=0.20` and `0.10 um` central
differences agree to about 1.16%, while `h=0.05 um` changes by about 30%.
Therefore this checkpoint remains fail-closed: the exact-binary Au route is
retained, but a numerical shape-adjoint is not promoted until an edge-local
50 nm mesh produces a forward-FD plateau.

Edge-local refinement is controlled independently from the 100 nm interior
mesh:

```bash
python 06_run_au_sharp_interface_width_control.py \
  --au-half-x-um 8.0 --edge-dxy-nm 25 --edge-band-um 0.5 \
  --output-dir /path/to/raw_case --gpu-device 'GPU 6'
python 09_summarize_au_sharp_interface_mesh_refinement.py
```

The 25 nm edge mesh gives a 0.716% difference between `h=0.10` and `0.05 um`,
so the within-mesh FD-step plateau passes. The `h=0.10 um` derivative still
changes by about 3.04% from edge-50 to edge-25 nm; mesh-independent shape
sensitivity and the numerical boundary adjoint therefore remain unvalidated.

The next isolated diagnostic evaluates the actual sharp-interface `P_Q`
shape-adjoint candidate against those independent central differences:

```bash
python 10_run_au_sharp_interface_pq_adjoint.py \
  --output-dir /path/to/raw_case --gpu-device 'GPU 6'
python 11_summarize_au_sharp_interface_pq_adjoint.py
```

The GPU FieldRegion adjoint source round trip, forward/adjoint component-grid
coordinates, and surface quadrature pass their numerical gates. The resulting
continuous boundary candidate does not: the strong `h=0.05 um` FD is
`-2.904123e-17 W/um`, whereas the candidate AD is `+4.079993e-12 W/um`.
The sign is wrong and the magnitude ratio is about `1.405e5`. Refining the
surface quadrature changes the candidate by only `0.435%`, so the discrepancy
is not repaired by integration refinement.

Status is therefore
`BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_UNVALIDATED`. The continuous pointwise
inside-Au loss trace is not a solver-consistent derivative of the discrete
conformal-Yee `P_Q` objective at the sharp metal edge. It is rejected without
fitting, normalization, sign changes, or gradient rescaling. Together with
the divergent uniform-`rho=1` `importnk2` endpoint, this means that neither
current Au representation permits production Au thermal/electrical/PTE
optimization yet.

The follow-up fixed-external-field diagnostic removes the explicit moving-Au
loss term and tests only the field-mediated boundary kernel:

```bash
python 12_run_au_sharp_interface_external_field_adjoint.py \
  --output-dir /path/to/raw_case --gpu-device 'GPU 0'
python 13_summarize_au_sharp_interface_external_field_adjoint.py
```

The independent `h=0.10` and `0.05 um` central differences agree to 0.154%.
The GPU adjoint has the correct sign and differs from the strong FD by 6.77%,
which is a major improvement over the rejected `P_Q` direct trace but still
fails the 1% gate. The boundary integral itself changes by 38.4% from 401 to
801 samples per vertical edge. The current published state is therefore
`BLOCKED_AU_SHARP_INTERFACE_BOUNDARY_QUADRATURE_UNRESOLVED`; this diagnostic
does not promote an Au optical gradient or permit optimization.

The completed engine HDF5 fields can be inspected without another Maxwell
solve or license checkout:

```bash
python 14_analyze_au_boundary_corner_localization.py
```

This offline localization finds that the two trapezoid endpoints at the sharp
Au corners (`y=+-10 um`) contribute 83.72% of the tangential-E proxy at 801
points per vertical face. The combined smooth-face interior over
`|y|<=9.5 um` changes by only 0.0047% from 201 to 6401 samples. The broad
vertical-face interior is therefore not the source of the tangential-E drift;
it is localized to the sharp metal corners sampled as polygon endpoints. This
does not by itself certify the complete normal-D/tangential-E derivative.

Moving the fixed y ends from `+-10` to `+-18 um` preserves a 0.0802% central-FD
plateau but does not fix the 3D derivative. The center-z rule still changes by
5.08%, and direct integration over the full lateral y-z surface changes by
19.75% and has the wrong sign. This distinguishes two edge classes: the
in-plane rectangle corners and the top/bottom rims of the extruded metal film.

A separate solver-discrete test remeshes `epsilon_x/y/z` at geometry steps of
100, 50, 25 and 12.5 nm without a Maxwell solve. The independently read index
and electric-field coordinates match to `6.78e-21 m`, but the resulting
derivative changes by 100.43% at the final refinement and misses the strong FD
by 68.13%. Thus a hidden E/index coordinate shift is not the explanation, and
conformal-mesh finite differences do not regularize this sharp metal edge.

The controlled remedy is a smooth closed exact-binary scalar-Au ellipse. It is
represented by 512 counter-clockwise vertices, but the boundary quadrature
uses endpoint-free Gauss-Legendre nodes and never samples a polygon vertex.
The x-semi-axis shape velocity is tested independently by recovering the exact
polygon area derivative. Run the forward controls and one adjoint with:

```bash
python 16_run_au_smooth_ellipse_width_control.py \
  --au-half-x-um <7.9|7.95|8.0|8.05|8.1> --au-half-y-um 10 \
  --output-dir /path/to/raw_case --gpu-device 'GPU 0'
python 17_run_au_smooth_ellipse_external_field_adjoint.py \
  --output-dir /path/to/raw_adjoint_case --gpu-device 'GPU 0'
python 18_summarize_au_boundary_root_cause_and_resolution.py
```

The completed smooth control does **not** pass. Its independent central FD
steps agree to `0.3366%`, but the endpoint-free boundary AD has the opposite
sign and differs from the strong FD by `108.69%`; its final quadrature change
is `1.325%`. The corresponding total-`P_Q` FD is also too weak and changes by
`22.62%` between steps. Therefore the original corner-localization result was
an amplifier diagnostic, not the complete root cause. The remaining blocker
is the exact high-contrast lossy-Au interface trace on the conformal Yee mesh,
and no Au optical shape gradient or production optimization is promoted.

A stricter follow-up removes the remaining top/bottom rims as well by replacing
the extruded film with a fully smooth 3-D ellipsoid. Five independent width
controls give a `0.2117%` central-FD plateau, but the GPU adjoint still has the
opposite sign and misses the strong FD by `121.30%`. The last boundary
quadrature refinement changes by `4.05%`. Thus neither in-plane corners nor
thin-film rims alone explain the failure. The validated fixed-geometry Au
material derivative and the failed moving-boundary controls together isolate
the blocker to the continuous high-contrast lossy-Au boundary derivative on
the v261 conformal Yee discretization. No empirical sign flip, normalization,
or gradient rescaling is allowed.

The complete smooth-3-D sequence is reproducible with:

```bash
python 29_run_smooth3d_ellipsoid_control_sequence.py --gpu-device 'GPU 0'
python 28_summarize_au_pva_rim_resolution.py
```

The final v261 GPU diagnosis also tests the documented temperature-grid
coupling as a **numerical optical carrier** and never as a physical thermal
temperature.  Conformal variant 1 reproduces a moderate `n=2, k=0.5` endpoint
on all three component grids with `0.015910%` six-face closure, but every
exact-Au 50-nm endpoint control diverges.  This remains true for forward and
reverse base directions, linear and nonlinear-table interpolation, and a
1000-K numerical carrier span.  A separate short control with the FDTD
stability factor reduced from `0.99` to `0.5` still diverges at the same
physical-time scale, excluding an overly large Courant step as the remedy.
PVA ignores the temperature coupling.

On the smooth 3-D ellipsoid, neither Au-inside nor air-outside one-sided field
traces recover the central-FD sign.  A separate solver-discrete conformal-Yee
diagonal-epsilon Jacobian is also not step converged and has the wrong sign.
The resulting promoted status is therefore:

```text
BLOCKED_AU_TOPOLOGY_OPTICAL_GRADIENT_NO_STABLE_GPU_DIFFERENTIABLE_AU_PATH
```

Generate the consolidated report, JSON, CSV, plot and raw-artifact manifest:

```bash
python 34_summarize_au_temperature_carrier_and_discrete_shape.py
```

Exact scalar Au remains valid for forward simulation, and the previously
validated fixed-geometry material derivative remains valid.  No Au topology
optimization is permitted because no tested v261 GPU representation is both
stable at the exact Au endpoint and differentiable with a validated gradient.

### Lossy-metal route audit and same-step control

The failure has since been narrowed further.  It persists with all six
boundaries changed from PML to Metal, `dt stability factor=0.5`, global
Conformal Variant 0, a sampled passive Ordal base, and even an exact-Au base
with a zero endpoint perturbation.  This excludes PML, the ordinary Courant
step, CV1 alone, and the exact scalar-Au forward material as root causes.  The
v261 Temperature/Index-perturbation carrier itself is not a stable exact-Au
topology representation in this control.

The installed v261 `lumopt2/parametrization/d_eps_calculator.py` was also
audited byte-for-byte.  Its geometry-difference path constructs
`real(index_c**2)` and later takes the real part of the sparse difference.  Its
wavelength-remapping helper clips negative real epsilon and fits a Cauchy
model under the documented source assumption `n>>k~0`.  This is a lossless
dielectric contract and is not valid for the 10 um Au endpoint
`epsilon=-4642.23+1674.64i`.  It must not be used silently for this problem.

Use the following no-Maxwell-solve controls to reproduce the source audit and
complex-epsilon step sweep:

```bash
python 35_validate_au_same_session_complex_deps.py \
  --steps-nm 100,50,25,10,5,2.5,1,0.5,0.25,0.1 \
  --output-dir /path/to/raw
python 36_summarize_au_differentiable_route_resolution.py
```

At a 0.1 nm CAD step the same-session complex-epsilon derivative recovers the
Maxwell-FD sign, but comparison with the older 50 nm FD still differs by
38.50%.  A final equal-step 1 nm central Maxwell FD is therefore evaluated by:

```bash
python 37_validate_au_same_step_local_maxwell_fd.py \
  --minus-dir /path/to/a7.999um \
  --plus-dir /path/to/a8.001um \
  --output-dir /path/to/raw
```

This last script only loads already solved FSP files and performs no Maxwell
solve.  The completed equal-step control **fails**: at `h=1 nm`, the Maxwell
central FD is `-2.916216e-30 J/um`, while the same-session complex diagonal
`d epsilon` contraction is `-9.629138e-31 J/um`.  The sign agrees but the
relative error is `66.9807%`.  A second exact pair at `h=0.5 nm` gives a
Maxwell FD of `-2.918610e-30 J/um`, only `0.0821%` from the `h=1 nm` FD, while
the matching complex `d epsilon` value is `-1.774631e-30 J/um` (`39.1960%`
error).  Thus the Maxwell local derivative is step-converged, but a diagonal
volume-permittivity contraction is not the complete derivative of v261's
conformal moving-Au update, even when the imaginary permittivity, component
coordinates and parameter step are all matched.

This result closes the generic v261 Au-adjoint route fail-closed.  Three paths
remain:

1. keep Au fixed and optimize only the TaIrTe4/dielectric design after a full
   coupled PTE AD--FD check;
2. optimize a few exact-binary Au shape parameters using independent central
   Maxwell differences (or a derivative-free trust-region method), after
   10/5/2.5-nm Au-interface forward-mesh convergence;
3. implement free-form Au topology in a solver with a discrete dispersive
   Drude/CCPR-ADE adjoint and auxiliary material-state gradient terms.

No empirical sign flip, normalization, or gradient rescaling is an accepted
repair.  The current published status is
`BLOCKED_AU_PRODUCTION_GRADIENT_REQUIRES_DISPERSIVE_DISCRETE_ADJOINT`.

### Working 3-D fixed-grid dispersive route

The separate fixed-grid route has now passed its first coupled optical gate.
Here Au is a **nanocube/nanoantenna design material**, not an electrode.  A
two-dimensional Au density is extruded through a fixed thickness above a
fixed anisotropic TaIrTe4 slab.  Au and the TaIrTe4 `a/b/c` components use
passive ADE poles whose finite-time-step harmonic response matches the 10-um
complex permittivity endpoints.  The solver-axis contract is `x=b, y=a,
z=c=b closure`.

Run and publish this control with:

```bash
env PYTHONPATH=/home/seunghyun/.local/au_fdtdx \
  CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  41_validate_au_on_fixed_tairte4_optical_adfd.py

/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  42_summarize_au_on_fixed_tairte4_optical_adfd.py
```

The promoted status is
`VALIDATED_AU_ON_FIXED_TAIRTE4_OPTICAL_ADFD_CONTROL`.  At the finest central
FD step, the maximum strong-direction total-power error is `0.01483%` and the
maximum multi-direction gradient-L2-normalized error is `0.00410%`.  Au and
TaIrTe4 absorbed powers and gradients are stored separately, and
`g_total=g_Au+g_TaIrTe4` closes to `6.89e-7` relative norm.

This does not unblock a v261 moving/conformal Au topology gradient.  It
validates the new causal fixed-grid route only.  Exact-binary Lumerical
endpoint cross-validation, substrate/mesh convergence, thermal/PTE coupling,
and production optimization remain subsequent fail-closed gates.

The exact-binary endpoint runner is:

```bash
python 43_run_lumerical_au_on_tairte4_binary_endpoint.py \
  --au-endpoint 0 --gpu-device "GPU N" --output-dir /raw/path/au0
python 43_run_lumerical_au_on_tairte4_binary_endpoint.py \
  --au-endpoint 1 --gpu-device "GPU N" --output-dir /raw/path/au1
```

It uses exact scalar material endpoints only: a fixed anisotropic TaIrTe4
slab and an optional exact Au nanostructure block.  It does not invoke
`importnk2`, gray Au, a moving conformal boundary, or a Lumerical adjoint.
The raw FSP/NPZ outputs are intentionally external to Git.  A successful run
requires a free GPU **and** solver license; runsetup alone is not promoted as
an endpoint validation.

### Discrete dispersive-adjoint repair control

The required mathematical repair is now represented by a separate offline
control rather than an empirical correction to the failed Lumerical gradient:

```bash
python 38_validate_discrete_drude_adjoint_control.py
```

This control fits a passive one-pole Drude model exactly to the frozen 10-um
Au endpoint, interpolates its pole strength on a fixed discrete grid, and
includes both the Maxwell-operator and direct material-loss derivatives.  It
passes five-direction central AD--FD with a maximum strong-direction relative
error of `4.351e-6` and a maximum linear residual of `4.067e-14`.

Its status is `VALIDATED_DISCRETE_PASSIVE_DRUDE_ADJOINT_CONTROL`, but its scope
is deliberately one-dimensional and algorithmic.  It proves that the
dispersive discrete-adjoint repair is numerically sound; it does not promote
the failed v261 moving-conformal-Au derivative or claim a 3-D production PTE
result.  Production free-form Au still requires the 3-D Yee/PML and
Drude/CCPR auxiliary-state extension plus exact-binary Lumerical endpoint
cross-validation.

## Material provenance

- Au optical `n,k`: [Ordal et al., Applied Optics 26, 744–752 (1987)](https://doi.org/10.1364/AO.26.000744).
- CC0 data transcription: [refractiveindex.info database](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml).
- Nonlinear interpolation: [Zeng, Venuthurumilli, and Xu, ACS Photonics (2021)](https://doi.org/10.1021/acsphotonics.1c00260).
- Bulk Au thermal reference: [NIST resistivity compilation](https://srd.nist.gov/JPCRD/jpcrd155.pdf).

Bulk transport values are references rather than certified thin-film values.
Film thickness, deposition, grain size, and Au/TaIrTe4 electrical/thermal
contacts remain explicit physical uncertainties.

## FDTDX quasi-uniform physical-thickness checkpoint

The post-release FDTDX main snapshot pinned at
`f26f84b70a8cceec9b889553955a868624736bf1` adds the rectilinear-grid route
needed to represent 50-nm Au on 100-nm TaIrTe4 without converting either film
to a coarse one-cell sheet.  The compact GPU control is generated by:

```bash
CUDA_VISIBLE_DEVICES=5 \
PYTHONPATH=/home/seunghyun/.local/fdtdx_main_src/src:/home/seunghyun/.local/au_fdtdx \
JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false \
python 45_validate_fdtdx_quasiuniform_au_tairte4_adfd.py --quick

python 46_validate_fdtdx_quasiuniform_source_direction.py
python 47_summarize_fdtdx_quasiuniform_au_tairte4.py
```

The five-direction compact dispersive-material AD--FD check passes at the 1%
level, and the independent source-only control verifies `-z/+z` reciprocity.
A separate production-width `w0=8.5 um` source-only control realizes a
primary-Ex mean waist of `8.4573 um` and passes the 0.5% closed-surface residual
gate at `0.3655%`.

The material-bearing production-width endpoint comparison is now also closed:
FDTDX and Lumerical absorbed fractions differ by `0.7204%` (TaIrTe4 only) and
`0.6405%` (Au/TaIrTe4), while the Au-present/Au-absent ratio differs by only
`0.0793%`.  FDTDX local native-Yee `Q` agrees with empty-subtracted six-face
power to `0.00785%` and `0.03644%`.  The required unit contract is
`E_SI=eta0*E_internal`, `H_SI=H_internal`, `S_SI=eta0*S_internal`; no empirical
gain is used.  The published state is
`VALIDATED_FDTDX_AU_OPTICAL_FORWARD_AND_COMPACT_MATERIAL_GRADIENT`.

This route is not yet the production thermal/PTE optimizer.  The next gate is
a production-width nonuniform-Au directional AD--FD smoke test followed by the
thermal/electrical chain validation.

## 10-um substrate/Yee-volume endpoint checkpoint

The 285-nm SiO2/lossless-Si diagnostic exposed a component-grid integration
error that the earlier air-only control could not reveal. `Ex` and `Ey` live
on z-edge dual volumes; using the cell-centered z width for every component
over-counted oxide absorption beside a coarse-Si/fine-SiO2 transition. The
corrected code uses component-specific dual volumes and a matched 15-nm
Si/SiO2 interface grid.

At 32 periods, direct material loss versus deep-box time-domain Poynting flux
closes to `0.4742%` for substrate-only, `0.1101%` for TaIrTe4/substrate, and
`0.1678%` for Au/TaIrTe4/substrate. All late-window changes are below
`0.016%`. The endpoint state is
`VALIDATED_FDTDX_DIAGNOSTIC_SUBSTRATE_BINARY_ENDPOINT_CLOSURE`.

This remains a diagnostic material contract because the installed Lumerical
Palik-Si readback is blocked; Si uses an explicitly cited lossless `n=3.4215`
value at 10 um.  A 20x20 nonuniform-Au substrate-bearing optical-Q gradient
also passes at the stable `h=0.02` and `h=0.01` plateau; `h=0.005` remains a
preserved small-step failure.  The combined optical/thermal/electrical PTE
gradient remains pending. See
`results_fdtdx_substrate_matched_interface_endpoints_32period_gpu3/` and
`results_fdtdx_substrate_gradient_plateau_gpu3/`.

## Spatial Maxwell-Q to explicit thermal/weighting forward

The 16-period, four-window substrate-bearing endpoint now exports native-Yee
`Qx`, `Qy`, and `Qz` with component-specific dual coordinates and volumes.
Independent reintegration differs by `3.62e-8` relative, matched-volume
closure is `0.12237%`, and Au/TaIrTe4/SiO2 absorb `22.1982%`, `61.6226%`, and
`16.1792%` of the literal source power.  The external raw artifact is recorded
by path, byte size, and SHA-256; it is not committed to Git.

The source is then transferred with an exact material-overlap conservative
operator.  No nearest-cell projection, power deletion, clipping, smoothing,
gain, or global rescaling is used.  Total power error is zero to printed
precision and the worst transpose dot error is `5.73e-15`.

The explicit `266x266x33` Au/TaIrTe4/SiO2/Si FVM and Au-aware two-layer
weighting operator pass their forward gates for both named TaIrTe4/SiO2
interface scenarios.  Relative to thermally grown `G=7.37e6 W/(m2 K)`, the
evaporated `G=7.37e4 W/(m2 K)` scenario gives about `16.42x` higher Tmax,
`28.05x` higher TaIrTe4 volume-average temperature rise, and `12.56x` higher
literal-normalization PTE current.  This is physical-parameter sensitivity,
not numerical error or an experimental prediction.

`G_Au/TaIrTe4` remains an explicitly labelled Au/MoS2 analogue and the
electrical contact remains a numerical scenario.  These forward results do
not authorize optimization: the next fail-closed gate is combined spatial
Maxwell-Q, explicit-thermal, and Au-aware electrical directional AD--FD.

The first derivative subgate is now independently closed by
`67_validate_explicit_thermal_weighting_fixed_spatial_q_adfd.py`. It holds
the certified spatial Maxwell source fixed and differentiates the full
`266x266x33` thermal material/contact operator plus the floating-Au
electrical/weighting operator. The transpose of both 500-nm-to-100-nm
lateral averaging and the TaIrTe4 thickness average is included explicitly.
Across both interface scenarios and five directions at three FD steps, the
worst strong-direction error is `2.30e-7` and the worst gradient-L2-normalized
error is `5.68e-8`. A separate thermal-matrix derivative audit differs by at
most `5.55e-7`. The remaining term is the thermal-source adjoint pulled back
through the conservative remap to the native-Yee spatial Maxwell Q.

That remap transpose is now certified by
`68_build_native_yee_thermal_source_adjoint_weights.py`. It applies the
transpose of both material-primal-to-explicit and component-Yee-to-primal
power maps. Ex, Ey, and Ez retain their own physical coordinates and dual
volumes. The worst two-stage dot-test error is `6.63e-15`, and the native-Yee
weighted source contraction matches the explicit thermal-grid contraction to
`5.48e-16` relative. The resulting unscaled weights have units `A/W`; the
next solve uses them as the spatial FDTDX objective.

The native-Yee spatially weighted Maxwell source derivative now also passes.
The thermally-grown source-adjoint weights are contracted directly with the
Au, TaIrTe4, and SiO2 component-native powers inside the 16-period GPU FDTDX
solve.  The adjoint-aligned derivative is `6.3285942e-18 A`; central FD at
`h=0.01` gives `6.3288875e-18 A`, a `0.004634%` relative error.  The matched
Q/flux closure is `0.123096%`, the late-Q change is `0.019832%`, and the
weighted-objective late change is `0.038413%`.  No objective or gradient
rescaling is used.  Status:
`VALIDATED_FDTDX_NATIVE_YEE_SPATIALLY_WEIGHTED_PTE_SOURCE_GRADIENT`.

This is the Maxwell/source branch only.  It does not yet certify the full PTE
gradient or authorize Au optimization.  The next fail-closed gate must add
this source term to the already validated direct thermal/contact and
electrical/weighting terms, then compare the sum against an end-to-end central
FD that recomputes Maxwell Q, conservative remap, explicit thermal transport,
and terminal current for both perturbed densities.

The first end-to-end combined directional smoke now passes as well.  At
`rho +/- 0.01 d`, where `d` is aligned with the sum of all three gradient
branches, each perturbation independently recomputes FDTDX native-Yee Q,
the two conservative material-overlap remaps, the explicit 3-D thermal
solution, and the Au-aware weighting/current solution.  The chain-rule
contributions are:

- Maxwell source: `6.1504913e-18 A`
- direct thermal/contact: `-3.0921740e-19 A`
- direct electrical/weighting: `6.1176116e-19 A`
- combined AD: `6.4530351e-18 A`
- end-to-end central FD: `6.4521531e-18 A`

The combined relative error is `0.013667%`; the worst linear residual is
`8.25e-10`, while thermal and terminal balances are near roundoff. Status:
`VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_DIRECTIONAL_ADFD`.
This is still one strongest-direction smoke, not the final multi-direction
or latent/filter/projection certificate.  It does not yet authorize
optimization.

The independent-direction extension now passes too.  The validated Stage-70
adjoint-aligned point is reused by exact SHA, while smooth-asymmetric,
central-localized, design-edge-localized, and fixed-seed-random directions
each recompute the complete forward chain at `rho +/- 0.01 d`.  Strong
directions have at most `0.12320%` relative AD--FD error, and the worst
near-null-safe error normalized by the full gradient norm is `0.01837%`.
The edge-localized direction is genuinely near-null (`0.0845%` of the full
gradient norm), so its ordinary `2.03%` relative error is reported but is not
misclassified as a strong-direction failure.  All optical closure, spatial-Q,
conservative-remap, residual, energy-balance, and terminal-balance gates pass.
Status:
`VALIDATED_FULL_COMBINED_FDTDX_THERMAL_WEIGHTING_PTE_MULTIDIRECTION_ADFD`.

This closes the physical-density gradient gate.  The latent/filter/projection
gates described below close the remaining derivative chain; they do not by
themselves select a fabrication contract or start an Au optimization.

## Latent/filter/projection closure

The finite nonperiodic conic-filter and tanh-projection mapping now passes its
solver-free transpose and finite-difference controls.  For the tested 20x20,
500-nm-pitch design grid and 750-nm filter-radius numerical scenario, the
worst JVP--VJP dot-test error is `2.64e-17`, the worst mapping-only FD error is
`8.72e-7`, opposite-edge wrap is zero, constant preservation is exact, and no
monotonicity regressions occur.  Status:
`VALIDATED_AU_LATENT_FILTER_PROJECTION_MAPPING`.

The full latent-chain central AD--FD then recomputes, at every `latent +/-
0.01 d`, the finite filter, beta-2 projection, substrate-bearing FDTDX native
Yee Q, exact material-overlap remap, explicit 3-D thermal solve, and Au-aware
electrical weighting/current solve.  The already certified physical baseline
is reconstructed from an interior latent field with `9.99e-16` maximum error,
so no clipping is needed.  Three independent latent directions pass:

- adjoint-aligned relative error: `0.002231%`;
- smooth-asymmetric relative error: `0.032211%`;
- fixed-seed-random near-null-safe gradient-norm error: `0.008539%`.

The worst residual is `9.86e-10`; thermal and terminal balances are
`2.79e-11` and `2.15e-12`.  No Q, density, objective, or gradient rescaling is
used.  Status:
`VALIDATED_FULL_LATENT_FILTER_PROJECTION_FDTDX_PTE_ADFD`.

This validates the differentiable chain for the stated numerical mapping
scenario.  The 750-nm radius is not yet a final fabrication/minimum-feature
contract, and no Au optimization has been run.  That contract and its beta/
optimizer continuation must be frozen before optimization is authorized.

## Checkpoint-free two-solve production adjoint

The earlier exact reverse-through-time certificate remains the immutable
reference, but it is not a viable optimization kernel: one Maxwell VJP took
`2920.02 s` and stored a checkpoint stack.  Stages 74--77 therefore validate a
different implementation of the same derivative.  It runs one settled-CW
forward solve and one reciprocal distributed-current adjoint solve, retains
only their final phasors, and contracts them on component-specific Yee grids.
It never calls reverse-mode AD through the FDTD time loop.

On the frozen `48 x 48 um`, `288 x 288 x 119`-cell, 16-period `E||b`
production contract, forward plus adjoint execution is `212.29 s`; the first
compile plus execution is `259.85 s`.  This is `13.75x` faster than the frozen
checkpointed VJP.  The checkpoint-free optical gradient agrees with the
reference to `0.15692%` in vector norm and `0.06873 deg` in direction.

Adding the independently validated thermal/contact and electrical/weighting
direct terms gives a combined physical-density vector error of `0.15389%`, a
norm error of `0.10700%`, and an angle of `0.06334 deg`.  Against the five
existing end-to-end finite-difference directions, the worst strong-direction
relative error is `0.25806%`; the worst full-gradient-normalized directional
error is `0.12047%`.  Status:
`VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_COMBINED_PTE_GRADIENT_EQUIVALENCE`.

The same replacement also passes through the certified finite conic filter
and beta-2 tanh projection.  The latent-gradient vector error is `0.13586%`,
the angle is `0.05003 deg`, and the worst existing latent directional-FD error
is `0.41182%`.  Status:
`VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_LATENT_PTE_GRADIENT`.

These certificates do not freeze the current source-adjoint weights during an
optimization.  A production iteration must recompute `Q`, the thermal and
weighting solutions, and `dI/dQ` at its current density before launching the
reciprocal Maxwell solve.  The inherited substrate provenance also remains
explicitly blocked as `BLOCKED_LUMERICAL_10UM_SI_PALIK_READBACK`; equivalence
to the frozen numerical contract is not a paper-material certification.

That dynamic update has now been executed once at production size as well.
Starting from the current forward phasors, the code builds native-Yee material
power in memory, conservatively maps it to the explicit thermal grid, solves
the current thermal/electrical systems and their adjoints, transposes the two
overlap maps, and launches the reciprocal Maxwell solve with the resulting
current `dI/dp_Yee`.  It does not read the frozen weights for the actual
adjoint source; those are used only as a post-run reference.

The regenerated Au/TaIrTe4/SiO2 weights differ from the frozen baseline by at
most `3.53e-8` relative.  Native-Yee and explicit-grid source-adjoint
contractions differ by `2.87e-16`, and the PTE objective differs from that
weighted-Q contraction by `4.44e-10`.  The dynamic combined gradient differs
from the frozen end-to-end gradient by `0.15399%`, with `0.10707%` norm error
and `0.06338 deg` angle.  The measured first-run pipeline after runsetup audit
was `426.62 s`; the two Maxwell solves themselves took `358.50 s`, still
`8.15x` faster than the frozen checkpointed reverse pass.  Status:
`VALIDATED_FDTDX_PRODUCTION_DYNAMIC_CHECKPOINT_FREE_PTE_ITERATION`.
