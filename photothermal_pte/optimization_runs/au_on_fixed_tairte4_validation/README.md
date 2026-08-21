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

The five-direction dispersive-material AD--FD check passes at the 1% level,
and the independent source-only control verifies `-z/+z` reciprocity.  The
subwavelength compact Gaussian used by the material-gradient control fails its
closed-surface flux audit.  A separate production-width `w0=8.5 um` source-only
control, however, realizes a primary-Ex mean waist of `8.4573 um` and passes
the 0.5% closed-surface residual gate at `0.3655%`.  The published state is
`PARTIAL_FDTDX_AU_GRADIENT_AND_W8P5_SOURCE_VALIDATED_PENDING_MATERIAL_CROSSCHECK`.
This route is not yet the production thermal/PTE optimizer; the next gate is a
material-bearing production-width comparison against the validated exact-binary
Lumerical endpoints.
