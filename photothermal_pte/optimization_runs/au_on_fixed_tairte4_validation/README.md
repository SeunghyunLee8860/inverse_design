# Au topology validation on a fixed TaIrTe4 flake

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

## Material provenance

- Au optical `n,k`: [Ordal et al., Applied Optics 26, 744–752 (1987)](https://doi.org/10.1364/AO.26.000744).
- CC0 data transcription: [refractiveindex.info database](https://raw.githubusercontent.com/polyanskiy/refractiveindex.info-database/main/database/data/main/Au/nk/Ordal.yml).
- Nonlinear interpolation: [Zeng, Venuthurumilli, and Xu, ACS Photonics (2021)](https://doi.org/10.1021/acsphotonics.1c00260).
- Bulk Au thermal reference: [NIST resistivity compilation](https://srd.nist.gov/JPCRD/jpcrd155.pdf).

Bulk transport values are references rather than certified thin-film values.
Film thickness, deposition, grain size, and Au/TaIrTe4 electrical/thermal
contacts remain explicit physical uncertainties.
