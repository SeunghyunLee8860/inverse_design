# TaIrTe4 photothermal / PTE validation

This repository export intentionally contains only the current photothermal/PTE work. Large Lumerical solver projects, raw fields, optimization runs, and unrelated inverse-design artifacts are excluded.

The runnable validation code is under `photothermal_pte/validation/photothermal_stage1`. Its minimal inherited FDTD geometry/runtime dependencies are under `photothermal_pte`, and compact scientific reports are under `photothermal_pte/reports`.

The latest GPU root-cause closure is documented in `photothermal_pte/reports/root_cause/ROOT_CAUSE_REPORT.md`. With the corrected broadband material/source fitting contract, no global uniform mesh override, auto non-uniform mesh, conformal variant 1, and mesh accuracy 5, the patterned disk gave `A_Q=0.466393700`, local-flux absorption `0.466118738`, and relative mismatch `0.0590%`. No Q clipping, flux gain, or HEAT run was used for that closure.

The source-bandwidth follow-up is under `photothermal_pte/reports/source_bandwidth`. Of 3.6–4.4, 3–6, 2.67–8, and 3–12 µm, the narrowest range satisfying flat-y flux/Q closure, flat-y TMM agreement, and disk-x closure below 0.5% was **3–6 µm**. The final flat x/y/45-degree and disk-x regression all passed. `production_optical_contract_proposed.patch` is a review-only proposal and has not been applied to the production model.

## Stage-1 validation

This directory is an isolated, forward-only validation of

`FDTD -> Q_on [W/m^3] -> HEAT steady state -> HEAT transient`.

It does not call the inverse-design optimizer, mapping/projection,
adjoint gradient, PTE-current, or electrode-weighting-field paths. The existing
production files outside this directory are not modified.

## What each file does

- `config_stage1.py`: SI-unit inputs, acceptance thresholds, and explicit
  thermal-property gates.
- `lumerical_api.py`: discovers `/opt/lumerical/v261` first and
  `/opt/lumerical/v251` second, loads the matching `lumapi.py`, and provides
  output/probe helpers.
- `00_probe_heat_api.py`: creates each requested DEVICE/HEAT object in a clean
  session, records every exposed property/default, and probes scalar versus
  diagonal thermal-conductivity round trips.
- `01_validate_heat_analytic.py`: solves the 1 um cubed, uniformly heated slab
  on three meshes and checks the analytic temperature, RMSE, and bottom heat
  outflow.
- `02_export_fdtd_qon.py`: reuses the repository geometry/material builder,
  inserts a fixed binary 1.5 um-radius disk without calling a mapping, and
  evaluates absorption with the installed `pabs_adv` object-library group. It also compares global R/T flux with a local flux box around TaIrTe4 and defaults to a 4 ps run (`STAGE1_FDTD_SIMULATION_TIME_S` can override it).
- `03_import_qon_heat_steady.py`: constructs the matching solid stack, imports
  the original rectilinear Q dataset with `importdataset`, and checks import and
  steady-state power balance.
- `04_validate_heat_scaling.py`: checks Q scales 0.5/1/2 and Si thicknesses
  3/6/12 um.
- `05_validate_heat_transient.py`: runs a guarded pulse and time-step
  refinement only when every solid has explicit `rho` and `cp`.
- `run_stage1.py`: timestamped fail-closed orchestration and final JSON/Markdown
  report generation.

## Important physical gates

All lengths are metres, Q is W/m³, thermal conductivity is W/(m K), density is
kg/m³, heat capacity is J/(kg K), and temperature is K.

The optical repository defines the design material only by optical index.
Therefore the default is:

```text
STAGE1_DESIGN_THERMAL_MODE=required
```

`STAGE1_DESIGN_K_W_MK`, `STAGE1_SIO2_K_W_MK`, and `STAGE1_SI_K_W_MK` must be
provided before a physical steady HEAT run. No silent material-database value
is accepted.

The optional plumbing-only mode is:

```bash
STAGE1_DESIGN_THERMAL_MODE=pipeline_placeholder
STAGE1_PLACEHOLDER_DESIGN_K_W_MK=1.0
```

Every such artifact is labeled
`NOT_PHYSICAL_PLACEHOLDER_DESIGN_THERMAL_PROPERTY` and must not be interpreted
as a scientific result.

Transient execution additionally requires:

```text
STAGE1_DESIGN_RHO_KG_M3       STAGE1_DESIGN_CP_J_KG_K
STAGE1_TAIRTE4_RHO_KG_M3      STAGE1_TAIRTE4_CP_J_KG_K
STAGE1_SIO2_RHO_KG_M3         STAGE1_SIO2_CP_J_KG_K
STAGE1_SI_RHO_KG_M3           STAGE1_SI_CP_J_KG_K
```

The requested TaIrTe4 conductivity is diagonal
`[14.4, 3.8, 1.0] W/(m K)` in repository x/y/z = crystallographic a/b/c.
The code requires that this vector round-trip through the installed HEAT
material property. If the installed version exposes only scalar conductivity,
the physical steady stage stops rather than silently using an average.

## Optical Q method and normalization

The installed v261 `pabs_adv` group:

1. reads Ex/Ey/Ez and the corresponding diagonal permittivity components on
   their native Yee locations;
2. computes each component's loss separately;
3. interpolates the component losses to the common result grid;
4. applies the explicitly enabled x/y periodic-boundary correction; and
5. divides the result by `sourcepower(f)`.

The script saves both native-source Q and

```text
Q_physical = Q_native * INCIDENT_INTENSITY_W_M2 / sourceintensity(f)
```

without clipping negative values or applying a power-matching gain. Integrals
use the actual x/y/z coordinates and nested trapezoidal quadrature.

The actual repository baseline differs from the approximate prompt in one
important value: its default Si substrate is 2.0 um, not approximately 3.0 um.
The optical boundary conditions are x/y periodic and z PML.

## Run

Use the environment that already contains NumPy, SciPy, Matplotlib, and the
repository dependencies:

```bash
cd /path/to/inverse_design
python \
  photothermal_pte/validation/photothermal_stage1/run_stage1.py \
  --lumerical-version auto
```

Run the optical control with the design region replaced by air:

```bash
python \
  photothermal_pte/validation/photothermal_stage1/02_export_fdtd_qon.py \
  --lumerical-version v261 \
  --output-dir /absolute/path/to/empty-output \
  --fixed-geometry none
```

Run selected stages only:

```bash
python \
  photothermal_pte/validation/photothermal_stage1/run_stage1.py \
  --run-probe --run-analytic
```

For an explicit new output directory:

```bash
python \
  photothermal_pte/validation/photothermal_stage1/run_stage1.py \
  --output-dir /absolute/path/to/new-empty-directory
```

The runner refuses a non-empty output directory. Outputs are otherwise written
under `output/<UTC timestamp>/`.

## Observed result on this host (v261)

- The HEAT-only analytic slab previously passed on the 25 nm mesh: `Tmax=304.9998768 K`, Delta-T relative error `2.46e-5`, NRMSE `6.48e-5`, and energy error `0.9466%`.
- The original narrow-source/global-uniform-mesh patterned disk failed closure: advanced-Pabs `0.508019448`, local flux `0.453709357`, mismatch `11.9702%`.
- The corrected broadband + auto non-uniform + conformal variant 1/accuracy 5 disk passed: advanced-Pabs `0.466393700`, local flux `0.466118738`, mismatch `0.0590%`. The full evidence and mesh/material contract are under `photothermal_pte/reports`.
- HEAT was not run during the current optical root-cause investigation. Physical FDTD-to-HEAT/PTE remains gated by explicit thermal properties and support for the requested diagonal TaIrTe4 thermal conductivity.

These are diagnostic validation results, not a validated PTE prediction.

## Pass/fail semantics

- The analytic stage is an absolute gate: ΔTmax error, normalized RMSE, and
  energy error must each be below 1%.
- FDTD volume absorption versus flux absorption must agree within 5%.
- FDTD-to-HEAT imported power must agree within 0.5%.
- Scaling requires R² above 0.999 and slope error below 2%.
- The 12 versus 6 um substrate comparison is reported separately; 3 um is
  never promoted automatically to a final physical result.
- Transient requires long-ON versus steady error below 1%, monotonic cooling,
  and time-step refinement error below 2%.

Successful execution alone never sets `validated=true`; every relevant numeric
criterion must pass.
