# Photothermal stage-1 validation

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

## Source-bandwidth validation

- `22_run_source_bandwidth_case.py` replays the corrected v261 optical contract while varying only the source range and records component-resolved Q, local/six-face flux, flat-stack TMM, source pulse properties, fitted epsilon, dt, and mesh coordinates.
- `23_summarize_source_bandwidth_sweep.py` applies the 0.5% selection gates and writes the compact sweep and selected-range regression report.
- `24_run_production_optical_regression.py` builds a fresh FSP through the production `eqc_lib.build_control_base(force=True)` entrypoint, asserts the realized source/material/monitor/mesh/solver contract before the solve, and records flat x/y/45-degree and fixed disk-x closure.
- The validated selection is a single broadband 3–6 µm source with analysis monitors and Pabs evaluated only at 4 µm. HEAT is not called by either script.

## Finite 2 µm optical-Q validation

- `27_validate_finite_2um_optical_q.py` is a separate, non-periodic v261
  builder for a finite 2 µm × 2 µm × 100 nm TaIrTe4 flake on
  285 nm SiO2/Si. Every FDTD boundary is PML.
- The source is a 3–6 µm finite Gaussian beam and all reported absorption
  quantities are evaluated at 4 µm. TFSF was rejected because the actual v261
  GPU engine does not support it. The required empty layered-stack and
  zero-amplitude controls use the same Gaussian builder.
- Finite absorption is measured with a six-face flux box. The script also
  exports unclipped component-resolved `Qx`, `Qy`, and `Qz`, normalizes to a
  measured incident intensity of 1 W/m², and records the volume-Q versus
  six-face closure.
- This path never calls HEAT, the optimizer, mapping/projection, adjoint, or
  gradient code, and it does not crop, tile, gain-correct, or rescale a
  periodic artifact.

Example contract-only build:

```bash
python \
  photothermal_pte/validation/photothermal_stage1/27_validate_finite_2um_optical_q.py \
  --output-dir /absolute/path/to/new-empty-directory \
  --case empty-stack --domain-um 8 --pml-layers 24 --flake-dz-nm 5 \
  --source-span-um 6 --waist-um 2 --gpu-device "GPU 0" --contract-only
```
