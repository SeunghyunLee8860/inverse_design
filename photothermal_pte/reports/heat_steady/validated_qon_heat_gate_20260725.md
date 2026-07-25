# Validated Q_on to HEAT steady-state gate (2026-07-25)

Baseline commit: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`.

## Outcome

The validated disk-x optical Q was exported without a Maxwell rerun and its
1 W/m2 unit-response normalization was verified.  The three-mesh analytic
steady-state HEAT control passed on the finest mesh.  The physical full-stack
solve was intentionally not started because mandatory physical inputs and the
TaIrTe4 conductivity-tensor capability gate failed.

No transient, density/heat-capacity model, PTE current, adjoint, gradient,
optimization, beta continuation, geometry update, clipping, gain, Q rescaling,
or smoothing was executed.

## Code audit

- `02_export_fdtd_qon.py` reads Lumerical `pabs_adv.Pabs`, whose stored result is
  source-power normalized.  It reconstructs native Q with `sourcepower(f)` and
  physical/unit-response Q with `I_target/sourceintensity(f)`.
- The completed production FSP is postprocessed only after checking its realized
  source, sampled material, monitor, mesh, flake dz, solver, and dt contract.
  The stored project resource is GPU 2 whereas the current host default expected
  GPU 1; resource validation is therefore not applied to postprocessing.  No
  solver run occurs in this path.  The saved FSP lacks the power-monitor d-cards,
  so R/T/local flux provenance is the validated `disk_x.json` that points to the
  exact same FSP.  Pabs remains extracted directly from the FSP.
- `03_import_qon_heat_steady.py` reads only `fdtd_qon/q_on_physical.npz` and
  requires the corresponding optical summary to be validated.  Coordinates are
  meters, Q is W/m3, and array order is explicitly `(x,y,z)`.
- The existing physical geometry is a 6 um by 6 um cell with 2.0 um Si,
  0.285 um SiO2, 0.1 um TaIrTe4, and a 0.6 um design disk.  Interfaces are
  perfect-contact geometry interfaces.
- The existing physical script fixes only the substrate bottom to 300 K.  It
  assumes unassigned top/lateral faces are adiabatic but does not read this back.
  It has no explicit physical HEAT mesh, saves only two temperature planes and
  one heat-flux plane, and checks only bottom outflow with a legacy 5% threshold.
  It therefore does not yet satisfy the requested full 3-D outputs, six-face
  energy balance, or two-mesh physical convergence contract.
- `run_stage1.py` runs all stages when invoked without stage flags, including
  scaling/transient.  It was not used.

## Optical Q and unit-response normalization

- Mode: `UNIT_RESPONSE_MODE_1_W_M2` (not an experimental temperature)
- Incident intensity: 1 W/m2
- Cell area: 3.6e-11 m2
- Incident power per cell: 3.6000000000000005e-11 W
- FDTD source intensity at 4 um: 1.3197216623094302e-3 W/m2
- Native integrated Q: 2.215909536713549e-14 W
- Unit-response integrated Q: 1.6790733985800054e-11 W
- A_Q from exported Pabs: 0.4664092773833347
- Validated Pabs reference A_Q: 0.46640927738333476
- A_Q P_incident: 1.679073398580005e-11 W
- Unit-response power-identity relative error: 1.9243794050443975e-16
- A_local: 0.46612396705472053
- Q/local closure: 0.06120910933136679%, below the 0.5% limit

The validated native-component total A_Q is 0.4664092803515875.  Its absolute
difference from the installed pabs_adv integral is 2.9682527e-9.

## Q-grid preflight

- Shape/order: `(241,241,36)`, `(x,y,z)`
- x/y range: -3 um to +3 um; z range: -150.0000000000023 nm to
  +48.944565493963 nm
- Coordinates: finite and strictly increasing
- Periodic endpoint relative maximum difference: x 2.65368135e-6,
  y 4.64345195e-12 (normalized by global max Q)
- Reintegrated/exported-power error: 0
- Power outside the TaIrTe4 bounds `[-100 nm,0]`: 0 W
- NaN/Inf: 0; negative voxels: 0; integrated negative power: 0 W
- Original samples and coordinates are retained; no seam deletion, clipping,
  smoothing, gain, or post-export rescaling is applied.

The Lumerical-side `integrated Q` result was not produced because it requires a
physical HEAT solve, which is blocked below.  Consequently the requested
FDTD-to-solved-HEAT `<0.5%` criterion has not yet been claimed as passed.

## Analytic v261 HEAT control

The 1 um uniform-heating slab used k=10 W/m/K, Q=1e14 W/m3, and a 300 K bottom.
Density and heat capacity were not set.

| max edge | DeltaT max error | normalized RMSE | energy error | pass |
|---:|---:|---:|---:|:---:|
| 100 nm | 0.049513% | 0.092878% | 3.743090% | no |
| 50 nm | 0.005598% | 0.024486% | 1.880824% | no |
| 25 nm | 0.002465% | 0.006480% | 0.946587% | yes |

The finest mesh satisfies all three 1% criteria and the errors converge with
mesh refinement.
