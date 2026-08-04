# Finite non-periodic TaIrTe4/SiO2 AD–FD contract

Status: `VALIDATED_EMPTY_AIR_CPU_TFSF_SOURCE_GATE_DEVICE_PENDING`

This contract supersedes the 6 µm periodic inverse-design certificate for the
new physical problem. The old result remains immutable numerical provenance;
it is not evidence for this finite problem.

## Geometry

- protected inverse-design/PTE ROI: 2 µm × 2 µm,
  `x,y=[-1,1] µm`;
- TaIrTe4 thickness: 100 nm, z=-100...0 nm;
- lateral TaIrTe4 flake: 4 µm and 6 µm named numerical scenarios until the
  fabrication footprint is supplied;
- inverse-designed SiO2: 2 µm × 2 µm × 600 nm, z=0...600 nm;
- bottom thermally-grown SiO2: 285 nm;
- optical Si depth: 2 µm; thermal Si-depth baseline: 20 µm;
- the radius-0.8-µm disk is a forward control, not the design.

## Optical problem

- actual optical endpoints: air n=1 and SiO2 n=1.38;
- the immutable physical/design/Q region is exactly
  `x,y=[-1,1] µm`; no padding or PML sample enters the objective;
- requested illumination: normal-incidence ideal plane wave, x polarization
  for the first certificate;
- 3–6 µm source support and 4 µm analysis;
- all six outer boundaries are PML; no periodic or Bloch boundary;
- a Bloch/periodic plane-wave source crossing transverse PML was tested and
  rejected; it is not a matched source/boundary pair;
- the official all-PML Diffracting plane-wave source was tested through a
  24 µm domain and 20 µm aperture and failed the exact central ROI gates;
- the installed v261 GPU engine rejects enabled TFSF;
- on 2026-07-26 the user explicitly authorized CPU TFSF forward solves;
- a `4×4 µm` empty-air CPU TFSF source gate with a `2.6 µm` transverse TFSF
  box passed the exact central-ROI field and energy gates at PML 24 and 32;
- the PML-24 native solver time was 3.27 s and complete Python `run()` time
  was 5.53 s for the 86×86×70 empty-air control;
- this does not yet promote a device source or authorize device AD–FD, because
  a finite 4 µm flake cannot fit inside a TFSF box inside a 4 µm FDTD domain;
- auto non-uniform mesh, conformal variant 1, accuracy 5;
- TaIrTe4 dz=5 nm with a 2.5 nm diagnostic;
- any future lateral padding will be selected by convergence and will never
  change the protected 2 µm design/PTE footprint.

The TaIrTe4 flake is now larger than the design ROI. Until fabrication
dimensions are supplied, `4 µm` and `6 µm` square flakes are named numerical
scenarios rather than final experimental geometry. Optical Q outside the
design ROI is retained for the subsequent full-flake thermal solve; it is not
deleted merely because the PTE objective is evaluated in the central ROI.
Gaussian illumination is not used. CPU TFSF is now the approved forward-source
candidate; the adjoint source remains a GPU-compatible FieldRegion source with
the TFSF object disabled. The GPU rejection evidence is reported in
`GPU_PLANE_WAVE_ROI_REPORT.md`, and the CPU empty-air source gate is reported
in `CPU_TFSF_4UM_ROI_REPORT.md`.

## Explicit thermal problem

- TaIrTe4 kappa=diag(14.4,3.8,1.0) W/(m K);
- SiO2 kappa=1.38 W/(m K), Si kappa=145 W/(m K);
- TaIrTe4/bottom-SiO2 G=7.37e6 W/(m2 K);
- TaIrTe4/deposited-design-SiO2 G=7.37e4 W/(m2 K);
- SiO2/Si G=1.1e9 W/(m2 K), retained as a named candidate;
- exposed TaIrTe4/air G=1 W/(m2 K);
- exposed SiO2/air Robin h=10 W/(m2 K), T_inf=300 K;
- far lateral substrate faces and bottom Si are 300 K numerical reservoirs;
- no physical device sidewall is silently adiabatic.

Gray design cells use the declared certificate relaxation
`k_air + rho*(k_SiO2-k_air)` and the top interface uses
`G_air + rho*(G_deposited_SiO2-G_air)`. These are inverse-design
relaxations, not measured gray-composite properties.

## PTE readout

The certificate uses a unit-potential uniform 45-degree weighting field:

`grad(psi)=(xhat+yhat)/(4 µm)`.

This corresponds to opposite diagonal equipotential lines spanning
2*sqrt(2) µm. It is a declared idealized electrode model, not a claim that a
fabricated contact fringe field is uniform.
