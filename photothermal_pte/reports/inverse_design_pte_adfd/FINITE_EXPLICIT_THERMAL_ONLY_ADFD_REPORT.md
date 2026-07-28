# Finite explicit thermal-only AD–FD

Status: `VALIDATED_FINITE_EXPLICIT_THERMAL_ONLY_ADFD`

This is an exact discrete fixed-Q certificate for the thermal branch only.
It is not a finite-device optical, combined, latent, or optimization result.

## Model

- finite TaIrTe4 flake: 4 µm square × 100 nm;
- design: 2 µm square × 600 nm;
- thermal domain: 32 µm square, Si depth
  20 µm;
- AD-certificate core grid: 100 nm,
  shape `(76, 76, 76)`;
- no periodic boundaries;
- far x/y and bottom: fixed DeltaT=0 K numerical reservoirs;
- top: Robin h=10 W/(m2 K);
- flake sidewalls: explicit TaIrTe4/air G=1
  W/(m2 K).

The fixed thermal control source is identical in every FD solve.  The
projected physical density changes both the full 3D design conductivity
`k_air + rho*(k_SiO2-k_air)` and the TaIrTe4/design contact conductance
`G_air + rho*(G_deposited_SiO2-G_air)`.

## Exact adjoint

`K(rho) theta = M_Q Q`, `K(rho)^T lambda = dI_PTE/dtheta`, and

`dI/drho = -lambda^T (dK/drho) theta`.

Every bulk, interface, and top-convection face derivative is formed from the
same two-half-cell series resistance used by the forward matrix.

## Directional AD–FD

| direction | h | adjoint [A] | centered FD [A] | relative error |
|---|---:|---:|---:|---:|
| uniform | 0.01 | -6.256361e-18 | -6.256364e-18 | 5.546112e-07 |
| uniform | 0.005 | -6.256361e-18 | -6.256362e-18 | 1.385290e-07 |
| uniform | 0.0025 | -6.256361e-18 | -6.256361e-18 | 3.405133e-08 |
| uniform | 0.00125 | -6.256361e-18 | -6.256361e-18 | 8.581265e-09 |
| sinusoidal | 0.01 | -9.402831e-18 | -9.402785e-18 | 4.799776e-06 |
| sinusoidal | 0.005 | -9.402831e-18 | -9.402819e-18 | 1.199701e-06 |
| sinusoidal | 0.0025 | -9.402831e-18 | -9.402828e-18 | 2.999658e-07 |
| sinusoidal | 0.00125 | -9.402831e-18 | -9.402830e-18 | 7.488931e-08 |
| center_edge | 0.01 | 2.001312e-18 | 2.001304e-18 | 4.133053e-06 |
| center_edge | 0.005 | 2.001312e-18 | 2.001310e-18 | 1.033089e-06 |
| center_edge | 0.0025 | 2.001312e-18 | 2.001312e-18 | 2.595660e-07 |
| center_edge | 0.00125 | 2.001312e-18 | 2.001312e-18 | 6.626416e-08 |
| seeded_random | 0.01 | 2.530225e-19 | 2.530215e-19 | 4.215438e-06 |
| seeded_random | 0.005 | 2.530225e-19 | 2.530223e-19 | 1.052169e-06 |
| seeded_random | 0.0025 | 2.530225e-19 | 2.530225e-19 | 2.727320e-07 |
| seeded_random | 0.00125 | 2.530225e-19 | 2.530225e-19 | 1.305626e-07 |

Worst primary-step (`h=0.005`) error:
`1.199701e-06`.

Gradient L2 norms [A]: combined
`1.957888e-18`, bulk-k
`3.563501e-20`, interface-G
`1.972115e-18`, and top-convection-k
`2.424932e-29`.

Worst linear residual: `1.070846e-11`.
Worst energy-balance error:
`1.583118e-12`.

## Scope boundary

The 100 nm grid certifies differentiation, not physical mesh convergence.
The source is a named synthetic fixed-Q thermal control, not a promoted
optical artifact.  Optical-only, combined physical-rho, and full latent
AD–FD remain required before the complete gradient can be called validated.
