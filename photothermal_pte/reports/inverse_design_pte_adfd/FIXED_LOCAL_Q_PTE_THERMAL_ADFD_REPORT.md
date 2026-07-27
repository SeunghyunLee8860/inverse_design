# Fixed-local-Q PTE thermal-only AD–FD

Status: `VALIDATED_FIXED_LOCAL_Q_PTE_THERMAL_ONLY_ADFD`

## What this validates

The matched \(dz=2.5\) nm native Yee \(Q\) is remapped once per named
thermal footprint and then held bitwise identical in the baseline, plus, and
minus thermal solves.  Maxwell and the optical-\(Q\) derivative are absent.
The differentiated system is

\[
K_T(\rho)\theta=M_VQ_{fixed},\qquad
\frac{dI}{d\rho}=
-\lambda_T^T\frac{dK_T}{d\rho}\theta .
\]

The discrete gradient contains all three implemented thermal paths:

1. bulk design \(k(\rho)\);
2. internal TaIrTe4/design \(G(\rho)\);
3. the design exposed-surface half-cell conductivity contribution.

The objective is the established uniform-45-degree weighting-field PTE
surrogate.  It is not yet a finite-contact solved weighting potential or a
terminal experimental current.

This stage intentionally uses the native 20×20 cell-centered thermal density
control at \(\rho=0.5\).  It does not claim the approved 81×81 nodal mapping;
that mapping and its JVP/VJP are the next separate gate.

## Baselines

| scenario | cells | PTE objective (A) | Tmax ΔT (K) | gradient norm (A) | energy error | forward residual | adjoint residual |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TaIrTe4_4um_footprint | 438976 | 3.467322427e-16 | 1.069958441e-07 | 2.271629343e-18 | 3.197e-12 | 1.015e-11 | 9.868e-12 |
| TaIrTe4_6um_footprint | 671536 | 9.511775517e-17 | 1.051871964e-07 | 4.297897222e-19 | 3.465e-12 | 1.006e-11 | 9.565e-12 |

## Centered AD–FD at selected step \(h=0.005\)

| scenario | direction | signal ratio | gated | adjoint (A) | FD (A) | relative error | bulk-k | interface-G | surface-k |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TaIrTe4_4um_footprint | adjoint_aligned | 6.764e-01 | True | 2.690834163e-17 | 2.690831501e-17 | 9.891022e-07 | -4.504289184e-19 | 2.735877055e-17 | 3.403005786e-28 |
| TaIrTe4_4um_footprint | seeded_random | 1.199e-02 | True | -4.769602888e-19 | -4.769596954e-19 | 1.244212e-06 | 1.921094959e-20 | -4.961712384e-19 | -5.494852606e-30 |
| TaIrTe4_4um_footprint | asymmetric_smooth | 5.122e-01 | True | -2.037690932e-17 | -2.037689932e-17 | 4.908791e-07 | 2.490454395e-19 | -2.062595476e-17 | -2.167267142e-28 |
| TaIrTe4_6um_footprint | adjoint_aligned | 6.505e-01 | True | 4.824496148e-18 | 4.824489363e-18 | 1.406291e-06 | -1.516929845e-19 | 4.976189132e-18 | 1.156663835e-28 |
| TaIrTe4_6um_footprint | seeded_random | 1.537e-02 | True | -1.139979057e-19 | -1.139976703e-19 | 2.065365e-06 | 7.188019953e-21 | -1.211859257e-19 | -2.323875857e-30 |
| TaIrTe4_6um_footprint | asymmetric_smooth | 5.815e-01 | True | -4.313109107e-18 | -4.313104833e-18 | 9.908173e-07 | 1.204326616e-19 | -4.433541768e-18 | -9.260175567e-29 |

The adjoint-aligned direction is always gated.  A fixed independent direction
is gated when its directional signal is at least
`1.000e-05` of the gradient L1 norm; lower-signal
directions remain published diagnostics rather than being divided by a
near-null slope.

## Gates

- Worst selected, conditioned AD–FD error:
  `2.065365e-06`
  (limit `5.000000e-03`).
- Worst energy-balance error:
  `3.467276e-12`
  (limit `1.000000e-02`).
- Worst forward/adjoint linear residual:
  `1.017511e-11`
  (limit `1.000000e-08`).
- Worst sum-of-gradient-components error:
  `4.037925e-16`.

No Maxwell solve, 81×81 mapping, transient calculation, or optimization is
claimed here.
