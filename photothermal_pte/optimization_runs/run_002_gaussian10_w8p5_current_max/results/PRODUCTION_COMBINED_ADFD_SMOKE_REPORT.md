# Production combined physical-rho PTE AD-FD smoke

Status: `VALIDATED_PRODUCTION_COMBINED_PHYSICAL_RHO_ADFD_SMOKE`

This checkpoint validates one nonuniform 201×201 physical-density baseline,
one adjoint-aligned direction, and centered finite difference at
`h=0.005`. It is a smoke certificate, not yet a multi-direction,
gray-law, latent/filter/projection, or optimization certificate.

## Result

| quantity | value | gate |
|---|---:|---:|
| adjoint directional derivative | 8.502570281382e-20 A | — |
| centered-FD directional derivative | 8.548619467411e-20 A | — |
| combined AD–FD relative error | 0.538674% | <1% |
| component-J transpose error | 6.372888e-16 | <1e-12 |
| Q-remap transpose error | 4.861811e-16 | <1e-12 |
| worst optical closure | 4.474819e-05 | <0.5% |
| worst Q mapping error | 0.000000e+00 | <0.5% |
| worst thermal residual | 8.612592e-11 | <1e-8 |
| worst thermal energy balance | 2.823629e-13 | <1% |

The base objective is `1.401741337812e-20 A` for incident power
`1.382226110302e-13 W`. Optical and thermal-material gradient
norms are `1.205023284459e-21 A` and
`1.086132454830e-22 A`, respectively. No
empirical normalization, gradient rescaling, Q clipping, smoothing, gain, or
rescaling was used.

## Mesh-parity fix

The first attempts were fail-closed before any FD pair because changing the
FieldRegion from monitor to source mode regenerated the auto-nonuniform mesh.
Deleting or disabling the Gaussian source produced maximum coordinate
mismatches of `87.497935 nm`. The successful contract keeps the
Gaussian source enabled with amplitude exactly zero during the adjoint. This
retains its mesh anchors but injects no forward illumination. The resulting
maximum mismatch is `6.776264e-12 nm`, with zero coordinates exceeding
`2e-18 m`. Forward and adjoint field arrays then have zero reported coordinate
mismatch.

The preserved v1–v5 failures are diagnostics and were not relabeled. They
contain no completed plus/minus FD sweep and no optimization iteration.

## Solver scope

- Maxwell: three forward solves total (the nonuniform base was reused) and
  one GPU adjoint solve; no CPU FDTD fallback.
- Thermal: three CUDA float64 forward solves and one CUDA float64 adjoint;
  no CPU linear-solve fallback.
- Optimization iterations: 0.

Before optimization, Run 002 still requires broader directional/step evidence,
gray-law sensitivity, the full latent/filter/projection pullback, and a
production design-window decision.
