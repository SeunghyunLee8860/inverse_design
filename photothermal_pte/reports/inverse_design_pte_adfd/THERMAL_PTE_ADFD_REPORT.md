# Fixed-K thermal/PTE AD–FD certificate

**Status: `VALIDATED_FIXED_K_THERMAL_PTE_ADFD`**

This certificate validates the discrete
`Q -> K_T^-1 -> temperature -> local PTE functional` chain. It is **not** a
final device prediction. The top optical design material has no thermal
material law in the repository, and the finite readout mask is a numerical
surrogate rather than an electrode/weighting-potential model.

## Frozen numerical contract

- Lateral cell: 6 um x 6 um, periodic x/y.
- Layers: 2 um Si, 285 nm SiO2, 100 nm TaIrTe4.
- Bottom: fixed DeltaT=0; top: adiabatic.
- TaIrTe4 kappa: diag(14.4, 3.8, 1.0) W/(m K).
- SiO2 kappa: 1.38 W/(m K); Si kappa: 145 W/(m K).
- Named interface scenario: G_top=7.37e6 and G_bottom=1.1e9 W/(m2 K).
- Synthetic positive asymmetric source power: 9.9999999999999998e-13 W.
- Grid: [24, 24, 8] cells; dx=dy=250 nm; nonuniform z.
- PTE functional unit: A m. Cell volume is applied exactly once.

## Discrete adjoint

`K_T theta = M_V Q + b`,
`K_T^T lambda = c_T`, and
`dF/dQ = M_V^T lambda`.

The adjoint source is the literal transpose of the same sparse local-PTE
functional used forward:
`c_T = -(sigma_a S_a D_x^T + sigma_b S_b D_y^T) V/sqrt(2)`.

## Gates

| Gate | Value | Limit | Pass |
|---|---:|---:|---|
| Linear residual | 8.315497e-12 | 1e-8 | True |
| Energy balance | 2.768712e-13 | 1% | True |
| Matrix asymmetry | 0.000000e+00 | 1e-13 | True |
| Minimum eigenvalue | 7.044210e-08 W/K | >0 | True |
| Temperature AD-FD, worst best-step | 5.352386e-16 | 1e-6 | True |
| Q AD-FD, worst best-step | 1.637661e-12 | 1e-6 | True |
| Volume transpose identity | 2.114507e-16 | 1e-13 | True |

## Physical blockers retained

- `BLOCKED_FULL_RHO_DEPENDENT_THERMAL_MATERIAL_MODEL`: no kappa(rho),
  interface-G(rho), or thermal topology interpolation is specified.
- `BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`: the optical
  model is periodic, while a nonzero device PTE current needs finite contacts,
  a finite flake readout region, or a solved weighting potential.
- Therefore the omitted term
  `-lambda^T (dK_T/drho) theta` is zero only in this fixed-K certificate, not
  in a future full multiphysics topology derivative.
