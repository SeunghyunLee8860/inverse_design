# Run 010 thermal and electrical AD-FD checkpoint

Status: `VALIDATED_TAIRTE4_FLAKE_THERMAL_AND_ELECTRICAL_SUBGRADIENTS`

This checkpoint validates the fixed-Q thermal branch and the
density-dependent electrical/weighting branch separately. It is not the
combined Maxwell/thermal/electrical gradient and it does not authorize an
optimization yet.

## Explicit thermal contract

- Lumerical coordinates: `x=b`, `y=a`, `z=c`
- TaIrTe4 kappa: `(3.8, 14.4, 1.0) W/(m K)`
- explicit air, 285 nm SiO2, and Si volumes
- TaIrTe4/SiO2 G: `7.37e6 W/(m2 K)` only on TaIrTe4 contact area
- SiO2/Si G: `1.1e9 W/(m2 K)`
- top of explicit air box: `h=10 W/(m2 K)` Robin boundary
- far x/y and bottom Si boundary: fixed zero temperature rise
- gray bottom contact: parallel area-fraction relaxation of the
  TaIrTe4/SiO2 and air/SiO2 paths
- Run 009's upper-SiO2 `G(rho)` law is not used
- the paper-reduced TaIrTe4/air `G=1` bath boundary is not added on top of
  explicit air, avoiding double counting

The thermal grid has 2,334,948 unknowns and 16,168,012 matrix nonzeros.

## Q attribution

Native component Q is deposited by literal optical-dual-cell/material-volume
intersection. No full cut cell is forced into TaIrTe4 and no nearest-cell
relocation, clipping, smoothing, gain, or rescaling is used.

- native Q: `3.25424260645e-14 W`
- material-attributed/thermal Q: `3.16155277531e-14 W`
- attributed fraction: `97.1517%`
- mapping conservation error relative to attributed power: `0`

The remaining 2.8483% is deliberately not reassigned to TaIrTe4.

## Fixed-Q thermal AD-FD

At the finest step `h=0.0025`:

| direction | relative error |
|---|---:|
| smooth asymmetric | 6.49787e-6 |
| central localized | 5.46781e-6 |
| design-edge localized | 5.65374e-6 |
| fixed-seed random | 3.78370e-6 |

Worst residual was `9.89809e-11`; worst energy-balance error was
`1.51467e-12`. After CUDA warm-up a thermal forward required approximately
1.0 s on GPU 5.

## Density-dependent electrical/weighting AD-FD

The 241x241 triangular-sheet model uses top potential 1, bottom potential 0,
and solves the anisotropic weighting field at every density. The direct
thermoelectric-density term and the implicit weighting-potential term are
both included.

The directional relative errors for `h=0.01, 0.005, 0.0025` were
`6.72438e-5, 1.63973e-5, 3.79066e-6`, decreasing monotonically. Weighting and
electrical-adjoint residuals were below `8.15e-11`.

The earlier diagnostic JSON is immutable. It was labeled failed because it
incorrectly required every coarse-step truncation error to be below `1e-5`.
The corrected certificate requires monotonic refinement and the finest-step
error below `1e-5`.

## Next gate

Build and validate the component-specific density-to-Yee material Jacobian,
then run a spatially weighted Maxwell adjoint and combined physical-density
AD-FD. Gray-law continuation and optimization remain blocked until that gate
passes.
