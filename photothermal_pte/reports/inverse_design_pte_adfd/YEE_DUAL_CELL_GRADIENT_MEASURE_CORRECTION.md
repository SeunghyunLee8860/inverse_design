# Yee dual-cell gradient-measure correction

Status: `VALIDATED_FULL_YEE_DUAL_CELL_GRADIENT_MEASURE`

The failed Stage 10 result remains an immutable diagnostic. No empirical
normalization, fitted factor, or gradient rescaling was used.

## Root cause

`J_c = d epsilon_Yee,c / d rho` already includes the conformal material fill
and exact design-support intersection. The old combined path multiplied the
forward/adjoint product by a Yee volume clipped again to the nominal
`[-1,1] um` design box. That applied the support fraction twice. The corrected
bilinear form uses the complete component-specific Yee dual-cell volume.

| thermal scenario | FD step | old clipped combined error | corrected full-Yee combined error |
|---|---:|---:|---:|
| 4um | 0.01 | 2.779940e-02 | 3.046885e-06 |
| 4um | 0.005 | 2.781062e-02 | 8.492219e-06 |
| 6um | 0.01 | 2.824562e-02 | 2.511642e-06 |
| 6um | 0.005 | 2.826331e-02 | 1.569769e-05 |

The worst corrected optical/combined directional error is
`1.569769e-05`. The old
clipped computation is reproduced with relative error
`0.000e+00`, so the
change is isolated to the integration measure.

## Support audit

The explicit `J_x`, `J_y`, and `J_z` operators remain unchanged. Their active
row counts and the number of active rows incorrectly reduced by clipping are:

- x: `49200` active,
  `5520` changed.
- y: `49200` active,
  `5520` changed.
- z: `48749` active,
  `6161` changed.

This checkpoint validates the corrected measure for the existing smooth
direction at both 4 and 6 um thermal scenarios. It does not claim completion
of the strong/five-direction combined physical-density gate. Gray-law
sensitivity, latent AD-FD, and optimization remain blocked.
