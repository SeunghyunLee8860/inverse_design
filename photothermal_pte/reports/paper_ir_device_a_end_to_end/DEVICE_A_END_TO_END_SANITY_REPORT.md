# Device A single-position Maxwell-to-PTE sanity check

Status: `COMPLETED_DEVICE_A_SINGLE_POSITION_SANITY_DISAGREES_WITH_DIGITIZED_CURRENT_RATIO`

The numerical chain completed for the pre-registered Figure-3 edge position. It is **not** a successful paper reproduction: the polarization dependence disagrees with the digitized measurement.

| metal thermalization diagnostic | simulated `|Ia|/|Ib|` | digitized measurement | relative difference |
|---|---:|---:|---:|
| isolated-metal absorption | 1.617656 | 0.836590 ± 0.008526 | +93.36% |
| perfect-to-flake transfer | 1.638590 | 0.836590 ± 0.008526 | +95.87% |

Both are diagnostic extremes, not two published interface models. A finite Au/Ti-to-TaIrTe4 thermal contact was not invented.

For the requested Figure-3G-style off-axis-edge comparator, the perfect-to-flake diagnostic gives `|grad_a|` a/b ratios of 2.709797 (raw one-cell maximum) and 4.367463 (P99). The raw maximum is retained as a diagnostic; the P99/RMS/mean metrics are the more robust 100 nm-grid comparators.

Optical `E||a / E||b` closure is 0.2364% / 0.0671%; final auto-shutoff is 9.461010e-06 / 9.968510e-06. Both optical gates pass and neither Q artifact contains a negative voxel.

## Case results

| scenario | polarization | mapped power (W) | Tmax rise (K) | flake average rise (K) | signed current (A) | residual | balance error |
|---|---|---:|---:|---:|---:|---:|---:|
| isolated | E||a | 3.674195405e-05 | 2.303991345e-01 | 2.485318908e-02 | 8.072622430e-09 | 1.036e-10 | 7.145e-12 |
| isolated | E||b | 5.046513024e-05 | 1.444754661e-01 | 3.418416877e-02 | 4.990319469e-09 | 1.069e-10 | 3.397e-14 |
| perfect | E||a | 3.694596918e-05 | 2.304092119e-01 | 2.498657845e-02 | 8.196197440e-09 | 1.039e-10 | 1.038e-11 |
| perfect | E||b | 5.047214284e-05 | 1.444758660e-01 | 3.418853698e-02 | 5.001980470e-09 | 1.062e-10 | 1.901e-13 |


## What is and is not validated

- Optical closure and auto-shutoff passed independently for both polarizations.
- Conservative Q remap, thermal residual, energy balance, and digitized-contact weighting solve passed.
- Full volumetric Maxwell Q and the existing explicit 3D thermal operator were used; Q was not collapsed to a sheet.
- No polarization matching, clipping, gain, global rescaling, beam-position tuning, adjoint, AD-FD, or optimization was used.
- The far-x/y and bottom Dirichlet fluxes are numerical truncation-boundary fluxes, not intrinsic heat-path fractions.
- Absolute current and the polarization ratio remain model-dependent because exact CAD, beam waist, hidden contact overlap, and metal-interface thermal data were not published.
