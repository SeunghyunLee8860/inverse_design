# Device A single-position Maxwell-to-PTE sanity check

Status: `COMPLETED_DEVICE_A_SINGLE_POSITION_SANITY_DISAGREES_WITH_DIGITIZED_CURRENT_RATIO`

This run uses the validated explicit-assumption scalar-Gaussian scenario
`target w0=8.75 um` (`source-object w0=8.610602974768 um`; realized
target-plane fit `8.739129/8.771047 um`). It is not a paper-certified beam
measurement, and the digitized Device-A geometry is not unpublished CAD.

The numerical chain completed for the pre-registered Figure-3 edge position. It is **not** a successful paper reproduction: the polarization dependence disagrees with the digitized measurement.

| metal thermalization diagnostic | simulated `|Ia|/|Ib|` | digitized measurement | relative difference |
|---|---:|---:|---:|
| isolated-metal absorption | 1.391604 | 0.836590 ± 0.008526 | +66.34% |
| perfect-to-flake transfer | 1.399008 | 0.836590 ± 0.008526 | +67.23% |

Both are diagnostic extremes, not two published interface models. A finite Au/Ti-to-TaIrTe4 thermal contact was not invented.

For the requested Figure-3G-style off-axis-edge comparator, the perfect-to-flake diagnostic gives `|grad_a|` a/b ratios of 2.367556 (raw one-cell maximum) and 3.673447 (P99). The raw maximum is retained as a diagnostic; the P99/RMS/mean metrics are the more robust 100 nm-grid comparators.

Optical `E||a / E||b` closure is 0.0194% / 0.0990%; final auto-shutoff is 9.927560e-06 / 9.925560e-06. Both optical gates pass and neither Q artifact contains a negative voxel.

## Case results

| scenario | polarization | mapped power (W) | Tmax rise (K) | flake average rise (K) | signed current (A) | residual | balance error |
|---|---|---:|---:|---:|---:|---:|---:|
| isolated | E||a | 4.214301605e-05 | 3.505916233e-01 | 2.851424421e-02 | 1.269312917e-08 | 1.026e-10 | 1.032e-11 |
| isolated | E||b | 5.833249583e-05 | 2.585673373e-01 | 3.949096415e-02 | 9.121224985e-09 | 1.048e-10 | 4.375e-12 |
| perfect | E||a | 4.218824235e-05 | 3.505939793e-01 | 2.854377398e-02 | 1.277167956e-08 | 1.037e-10 | 1.198e-11 |
| perfect | E||b | 5.833406851e-05 | 2.585674322e-01 | 3.949195472e-02 | 9.129096731e-09 | 1.045e-10 | 4.529e-12 |


## What is and is not validated

- Optical closure and auto-shutoff passed independently for both polarizations.
- Conservative Q remap, thermal residual, energy balance, and digitized-contact weighting solve passed.
- Full volumetric Maxwell Q and the existing explicit 3D thermal operator were used; Q was not collapsed to a sheet.
- No polarization matching, clipping, gain, global rescaling, beam-position tuning, adjoint, AD-FD, or optimization was used.
- The far-x/y and bottom Dirichlet fluxes are numerical truncation-boundary fluxes, not intrinsic heat-path fractions.
- Absolute current and the polarization ratio remain model-dependent because exact CAD, beam waist, hidden contact overlap, and metal-interface thermal data were not published.
