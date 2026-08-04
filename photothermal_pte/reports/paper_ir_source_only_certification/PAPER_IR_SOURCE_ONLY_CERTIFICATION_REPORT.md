# Paper-IR source-only beam certification

Status: `VALIDATED_PAPER_LIKE_SCALAR_GAUSSIAN_SOURCE_ONLY`

This is a **paper-like scalar-Gaussian scenario with an explicitly assumed
waist**.  It is not an experimentally reproduced or paper-certified beam.
The physical target-plane 1/e² radius remains 12 µm.  The Lumerical
source-object input is 11.916864890 µm,
obtained from the SHA-pinned uncalibrated field as a numerical source
calibration.  It does not rescale incident power or Q.

## Final GPU source-only certificate

- v261 internal version: `8.35.4522`
- wavelength/source/domain: 11 / 50 / 60 µm
- six boundaries: PML; periodic/Bloch: none
- source-object input w0: 11.916864890 µm
- realized target w0 x/y: 11.996332674 /
  12.005906748 µm
- target error x/y: 0.030561% /
  0.049223%
- linear-intensity Gaussian-fit NRMSE:
  0.075808%
- x/y ellipticity: 0.079777%
- target incident power: 2.902892912754782e-13 W
- incident-power closure: 0.052937%
- source boundary max/mean: 1.74501027e-04 /
  5.22539338e-05
- auto-shutoff: 4.70241000e-06
- solver grid: [237, 237, 134]
- post-run native mesh: [190, 190, 87]
- precise GPU memory: 0.277 GiB
- API wall time: 5.618 s
- GPU engine: yes; CPU FDTD fallback: no

All mandatory gates pass.  The final raw NPZ SHA-256 is
`61b01be39fa588e01658297f3e0dc87de4c4db5a48d9cb218a92d809f3856ff0`.  Raw NPZ/FSP files remain outside Git.

## Controls and corrections

The original `BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE` interpretation was a
sandbox-network false negative.  Host execution checked out v261
successfully.  A later `lum_fdtd_solve` task shortage was transient license
contention, not missing entitlement; the final GPU solve used three host
orchestration threads without changing the GPU engine or numerical model.

The uncalibrated accuracy-5/6 controls realized approximately 12.08 µm, so
mesh refinement did not remove the small waist offset.  A positive
`distance from waist` diagnostic worsened the error above 3%; it is not
promoted.  The negative-distance calibrated case is the sole promoted
source-only contract.

## Decision

The source-only gate authorizes the ordered successor cases: planar
TaIrTe4 a/b, followed by straight-45-degree finite-edge a/b.  None of those
material cases, nor thermal/PTE/adjoint/optimization, was executed by this
certificate.
