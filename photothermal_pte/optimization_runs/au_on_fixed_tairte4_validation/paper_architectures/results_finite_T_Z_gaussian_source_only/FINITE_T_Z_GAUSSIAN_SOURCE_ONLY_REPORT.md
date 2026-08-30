# Finite T/Z Gaussian source-only report

Status: `VALIDATED_FINITE_T_Z_GAUSSIAN_SOURCE_ONLY`

This certificate contains no material, Q, thermal, electrical, PTE, adjoint, or optimization result.

| case | wx (um) | wy (um) | fit NRMSE | ellipticity | power closure | shutoff | runtime | GPU memory |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T | 3.990275 | 4.011085 | 0.074595% | 0.520184% | 0.229622% | 7.83046e-07 | 24.92 s | 0.445 GiB |
| Z | 3.984605 | 4.018039 | 0.081371% | 0.835560% | 0.332457% | 8.80117e-07 | 31.02 s | 0.584 GiB |

The first uncalibrated attempts are retained as fail-closed diagnostics in the CSV. Source-object waist calibration is not Q or power rescaling.

The sub-1% target ellipticity is retained. It is not removed by averaging or symmetrization.
