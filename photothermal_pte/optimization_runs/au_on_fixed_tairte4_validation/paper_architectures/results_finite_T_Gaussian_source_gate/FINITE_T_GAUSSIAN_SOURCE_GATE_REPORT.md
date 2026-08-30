# Finite multi-T Gaussian source gate

Status: `VALIDATED_FINITE_T_GAUSSIAN_SOURCE`

The promoted physical target is `w0=12 um` at `lambda=11.825 um`. The Lumerical source-object input is `11.8575713844 um`; this is a numerical source calibration, not power or Q rescaling.

| case | source-object w0 (um) | fitted wx | fitted wy | fit NRMSE | ellipticity | strict pass |
|---|---:|---:|---:|---:|---:|---|
| uncalibrated target 4.0 um | 4.000000 | 4.90663 | 5.10088 | 2.0503% | 3.8821% | False |
| uncalibrated target 8.5 um | 8.500000 | 8.61229 | 8.68475 | 0.1569% | 0.8379% | False |
| 12 um first calibration | 11.916865 | 12.04741 | 12.07262 | 0.0961% | 0.2090% | False |
| 12 um second calibration (promoted) | 11.857571 | 11.98755 | 12.01319 | 0.0908% | 0.2137% | True |

The promoted case realizes `11.98755 x 12.01319 um`; all source-only gates pass. The 4 and 8.5 um failures remain as fail-closed diagnostics. No thermal, PTE, adjoint, or optimization solve was run.
