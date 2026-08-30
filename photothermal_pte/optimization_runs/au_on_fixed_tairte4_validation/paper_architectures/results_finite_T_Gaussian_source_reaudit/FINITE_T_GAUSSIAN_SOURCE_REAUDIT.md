# Finite-T Gaussian source re-audit

Status: `BLOCKED_FINITE_T_GAUSSIAN_SOURCE_DISTORTION`

The primary comparator is the downward transverse incident wave, not total |E|^2 including longitudinal Ez.

| requested w0 (um) | fitted wx | fitted wy | NRMSE | ellipticity | pass |
|---:|---:|---:|---:|---:|---|
| 4.0 | 4.9066 | 5.1009 | 2.0503% | 3.8821% | False |
| 8.5 | 8.6123 | 8.6848 | 0.1569% | 0.8379% | False |

No full finite-array Q, thermal, PTE, adjoint, or optimization solve was run.
