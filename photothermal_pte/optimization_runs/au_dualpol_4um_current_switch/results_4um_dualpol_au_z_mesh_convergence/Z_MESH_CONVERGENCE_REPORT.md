# 4 um Au/TaIrTe4 z-mesh convergence

Status: `STALE_HISTORICAL_PARTIAL_Z_NOT_CURRENT_CONTRACT`

Do not use the numerical tables below as evidence for the current code. They
were generated with the historical optical-rho^3/thermal-electrical-rho law,
the pre-correction Shockley-Ramo current sign, and a cache key bound only to
the density checkpoint. They are retained only as historical evidence. The
replacement diagnostic is hash-bound to the implementation, shared material
law, device contract, time contract, and checkpoint.

The exact current Au/FDTDX checkpoint had no prior z-mesh convergence certificate.
AD-FD on the baseline grid certifies differentiation of that discrete grid only.

The density, x/y mesh, source, material endpoints, and historical O3/TE1 gray law are frozen.
The gray law remains diagnostic and is not promoted as physical Au.

| factor | Au dz (nm) | TaIrTe4 dz (nm) | SiO2 dz (nm) | Yee cells |
|---:|---:|---:|---:|---:|
| 1 | 25.000 | 20.000 | 95.000 | 1383840 |
| 2 | 12.500 | 10.000 | 47.500 | 1729800 |
| 4 | 6.250 | 5.000 | 23.750 | 2421720 |
| 8 | 3.125 | 2.500 | 11.875 | 3805560 |

## Independent physics-gate failures

Overall physics gates pass: `False`. The following factor-8 cases exceed the
0.5% absorption-Q/closed-flux closure gate; the conservative remap and linear
solver residual gates pass.

| factor | density | pol | Q/flux closure |
|---:|---|---|---:|
| 8 | eta_0.35 | Ea | 1.7133% |
| 8 | eta_0.50_nominal | Eb | 0.8711% |
| 8 | eta_0.35 | Eb | 1.9569% |
| 8 | eta_0.65 | Eb | 0.8428% |

## Final refinement-pair comparison

| density | pol | dP_Q | Q NRMSE | T NRMSE | dTmax | dI | sign | pass |
|---|---|---:|---:|---:|---:|---:|:---:|:---:|
| eta_0.50_nominal | Ea | 2.6186% | 10.7994% | 2.7665% | 2.9807% | 2.1675% | True | False |
| eta_0.50_nominal | Eb | 1.5339% | 19.6947% | 2.0609% | 2.4610% | 5.8517% | True | False |
| eta_0.35 | Ea | 3.0140% | 10.7408% | 3.0785% | 2.0881% | 1.1256% | True | False |
| eta_0.35 | Eb | 2.4219% | 20.5949% | 2.7178% | 0.6477% | 20.8118% | True | False |
| eta_0.65 | Ea | 2.5948% | 10.7369% | 2.7603% | 2.9508% | 0.1162% | True | False |
| eta_0.65 | Eb | 1.7618% | 19.7581% | 2.2031% | 2.6816% | 7.5963% | True | False |

No Q clipping, smoothing, gain, polarization matching, or closure rescaling is used.
Each optical mesh is normalized only by its own all-air incident-power calibration.

This was a partial z sweep: it refined Au, TaIrTe4, and SiO2 only. The Si
substrate, air regions, and z-PML discretization were fixed, so this failure is
not a certificate for the full optical z domain. Time-window stationarity was
also not measured and must be separated from spatial error before extension.

Next gate: `DIAGNOSE_TIME_AND_ABSORPTION_CLOSURE_THEN_DEFINE_FULL_Z_SWEEP`.
