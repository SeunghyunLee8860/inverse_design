# 4 um Au/TaIrTe4 z-mesh convergence

Status: `BLOCKED_SHARED_LINEAR_FULL_DOMAIN_Z_CONVERGENCE`

The exact current Au/FDTDX checkpoint had no prior z-mesh convergence certificate.
AD-FD on the baseline grid certifies differentiation of that discrete grid only.

The density, x/y mesh, source, material endpoints, and shared-linear Au law are frozen.
Every physical z region and both z-PMLs are refined together.

| factor | Au dz (nm) | TaIrTe4 dz (nm) | SiO2 dz (nm) | Yee cells |
|---:|---:|---:|---:|---:|
| 1 | 25.000 | 20.000 | 95.000 | 1383840 |
| 2 | 12.500 | 10.000 | 47.500 | 2767680 |
| 4 | 6.250 | 5.000 | 23.750 | 5535360 |

## Independent physics-gate failures

Overall physics gates pass: `True`.  The table lists every case that failed before pairwise convergence was considered.

| factor | density | pol | Q/phasor | E stationarity | mapping | thermal balance | thermal residual | electrical residual |
|---:|---|---|---:|---:|---:|---:|---:|---:|

## Final refinement-pair comparison

| density | pol | dP_Q | Q NRMSE | T NRMSE | dTmax | dI | sign | pass |
|---|---|---:|---:|---:|---:|---:|:---:|:---:|
| eta_0.50_nominal | Ea | 3.1997% | 18.3332% | 3.5193% | 4.0605% | 3.3041% | True | False |
| eta_0.50_nominal | Eb | 0.8576% | 34.0715% | 2.7290% | 12.8248% | 17.4330% | True | False |
| eta_0.35 | Ea | 3.3142% | 18.3495% | 3.6339% | 4.6298% | 5.1845% | True | False |
| eta_0.35 | Eb | 1.8271% | 31.2092% | 3.3001% | 1.7450% | 34.7631% | True | False |
| eta_0.65 | Ea | 3.2117% | 18.3111% | 3.5032% | 4.0044% | 2.9783% | True | False |
| eta_0.65 | Eb | 1.2929% | 34.0510% | 3.4814% | 30.1502% | 37.6644% | True | False |

No Q clipping, smoothing, gain, polarization matching, or closure rescaling is used.
Each polarization on each optical mesh is normalized only by its own all-air incident-power calibration.
Every material, Si, air, and z-PML segment is refined. Each material run must also pass
previous-versus-late field stationarity and Q/TD/phasor closed-flux consistency before
the final spatial pair is considered.

Next gate: `RUN_OPTICAL_XY_CONVERGENCE`.
