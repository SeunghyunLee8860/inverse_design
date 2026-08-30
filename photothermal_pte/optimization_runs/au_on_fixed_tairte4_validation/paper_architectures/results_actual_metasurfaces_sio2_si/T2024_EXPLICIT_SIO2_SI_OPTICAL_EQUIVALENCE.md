# Explicit SiO2/Si substrate optical equivalence

Status: `VALIDATED_T2024_EXPLICIT_SIO2_SI_OPTICAL_EQUIVALENCE`

The physical periodic stack is air / Au inverse-T / 100-nm TaIrTe4 / 35-nm Al2O3 / 200-nm Au mirror / 1.5-um thermally grown SiO2 / intrinsic Si. The older Au-to-bottom-PML artifacts are retained only as a numerical optical control.

| case | total P_Q change | TaIrTe4 Q change | max lateral shape NRMSE | bottom transmission |
|---|---:|---:|---:|---:|
| T_Eb | -0.000000% | -0.000000% | 0.000260% | 4.043e-10 |
| T_Ea | -0.000002% | -0.000002% | 0.000324% | 6.536e-11 |
| bare_Eb | 0.000018% | 0.000009% | 0.000081% | 7.511e-10 |
| bare_Ea | 0.000003% | 0.000004% | 0.000052% | 9.099e-11 |

All explicit-substrate forward, closure, auto-shutoff, finite-Q and nonnegative-Q gates passed. No clipping, smoothing, gain, global rescaling, or polarization matching was used. This validates only the optical insensitivity below the opaque Au mirror; the thermal model still retains explicit SiO2 and Si.
