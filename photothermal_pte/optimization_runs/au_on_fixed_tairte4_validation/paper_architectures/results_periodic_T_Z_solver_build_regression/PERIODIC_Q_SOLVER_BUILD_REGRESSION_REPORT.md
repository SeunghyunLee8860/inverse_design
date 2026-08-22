# Periodic optical-Q solver-build regression

Status: `VALIDATED_PERIODIC_Q_REQUIRES_V261_R1P2_BUILD_8P35P4522`

The same `v261` directory name referred to two different solver builds. The older R1 build fails the flux/local-loss identity for both compact T and centered Z, whereas R1.2 passes. This is not a 10-versus-25-nm mesh effect: the two R1 T fluxes agree within 0.5% and both fail closure.

| case | solver | mesh (nm) | A_flux | A_Q | closure | status |
|---|---:|---:|---:|---:|---:|---|
| T_R1p2_10nm | 8.35.4522 | 10 | 0.195081 | 0.194752 | 0.169% | COMPLETED_T2024_TAIRTE4_OPTICAL_SMOKE |
| T_R1_10nm | 8.35.4413 | 10 | 0.010055 | 0.227466 | 95.579% | FAILED_T2024_TAIRTE4_OPTICAL_SMOKE_GATE |
| T_R1_25nm | 8.35.4413 | 25 | 0.010047 | 0.225017 | 95.535% | FAILED_T2024_TAIRTE4_OPTICAL_SMOKE_GATE |
| Z_R1_25nm | 8.35.4413 | 25 | 0.676072 | 4.307517 | 84.305% | FAILED_Z2022_M2_SELECTED_Q_GATE |
| Z_R1p2_25nm | 8.35.4522 | 25 | 0.747699 | 0.746514 | 0.159% | COMPLETED_Z2022_M2_CENTERED_EXPANDED_SELECTED_Q |

Production rule: use and record `8.35.4522` (2026 R1.2) for all periodic local-Q certificates. Raw FSP/NPZ files remain outside Git.

No thermal, weighting-field, PTE, adjoint, or optimization calculation is included in this audit.
