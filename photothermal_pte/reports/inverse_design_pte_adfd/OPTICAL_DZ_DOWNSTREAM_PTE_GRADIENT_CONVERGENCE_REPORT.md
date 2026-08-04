# Optical dz downstream PTE/gradient convergence

Status: `VALIDATED_OPTICAL_DZ_DOWNSTREAM_PTE_GRADIENT_CONVERGENCE`

The nonuniform 81×81 physical-density forward source was solved at
`dz = 2.5, 1.25, 0.625 nm`. Each source was conservatively remapped into the
same explicit 4 µm and 6 µm thermal scenarios. The spatially weighted Maxwell
adjoint used

`dI_PTE/dQ_thermal -> R_Q^T -> native PABS Yee vector source`.

The optical density gradient used explicit component operators
`J_c = d epsilon_Yee,c / d rho_81x81`. Forward field, adjoint field,
`epsilon_c`, and clipped `dV_c` were paired on the same component-specific
PABS coordinates. The removed DESIGN-monitor same-index path was not used.

## Per-mesh values

| optical dz | thermal scenario | P_Q (W) | six-face closure | PTE (A) | optical directional gradient (A) | combined directional gradient (A) |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5nm | 4um | 1.692523974e-12 | 2.014e-04 | 4.626383564e-16 | -7.798458252e-16 | -7.696255670e-16 |
| 2.5nm | 6um | 1.692523974e-12 | 2.014e-04 | 1.256259220e-16 | -9.693606489e-17 | -9.620440635e-17 |
| 1.25nm | 4um | 1.692526176e-12 | 2.196e-04 | 4.625279224e-16 | -7.798085388e-16 | -7.695875180e-16 |
| 1.25nm | 6um | 1.692526176e-12 | 2.196e-04 | 1.255959323e-16 | -9.692480255e-17 | -9.619304516e-17 |
| 0.625nm | 4um | 1.692537364e-12 | 2.272e-04 | 4.625087035e-16 | -7.798023386e-16 | -7.695811074e-16 |
| 0.625nm | 6um | 1.692537364e-12 | 2.272e-04 | 1.255907193e-16 | -9.692282681e-17 | -9.619104527e-17 |

## Convergence

| dz comparison (nm) | scenario | remapped-Q NRMSE | TaIrTe4 T-field NRMSE | raw PTE relative change | optical gradient relative change | combined gradient relative change |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5→1.25 | 4um | 2.512e-03 | 1.131e-04 | 2.387e-04 | 4.781e-05 | 4.944e-05 |
| 2.5→1.25 | 6um | 2.512e-03 | 1.176e-04 | 2.387e-04 | 1.162e-04 | 1.181e-04 |
| 1.25→0.625 | 4um | 5.087e-04 | 2.152e-05 | 4.155e-05 | 7.951e-06 | 8.330e-06 |
| 1.25→0.625 | 6um | 5.087e-04 | 2.216e-05 | 4.151e-05 | 2.038e-05 | 2.079e-05 |
| 2.5→0.625 | 4um | 3.021e-03 | 1.339e-04 | 2.802e-04 | 5.576e-05 | 5.777e-05 |
| 2.5→0.625 | 6um | 3.021e-03 | 1.391e-04 | 2.802e-04 | 1.366e-04 | 1.389e-04 |

The production optical mesh is therefore:

`flake_dz_nm = 2.5`

This is the coarsest mesh whose raw PTE, optical directional gradient, and
combined directional gradient are all within 0.5% of the 0.625 nm reference
for both named thermal footprints. No empirical normalization, gradient
rescaling, clipping, smoothing, gain, global Q rescaling, tiling, or Q-source
deletion was used.

Worst layout JVP/VJP dot error:
`6.263608780e-16`.

Raw FSP/NPZ/J artifacts remain outside Git and are SHA-256 pinned in the
manifest. This checkpoint does not run gray-law sensitivity, latent AD-FD, or
optimization.
