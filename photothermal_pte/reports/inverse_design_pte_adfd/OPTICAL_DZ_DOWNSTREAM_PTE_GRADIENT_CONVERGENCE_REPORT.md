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
| 2.5nm | 4um | 1.692523974e-12 | 2.014e-04 | 4.626383564e-16 | -7.584484158e-16 | -7.482281575e-16 |
| 2.5nm | 6um | 1.692523974e-12 | 2.014e-04 | 1.256259220e-16 | -9.421847743e-17 | -9.348681889e-17 |
| 1.25nm | 4um | 1.692526176e-12 | 2.196e-04 | 4.625279224e-16 | -7.585748901e-16 | -7.483538693e-16 |
| 1.25nm | 6um | 1.692526176e-12 | 2.196e-04 | 1.255959323e-16 | -9.422926300e-17 | -9.349750560e-17 |
| 0.625nm | 4um | 1.692537364e-12 | 2.272e-04 | 4.625087035e-16 | -7.586013655e-16 | -7.483801344e-16 |
| 0.625nm | 6um | 1.692537364e-12 | 2.272e-04 | 1.255907193e-16 | -9.423168032e-17 | -9.349989878e-17 |

## Convergence

| dz comparison (nm) | scenario | remapped-Q NRMSE | TaIrTe4 T-field NRMSE | raw PTE relative change | optical gradient relative change | combined gradient relative change |
|---:|---:|---:|---:|---:|---:|---:|
| 2.5→1.25 | 4um | 2.512e-03 | 1.131e-04 | 2.387e-04 | 1.667e-04 | 1.680e-04 |
| 2.5→1.25 | 6um | 2.512e-03 | 1.176e-04 | 2.387e-04 | 1.145e-04 | 1.143e-04 |
| 1.25→0.625 | 4um | 5.087e-04 | 2.152e-05 | 4.155e-05 | 3.490e-05 | 3.510e-05 |
| 1.25→0.625 | 6um | 5.087e-04 | 2.216e-05 | 4.151e-05 | 2.565e-05 | 2.560e-05 |
| 2.5→0.625 | 4um | 3.021e-03 | 1.339e-04 | 2.802e-04 | 2.016e-04 | 2.031e-04 |
| 2.5→0.625 | 6um | 3.021e-03 | 1.391e-04 | 2.802e-04 | 1.401e-04 | 1.399e-04 |

The production optical mesh is therefore:

`flake_dz_nm = 2.5`

This is the coarsest mesh whose raw PTE, optical directional gradient, and
combined directional gradient are all within 0.5% of the 0.625 nm reference
for both named thermal footprints. No empirical normalization, gradient
rescaling, clipping, smoothing, gain, global Q rescaling, tiling, or Q-source
deletion was used.

Worst layout JVP/VJP dot error:
`9.569081290e-16`.

Raw FSP/NPZ/J artifacts remain outside Git and are SHA-256 pinned in the
manifest. This checkpoint does not run gray-law sensitivity, latent AD-FD, or
optimization.
