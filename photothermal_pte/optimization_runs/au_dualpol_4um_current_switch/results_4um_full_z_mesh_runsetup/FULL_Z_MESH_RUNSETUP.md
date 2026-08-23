# Full-domain optical z-mesh runsetup

Status: `AUDITED_SHARED_LINEAR_FULL_Z_MESH_VARIANTS_NOT_SOLVED`

The historical sweep refined only SiO2, TaIrTe4, and Au. These new
variants refine every z segment, including resolved Si, air, and both
z-PML regions, while the x/y grid and lateral PML remain fixed.
Factor 1 is bitwise identical to the current baseline edge arrays.

| factor | grid shape | Yee cells | z-PML cells/face | Si dz (nm) | SiO2 dz (nm) | TaIrTe4 dz (nm) | Au dz (nm) | near-air dz (nm) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 186x186x40 | 1383840 | 8 | 203.000 | 95.000 | 20.000 | 25.000 | 50.000 |
| 2 | 186x186x80 | 2767680 | 16 | 101.500 | 47.500 | 10.000 | 12.500 | 25.000 |
| 4 | 186x186x160 | 5535360 | 32 | 50.750 | 23.750 | 5.000 | 6.250 | 12.500 |

This file is not a convergence certificate. Every variant needs its
own source calibration, time/closure gates, and downstream comparison.
The physical-device contract must also be confirmed first.
