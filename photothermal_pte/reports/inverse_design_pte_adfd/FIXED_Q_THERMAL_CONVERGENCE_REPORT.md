# Fixed-local-Q thermal domain/depth/mesh convergence

Status: `VALIDATED_FIXED_Q_THERMAL_DOMAIN_DEPTH_MESH_CONVERGENCE`

## Scope and interpretation

This certificate reuses only the immutable native Yee-grid absorption arrays
from the matched \(dz=2.5\) nm optical run.  Every thermal target grid is
remapped independently.  It does not reuse a previously mapped source and
does not run Maxwell, an adjoint, finite differences, a transient solve, or
optimization.

The 4 µm and 6 µm TaIrTe4 footprints are separate named numerical scenarios.
Neither is promoted as fabrication truth or as a final experimental
prediction.  Lateral and bottom Dirichlet power entries are numerical
truncation-boundary fluxes, not intrinsic physical heat-path fractions.

Native optical power is `1.6887880194040323e-12 W`.  There is no
clipping, smoothing, gain, global rescaling, tiling, or source deletion.

## Thermal physical model held fixed

- TaIrTe4:
  \(\boldsymbol{\kappa}=\operatorname{diag}(14.4,3.8,1.0)\)
  W/(m K).
- Bulk SiO2 / Si / air:
  `1.38 / 145 / 0.026 W/(m K)`.
- TaIrTe4/bottom-SiO2:
  `G=7.37e6 W/(m2 K)`; SiO2/Si:
  `G=1.1e9 W/(m2 K)`.
- TaIrTe4/air: `G=1 W/(m2 K)`.
- Deposited design-SiO2 endpoint:
  `G=7.37e4 W/(m2 K)`.
- Exposed SiO2/air: `h=10 W/(m2 K)`.
- This checkpoint holds \(\rho=0.5\), with
  \(k(\rho)=k_{air}+\rho(k_{SiO2}-k_{air})\) and
  \(G(\rho)=G_{air}+\rho(G_{deposited\ SiO2}-G_{air})\).
  Those gray laws are numerical relaxations, not measured gray-composite
  properties.  Their sensitivity is a later, separate gate.

## Independent controls

- Native: 32 µm lateral, 20 µm Si depth, 100/25/100 nm
  core-xy/flake-z/design-z grid.
- Lateral: only the lateral domain is enlarged to 40 µm.
- Depth: only Si depth is enlarged to 30 µm.
- Refined: the domain/depth remain 32/20 µm and the grid becomes
  50/12.5/50 nm.
- The complete TaIrTe4 field is compared on a fixed 100 nm by 100 nm by
  25 nm common probe grid using trilinear cell-center interpolation.

## Results

| scenario | case | cells | Tmax ΔT (K) | TaIrTe4 average ΔT (K) | PTE objective (A) | field NRMSE | worst gated difference |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TaIrTe4_4um_footprint | native | 438976 | 1.069958441e-07 | 4.086232470e-08 | 3.467322427e-16 | baseline | baseline |
| TaIrTe4_4um_footprint | lateral_40um | 536256 | 1.070110685e-07 | 4.087759119e-08 | 3.467340882e-16 | 0.030551% | 0.037347% |
| TaIrTe4_4um_footprint | si_depth_30um | 554496 | 1.069981802e-07 | 4.086464181e-08 | 3.467322744e-16 | 0.004638% | 0.005670% |
| TaIrTe4_4um_footprint | refined | 1238400 | 1.072998621e-07 | 4.076778839e-08 | 3.451468035e-16 | 0.200922% | 0.283335% |
| TaIrTe4_6um_footprint | native | 671536 | 1.051871964e-07 | 1.873295475e-08 | 9.511775517e-17 | baseline | baseline |
| TaIrTe4_6um_footprint | lateral_40um | 790704 | 1.052024371e-07 | 1.874831358e-08 | 9.512347042e-17 | 0.048231% | 0.081921% |
| TaIrTe4_6um_footprint | si_depth_30um | 848256 | 1.051895249e-07 | 1.873523043e-08 | 9.511784848e-17 | 0.007149% | 0.012146% |
| TaIrTe4_6um_footprint | refined | 2146904 | 1.055190380e-07 | 1.870106367e-08 | 9.451895460e-17 | 0.139010% | 0.314485% |

The raw relative PTE change is retained in the JSON/CSV.  Because a uniform
45-degree weighting field can strongly cancel a nearly symmetric
temperature field, the convergence gate uses
\(|I-I_0|/\max(\sum |w_i T_i|)\), not a potentially ill-conditioned division
by a near-zero signed current.

## Gates

- Worst Q mapping power error: `3.587455e-16`
  (limit `5.000000e-03`).
- Worst energy-balance error:
  `5.636849e-12` (limit
  `1.000000e-02`).
- Worst linear residual: `1.953723e-11` (limit
  `1.000000e-08`).
- Worst temperature/PTE convergence metric:
  `3.144850e-03` (limit
  `1.000000e-02`).

The next gate is fixed-local-Q PTE thermal-only AD–FD.  This report does not
claim that gate has run.
