# Circular-design Pabs diagnostic report

## Scope and normalization

- Optical FDTD only; HEAT was not run.
- No geometric clipping of Q and no flux/Q gain were applied.
- All reported Python Q integrals use the exported solver grid and nested trapezoidal quadrature.

## Same-grid DeltaQ result

- `DeltaA_Q = 0.251391487761`
- `DeltaA_flux = 0.197084432245`
- `excess = 0.054307055516`
- Same grid verified: `True`

| radial region | Delta absorptance | fraction of DeltaA_Q |
|---|---:|---:|
| disk interior | 0.205804762562 | 81.866242% |
| disk edge band | 0.030931032837 | 12.303930% |
| exterior | 0.014655692361 | 5.829828% |

| z region | Delta absorptance | fraction of DeltaA_Q |
|---|---:|---:|
| TaIrTe4 bottom boundary | 0.024543775230 | 9.763169% |
| TaIrTe4 interior | 0.182027690369 | 72.408056% |
| TaIrTe4 top boundary | 0.044820022158 | 17.828775% |

DeltaQ is predominantly inside the disk footprint and in the TaIrTe4 interior. The edge band is secondary. The scalar flux difference has no unique voxelwise allocation, so these values localize DeltaQ rather than artificially assigning the complete scalar excess to voxels.

## Geometry matrix

| case | geometry | A_Q internal | A_Q Python | A_local_flux | A_Q - A_local_flux | signed relative mismatch |
|---|---|---:|---:|---:|---:|---:|
| A | no design | 0.256627959743 | 0.256627959743 | 0.256624925177 | 0.000003034566 | 0.001182% |
| B | full-cell uniform n=4 film, touching | 0.534846027743 | 0.534846027743 | 0.534837340196 | 0.000008687546 | 0.001624% |
| C | grid-aligned square n=4 block, touching | 0.472305352679 | 0.472305352679 | 0.363376055266 | 0.108929297413 | 29.977016% |
| D | circular n=4 disk, touching | 0.508019447504 | 0.508019447504 | 0.453709357422 | 0.054310090082 | 11.970238% |
| E | circular n=4 disk, 50 nm air gap | 0.490049830724 | 0.490049830724 | 0.391220073794 | 0.098829756929 | 25.261934% |

The no-design and full-cell uniform-film controls close at approximately 1e-5 relative error, whereas both finite lateral n=4 structures fail. The square fails more strongly than the circle, and a 50 nm gap does not remove the circle mismatch. This isolates the discrepancy to finite lateral high-index discontinuities/associated field interpolation rather than direct design–TaIrTe4 volume overlap.

## Independent mesh sweeps

| TaIrTe4 dz [nm] | A_Q | A_local_flux | signed mismatch |
|---:|---:|---:|---:|
| 10.000 | 0.508152084081 | 0.453930358590 | 11.944944% |
| 5.000 | 0.508019447504 | 0.453709357422 | 11.970238% |
| 2.500 | 0.507483258895 | 0.453674000528 | 11.860776% |

| design edge step [nm] | A_Q | A_local_flux | signed mismatch |
|---:|---:|---:|---:|
| 50.000 | 0.508448696906 | 0.454503389674 | 11.869066% |
| 25.000 | 0.508019447504 | 0.453709357422 | 11.970238% |
| 12.500 | 0.506614806292 | 0.451986153385 | 12.086355% |

## Exact disk/interface and material settings

- TaIrTe4 z: `-1e-07` to `0` m
- disk z: `0` to `6e-07` m
- signed gap: `0` m; volume overlap: `0` m
- design/TaIrTe4 mesh order: `2.0` / `2.0`
- conformal mesh refinement: `precise volume average`; mesh type: `auto non-uniform`
- Im(epsilon) at 4 um, TaIrTe4 axes: `50.8480861078`, `9.28919488762`, `0`
- Im(n) at 4 um, TaIrTe4 axes: `6.71257138197`, `1.12128175229`, `0`
- design n=4 imaginary part: `0.0`

The Lumerical internal Pabs integral and the Python exported-Q integral agree to floating-point precision in every newly generated case. Therefore the observed mismatch is not introduced by NPZ export or Python quadrature.
