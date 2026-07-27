# 81×81 nodal physical-density optical/thermal coupling

Status: `VALIDATED_81X81_NODAL_OPTICAL_THERMAL_MAPPING_JVP_VJP`

## Coordinate contract

The physical design variable is exactly 81×81 **nodes** on
\([-1,1]\) µm × \([-1,1]\) µm at 25 nm spacing.  It is not 81
finite-width pixels and has no periodic fencepost or wrap.

The optical map is identity on those x-y nodes and exact repetition on 13
z nodes from 0 to 600 nm at 50 nm spacing.  Its VJP is the literal sum over
the same z copies.

The thermal map is the exact area average of the nonperiodic
piecewise-bilinear nodal interpolant over each Cartesian control volume.
The transpose is the literal sparse-matrix transpose.  Target and source
bounds must match exactly, so cropping, padding, gain, or tiling fail closed.

## Endpoint, affine, conservation, and non-wrap controls

Optical rho=0 / rho=1 / z-extrusion maximum errors:
`0.000e+00 /
0.000e+00 /
0.000e+00`.

| thermal cell (nm) | shape | rho=1 error | affine-average error | area-integral error | opposite-corner leakage |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100 | 20×20 | 2.665e-15 | 1.221e-15 | 2.019e-16 | 0.000e+00 |
| 50 | 40×40 | 5.995e-15 | 3.109e-15 | 0.000e+00 | 0.000e+00 |

## JVP, centered FD, and VJP

| target | resolution (nm) | direction | JVP–FD relative error | JVP–VJP dot error |
| --- | ---: | --- | ---: | ---: |
| optical_81x81x13_nodes | 25 | smooth | 4.368814e-12 | 2.935354e-16 |
| optical_81x81x13_nodes | 25 | seeded_random | 1.179142e-11 | 2.635065e-16 |
| thermal_cell_average | 100 | smooth | 4.783268e-12 | 2.049526e-16 |
| thermal_cell_average | 100 | seeded_random | 6.185837e-11 | 1.958998e-16 |
| thermal_cell_average | 50 | smooth | 4.920849e-12 | 1.985725e-16 |
| thermal_cell_average | 50 | seeded_random | 3.663059e-11 | 0.000000e+00 |

## Gates

- Worst JVP–FD error:
  `6.185837e-11`
  (limit `1.000000e-09`).
- Worst JVP–VJP transpose error:
  `2.935354e-16`
  (limit `1.000000e-12`).
- Worst endpoint constant error:
  `5.995204e-15`.
- Worst area-integral error:
  `2.019484e-16`.
- Opposite-boundary leakage:
  `0.000000e+00`.

This is a solver-free coupling certificate.  Imported-permittivity endpoint
equivalence is the next fail-closed gate.
