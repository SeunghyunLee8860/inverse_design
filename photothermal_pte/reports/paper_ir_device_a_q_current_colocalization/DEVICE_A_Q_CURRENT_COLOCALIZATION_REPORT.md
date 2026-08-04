# Device-A mapped-Q/current co-localization

Status: `COMPLETED_DEVICE_A_Q_CURRENT_COLOCALIZATION`

This checkpoint reads the immutable material-overlap mapped TaIrTe4
`Q_W_m3` and the co-located PTE fields on the same explicit-3D thermal grid.
No new solver was run.

## Same-position source-to-current chain

| d (um) | total Pb/Pa | total Ib/Ia | efficiency ratio `(Ib/Pb)/(Ia/Pa)` | free-edge Q fraction a | free-edge Q fraction b | edge-fraction a/b |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.136726 | 0.812955 | 0.715172 | 39.660806% | 23.895415% | 1.659766 |
| 3 | 1.116961 | 0.836724 | 0.749108 | 41.811698% | 26.870027% | 1.556072 |
| 5 | 1.093917 | 0.844703 | 0.772182 | 43.411870% | 29.732849% | 1.460064 |


`Pb/Pa>1` means the `b` polarization absorbs more total TaIrTe4 power. A
current or efficiency ratio below one means the downstream response still
favors `a`. The free-edge fractions use the same exclusive one-micrometre
device partition as the preceding current-decomposition checkpoint.

The result is not a total-power effect: `b` absorbs `9.39--13.67%` more total
power, while its current efficiency is only `71.52--77.22%` of `a`. The
equal-power Q maps place the missing efficiency at the illuminated free edge.

## Nearest-edge and depth localization

| d (um) | Q fraction a within 0.25 um | Q fraction b within 0.25 um | a/b nearest-edge fraction | top-third Q fraction a | top-third Q fraction b |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.456366% | 4.578249% | 3.594467 | 36.087927% | 31.306249% |
| 3 | 16.619809% | 5.494905% | 3.024585 | 36.076539% | 31.301733% |
| 5 | 17.270907% | 6.522669% | 2.647828 | 36.183266% | 31.308523% |


The closest quarter-micrometre edge band is enriched by `2.65--3.59x` for
`a` after each polarization is normalized by its own absorbed power. The
effect therefore survives equal-power normalization. A smaller but systematic
depth redistribution is also present: `a` places about `36.1%` in the top
third, versus about `31.3%` for `b`.

## Interpretation boundary

This analysis establishes spatial co-localization only. Current generated at
one cell depends nonlocally on Q throughout the device through the thermal
Green function. Therefore `region current / region Q` is not reported as a
causal local material coefficient. Exact causal source attribution would
require new thermal solves with complementary edge/interior Q sources, whose
sum must reconstruct the immutable full-source result.

Accordingly, the next causal control is **not another FDTD run**. It is a
linear superposition check using the unchanged thermal operator:
`Q_full = Q_free-edge + Q_remainder`. Solving the edge term and verifying that
the inferred remainder reconstructs the immutable full temperature/current
will quantify how much of `a>b` is causally driven by edge-localized Q.

All mapped-Q arrays are finite and nonnegative. Reintegrated mapped power and
PTE current close below `1e-12`. No Q clipping, smoothing, gain, rescaling,
tiling, nearest relocation, or source deletion was used. Raw NPZ files remain
outside Git and are SHA-256 pinned in the manifest.
