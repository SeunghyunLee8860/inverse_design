# Run061/062 shared-geometry dual-polarization inverse design

## Objective

Each electrode geometry has one density field shared by `E||a` and `E||b`.
Every MMA evaluation solves both polarizations on that same density. The
optimized objective is a smooth worst-case current,

`-tau * log(mean(exp(-I/tau)))`,

with `tau=5 nA` at 285 uW. This gives more gradient weight to whichever
polarization has the lower signed terminal current. It does not average two
independently optimized structures.

## Cases

| Run | Electrodes | Flake | Designable region | Fixed TaIrTe4 contact overlap |
|---|---|---|---|---|
| 061 | top-bottom | 24 x 24 um, `x=b`, `y=a` | central 24 x 20 um | 2 um at top and bottom |
| 062 | +45-degree diagonal | same 24 x 24 um flake rotated in the crystal frame | central 20 x 24 um in local device coordinates | 2 um at both diagonal terminals |

Both cases use the thermally-grown TaIrTe4/SiO2 interface,
`G=7.37e6 W/m2/K`. Au is absent from the optical and thermal models; the
electrodes are ideal equipotential regions only in the electrical solve.
Run061 and Run062 execute sequentially on one GPU under one nine-license
`runres` reservation.

## Exact-binary gate

The final geometry must pass the independent 500 nm solid/void audit. Every
exact candidate receives fresh `E||a` and `E||b` Maxwell, thermal, and
electrical evaluations. Final PASS requires the exact candidate to preserve
each polarization's continuous current within 1%.

## Status

Prepared for the fresh Run061 optimization. Run062 follows automatically.
Final fields, signed local contributions, dense current-density maps, and the
two same-device polarization response summaries will be published here.
