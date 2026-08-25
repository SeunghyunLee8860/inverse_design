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

### Run061: complete

At the user's request, Run061 was finalized from the completed beta-64
checkpoint instead of spending the remaining budget at beta 128. The
thresholded checkpoint had 22 exact-audit violations. Gradient-aware discrete
repair produced four independently audited candidates with zero violations;
candidate rank 3 had the largest recomputed dual soft-min current.

| Quantity at 285 uW | Continuous beta 64 | Selected exact binary |
|---|---:|---:|
| `E||a` signed current | +53.913103 nA | +53.605863 nA |
| `E||b` signed current | +53.402246 nA | +53.333744 nA |
| dual signed soft-min | +53.651153 nA | +53.467953 nA |
| exact 500 nm bad cells | 22 after thresholding | **0** |

The exact soft-min change is -0.3415%. Both per-polarization 1% preservation
gates pass. The selected density and independent audit are in
`run_061_top_bottom_thermally_grown_sio2_dual_polarization/results/` as
`FINAL_EXACT_BINARY_STRUCTURE.npz`, `FINAL_EXACT_BINARY_STRUCTURE.png`, and
`FINAL_RESULT.json`.

### Run062: running

The +45-degree diagonal case started automatically under the same nine-license
`runres` reservation on GPU 3 after Run061 finalization. It retains the full
beta schedule and will undergo the same exact-zero, two-polarization final
validation.
