# Fresh FDTDX anchor placement certificate

Status: **VALIDATED_PLACEMENT_ONLY_NOT_A_MESH_CERTIFICATE**

This certificate covers object placement and material/PML readback only. It
does not claim field, absorption, thermal, electrical, or mesh convergence.
No Maxwell timestep was advanced.

## Provenance

```text
inverse-design commit  e7d9e89623559af34b49e6be7767bb68f300246d
FDTDX commit           f26f84b70a8cceec9b889553955a868624736bf1
FDTDX tree             43687e561d4bd2735f188149b2fc1bc50da82c47
GPU                     GPU-b288c55e-827d-e6b4-d05a-4b27eb65477f (B200)
runtime date            2026-08-24
```

The repository and pinned FDTDX checkout were both clean. The raw JSON remains
outside Git at:

```text
/home/seunghyun200/fdtdx_results/anchor_placement_e7d9e896_20260824/FDTDX_FRESH_ANCHOR_PLACEMENT.json
sha256 b3ddcadf923ca1d98d672cec4521ce5043c4185744a6f2f1ff9fb8a19c297fb2
```

## What passed

All 59 checks passed in 18.976 s:

- anchor grid `196 x 196 x 160 = 6,146,560` Yee cells;
- exact slices and physical bounds for Si, 285 nm SiO2, 100 nm TaIrTe4,
  50 nm Au window, Gaussian source, incident/target planes, and closed monitors;
- all six PML objects, slices, 4 um alpha, face-specific sigma, finite lossy
  coefficient arrays, and unity kappa;
- centered 2 um square integer mask, containing 400 Au design cells and air
  everywhere else in the 80 x 80 window;
- exact readback of Au ADE `c1/c2/c3` as only zero or the locked ordinary-Au
  endpoint; inverse epsilon-infinity was exactly one throughout the window;
- Ea source polarization `(0, 1, 0)` under the frozen x=b, y=a convention;
- no adjoint source and zero calls to `fdtdx.run_fdtd`.

FDTDX stores the realized nonuniform edge metrics as float32 in this runtime.
They equal the contract edges after the solver dtype cast. Maximum error versus
the float64 contract was `4.453e-13 m` in x/y and `1.061e-13 m` in z. The
realized lateral PML thickness was `9.999994e-7 m`; the z PML thickness was
`1.6000001e-6 m`.

## What this does not authorize

Placement success does not authorize a reference sweep or optimization. The
next required step is a source-only all-air solve for each polarization on the
same exact grid/PML/time contract, with incident-power positivity, polarization
purity, flux sign, and previous/late complex-field stationarity checks. After
that, all six exact-binary references must pass every independent mesh axis and
two successive comparisons described in `FDTDX_EXACT_BINARY_CONVERGENCE_PLAN.md`.

Thermal and electrical discretizations remain downstream blockers. They must
not be treated as converged merely because the Maxwell placement passed.
