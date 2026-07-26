# FVM internal-interface-G control report

**Status: `VALIDATED_FVM_INTERNAL_INTERFACE_G_CONTROLS`.**

This is an independent conservative Cartesian Python/SciPy finite-volume
result, not a Lumerical HEAT result. No optical Q or full-device geometry was
used in this control.

The internal face resistance used by both matrix assembly and flux recovery
is

`R'' = dz_1/(2 k_1) + 1/G + dz_2/(2 k_2)`.

The slabs use `k1=5 W/(m K)`, `k2=20 W/(m K)`,
`t1=t2=1 um`, and fixed `310 K -> 300 K` boundary temperatures.

| G (W/m2 K) | mesh (nm) | analytic q'' | numerical q'' | flux error | analytic jump (K) | numerical jump (K) | jump error | k1/k2 flux mismatch | energy error | status |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 7.37e+06 | 100 | 25927880.387 | 25927880.387 | 2.96697e-11% | 3.51802990325 | 3.51802990325 | 7.58656e-12% | 1.21696e-11% | 1.9732e-10% | PASSED |
| 7.37e+06 | 50 | 25927880.387 | 25927880.387 | 5.46411e-11% | 3.51802990325 | 3.51802990326 | 8.19122e-11% | 3.02329e-10% | 4.38481e-10% | PASSED |
| 7.37e+06 | 25 | 25927880.387 | 25927880.3871 | 5.36957e-10% | 3.51802990325 | 3.51802990325 | 1.74996e-10% | 9.38813e-10% | 2.28006e-09% | PASSED |
| 1.1e+09 | 100 | 39855072.4638 | 39855072.4638 | 1.12165e-11% | 0.036231884058 | 0.0362318840583 | 8.57176e-10% | 5.55217e-12% | 4.27777e-11% | PASSED |
| 1.1e+09 | 50 | 39855072.4638 | 39855072.4638 | 9.27605e-11% | 0.036231884058 | 0.0362318840589 | 2.42605e-09% | 1.93672e-10% | 5.13451e-10% | PASSED |
| 1.1e+09 | 25 | 39855072.4638 | 39855072.4638 | 1.60583e-10% | 0.036231884058 | 0.0362318840557 | 6.20278e-09% | 5.60227e-10% | 9.128e-10% | PASSED |
| perfect | 100 | 40000000 | 40000000 | 1.71736e-11% | 0 | 2.27373675443e-13 | 2.27374e-12% | 7.89762e-12% | 9.94755e-11% | PASSED |
| perfect | 50 | 40000000 | 40000000 | 1.76579e-11% | 0 | 6.8212102633e-13 | 6.82121e-12% | 2.54288e-10% | 1.13689e-10% | PASSED |
| perfect | 25 | 40000000 | 40000000.0002 | 4.32245e-10% | 0 | -2.21689333557e-12 | 2.21689e-11% | 6.19423e-10% | 2.16004e-09% | PASSED |

## Independent checks

- The one-sided interface temperatures are obtained by independently fitting
  the cell-center temperature profile in each material and extrapolating each
  fit to `z=0`.
- Heat flux is recovered independently at the hot boundary, in material 1,
  on the interface face, in material 2, and at the cold boundary.
- Every finite-G jump, analytic series-resistance flux, material-to-material
  flux transmission, temperature profile, and global energy balance error is
  below 1%.
- The linear residual limit is `1e-09`.

## Mesh refinement and perfect contact

All three G conditions pass at 100, 50, and 25 nm. The finite-G jump and
transmitted-flux spreads across the three meshes are below 1%.

For perfect contact, the extrapolated one-sided interface jump remains at
roundoff while the raw adjacent-cell difference decreases as the cell
centers approach the interface:

`[0.5, 0.25, 0.12500000000005684]` K.

The finest/coarsest raw-jump ratio is
`0.25`;
the expected first-order geometric ratio for 25/100 nm is 0.25.

## Gate

The finite-G FVM analytic gate is closed successfully. The next required
step is a common 3D isotropic, perfect-contact, heterogeneous-material,
volumetric-Q control solved by both v261 Lumerical HEAT and this FVM. The
validated finite optical Q must not be imported until that cross-validation
passes.
