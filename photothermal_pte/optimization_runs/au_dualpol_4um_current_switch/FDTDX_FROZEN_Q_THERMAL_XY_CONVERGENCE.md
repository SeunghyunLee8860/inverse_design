# FDTDX frozen-Q thermal x/y convergence

## Decision and boundary

The current prototype thermal operator is spatially converged in x/y and z
for the frozen exact-binary FDTDX z32 Ea/Eb heat sources.  The prior thermal-z
certificate selected z factor 2.  With that z grid fixed, both x/y factor
pairs 1 to 2 and 2 to 4 pass for both polarizations.  The selected diagnostic
thermal mesh is therefore x/y factor 2, z factor 2, with shape
`532 x 532 x 66`.

This is not a production multiphysics mesh.  The optical mesh is still
blocked, the thermal domain and boundary-condition assumptions are not
converged, the actual flake/electrode geometry is not encoded, the electrical
mesh/contact/floors are not converged, and optimization remains forbidden.
No Lumerical file or result was edited, launched, or reinterpreted.

## Provenance chain

The x/y certificate first revalidates the prior thermal-z certificate by
bytes:

- prior certificate:
  `/home/seunghyun200/fdtdx_results/frozen_q_thermal_z_certificate_94a4e593/FDTDX_FROZEN_Q_THERMAL_Z_CERTIFICATE.json`
- prior SHA-256:
  `c333d4a3050c4e9ac18f28c8aa6db8d377b4e0a9d33de3b80f6f905cb37b6f0e`
- prior selected diagnostic z factor: 2
- prior thermal x/y selected: false
- prior production mesh selected: false
- prior optimizer start allowed: false

The new x/y factor-1 Ea/Eb baseline was recomputed with the new runner.  Its
160 x 160 TaIrTe4 temperature map, x gradient, y gradient, x/y coordinates,
and 266 x 266 x/y-integrated source-power map are exactly array-equal to the
prior z-factor-2 artifacts.  This prevents the x/y ladder from silently
changing z, Q, or the observation operator.

Every x/y case also revalidates the blocked optical z32 certificate SHA
`079a6fbbb78aeab29d5e7460815f22208708a307f02572dc956f244433b9bb97`,
the exact-binary 375-cell mask, common 285-uW normalization, conservative Q
mapping, clean runner commit, one exclusive GPU, residual, and energy balance.

## Refinement and comparison rule

Every original lateral thermal interval is subdivided uniformly by the x/y
factor.  No original face moves.  The +/-4-um Au window, +/-8-um TaIrTe4
footprint, material interfaces, outer shoulders, and +/-32-um domain faces are
preserved exactly.  The 80 x 80 exact-binary material mask is replicated into
the refined Au cells; it is not interpolated into gray material.

The native Ta temperature map grows from 160 x 160 to 320 x 320 and 640 x
640.  For comparison, refined temperature is area-averaged back to the
original 100-nm Ta observation cells.  Native source power is block-summed,
not averaged, back to the original 266 x 266 thermal observation grid.  The
source total and x/y distribution are therefore conserved.  Gradients are
computed after restriction on the common physical coordinate grid.

The limits were fixed before certification:

- Ta temperature-map NRMSE: at most 2%;
- Ta maximum-temperature relative change: at most 2%;
- Ta mean-temperature relative change: at most 2%;
- combined x/y temperature-gradient NRMSE: at most 5%;
- x/y source-power-map NRMSE: at most `5e-12`;
- reconstructed base-center coordinate tolerance: `2e-18 m`;
- both successive pairs must pass for Ea and Eb.

The factor-2-to-4 base-center difference is only
`8.470329472543003e-22 m`, well inside the pre-existing runner tolerance.
Temperature, gradient, and source limits were not relaxed.

## Mesh ladder and practical cost

The z factor remains 2 in every row.

| x/y factor | shape | unknowns | matrix nonzeros | role |
|---:|---:|---:|---:|---|
| 1 | 266 x 266 x 66 | 4,669,896 | 32,477,536 | exact rebound baseline |
| 2 | 532 x 532 x 66 | 18,679,584 | 130,050,592 | selected diagnostic mesh |
| 4 | 1064 x 1064 x 66 | 74,718,336 | 520,483,264 | finer tail check only |

| factor | pol. | total | assembly | remap | CUDA PCG | iterations | residual | energy error |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Ea | 8.15 s | 1.56 s | 0.33 s | 1.61 s | 2,525 | 7.30e-10 | 3.58e-11 |
| 1 | Eb | 8.00 s | 1.48 s | 0.32 s | 1.84 s | 2,500 | 8.46e-10 | 3.86e-11 |
| 2 | Ea | 18.37 s | 6.14 s | 0.57 s | 6.02 s | 3,750 | 8.45e-10 | 1.74e-11 |
| 2 | Eb | 19.23 s | 6.14 s | 0.56 s | 5.82 s | 3,750 | 8.78e-10 | 1.03e-11 |
| 4 | Ea | 71.52 s | 25.68 s | 1.55 s | 34.21 s | 6,775 | 9.33e-10 | 1.13e-11 |
| 4 | Eb | 70.41 s | 24.41 s | 1.46 s | 34.57 s | 6,725 | 9.34e-10 | 1.51e-11 |

Ea and Eb ran concurrently on physical GPUs 6 and 7, so pair wall time is the
slower case rather than the sum.  Factor 2 peaked near 7.1 GiB host memory per
service; factor 4 peaked near 27.7 GiB.  Factor 4 is acceptable as a one-time
certificate but is not selected for repeated work.  The selected factor-2
thermal case is about 19 seconds cold per polarization pair and about 6
seconds for PCG.  This remains much cheaper than the blocked z32 Maxwell
forward, which is about 18.5 minutes per polarization.

## Convergence results

| pair | pol. | T-map NRMSE | Tmax relative | Tmean relative | combined-gradient NRMSE | source-xy NRMSE |
|---|:---:|---:|---:|---:|---:|---:|
| 1 to 2 | Ea | 0.10346% | 0.04889% | 0.01167% | 1.01590% | 1.12e-16 |
| 1 to 2 | Eb | 0.17631% | 0.07877% | 0.01181% | 2.39445% | 1.10e-16 |
| 2 to 4 | Ea | 0.02878% | 0.01316% | 0.00251% | 0.27171% | 1.21e-16 |
| 2 to 4 | Eb | 0.04690% | 0.02045% | 0.00249% | 0.62966% | 1.19e-16 |

All gates pass.  Every second-pair error is smaller than the corresponding
first-pair error.  The selected factor-2 base-grid Ta results are:

| pol. | base Tmax rise | base mean rise | native refined Tmax rise |
|:---:|---:|---:|---:|
| Ea | 0.98822355 K | 0.11395105 K | 0.98852789 K |
| Eb | 1.64570484 K | 0.19354907 K | 1.64687077 K |

These temperatures are properties of the frozen, optically unconverged Q and
the current assumed thermal geometry.  They are not validated detector
predictions.

## External artifacts

Case root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_xy_6ccfb792/`

All cases use clean runner commit
`6ccfb792bac9e42e17c95272e04a4ecedf7da4c3` and runner SHA-256
`20d9e3bd6388619b11eebfdc3de0bef933e7684d8b8845a3dab300874b6954c2`.

| factor | pol. | report SHA-256 | raw NPZ SHA-256 |
|---:|:---:|---|---|
| 1 | Ea | `755492dc353e42c74b1a051d11c7bd3e2aa55b02c1f124249487b7c0e889a152` | `1c1939c8a0bc0f0a0841d96d6df7939c43223a2d48888ae379f672f7ca90a55e` |
| 1 | Eb | `6162a6c147e59b3e914316f0e92e69a162fbbfa492a5741172f1ddfebb88f613` | `f52bbf880fa7e79e867f0b2bd50cee5ab0ec329f4892d677ee1cc1347e922082` |
| 2 | Ea | `b54eaffe50a05daca24ed67cf31f45b91efad81b760c35e464d3371954dbef79` | `2bf49a265c2e3eaa8015d8e4b6b19318d4541bbbcc9b2d10c81c0736b92f5fcf` |
| 2 | Eb | `5f8eec60757626696e2749f24129ffbe834df18ba1bd2bd4928416c944b175b8` | `b8907b383df2ca420bbe63bbb90565cd50bfadee2a557d5f31daf6cdeb30d5d5` |
| 4 | Ea | `560315ab6331c5ddfcbeed88db847bf774853937902b07086837f1dbcbc3ae4d` | `d2acfde2949c93593f28175fc8586ce0bcdab5612a98492965d68b4f5e92a348` |
| 4 | Eb | `748fd74a75dfb24a5a8ec76961414858ac7fa327d050a0973fa16d745dc7f25b` | `0ed4511ffd79f0f86d98cf6eb3160364aa136987bbaf3f5cae2a2384abc5ab93` |

Certificate root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_xy_certificate_65f2d44e/`

- certificate: `FDTDX_FROZEN_Q_THERMAL_XY_CERTIFICATE.json`
- SHA-256:
  `811c79ded3ba1b7cfe70d23f75bfad76c665566815e881645eee8e4cdbfae96f`
- generator commit:
  `65f2d44e`
- status:
  `VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_XY_CONVERGENCE`
- ready: true
- selected diagnostic shape: 532 x 532 x 66
- production mesh selected: false
- optimizer start allowed: false

No raw NPZ, image, or iteration artifact is committed to Git.

## Remaining blockers and next actions

1. Keep x/y factor 2 and z factor 2 fixed only for subsequent frozen-Q
   diagnostics of the current prototype thermal geometry.
2. Test thermal lateral-domain extent and substrate depth separately.  A mesh
   can be converged while a finite Dirichlet boundary remains too close.
3. Audit the assumed side/bottom ambient Dirichlet conditions, top convection,
   and Au-TaIrTe4/TaIrTe4-SiO2 interface-conductance uncertainty.  These are
   model-form and parameter questions, not mesh questions.
4. Obtain the real flake outline/thickness, a-axis angle, electrode and pad
   polygons, signed terminal assignment, and patterned-Au electrical role.
   The present full-edge terminal rectangle cannot be promoted as the device.
5. Build actual-geometry electrical pitch/contact/void-floor tail pairs and a
   signed Shockley-Ramo current audit for Ea/Eb.
6. Do not start inverse design until an independent Maxwell route, thermal
   domain/boundary checks, actual-geometry electrical checks, and complete
   coupled AD-FD all pass.
