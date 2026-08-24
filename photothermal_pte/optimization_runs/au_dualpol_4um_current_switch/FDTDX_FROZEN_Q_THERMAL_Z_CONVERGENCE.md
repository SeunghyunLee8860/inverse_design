# FDTDX frozen-Q thermal z convergence

## Decision and strict scope

The explicit thermal solver's z discretization is converged for the frozen
Ea/Eb heat sources taken from the exact-binary FDTDX z32 artifacts.  Both
successive pairs, factor 1 to 2 and factor 2 to 4, pass for both
polarizations.  Thermal z factor 2 (`266 x 266 x 66`) is selected for this
diagnostic because it is the common member of the two passing tail pairs.

This result does **not** repair or override the failed FDTDX optical mesh
certificate.  It does not converge thermal x/y, the thermal domain or
boundary conditions, the actual electrode/flake electrical geometry, the
electrical mesh/contact/floors, or the coupled objective.  No production
multiphysics mesh is selected and optimization remains forbidden.

The concurrent Lumerical work is independently owned.  This checkpoint did
not edit, launch, or reinterpret any Lumerical file or result.

## Why the optical Q was frozen

The final practical FDTDX z16-to-z32 optical pair failed component-Q and
material-region complex-field gates.  z32 also costs about 18.5 minutes per
polarization forward and cannot support an inverse-design loop.  Running new
FDTDX adjoints or an optimizer would therefore be both numerically unjustified
and impractical.

The z32 material artifacts themselves remain valid exact-binary, stationary,
energy-audited inputs.  Freezing their native component-Yee Q fields lets the
thermal discretization be tested independently without claiming that the
optical source is converged.  Every thermal case records
`diagnostic_only=true`, `optical_mesh_blocked=true`,
`production_mesh_selected=false`, and `optimizer_start_allowed=false`.

The frozen optical certificate is:

- path:
  `/home/seunghyun200/fdtdx_results/increment_state_z32_extension_certificate_1cebc11e/FDTDX_INCREMENT_STATE_FULL_Z32_EXTENSION_CERTIFICATE.json`
- SHA-256:
  `079a6fbbb78aeab29d5e7460815f22208708a307f02572dc956f244433b9bb97`
- status: `BLOCKED_FDTDX_INCREMENT_STATE_Z16_TO_Z32`
- global artifact/provenance checks: all true
- selected optical mesh: none

The exact-binary mask contains 375 solid 100-nm design cells.  No gray Au
law, `rho**3`, clipping, smoothing, gain, or per-polarization power matching
is used.  The same common 285-uW incident-power scale is applied to Ea and Eb.
The resulting mapped absorbed powers are `67.77223817417422 uW` for Ea and
`116.28755821846876 uW` for Eb.

## Code and environment

The diagnostic path consists of:

- `multiphysics_4um.py`: `thermal_edges(z_refinement_factor)` and
  `build_thermal_state(..., z_refinement_factor=...)`;
- `fdtdx_frozen_q_thermal_z_case.py`: byte-bound frozen-Q remap and one
  thermal solve;
- `fdtdx_frozen_q_thermal_z_certificate.py`: six-case revalidation and two
  successive-pair certificate;
- `test_thermal_z_refinement.py`,
  `test_fdtdx_frozen_q_thermal_z_case.py`, and
  `test_fdtdx_frozen_q_thermal_z_certificate.py`.

The thermal solve uses an isolated environment rather than modifying the
FDTDX/JAX environment:

- environment: `/home/seunghyun200/.venvs/thermal-cu130-py312`
- Python: 3.12
- PyTorch: `2.11.0+cu130`
- PyTorch CUDA: 13.0
- GPU: NVIDIA B200, exactly one visible device per process
- environment distribution-manifest SHA-256:
  `4a0310fb3c723dd18d5841429d8d69c3498fbc650c6564fc3aca19d0fa6c6d6d`

The original FDTDX environment still has no PyTorch, so its JAX/NVIDIA
dependencies were not replaced.  The CUDA CSR smoke test returned the exact
expected vector result before any solve.

The first factor-1 attempt stopped before GPU solving because
`multiphysics_4um.py` imported the numerical overlap functions from a legacy
validation script whose module-level imports unnecessarily required
Matplotlib.  Commit `dc0d7397` moved plotting imports inside that script's
`main()` and added a regression test proving that the numerical overlap kernel
imports without Matplotlib.  This is a dependency-boundary fix; no overlap
coefficient or physical equation changed.  The failed output is retained at
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_z_4b90a1b7/` and was not
overwritten.

All successful cases use runner commit
`dc0d7397c1f3ad0bd22dd30a0588667833111cd1` and runner SHA-256
`c754a5de460954d624827ab8e0c5caac7864cd83c45a6d36bd3cc1407f40c0eb`.
The worktree was clean before and after every case.  Ea and Eb ran in parallel
on physical GPUs 6 and 7.  A separate user's Lumerical process on physical GPU
4 was observed and never touched.

## Mesh ladder

Only z edges change.  Every original thermal interval, including Si, SiO2,
TaIrTe4, Au, air and exterior-domain intervals, is subdivided by the stated
factor.  All original material and interface faces remain exact.

| factor | shape | unknowns | matrix nonzeros | TaIrTe4 dz | Au dz sequence |
|---:|---:|---:|---:|---:|---:|
| 1 | 266 x 266 x 33 | 2,334,948 | 16,168,012 | 10 nm | 10, 10, 30 nm |
| 2 | 266 x 266 x 66 | 4,669,896 | 32,477,536 | 5 nm | 5, 5, 5, 5, 15, 15 nm |
| 4 | 266 x 266 x 132 | 9,339,792 | 65,096,584 | 2.5 nm | factor-1 intervals / 4 |

The selected diagnostic factor is 2.  Factor 4 is the independent finer tail
check, not the selected run mesh.

## Artifact roots and byte identities

Successful case root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_z_dc0d7397/`

| factor | pol. | report SHA-256 | compact raw NPZ SHA-256 |
|---:|:---:|---|---|
| 1 | Ea | `f05d33df20d14229e93013d8444faeefc45de628a8796dae4531fb29aa929702` | `a88c01f9498a4e0a643394fb0ca32b9f3d708e388d502eadcc5ff170447f6785` |
| 1 | Eb | `5d7e785e2ce66ac1c2abb27e0b621c20b1687bad9d0ed5153116ed1097cc9e19` | `21e372a7a095bd840c873f74326c50fc8814a9fe6acf1a11ae7e90d5131fd4b6` |
| 2 | Ea | `405de2ddfa65ceee353227ffb2d84426caf5cce32f4aeb1ed92db169f27d5db0` | `23b2b05c346536c0d6a33ba624820a89135ba379acbf75fad969c35c91ed8bab` |
| 2 | Eb | `b7b38ed306ed953b673fdb90563c603c48931f418847f934c13bd6ad0423f234` | `9b5e61b29cf7ef25b7368e92387d277a3212873a1596f3e317f4ec232cdd43cd` |
| 4 | Ea | `0b0234cace94094c45ff02f97d25f5641186d401afb52f126804ff8797ba6070` | `3d2a97e396e4b53b66565e9e05de65b6fc13252ff90d74717e2dc5ec827e73b1` |
| 4 | Eb | `0b63307c7c033be377fec11acfe09e60299892bb4104db74a80f0d134ef22c93` | `3a338bcea7f471020d13293b5f43cbc3d8a1d77354b8722c22ed4e4ea0bfb66b` |

The compact NPZ files retain the 160 x 160 thickness-averaged TaIrTe4
temperature, x/y temperature gradients, coordinates, center z profile, and
x/y-integrated source power.  The full multi-million-cell temperature field is
not committed or copied into the repository.

Certificate root:
`/home/seunghyun200/fdtdx_results/frozen_q_thermal_z_certificate_94a4e593/`

- certificate: `FDTDX_FROZEN_Q_THERMAL_Z_CERTIFICATE.json`
- SHA-256:
  `c333d4a3050c4e9ac18f28c8aa6db8d377b4e0a9d33de3b80f6f905cb37b6f0e`
- generator commit:
  `94a4e593626e528ec854f1d3370db4322bf9651f`
- status:
  `VALIDATED_DIAGNOSTIC_FDTDX_FROZEN_Q_THERMAL_Z_CONVERGENCE`
- ready: true
- all six case artifacts revalidated: true
- both successive pairs pass for both polarizations: true
- production mesh selected: false
- optimizer start allowed: false

## Convergence gates and results

The limits were declared in code before the certificate was generated:

- TaIrTe4 thickness-averaged temperature-map NRMSE: at most 2%;
- TaIrTe4 maximum-temperature relative change: at most 2%;
- TaIrTe4 mean-temperature relative change: at most 2%;
- combined x/y temperature-gradient NRMSE: at most 5%;
- x/y-integrated source-power map change: at most `5e-12` NRMSE;
- both successive pairs must pass for both polarizations.

| pair | pol. | T-map NRMSE | Tmax relative | Tmean relative | combined-gradient NRMSE | source-xy NRMSE |
|---|:---:|---:|---:|---:|---:|---:|
| 1 to 2 | Ea | 0.10064% | 0.08684% | 0.12924% | 0.09490% | 1.07e-16 |
| 1 to 2 | Eb | 0.10927% | 0.09597% | 0.13791% | 0.11543% | 1.05e-16 |
| 2 to 4 | Ea | 0.02728% | 0.02369% | 0.03468% | 0.02566% | 1.01e-16 |
| 2 to 4 | Eb | 0.03018% | 0.02678% | 0.03756% | 0.03199% | 1.00e-16 |

Every gate passes.  The refinement differences decrease from the first pair
to the second pair for both polarizations.

## Solver quality and measured runtime

| factor | pol. | total | assembly | CUDA PCG | iterations | residual | energy-balance error | Ta Tmax | Ta Tmean |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Ea | 6.33 s | 0.71 s | 1.30 s | 1,850 | 8.29e-10 | 2.72e-12 | 0.98957 K | 0.11411 K |
| 1 | Eb | 6.47 s | 0.74 s | 0.98 s | 1,850 | 9.72e-10 | 6.98e-14 | 1.64858 K | 0.19384 K |
| 2 | Ea | 8.45 s | 1.49 s | 1.96 s | 2,525 | 7.30e-10 | 3.58e-11 | 0.98871 K | 0.11396 K |
| 2 | Eb | 8.87 s | 1.51 s | 1.61 s | 2,500 | 8.46e-10 | 3.86e-11 | 1.64700 K | 0.19357 K |
| 4 | Ea | 11.74 s | 2.92 s | 3.90 s | 4,375 | 9.68e-10 | 3.36e-11 | 0.98847 K | 0.11392 K |
| 4 | Eb | 12.04 s | 2.90 s | 3.70 s | 4,400 | 8.86e-10 | 2.83e-11 | 1.64656 K | 0.19350 K |

Ea/Eb were concurrent, so pair wall time is approximately the slower total,
not the sum.  Factor 4 used about 2.58 GiB of GPU memory per process and about
3.7 GiB peak host memory per service.  Thermal z convergence is therefore not
the multi-hour bottleneck.  The blocked FDTDX Maxwell solve remains the
dominant cost: a nominal forward-plus-adjoint optical iteration at z32 has a
lower bound near 37 minutes even with the two polarizations parallelized.

## What may and may not use factor 2

Factor 2 may be used as the z discretization in the next frozen-Q diagnostic
of the existing prototype thermal geometry.  It provides 5-nm TaIrTe4 cells
and subdivides every original z interval without moving an interface.

It may not be called a production mesh because:

1. the thermal lateral core is still 100 nm and has no x/y tail-pair
   certificate;
2. the +/-32-um lateral domain, -20-um substrate depth, ambient Dirichlet
   faces, top convection, and interface conductances have no convergence or
   uncertainty certificate;
3. the z32 optical fields still fail their mesh gates;
4. the electrical model still uses an assumed rectangular flake, complete
   ideal left/right edge terminals, no measured crystal rotation, no actual
   contact polygons, and untested void/contact floors;
5. the target physical-device contract remains unconfirmed.

## Required next actions

1. Preserve this certificate and use thermal z factor 2 only within its frozen
   diagnostic scope.
2. Add a thermal x/y refinement API that preserves all material boundaries
   and changes no z edge, then run factor 1/2/4 tail pairs with the selected
   z factor 2.  Stop if runtime or memory becomes impractical.
3. Separately test domain size, substrate depth, boundary conditions, and
   interface-conductance uncertainty; mesh convergence alone does not validate
   assumed boundary physics.
4. Obtain and encode the actual flake outline/thickness, a-axis angle,
   electrode/pad polygons, signed readout contacts, and whether patterned Au
   is electrically floating/contacting.  Do not certify the present ideal
   edge-terminal electrical model as the device.
5. Once the actual geometry is fixed, run electrical pitch/contact/void-floor
   ladders and a signed Shockley-Ramo current audit for both polarizations.
6. Do not start inverse design until an independently validated Maxwell route,
   thermal x/y/z/domain checks, actual-geometry electrical checks, and complete
   coupled AD-FD all pass.
