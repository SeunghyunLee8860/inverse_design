# User-requested balanced FDTDX baseline

## Result

The user-requested mesh has passed a new all-air Ea/Eb source control and a
source-bound exact-binary Au material forward for both polarizations.  This is
a validated **single-mesh Maxwell baseline**, not a mesh-convergence
certificate and not a photocurrent result.  No Lumerical file or process was
modified.

The baseline is tied to repository commit `ae571336` and patched FDTDX commit
`6cc0e97252ee0b95de5016e8db1a5b414177efa4`.  The complete project test suite
at this checkpoint is `441 passed, 7 subtests passed`.

## Realized mesh and time contract

- physical domain: `20 um x 20 um x 6 um`, z bounds `-3 um` to `+3 um`
- complete `16 um x 16 um` TaIrTe4 flake and the Au design window: `100 nm`
  x/y cells
- outer non-PML air margin: `200 nm` x/y cells
- SiO2, TaIrTe4, and Au: `5 nm` z cells
- non-PML air: `50 nm` z cells
- frozen resolved-Si buffer: 20 cells at `50.75 nm`; its `1.5%` deviation from
  50 nm is explicit because the fixed `1.015 um` buffer cannot be divided
  exactly without moving a physical or PML boundary
- PML: eight cells on every face; lateral PML cells are `125 nm`, and z-PML
  cells are `200 nm`
- grid: `186 x 186 x 150 = 5,189,400` Yee cells
- time: 24 optical periods, four-period late/previous windows, Courant `0.5`,
  `38,496` time steps
- topology used for this control: exact binary
  `l_shape_4um_with_500nm_arms`, 375 solid `100 nm x 100 nm` design cells and
  ten Au z cells

The mesh contract is implemented independently in
`fdtdx_user_balanced_mesh.py`; historical mesh certificates were not edited or
reinterpreted.

## Source-pair result

Ea and Eb source-only cases ran concurrently on verified-idle physical B200
GPUs 6 and 7.  Physical GPUs 0 and 4 had pre-existing Lumerical `fdtd-engine`
processes and were not used.

- Ea unscaled incident power: `1.8822944505142436e-12 W`
- Eb unscaled incident power: `1.8822942336738090e-12 W`
- relative mismatch: `1.1520006709251907e-7`, below the `5e-3` gate
- common reference: arithmetic mean `1.8822943420940264e-12 W`
- one shared reporting power scale: `151410963.53875315`
- one shared field-amplitude scale: `12304.916234528098`

Per-polarization power matching is forbidden.  With the shared scale, the
reported source powers are `285.000016416 uW` and `284.999983584 uW`.

The source-pair certificate is external to Git:

```text
/home/seunghyun200/fdtdx_results/user_balanced_source_a0a286a8/
  FDTDX_USER_BALANCED_SOURCE_PAIR_ae571336.json
```

Its SHA-256 is
`577294187bde149af928e62e45ebc389aab38720b1e0fe2e7dec6e2b232210f4`.
The underlying Ea/Eb report SHA-256 values are
`d85972bde5763287305d587670a30f001c3327e373d58b851066ae0da5c8cc5f`
and `125eb89a742060e42d7f65b5ec38c1383bd059133c4c3d280526d9171221bf33`.

## Exact-binary Au material result

The material cases used the same source, time, placement, PML, and mesh
contracts as the source pair.  Au was assigned only through exact air/Au
finite-dt increment-state ADE endpoints.  There is no gray density, no
`rho**3`, and no optical/thermal/electrical density mismatch in this control.
All material readback, provenance, stationarity, Q, and flux gates pass.

| metric | Ea | Eb |
| --- | ---: | ---: |
| total unscaled absorbed power | `4.6398409177e-13 W` | `8.4209898019e-13 W` |
| Au absorbed power | `5.0376281814e-15 W` | `1.2952572940e-14 W` |
| TaIrTe4 absorbed power | `4.5894646359e-13 W` | `8.2914640725e-13 W` |
| total absorbed fraction of all-air source | `0.24649922` | `0.44737901` |
| maximum late/previous complex-E NRMSE | `8.2538e-5` | `1.7783e-4` |
| total-Q late/previous relative change | `7.5284e-7` | `3.1761e-6` |
| spatial-Q late/previous NRMSE | `1.9409e-5` | `5.1059e-5` |
| Q versus closed phasor relative closure | `9.8233e-4` | `5.3147e-5` |
| shared-scale total absorbed power | `70.252278 uW` | `127.503018 uW` |
| total process time | `60.246 s` | `59.166 s` |
| cold compile plus forward | `35.000 s` | `34.845 s` |

For this L-shaped reference, Eb total absorption is `1.81493` times Ea.  Au
accounts for only `1.086%` of Ea and `1.538%` of Eb absorption; most absorption
is in TaIrTe4.  This polarization-dependent optical absorption does **not**
establish the sign or magnitude of PTE current.  Thermal and electrical solves
have not been applied to this new baseline.

External material artifacts are under:

```text
/home/seunghyun200/fdtdx_results/user_balanced_material_ae571336/{Ea,Eb}/
```

| artifact | Ea SHA-256 | Eb SHA-256 |
| --- | --- | --- |
| JSON report | `eafddef006178f4a729a09c9c9560243c64ece8c8478ade22427af9ba0ed749d` | `b66cfd18150c3f4e0c6a2bf7a118386934692d50aebba51b39e03c92dbf1ca02` |
| raw NPZ | `5a5f217f4e3b15d8f54dd37dbcfa0c7f5fdc14faa7067575eeaa2ac7500efab7` | `b6dcbd87b0104f5959f96d9d0c13d5992513bc620dbd42132a1250bb206c9f6f` |

Raw NPZ files are intentionally outside Git.

## What is allowed next

Do not start inverse design from this one passed mesh.  The next step is a
new, hash-separated refinement family around this anchor:

1. refine the complete z domain while holding x/y at the user baseline;
2. give every refined z case its own all-air Ea/Eb source pair before its Au
   material pair;
3. compare total and component Q, conservative 3-D Q, fixed-probe fields, and
   Au/TaIrTe4 region complex fields without relaxing existing gates;
4. separately refine the complete x/y domain while holding the selected z
   level;
5. measure the first refinement runtime before authorizing any further level.

At least one refinement result is required before deciding whether a third
level is scientifically necessary and computationally practical.  A two-level
comparison must not be relabeled as a final convergence certificate.  Gray
material interpolation, adjoint timing, thermal/electrical coupling, and every
optimizer remain blocked meanwhile.
