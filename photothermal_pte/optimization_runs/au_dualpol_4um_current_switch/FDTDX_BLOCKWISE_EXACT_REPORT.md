# FDTDX blockwise exact-adjoint report

Date: 2026-08-25 (Asia/Seoul)

Small full-state prototype commit: `467a9fcb69acff71a9335e708fffaa2d9441fd71`

Sparse production-bound implementation/execution commit:
`44a477864c0701f1aa98f9af3c8479cdbda5c63e`

Status: `PASS_EXACT_GRADIENT_CONNECTIVITY`; `BLOCKED_30_MIN_RUNTIME`

## Change and correctness proof

The blocked reversible VJP evaluated local Jacobians at E/H/ADE-P/CPML states
obtained by algebraic time reversal.  The new blockwise path never reconstructs
a previous state.  It saves exact forward block starts, restarts from those
states in backward, and differentiates the ordinary pinned FDTDX forward step.

The first prototype retained full states and matched direct unrolled AD on the
24-step and 70-step CPML/ADE/late-PhasorDetector scenes.  The latter includes a
partial final block.  The production-bound implementation then retained P only
on the already certified TaIrTe4/Au support and used exact Equinox checkpointed
recomputation within each block.  It passed the same two direct-gradient tests.
The complete CPU suite passed `242` tests in 219.71 seconds.

## Exact-grid 4,096-step result

Ea and Eb ran in parallel on verified-idle GPUs 6 and 7.  Both used the exact
`186 x 186 x 286` grid, beta-4 canonical 81 x 81 latent point/direction,
production Au/TaIrTe4 late phasor states, block length 4,096, and 96 inner
checkpoints.  The diagnostic loss remained final Au-region electric-field
energy; this was not a Q, PDE, current, or optimizer run.

| polarization | directional AD | centered FD | relative error | VAG | linear 40-period projection |
|---|---:|---:|---:|---:|---:|
| Ea | `-8.5373148285e-8` | `-8.5363804914e-8` | `5.4724e-5` | 34.466 s | 35.93 min |
| Eb | `-1.9745947803e-7` | `-1.9747243840e-7` | `3.2817e-5` | 34.578 s | 36.04 min |

Every one of the 6,561 latent-gradient entries is finite and nonzero.  Relative
L2 differences from the prior independent checkpointed 4,096-step gradient
are `2.3342e-5` for Ea and `1.0886e-5` for Eb, consistent with float32 loop
grouping.  Removing algebraic reverse therefore closes the earlier Eb
long-horizon bias mechanism; this bounded result does not yet certify the
40-period gradient.

With the retained late detector states, one sparse checkpoint is 384,151,876
bytes.  A full 256,163-step schedule with 4,096-step blocks has 63 outer
checkpoints (24.20 GB) and 96 inner checkpoints (36.88 GB), or 61.08 GB of
checkpoint payload before transient/XLA buffers.  The bounded measured peak
was about 68.15 GB.

## Frozen checkpoint-count comparison

The only follow-up changed inner checkpoints from 96 to 192.  Values,
gradients, FD directionals, and raw NPZ hashes were identical, but runtime got
slightly worse and peak memory rose sharply:

| checkpoints | Ea VAG | Eb VAG | peak XLA bytes | projected parallel wall |
|---:|---:|---:|---:|---:|
| 96 | 34.466 s | 34.578 s | about 68.15 GB | 36.04 min max |
| 192 | 34.843 s | 34.914 s | about 127.82 GB | 36.39 min max |

For the full horizon, 192 outer-plus-inner checkpoint payload alone is
97.96 GB.  Its bounded largest allocation was 120.48 GB, so it is neither
faster nor safely composable with the full outer checkpoint array under the
reported 143.63-GB XLA limit.  Further checkpoint-count tuning is closed.

## Decision and next gate

The exact-gradient algorithm is connected, but its measured 4,096-step linear
projection fails the user runtime gate of 30 minutes per parallel Ea/Eb
iteration.  The true multi-block schedule may be slower, so the 36-minute
number is not an upper bound.  Do not run a 16,384-step or full-horizon probe,
complete current gradient, or optimizer on this implementation.

The next bounded implementation may change only the exact recomputation
schedule, not the Maxwell/material equations: construct an offline two-level
segmented VJP that saves exact sparse outer/inner starts and uses short direct
segment VJPs.  Prove 24/70-step direct parity first, audit the complete payload,
and run at most one 4,096-step Ea/Eb comparison if its static memory estimate
fits.  Reject it unless the linear parallel-wall projection is below 30 minutes.

## External raw artifacts

| setting | artifact | SHA-256 |
|---|---|---|
| c96 Ea | JSON | `99358d4f581187111c78df7611bd286880fba4ef39b672e32e2000c346d640ac` |
| c96 Ea | NPZ | `5f5ffa1c7570a3d854dd91b5f25897e61a843621856b1436f10d6cd52fb6459d` |
| c96 Eb | JSON | `1a817806b31b940951b5dbb2792486014202ad2e593f017c1e994288267e26a9` |
| c96 Eb | NPZ | `c728a6cd24c68c20583844695527095061bd849fb977f780c6f3245816b00bf6` |
| c192 Ea | JSON | `448ec5fde6022f70a182ef3e2b7e5386d9fc926eecaf23d37aa15f2a8e966152` |
| c192 Ea | NPZ | `5f5ffa1c7570a3d854dd91b5f25897e61a843621856b1436f10d6cd52fb6459d` |
| c192 Eb | JSON | `3abe894c08a588ef885eff4c99e4e5c312b742f649800415d97b90a43186c9ab` |
| c192 Eb | NPZ | `c728a6cd24c68c20583844695527095061bd849fb977f780c6f3245816b00bf6` |

No raw result is stored in Git.  No full gradient, Q/PDE/current evaluation,
optimizer, Lumerical, HEAT, or CHARGE call was made.
