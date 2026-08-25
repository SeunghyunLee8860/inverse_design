# FDTDX offline two-level exact-adjoint report

Date: 2026-08-25 (Asia/Seoul)

Implementation/execution commit: `07ccae9eea6d9303abd09fd7c49432b8f3d5bb83`

Status: `PASS_EXACT_GRADIENT_CONNECTIVITY`; `BLOCKED_30_MIN_RUNTIME`

## Tested alternative

The preceding exact blockwise VJP passed gradient gates but projected to about
36 minutes per polarization because its online checkpoint schedule recomputed
too much.  This follow-up removed that online loop.  It saves exact sparse
outer starts, regenerates exact segment starts once per outer block, and uses a
short direct JAX VJP for each segment.  It never calls the algebraic E/H/ADE-P/
CPML reverse functions and never changes a Maxwell or material equation.

The implementation matched direct unrolled AD on the 24-step and 70-step
CPML/ADE/late-PhasorDetector scenes, including partial final outer blocks and
segments.  The complete CPU suite passed `245` tests in 259.81 seconds.

## Exact-grid bounded result

Ea and Eb ran in parallel on verified-idle GPUs 6 and 7 from a clean commit.
Both used the exact `186 x 186 x 286` grid, 4,096 steps, outer block length
4,096, direct segment length 64, canonical beta-4 81 x 81 latent point and
direction, and retained production Au/TaIrTe4 late phasor states.  The
diagnostic loss was final Au-region electric-field energy, not Q, PDE, current,
or the optimizer.

| polarization | directional AD | centered FD | relative error | VAG | linear 40-period projection |
|---|---:|---:|---:|---:|---:|
| Ea | `-8.5373153683e-8` | `-8.5363893731e-8` | `5.4235e-5` | 36.739 s | 38.295 min |
| Eb | `-1.9745976840e-7` | `-1.9747243840e-7` | `3.2081e-5` | 36.010 s | 37.534 min |

All 6,561 latent-gradient entries are finite and nonzero.  The exact-gradient
connectivity remains sound, but the offline schedule is slower than the
34.466/34.578-second online c96 blockwise result.

One sparse state is 384,151,876 bytes.  The full-horizon 4,096/64 schedule has
63 outer starts (24.20 GB) and 64 reused inner starts (24.59 GB), or 48.79 GB
before direct-segment residuals and XLA work buffers.  The bounded measured
peak was about 111.22 GB and the largest allocation was 103.89 GB.  Adding the
remaining full-horizon outer starts leaves little margin under the reported
143.63-GB XLA limit; full-horizon memory is not certified.

## Decision

This experiment was the single bounded comparison authorized by
`FDTDX_BLOCKWISE_EXACT_REPORT.md`.  It fails the 30-minute runtime gate and is
not a production candidate.  Do not search segment sizes, outer block sizes,
checkpoint counts, or reversible slice lengths.  Do not run a 16,384-step or
full-horizon gradient, complete Q/PDE/current gradient, smoke optimization, or
MMA on any of these FDTDX reverse paths.

Under the frozen 40-period, exact-grid contract, the audited FDTDX routes now
separate cleanly:

- algebraic reverse: fast enough in projection, but Eb gradient is biased;
- online blockwise exact: accurate, but projects to about 36 minutes;
- offline two-level exact: accurate, but projects to about 38 minutes.

Therefore FDTDX inverse-design iteration is currently infeasible under the
user runtime requirement.  A next route must change a higher-level premise
with independent physical validation, such as a proven shorter steady-state
horizon or a different exact frequency-domain/adjoint solver.  It must not
silently relax the selected mesh, material law, two-polarization objective, or
AD-FD gate.  The separate Lumerical work must continue independently and must
not be overwritten by this FDTDX audit.

## External raw artifacts

| artifact | SHA-256 |
|---|---|
| Ea final JSON | `a0518a8afb90b711b867648a1934e90ed92e43b62fc1aae34c90eff69c53cc00` |
| Ea NPZ | `9be6f100f248ffee95b779204224452d5ff553dcbe4f17362af77930e957df3d` |
| Ea adapter JSON | `cbd575518bf635c7af4ad9b39a7fffa25c7015d7b4fb5459c9375357ce8f2fc7` |
| Eb final JSON | `20ca3ae881f50267c3b3062d56b932512506198b81930b6283863a14b50cc0b2` |
| Eb NPZ | `48021c0ab52a4bb59165cb10acf055ebf85ab6d5e55b01e01cb8d241307b4942` |
| Eb adapter JSON | `2a134a04b52933d149638385acec250db7202a71c318398c29ad229e5528c751` |

No raw result is stored in Git.  No full gradient, Q/PDE/current evaluation,
optimizer, Lumerical, HEAT, or CHARGE call was made.
