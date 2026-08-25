# FDTDX optical AD probe report

Date: 2026-08-25 (Asia/Seoul)

Code commit: `591ad5be75c482ee20bb671bf16b0720e650cac0`

This report is a bounded connectivity and resource result.  It is not a
40-period optical-gradient certificate and it does not enable the optimizer.

## What was differentiated

The probe used the exact parity grid (`186 x 186 x 286`) and the committed
beta-4 mapping

`81 x 81 latent -> 500-nm finite conic filter -> tanh projection -> 80 x 80 cells`.

The loss was only

`mean(final E^2 inside the Au design slab)`.

It contained no explicit rho or target-Q term.  Therefore a nonzero derivative
had to traverse

`latent rho -> shared mapping -> nonlinear Au ADE c3 -> checkpointed Maxwell loop -> field loss`.

All runs used the pinned FDTDX/JAX source hashes frozen by
`fdtdx_parity_ad_contract.py`, Courant `0.25`, thin-stack z cells `2.5 nm`,
bulk/air z cells at most `50 nm`, design/flake xy cells `100 nm`, and the full
physical material stack.  Only the time horizon was bounded to 4,096 steps,
or `0.6395927593` optical period.

## Passing AD-FD results

| polarization | checkpoints | JAX memory limit | value-and-grad | AD directional | centered FD | symmetric error | status |
|---|---:|---:|---:|---:|---:|---:|---|
| Ea | 16 | default 75% | 122.2090 s | -8.5373617e-8 | -8.5370733e-8 | 1.6893e-5 | PASS |
| Ea | 32 | 95% | 107.7234 s | -8.5373617e-8 | -8.5370822e-8 | 1.6373e-5 | PASS |
| Eb | 32 | 95% | 107.7629 s | -1.9745934e-7 | -1.9746906e-7 | 2.4620e-5 | PASS |

Each passing run produced a finite, nonzero gradient at all 6,561 latent
nodes.  The 32-checkpoint Ea gradient is numerically identical to the
16-checkpoint Ea gradient at the reported precision.  This validates bounded
Ea and Eb field-mediated reverse-mode connectivity; it does not validate the
late-window absorbed-power objective over 40 periods.

## Measured memory and failed checkpoint counts

The original resource table was explicitly a lower bound and substantially
underestimated the actual differentiated XLA program.

| checkpoints | XLA peak bytes in use | allocator pool / observed nvidia-smi | outcome |
|---:|---:|---:|---|
| 16 | 71,768,244,736 B = 66.84 GiB | 83,762 MiB = 81.80 GiB | PASS |
| 32 | 134,850,899,968 B = 125.59 GiB | 149,298 MiB = 145.80 GiB | PASS only with 95% JAX limit |
| 64 | requested one 238.26-GiB allocation | exceeds the 179.06-GiB device | OOM |

With the default approximately 75% JAX allocator limit, both Ea and Eb
32-checkpoint runs failed while requesting a `120.76 GiB` buffer.  With
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` and preallocation still disabled, both
passed.  The environment change was process-local on verified-idle GPU UUIDs.
It did not touch GPUs used by Lumerical or other users.  A 64-checkpoint run
requested `238.26 GiB`; no allocator fraction can make that fit on this B200.

## Runtime feasibility gate

The 32-checkpoint value-and-gradient was only about 11.9% faster than the
16-checkpoint result while consuming about 64 GiB more observed GPU memory.

A deliberately optimistic constant-cost-per-step projection from the two
32-checkpoint measurements gives:

- Ea: `107.7234 s * 256163 / 4096 = 112.28 min` per value-and-gradient.
- Eb: `107.7629 s * 256163 / 4096 = 112.32 min` per value-and-gradient.

This already exceeds the frozen 30-minute feasibility limit by about 3.74x.
It is not an upper bound.  The online checkpoint binomial-capacity level rises
from 3 at 4,096 steps to 5 at 256,163 steps for 32 checkpoints, so the full run
has a deeper recomputation schedule.  The production late-phasor/Q objective
also does no less work than this final-field microprobe.

The current upstream checkpointed FDTDX implementation is therefore
`BLOCKED_PRODUCTION_RUNTIME`.  Do not spend roughly two hours per polarization
on a full gradient, do not run the 16-forward full Ea/Eb certificate, and do
not start the two-iteration optimizer with this implementation.

## Raw artifacts outside Git

| artifact | SHA-256 |
|---|---|
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_591ad5be_Ea_c16.json` | `9f5df553ef5d7c1d91576a8b4ab7ef7d7e2125690d7ae7e534fee3f762298d28` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_591ad5be_Ea_c16.npz` | `f0a3d8f95a8c4378504b66c3728cd32b1d4461690d3cd38f0b992ab098f29c54` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_591ad5be_Ea_c32_mf95.json` | `a7316d0a4a11c4d396d99eea5841490f1351177b4d8014c79d9ca8f8c74f65cb` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_591ad5be_Ea_c32_mf95.npz` | `6d642834d4fb49bc34ac4c0a85e47d6837e2042dea7a6ba30ed4b188a4192c09` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_591ad5be_Eb_c32_mf95.json` | `a7f384b59bda29b3c652c08bf8bbb23980ca3251df85b3ffb0ffc9576af3e53c` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_591ad5be_Eb_c32_mf95.npz` | `48d9ba537ce866b626bdba51a9b81f4406cc90db01ebae4d1786eb9719d44c24` |

No raw arrays, logs, images, or iteration results are in Git.

## Required next implementation gate

Before another full-grid derivative run, inspect and reduce the checkpointed
loop state.  In particular:

1. Separate time-varying E/H/P/PML state from immutable full-grid material
   arrays so fixed c1/c2, fixed portions of c3, inverse permittivity, and other
   unchanged leaves are not copied into every checkpoint.
2. Build a gradient-only object set containing only the detectors required for
   late Au/TaIrTe4 absorption; do not carry the energy-closure control detector
   suite through every optimizer evaluation.
3. Preserve differentiation through the shared mapping and the spatially
   varying Au c3 carrier.  A memory shortcut that stop-gradients c3 is invalid.
4. Re-run the bounded Ea/Eb AD-FD probe and require a defensible projection
   below 30 minutes before any 40-period AD-FD certificate.
5. Keep `optimizer_enabled=false` until the complete physical objective,
   custom thermal/electrical PDE residuals, signed-current gradients, and all
   four Ea/Eb latent directions pass.

Lumerical, HEAT, and CHARGE were not called during this work.  Final candidates
still require independent Lumerical CV0 and finer-mesh validation when the
separate license workflow is available.
