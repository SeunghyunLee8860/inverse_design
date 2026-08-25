# FDTDX reversible exact-grid runtime report

Date: 2026-08-25 (Asia/Seoul)

Implementation commit: `9130d20655fe2be61caa511518ef03365c0719f1`

Status: `PASS_4096_STEP_EA_EB_BOUNDED_CONNECTIVITY`; long-slice stability,
late-window Q, and production runtime remain unvalidated.

## Execution contract

Ea and Eb ran concurrently on two separately verified-idle B200 UUIDs:

- Ea: GPU 6, `GPU-0e94c58d-ebdd-2b12-ce98-28159e8dd756`;
- Eb: GPU 7, `GPU-b288c55e-827d-e6b4-d05a-4b27eb65477f`.

Each process independently required zero memory, zero utilization, and no
compute process on its exact UUID immediately before importing GPU-sensitive
JAX/FDTDX code. GPUs 0, 2, 3, and 5 were occupied by other users and were not
touched. Both selected GPUs returned to zero memory after the probes.

Both runs used the exact `186 x 186 x 286` grid, 4,096 steps, beta 4, the
deterministic 81 x 81 gray latent field, the uniform latent direction, 256
steps per slice, and 16 sparse regional-P slice checkpoints. The differentiated
loop retained only the production `au_late` and `tairte4_late` phasor states.
Because this early horizon precedes their physical late windows, the scalar
probe remained the final Au-region field energy. It validates the
latent-to-ADE-to-Maxwell path and resources, not the late Q derivative.

## Results

| metric | Ea | Eb |
|---|---:|---:|
| status | PASS | PASS |
| value-and-grad | 21.5364 s | 21.6001 s |
| centered AD-FD error | 6.0881e-5 | 3.0847e-4 |
| finite/nonzero latent gradients | 6,561 | 6,561 |
| peak device bytes | 15,665,398,784 | 15,669,544,192 |
| largest allocation | 10,197,115,648 | 10,197,115,648 |
| build | 32.8037 s | 32.7107 s |
| AD compile | 27.7600 s | 27.8097 s |
| primal compile | 16.6001 s | 16.7441 s |
| two FD forwards | 13.7964 s | 13.7997 s |

The AD and FD directionals are nonzero with the same sign. The error gate was
frozen at `5e-3`, so both results pass with margin. The exact placed support
audit again found 4,224,000 nonzero entries for each of `c1/c2/c3`, with every
coefficient exactly zero outside the TaIrTe4/Au region union.

## Runtime interpretation

A single-depth linear ratio gives:

| projection | Ea | Eb |
|---|---:|---:|
| one polarization | 22.4481 min | 22.5144 min |
| sequential Ea+Eb | 44.8961 min | 45.0288 min |

These are not measured full-horizon times or certificates. With Ea and Eb on
two idle GPUs as requested, the optical-gradient wall-time projection is the
slower single-polarization value, about 22.51 minutes, rather than the
sequential sum. This is below 30 minutes but leaves limited margin for thermal,
electrical, communication, and optimizer work.

The prior production-detector checkpointed route measured 27.4033 seconds at
4,096 steps and projected 43.675 minutes per polarization at 65,536 steps. The
new 4,096-step reversible value-and-grad is about 21.4% faster, while using far
less peak memory at this checkpoint count.

## Slice-length blocker

Slice 256 is a correctness/runtime probe only. On 256,163 steps it would emit
1,001 sparse checkpoints and 356,860,375,872 checkpoint bytes, which exceeds a
B200. Candidate full-horizon payloads are:

| slice steps | slices | checkpoint bytes |
|---:|---:|---:|
| 1,024 | 251 | 89,482,471,872 |
| 2,048 | 126 | 44,919,487,872 |
| 4,096 | 63 | 22,459,743,936 |

Inverse damping grows within a slice, so the longest memory-efficient slice
cannot be selected without AD-FD evidence. The next bounded experiment is
16,384 steps with slice 1,024 for Ea/Eb in parallel. It keeps 16 checkpoints,
matching the measured short-run checkpoint count, while testing four times
longer algebraic reconstruction between exact resets. Only after that passes
may a larger slice or deeper horizon be considered.

## External artifacts

| artifact | SHA-256 |
|---|---|
| Ea JSON | `02c9be2448af871197b4f67ee66ca9d36efdb0b4b7ebd61b7af8b452e7537e6b` |
| Ea NPZ | `f4161c95e62a0d3cb4b592b081ab9dd7087a245aaea72853aee1c83293f2d195` |
| Eb JSON | `705f19e95a16c977f945402a822c3b899f1145f197f15342cd16883b48f27880` |
| Eb NPZ | `36c22f1ffeae23311c3d73e85c94b0dbe47f62700ea18202ae084f2064c6f47b` |

Raw artifacts remain under `/home/seunghyun200/fdtdx_parity_raw` and are not in
Git. No full 40-period gradient, PDE/current solve, optimizer, Lumerical, HEAT,
or CHARGE call was made.
