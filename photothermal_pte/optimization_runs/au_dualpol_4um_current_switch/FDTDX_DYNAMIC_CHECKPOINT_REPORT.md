# FDTDX dynamic checkpoint report

Date: 2026-08-25 (Asia/Seoul)

Implementation commit: `a4229081c0ac5cc2bb5ea38403b473049d9e9daa`

This is an exact-grid bounded AD connectivity/resource result.  It does not
enable a 40-period gradient or the optimizer.

## Change

The generic FDTDX checkpointed loop uses

`(time_step, complete ArrayContainer)`

as the Equinox loop value.  Equinox documents that every checkpoint stores a
copy of the loop `init_val`.  The generic route therefore copied full-grid
material leaves that never change with time.

`fdtdx_parity_dynamic_checkpoint.py` keeps only

`(time_step, FieldState, detector_states)`

in the checkpoint value.  Inverse permittivity, conductivity, c1/c2/c3/c4,
and other immutable material leaves are differentiable closure inputs to the
unchanged FDTDX `forward()` step.  No Maxwell or ADE update equation was
modified and c3 was not stop-gradiented.

## Correctness gates

On a small real dispersive FDTD scene, the generic and dynamic loops matched
in both final-field loss and the complete c3 gradient.  The full target-folder
suite passed `199` tests.

On the exact `186 x 186 x 286` parity grid, every dynamic result reproduced
the prior generic result:

| polarization | dynamic checkpoints | value-and-grad | AD-FD error | raw NPZ SHA-256 |
|---|---:|---:|---:|---|
| Ea | 32 | 84.9653 s | 1.6893e-5 | `f0a3d8f95a8c4378504b66c3728cd32b1d4461690d3cd38f0b992ab098f29c54` |
| Ea | 64 | 74.0574 s | 1.6893e-5 | `f0a3d8f95a8c4378504b66c3728cd32b1d4461690d3cd38f0b992ab098f29c54` |
| Eb | 64 | 73.9963 s | 2.4620e-5 | `48d9ba537ce866b626bdba51a9b81f4406cc90db01ebae4d1786eb9719d44c24` |

The dynamic Ea NPZ hashes are exactly the same as generic Ea/16 and Ea/32.
The dynamic Eb hash is exactly the same as generic Eb/32.  Each NPZ contains
the latent array, direction, complete 6,561-node gradient, primal value, and
centered-FD values.  Thus the loop-state optimization changed neither value
nor gradient bits in these saved arrays.

## Carry and GPU memory

The exact physical ArrayContainer audit reported:

| item | bytes |
|---|---:|
| complete ArrayContainer | 2,161,538,560 |
| dynamic FieldState | 985,960,704 |
| dynamic detector states | 67,398,784 |
| dynamic time step | 4 |
| dynamic checkpoint total | 1,053,359,492 |
| excluded immutable leaves | 1,108,179,072 |

The dynamic checkpoint value is `48.73%` of the generic full-container value.

| route | checkpoints | XLA peak bytes in use | allocator / observed nvidia-smi |
|---|---:|---:|---:|
| generic | 32 | 134,850,899,968 B = 125.59 GiB | 149,298 MiB = 145.80 GiB |
| dynamic | 32 | 64,441,950,464 B = 60.02 GiB | XLA peak pool 80.00 GiB |
| dynamic | 64 | 120,944,277,760 B = 112.64 GiB | 150,016 MiB = 146.50 GiB |

Generic 64 checkpoints previously required a 238.26-GiB allocation and OOMed.
Dynamic 64 passes at roughly the same observed process memory as generic 32.
Dynamic 96 still OOMed while requesting a 160.43-GiB buffer; it is not a safe
production setting on this 179.06-GiB B200.

## Runtime gate remains blocked

Dynamic 32 is about 21% faster than generic 32, and dynamic 64 is about 31%
faster than generic 32.  Nevertheless, the deliberately optimistic linear
4,096-to-256,163-step projections are:

- Ea dynamic 32: `88.56 min`.
- Ea dynamic 64: `77.19 min`.
- Eb dynamic 64: `77.13 min`.

The full online checkpoint schedule is deeper than the bounded probe, so these
are not upper bounds.  The current route remains
`BLOCKED_PRODUCTION_RUNTIME`; do not run the full gradient, the full 16-forward
AD-FD certificate, or the optimizer.

## Next bottleneck

Of the `985,960,704` FieldState bytes, the full-domain current and previous ADE
polarization arrays alone occupy `712,400,832` bytes.  FDTDX allocates them for
all 9,894,456 cells even though c3 is nonzero only in the thin TaIrTe4/Au
material region.  They are now the dominant checkpoint payload.

The next implementation gate is a parity-only sparse ADE-state carry:

1. Store P-current/P-previous only on the exact union of the TaIrTe4 and Au
   slices.
2. Reconstruct/extract full-grid P only inside the unchanged single-step
   update, or implement an algebraically identical regional ADE correction.
3. Prove small-scene forward/c3-gradient parity and exact-grid bounded Ea/Eb
   AD-FD parity before accepting any timing result.
4. Reject the change if scatter/expand overhead or a changed update equation
   compromises correctness.

## External raw artifacts

| artifact | file SHA-256 |
|---|---|
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_a4229081_Ea_dynamic_c32.json` | `e7ff053a75d22452a89f5db53ccf2321bb666677a76d55e6ddd2ac7579df5768` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_a4229081_Ea_dynamic_c32.npz` | `f0a3d8f95a8c4378504b66c3728cd32b1d4461690d3cd38f0b992ab098f29c54` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_a4229081_Ea_dynamic_c64.json` | `327f5f940007add707df5f543aa7916a33239b0f4c82e88df11a64d5fa6bf79d` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_a4229081_Ea_dynamic_c64.npz` | `f0a3d8f95a8c4378504b66c3728cd32b1d4461690d3cd38f0b992ab098f29c54` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_a4229081_Eb_dynamic_c64.json` | `021fa207eb09d5a769502bdec3cbc525c2d39e54b98932b40024e97044c998fb` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_a4229081_Eb_dynamic_c64.npz` | `48d9ba537ce866b626bdba51a9b81f4406cc90db01ebae4d1786eb9719d44c24` |

No raw result is stored in Git.  Lumerical, HEAT, and CHARGE were not called.
