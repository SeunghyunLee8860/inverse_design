# FDTDX sparse ADE checkpoint report

Date: 2026-08-25 (Asia/Seoul)

Implementation commit: `1c99676219b6b954b52338cc1b89ea420adfe799`

Probe-budget commits: `ef3319d1`, `4e777d40`

This is an exact-grid bounded AD connectivity and scaling result.  It does not
validate a 40-period gradient, the complete Q/current objective, or an
optimizer.

## Change

The dynamic checkpoint loop still stored full-domain ADE polarization current
and previous states.  They occupied `712,400,832` bytes per checkpoint even
though dispersion is confined to two disjoint material regions.

`fdtdx_parity_sparse_ade_checkpoint.py` stores P-current and P-previous only on
the exact support

- TaIrTe4: `x=13:173`, `y=13:173`, `z=167:207` (`160 x 160 x 40` cells),
- Au design: `x=53:133`, `y=53:133`, `z=207:227` (`80 x 80 x 20` cells).

Immediately before each time step it reconstructs full-grid P, calls the
unchanged pinned FDTDX `forward()` step, and extracts the same two regions.
The Maxwell and ADE update equations were not modified.  Immutable c1/c2/c3/c4
arrays remain differentiable closure inputs and c3 is not stop-gradiented.

## Fail-closed support and correctness gates

The coefficient-support audit requires c1, c2, c3, and c4 to be exactly zero
outside the two certified regions.  The exact parity model passed: every
outside maximum was exactly zero, while c1/c2/c3 each had `4,224,000` nonzero
entries inside the union.  c4 is zero by material-law construction.

On a small real dispersive FDTD scene, generic and sparse loops matched the
final-field loss and the complete gradient with respect to the regional c3
parameter.  This is the relevant derivative contract: the 6,561 latent nodes
can alter Au coefficients only inside the Au region.  A hypothetical c3
perturbation outside the certified material support is deliberately rejected
rather than silently represented.

The target-folder CPU suite passed `204` tests after the implementation and
probe-budget changes.

## Checkpoint payload

| item | bytes |
|---|---:|
| non-P dynamic FieldState | 273,559,872 |
| full-domain P-current/P-previous | 712,400,832 |
| sparse regional P-current/P-previous | 82,944,000 |
| removed P bytes per checkpoint | 629,456,832 |
| detector states retained in this probe | 67,398,784 |
| prior dynamic checkpoint total | 1,053,359,492 |
| sparse checkpoint total | 423,902,660 |

The sparse carry is `40.24%` of the prior dynamic carry and `19.61%` of the
original full-ArrayContainer carry.  These probes intentionally still retain
the complete forward-control detector suite; detector pruning is a separate
gate.

## Exact-grid Ea/Eb AD-FD results

All runs used the exact `186 x 186 x 286` grid and a field-only loss whose
dependence on rho is exclusively

`81 x 81 latent -> filter -> projection -> 80 x 80 Au cells -> c3 -> Maxwell field`.

Every saved 6,561-node gradient was finite and nonzero.  The 4,096-step sparse
NPZ files are bit-identical to the earlier generic/dynamic results.

| steps | pol. | checkpoints | value-and-grad | relative centered AD-FD error | XLA peak bytes |
|---:|---|---:|---:|---:|---:|
| 4,096 | Ea | 96 | 29.6460 s | 1.6893e-5 | 70,170,189,568 |
| 4,096 | Eb | 96 | 29.9209 s | 2.4620e-5 | 70,168,187,136 |
| 16,384 | Ea | 128 | 176.5189 s | 1.8264e-4 | 91,332,016,128 |
| 16,384 | Eb | 192 | 117.7795 s | 1.6441e-4 | 133,659,670,528 |
| 32,768 | Ea | 192 | 371.1970 s | 1.9446e-4 | 133,659,673,600 |

The longer-horizon errors remain well below the probe's `5e-3` acceptance
gate.  The nonmonotonic timing versus checkpoint count is expected from the
online checkpoint schedule: more checkpoints reduce recomputation until the
memory ceiling is reached.

A 65,536-step Ea run with 256 checkpoints did not start its numerical loop:
XLA requested a single `159.07 GiB` allocation and OOMed.  There is no result
artifact for that failed compile/run.  The largest presently safe setting on
the shared 179.06-GiB B200 is 192 checkpoints with the full detector suite.

## Runtime decision

The deepest successful measurement gives the deliberately optimistic linear
projection

`371.1970 s * 256,163 / 32,768 = 48.36 min per polarization`.

This is not a measured full-gradient runtime and is not an upper bound.  The
production loss must also accumulate late-window material phasors/Q and then
run thermal/electrical solves.  Therefore the current full-control-object route
is `BLOCKED_PRODUCTION_RUNTIME`; do not run the 40-period gradient, complete
16-forward AD-FD certificate, or optimizer.

The detector-pruning experiment is complete; read
`FDTDX_GRADIENT_DETECTOR_REPORT.md`.  Retaining only late Au/TaIrTe4 states
makes 256 checkpoints fit, but the 65,536-step production-profile result still
projects to 43.68 minutes per polarization.  Even the detector-free lower bound
projects to 42.47 minutes.  The 30-minute feasibility gate therefore fails,
and further checkpoint-count tuning is closed.  An independently derived and
tested adjoint or other reverse-mode algorithm is required before any full run.

## External raw artifacts

| artifact | file SHA-256 |
|---|---|
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_1c996762_Ea_sparse_c96.json` | `6f18895844be4b99558bc2f462bd42e7ae068d1d5744588693128124df827ec6` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_1c996762_Ea_sparse_c96.npz` | `f0a3d8f95a8c4378504b66c3728cd32b1d4461690d3cd38f0b992ab098f29c54` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_1c996762_Eb_sparse_c96.json` | `481d8ebd056a30b8047f905ef14de0aeadc5bfa82d2105f035d118740c10619b` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_1c996762_Eb_sparse_c96.npz` | `48d9ba537ce866b626bdba51a9b81f4406cc90db01ebae4d1786eb9719d44c24` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_ef3319d1_Ea_sparse_c128_s16384.json` | `2ada0c9c58ad1d7e12b945efee49ccac82acc128caf8d7de5d66ddb2c30e08b7` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_ef3319d1_Ea_sparse_c128_s16384.npz` | `eeb7b53ac65e7b82b9d72399e6a4e683e5d5211f44c27a0159bcdf2a3a6b4fa1` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_ef3319d1_Eb_sparse_c192_s16384.json` | `9dfee8deae019cc52df184b6f33d54e6c7c852e6b73b03365d3c18e06f0ef564` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_ef3319d1_Eb_sparse_c192_s16384.npz` | `0da1257aefbc8a449df95456a78c992b9007e4327c1d76ebdb50fa9d5e418668` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_4e777d40_Ea_sparse_c192_s32768.json` | `23bfb94ff1676db02d14303fd9e4449e1e5426227964221ebf96804037a17ecf` |
| `/home/seunghyun200/fdtdx_parity_raw/ad_microprobe_4e777d40_Ea_sparse_c192_s32768.npz` | `112c251fdfacd4189f19cf7cf8ae5d0fbd9726edaa90e77d6ebc8e39b0ca3452` |

No raw result is stored in Git.  Lumerical, HEAT, and CHARGE were not called.
