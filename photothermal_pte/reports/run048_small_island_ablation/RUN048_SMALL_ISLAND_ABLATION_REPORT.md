# Run 048 single-small-island ablation

Status: `COMPLETED_RUN048_SMALL_FLOATING_ISLAND_ABLATION`

The immutable selected Run 048 exact-binary density was preserved. Only the 18-node floating TaIrTe4 component near `(x=b,y=a)=(-6.65,-3.45) um` was changed to air, followed by one fresh GPU Maxwell solve and the unchanged CUDA thermal/electrical calculation.

## Geometry audit

- Removed solid nodes: `18` (`0.180 um^2`)
- All other nodes: bitwise identical
- Exact 500 nm solid/void bad nodes after removal: `0`

## Fresh end-to-end comparison at 285 µW

| Metric | Original | Small island removed | Relative change |
|---|---:|---:|---:|
| Raw Maxwell P_Q | 2.643517771e-14 W | 2.643040743e-14 W | -0.0180% |
| Mapped P_Q | 5.340466300e-05 W | 5.339515906e-05 W | -0.0178% |
| TaIrTe4 Tmax rise | 0.19745568 K | 0.19747214 K | 0.0083% |
| Max strict grad T | 67319.6767 K/m | 67317.6178 K/m | -0.0031% |
| Terminal current | 22.606415 nA | 22.605564 nA | -0.0038% |
| Terminal conductance | 5.575808635e-03 S | 5.575808634e-03 S | -0.000000% |

All optical closure, conservative mapping, thermal residual/energy, electrical residual, finite-value, and GPU-only physical gates passed: `True`. No clipping, smoothing, gain, polarization matching, or rescaling was used.
