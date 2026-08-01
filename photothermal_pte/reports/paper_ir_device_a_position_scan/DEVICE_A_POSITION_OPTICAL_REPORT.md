# Device-A three-position optical checkpoint

Status: `VALIDATED_DEVICE_A_THREE_POSITION_OPTICAL_GATE`

Four new GPU FDTD solves and the two immutable `s0` artifacts were audited on
one fixed Device-A/PML/monitor/material/mesh contract. Only source center and
polarization change. No CPU FDTD fallback or Q clipping, smoothing, gain,
rescaling, or polarization matching was used.

| signed s (um) | polarization | P_Q (W) | P_six (W) | closure | auto-shutoff |
|---:|---|---:|---:|---:|---:|
| 2.0 | E||a | 2.967404434e-11 | 2.974501756e-11 | 0.2386% | 9.995150e-06 |
| 2.0 | E||b | 4.016286976e-11 | 4.021141798e-11 | 0.1207% | 9.718390e-06 |
| 3.0 | E||a | 3.189657093e-11 | 3.197216283e-11 | 0.2364% | 9.461010e-06 |
| 3.0 | E||b | 4.361628833e-11 | 4.364555405e-11 | 0.0671% | 9.968510e-06 |
| 4.0 | E||a | 3.386492893e-11 | 3.389513886e-11 | 0.0891% | 9.782220e-06 |
| 4.0 | E||b | 4.670616571e-11 | 4.673525513e-11 | 0.0622% | 9.897370e-06 |

The legacy top-level `validated=false` field in each production case result is
preserved. This checkpoint uses the explicit acceptance items plus an
independent raw-NPZ dual-cell reintegration; it does not rewrite raw metadata.
