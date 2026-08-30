# FDTDX 16-period/4-window fast-contract screening

Status: **VALIDATED_FDTDX_DIAGNOSTIC_16PERIOD4WINDOW_OBJECTIVE_DIRECTIONAL_GRADIENT_EQUIVALENCE**

The candidate and reference use the same 20x20 nonuniform Au density, fixed
TaIrTe4, matched Si/SiO2 interface grid, material models, source, and spatial
grid. Only the simulated duration/window changes from 32/4 to 16/4 periods.

| comparison | relative difference |
|---|---:|
| total P_Q | 0.000503% |
| gradient L2 norm | 0.002810% |
| same smooth directional AD | 0.000870% |

The candidate's internal AD--FD error is
`0.007423%`, total-Q/flux closure
is `0.123109%`, and the
substrate-only closure is
`0.475776%`.

AD execution falls from `5773.955 s` to
`2919.009 s`, a
`1.978x` speedup.

The immutable 32-period checkpoint did not store its 20x20 gradient vector,
so a full gradient angle cannot be reconstructed. This result therefore
promotes only objective/norm/same-direction equivalence. It does not validate
the spatially weighted Maxwell source required by PTE, coupled thermal or
electrical chain-rule terms, Au thermopower, or an inverse-design run.
No clipping, smoothing, gain, result rescaling, or gradient rescaling is used.
