# HEAT internal-interface-G solver report

**Status: `BLOCKED_INTERFACE_G_UNVERIFIED`.**

The tested v261 candidate used a temperature boundary on the shared
`material:material` surface with `thermal impedance = 1/G`. The
surface/material selection and exact insulance survived save/reload,
but the numerical solution did not realize a two-sided contact
resistance.

| G (W/m2 K) | 1/G (m2 K/W) | expected jump | numerical jump | jump error | flux error | transmission error | energy error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `7.37e+06` | `1.3568521e-07` | `6.55976875 K` | `1.13686838e-13 K` | `100%` | `86.4614%` | `57.5407%` | `28.7703%` |
| `1.1e+09` | `9.09090909e-10` | `0.0565883555 K` | `2.27373675e-13 K` | `100%` | `56.1839%` | `119.134%` | `59.5668%` |

Both finite-G cases remained temperature-continuous and introduced a
third 305 K boundary power. They therefore behave as a
fixed-temperature reservoir with insulance, not as the requested
two-sided law `DeltaT_int = q''/G`.

## Perfect-contact mesh control

| max edge | numerical jump | heat-flux error | energy error |
|---:|---:|---:|---:|
| `100 nm` | `2.27373675e-13 K` | `7.20844e-12%` | `8.36869e-12%` |
| `50 nm` | `1.13686838e-13 K` | `1.07475e-11%` | `2.81215e-11%` |
| `25 nm` | `2.84217094e-13 K` | `7.89762e-12%` | `1.6568e-11%` |

The perfect-contact control reproduces the analytic
`4.0e7 W/m2` flux and a machine-zero jump at all three meshes. This
validates the two-slab geometry and extraction method, but does not
validate finite internal G.

The official HEAT documentation defines thermal impedance on a
temperature boundary as boundary thermal insulance, and states that
internal material interfaces otherwise enforce continuity:
https://optics.ansys.com/hc/en-us/articles/360034398314-Boundary-Conditions-in-HEAT-Simulation-Object
https://optics.ansys.com/hc/en-us/articles/360034917713-HEAT-solver-introduction

No thin-layer substitute or perfect-contact substitution is reported
as a validated internal-G path. The full device remains blocked.
