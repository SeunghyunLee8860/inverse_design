# Run 002 production-candidate GPU forward

Status: `VALIDATED_RUN002_PRODUCTION_CANDIDATE_FORWARD`

This is the first actual GPU Maxwell forward for the matched-volume rho=0.5
coarse production-candidate stack.  It is not a thermal, PTE, adjoint, or
optimization result.

| metric | value |
|:--|--:|
| source power | 1.382226110302e-13 W |
| P_Q | 7.296954820427e-14 W |
| P_six | 7.296652586386e-14 W |
| six-face closure | 0.004142% |
| final auto-shutoff | 7.811230e-08 |
| solver wall time | 122.150 s |
| GPU memory | 1.148 GiB |
| logged grid points | 36,551,664 |

| component | power (W) | fraction of P_Q (%) | native hotspot xyz (m) | negative cells |
|:--:|--:|--:|:--|--:|
| x | 7.244527560956e-14 | 99.281519 | [-4.999999999998962e-08, 1.037615360386518e-20, -1.000000000000002e-08] | 0 |
| y | 9.958403884717e-18 | 0.013647 | [9.90000000000002e-06, 9.950000000000022e-06, -1.000000000000002e-08] | 0 |
| z | 5.143141908317e-16 | 0.704834 | [-1e-05, 1.037615360386518e-20, 1.664526907557124e-08] | 0 |

No Q clipping, smoothing, gain, or rescaling was used.  The three component
maps remain on their native staggered Yee coordinates; they were not summed by
array index for the plots.

## Runtime implication

One forward required about 122.2 seconds.  A
forward+adjoint iteration on the full 20×20 µm coarse canvas will therefore be
on the order of minutes.  The reviewed gradient-L1 window-selection step is
still required before the 50 nm production optimizer is enabled.
