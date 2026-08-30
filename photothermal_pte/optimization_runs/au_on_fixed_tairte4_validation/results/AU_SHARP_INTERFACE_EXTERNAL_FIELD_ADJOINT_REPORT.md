# Au sharp-interface external-field boundary-kernel diagnostic

Status: `BLOCKED_AU_SHARP_INTERFACE_BOUNDARY_QUADRATURE_UNRESOLVED`

This diagnostic removes the explicit moving-Au loss term. Its objective is a
fixed smooth electric-field-energy proxy in air, at least 150 nm below the Au
film. Therefore the derivative contains only the field-mediated sharp-interface
boundary kernel; it is not a `P_Q`, thermal, electrical, PTE, or optimization
result.

The independent central differences are `-3.012113323932e-30`
and `-3.016746598291e-30 J-proxy/um` at `h=0.10` and `0.05 um`. Their relative
change is `0.153585%`, and
the strong perturbation changes the baseline observable by
`3.936734%`. The FD
signal is therefore neither near-null nor step-size dominated.

The GPU adjoint completed in `278.591 s` on
`GPU 0` with final auto-shutoff
`9.717290e-08`. The source
round trip and forward/adjoint coordinate mismatch are exactly zero. The Au
fitted-epsilon readback relative error is
`4.346539e-16`.

The selected 801-point boundary result is `-2.812540252377e-30 J-proxy/um`. Its sign
agrees with FD, but its strong-direction relative error is
`6.769092%`, above the 1% gate. More importantly, increasing
the vertical-edge quadrature from 201 to 401 to 801 samples changes the
derivative from `-8.073121942263e-30` to
`-4.567084428374e-30` to
`-2.812540252377e-30 J-proxy/um`; the final
change is `38.417161%`.
The near-halving pattern is consistent with a grid-point/corner-dominated
sample, but the present artifact did not store the kernel profile, so that
mechanism is explicitly a hypothesis rather than a validated conclusion.

This is a useful improvement over the rejected `P_Q` trace: the sign is now
correct and the error is 6.77%, not five orders of magnitude. It is not a
certificate. No empirical normalization, FD fitting, sign change, or gradient
rescaling is used. Au topology optimization remains prohibited until the
boundary integral converges and the discrete conformal-Yee `P_Q` material-loss
derivative is separately certified.
