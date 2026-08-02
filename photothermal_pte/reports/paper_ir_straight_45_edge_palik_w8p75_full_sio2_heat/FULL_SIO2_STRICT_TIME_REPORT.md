# Full-SiO2 strict-time FDTD diagnostic

Status: `BLOCKED_LUMERICAL_FDTD_LATE_TIME_DIVERGENCE`

The old `1e-5` run stopped at `0.736725 ps`
(`18.418%` of the requested 4 ps window).  The matched
strict run continued through the same apparent minimum, then its logged
auto-shutoff observable rose to `93861.7`
and the GPU solver terminated for diverging electromagnetic fields at
`1.626320 ps` (`40.658%`).

The log alone does not distinguish delayed source content from growth of a
numerically unstable electromagnetic mode.  It does establish that the old
early-stop monitor result did not test this late-time interval.

The strict run's Pabs bounds were on native mesh planes within 1 fm.  It did
not produce a converged final Q or face-flux result, so no strict closure is
reported and no thermal, PTE, adjoint, or optimization run follows.

The old `P_Q=1.261110455146e-11 W` and closure
`1.249223%` are preserved as early-stop
diagnostics only.  They are not promoted or rescaled.

The time trace alone does not distinguish delayed source content from the
onset of numerical-instability growth.  It does establish that the old
early-stop artifact never tested the interval in which the strict run failed.

The engine log prints a generic successful-completion footer after explicitly
reporting divergence; the LumAPI run exception and explicit divergence line,
not that footer, determine the fail-closed status.
