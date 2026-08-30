# Au PVA boundary-rim diagnosis

Status: `BLOCKED_AU_MOVING_BOUNDARY_ADJOINT_AFTER_SMOOTH3D_CONTROL`

## Result

The fixed-geometry material derivative passes AD--FD with
`0.003896%` error. This independently
validates the FieldRegion source normalization, the official unconjugated
`E_f E_a` convention, component-specific Yee coordinates, and volume
integration. The same chain fails only when the Au boundary moves.

For the fixed-grid 50-nm PVA film, the central FD steps agree to
`0.701969%`, but the continuous
boundary AD has the wrong sign and `142.794779%`
error. Native component-grid `d-epsilon` also has the wrong sign and changes by
`41.708996%` at the final step, despite
a maximum E/index coordinate mismatch of only
`5.082198e-21 m`.

Sampling 50 nm inside Au, exactly on the boundary, or 50 nm outside air never
recovers the FD sign. At the geometric trace, the film-depth region after
removing the top and bottom 10 nm contributes
`-8.661696783707e-25 J-proxy/m`, which has
the FD sign. The two 10-nm rims contribute
`5.833595243493e-24 J-proxy/m` and reverse
the total. The rim magnitude is
`117.437%` of the absolute final
integral.

The exact Au 50-nm sheet-conductivity endpoint was also constructed. Its fitted
surface conductivity differs from the requested value by only
`0.394771%`, but v261 GPU FDTD explicitly
rejects the sampled-2D material. No CPU fallback was run. The official GPU
limitation states that 2-D optical-conductivity materials are unsupported
except PEC; PEC cannot supply lossy Au absorption.

The fully smooth 3-D result is `FAILED_AU_SMOOTH3D_ELLIPSOID_BOUNDARY_ADJOINT` with 121.298438% strong-direction error.

Production Au PTE optimization remains prohibited. The failed smooth-3-D
control rules out the narrower hypothesis that only the non-smooth finite-film
rim causes the mismatch. The unresolved defect is the continuous
moving-boundary derivative of high-contrast lossy Au on the v261 conformal Yee
discretization. A realistic rounded thin-Au endpoint and the direct
moving-material `P_Q` term therefore remain uncertified. No thermal,
electrical, PTE, or optimization solve is part of this checkpoint.

Official GPU limitation: https://optics.ansys.com/hc/en-us/articles/17518942465811-Getting-started-with-running-FDTD-on-GPU
