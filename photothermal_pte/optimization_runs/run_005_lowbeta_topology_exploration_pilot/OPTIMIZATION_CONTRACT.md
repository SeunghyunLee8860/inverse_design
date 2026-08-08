# Run 005 bounded optimization contract

## Immutable physics

The optical, material-Jacobian, thermal, PTE, and objective contracts are
identical to Run 002/Run 004: 10 um scalar Gaussian illumination, target-plane
waist 8.5 um, 18.6 x 18.6 um finite nonperiodic design window, 373 x 373 nodal
density at 50 nm spacing, 1 um design height, signed `+I_PTE/P_incident`, GPU
Maxwell forward/adjoint, and CUDA float64 thermal/PTE forward/adjoint. No CPU
fallback, density repair, clipping, or empirical gradient scaling is allowed.

## Bounded beta=2 experiment

The initial density is the immutable original beta=2 state with SHA-256
`d3617baf54d54e735feba9d85c439ee77bcdf5ddaeec47e12c812bf036b2c87e`.
Exactly one `move=0.01` proposal may receive a solver evaluation. A failed
offline or solver-backed gate stops the run; it cannot retry a smaller move.

The beta=2 smooth caps are:

| phase | initial value | cap | initial/cap |
|---|---:|---:|---:|
| solid | 1.192175674e-3 | 1.26e-3 | 0.9462 |
| void | 2.563430042e-5 | 5.00e-5 | 0.5127 |

The void occupancy is intentionally below the general 0.8--0.95 calibration
target because this specific pilot tests the previously observed `move=0.01`
topology candidate without forcing it into a tiny-move repair. This exception
is recorded, is not a production cap schedule, and must be reviewed after the
one point.

Acceptance requires both smooth constraints feasible and actual FOM retention
of at least 99.8%. Exact thresholded 500 nm bad cells are recorded but are not
a monotone veto at beta=2. A candidate exceeding both 1.5 times the current
exact total and 25 additional cells halts before Maxwell as a catastrophic
safety guard.

## Gates after this pilot

No beta promotion occurs here. Before any 3--5 update pilot, the point must show
genuine FOM/topology movement rather than constraint-only repair. Before every
later beta, the same checkpoint must be reprojected and its smooth-cap occupancy
audited. At beta >= 32 the exact total nonincrease veto is restored, and the
final binary design must have exactly zero solid and void bad cells. Two
consecutive accepted moves below 0.0025 require an automatic halt and cap
recalibration.
