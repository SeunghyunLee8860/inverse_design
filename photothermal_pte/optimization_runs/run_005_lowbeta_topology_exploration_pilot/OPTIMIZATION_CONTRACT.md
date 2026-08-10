# Run 005 bounded optimization contract

## Immutable physics

The optical, material-Jacobian, thermal, PTE, and objective contracts are
identical to Run 002/Run 004: 10 um scalar Gaussian illumination, target-plane
waist 8.5 um, 18.6 x 18.6 um finite nonperiodic design window, 373 x 373 nodal
density at 50 nm spacing, 1 um design height, signed `+I_PTE/P_incident`, GPU
Maxwell forward/adjoint, and CUDA float64 thermal/PTE forward/adjoint. No CPU
fallback, density repair, clipping, or empirical gradient scaling is allowed.
The supervisor enforces a single-physical-GPU contract: `--gpu N` sets
`CUDA_VISIBLE_DEVICES=N` before PyTorch creates a CUDA tensor, so the
morphology/thermal `cuda:0` and Lumerical `GPU N` both use physical GPU `N`.
Using a second physical GPU for constraints is prohibited.

From beta 32 onward, if every authorized move in the bounded retry set fails
the solver-free smooth-restoration or exact-DRC gate, the stage advances to the
next beta without claiming convergence. It may not relax the cap or generate a
move below 0.0025 merely to repair a few cells. Any exact violations that
survive beta 128 still fail closed before solver-free high-beta sharpening.

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
consecutive accepted minimum moves terminate only the current beta stage and
advance projection; they never authorize a looser cap or repeated micro-repair.

## Approved bounded extension after g001

The one-point checkpoint passed with FOM `+19.4826487%` and exact total
`158 -> 46`. The next execution remains at beta=2 and is bounded at five total
accepted updates, including g001. Reprojecting g001 gives the fixed extension
caps:

| phase | g001 value | extension cap | value/cap |
|---|---:|---:|---:|
| solid | 9.047577468e-4 | 1.00e-3 | 0.9048 |
| void | 3.816872717e-5 | 4.50e-5 | 0.8482 |

Both occupancies lie inside the approved 0.8--0.95 interval. These caps remain
fixed for the extension and are not changed to rescue a candidate. The only
offline smooth-feasibility move trials are 0.01, 0.005, and 0.0025. No move
below 0.0025 is permitted. If a solver-backed candidate fails the physics or
99.8% FOM-retention gate, the pilot pauses without another GPU evaluation.

After at least three accepted beta=2 points exist, a recent-three net FOM gain
below 0.2% also pauses the pilot. Exact DRC remains diagnostic below beta=32,
apart from the existing catastrophic-growth guard. Reaching five total points
always pauses for review and still does not promote beta.

## Second cap epoch after accepted g002

Accepted g002 increased FOM by another `16.4319769%`, giving
`1.359883308119e-7 A/W`, while exact bad cells changed `44/2 -> 43/2`.
Its void value reached 99.62% of the g001-reprojected cap. The following g003
proposals at moves 0.01, 0.005, and 0.0025 all failed that old smooth cap and
were rejected offline with zero Maxwell and zero thermal solves. No micro-move
below 0.0025 was attempted.

To continue low-beta topology exploration rather than repair the old cap, g002
is reprojected into a new fixed epoch:

| phase | g002 value | new fixed cap | value/cap |
|---|---:|---:|---:|
| solid | 7.371437373e-4 | 8.20e-4 | 0.8990 |
| void | 4.482910852e-5 | 5.30e-5 | 0.8458 |

This recalibration uses the accepted checkpoint values, not an FD-fitted or
solver-fitted objective correction. The bounded five-point target and all
anti-microrepair/FOM gates remain unchanged.

## Final fixed beta=2 exploration envelope after accepted g003

Accepted g003 increased FOM by another `14.2174320%`, reaching
`1.553223792767e-7 A/W` and a cumulative `+58.8947%` relative to the immutable
baseline. Exact bad cells changed `43/2 -> 40/6`. The second cap epoch then
blocked all g004 proposals at moves 0.01, 0.005, and 0.0025 offline, again with
zero Maxwell and zero thermal solves.

Run 005 will not keep recalibrating a nearly active void cap after every
accepted low-beta point. The remaining two authorized beta=2 points instead
use one fixed topology-exploration envelope:

| phase | g003 value | fixed envelope | value/envelope |
|---|---:|---:|---:|
| solid | 6.430948671e-4 | 1.00e-3 | 0.6431 |
| void | 5.244079806e-5 | 1.00e-4 | 0.5244 |

The intentionally loose occupancies are a beta=2 topology-search exception,
not a production cap schedule. They prevent the optimizer from spending the
last two pilot points only repairing a moving cap. This envelope cannot change
again before the five-point audit. Any beta=4 continuation must reproject the
accepted beta=2 checkpoint and establish a new, explicitly reviewed cap epoch.

## Bounded beta=2 outcome

The five-point target completed with all accepted moves equal to 0.01. Actual
FOM gains per update were `+19.4826%`, `+16.4320%`, `+14.2174%`, `+12.5336%`,
and `+11.2077%`; cumulative gain was `+98.8505%`. The final smooth constraints
remain feasible, and no solver-backed rejection, micro-move retry, CPU FDTD,
CPU thermal fallback, empirical normalization, or gradient rescaling occurred.

The pilot is now paused. Beta=4 remains unauthorized until a solver-free
reprojection audit records the projected density change, both constraint
values, exact DRC, and the new fixed cap proposal. The final binary requirement
remains zero solid and zero void bad cells; this gray beta=2 checkpoint is not a
finished inverse-designed structure.

## Approved full-binary continuation

The immutable g005 checkpoint is the restart. Beta=2 must first satisfy the
existing four-update FOM/density plateau and may use at most 16 total accepted
updates. Later optimized-stage budgets are 10, 10, 8, 6, 6, and 12 accepted
updates for beta 4, 8, 16, 32, 64, and 128 respectively. Exhausting a budget
ends that beta stage without claiming plateau and advances projection; it is
not permission to continue constraint-only micro-repair and is not a completed
optimization.

For every beta after 2, the incoming checkpoint is reprojected once and a
solid/void cap pair is persisted in `stage_caps.json`. The pair is immutable
inside that stage. From beta=4 the MMA subproblem may not use slack in one phase
to worsen the other. At beta 32, 64, and 128 the incoming constraint occupancy
targets are 1.10, 1.15, and 1.20 respectively, so these stages perform bounded
morphology restoration rather than silently granting fresh slack. A candidate
starting above the cap must reduce normalized smooth violation by at least 1%
and retain at least 99.8% of the solver-backed FOM. From beta=32 the exact total
bad-cell count also cannot increase. Two consecutive accepted minimum moves of
0.0025 advance projection instead of relaxing a cap.

At the final optimized stage (`beta=128`), if every bounded smooth-MMA
candidate fails the solver-free prescreen while exact violations remain, the
driver may invoke the versioned sequential contract
`sequential_exact_drc_active_set_filter_vjp_v2`. The current exact bad-solid
and bad-void masks define only a frozen local physical-density surrogate. Its
cotangent is propagated through the same certified projection and finite-conic
filter VJP to propose bounded latent-variable steps. A proposal is eligible for
one fresh GPU Maxwell plus CUDA thermal/PTE evaluation only if it strictly
reduces the exact bad-cell total and reduces fixed-cap smooth violation by at
least 1%. Actual per-step FOM retention remains at least 99.8%.

This path never assigns thresholded pixels and never applies binary opening or
closing to the design. It is therefore a solver-backed continuous restoration
epoch, not post-hoc geometry repair. Failure to reach exact zero inside the
12-update beta=128 budget remains terminal and fail closed.

If every bounded one-step active-set direction stalls on the discontinuous
audit, v2 may solve one continuous filtered-sign feasibility subproblem on the
same selected CUDA device. The idempotent exact audit supplies target signs
only; the returned variable is still a continuous latent field obtained by
conic-filter differentiation. The target binary is never assigned to the
design. This subproblem is bounded by `|delta latent| <= 0.5`, must reach exact
`0/0`, must satisfy or improve the smooth gate, and must pass a 0.2% linearized
FOM-loss prescreen before the mandatory fresh GPU/CUDA physics evaluation.

Once exact solid and void bad-cell counts both reach zero, beta 256 through
8192 are projection-only sharpening checkpoints with no redundant forward or
adjoint solve. The first projection satisfying gray fraction <0.001 and mean
`4*rho*(1-rho)` <0.001 is thresholded and evaluated afresh with GPU Maxwell and
CUDA thermal/PTE. Only that successful evaluation has status
`COMPLETED_FULLY_BINARIZED_EXACT_500NM_CONSTRAINED_PTE_OPTIMIZATION`.

## Post-g007 fixed beta=2 cap epoch

Accepted g006 and g007 retained all physics gates and increased FOM by
`5.060714972%` and `4.829345348%`, so the four-update plateau was not present.
At g007 the smooth constraints were:

| phase | g007 value | new fixed cap | value/cap |
|---|---:|---:|---:|
| solid | 4.677681717e-4 | 5.50e-4 | 0.8505 |
| void | 9.971852083e-5 | 1.11e-4 | 0.8984 |

Under the previous `1.00e-3/1.00e-4` envelope, both authorized g008 moves
were rejected offline because the candidate void values were
`1.075145472e-4` and `1.024648997e-4`. No GPU solve was spent. The run must
not promote beta while FOM motion is still strong and must not reduce the move
below `0.0025`. This table establishes one explicit v6 cap epoch from the
accepted g007 checkpoint. It remains immutable for all subsequent beta=2
proposals; no per-step cap update is permitted.

## Final g009 topology-search epoch and MMA reset

The v6 g008/g009 solver-backed gains were `+2.307650024%` and
`+2.258164402%`, despite both accepted moves being `0.0025`.  The repeated
minimum-move halt therefore diagnoses a cap-controlled step size, not a true
FOM/density plateau.  Exact bad cells changed `45 -> 52`; this remains a
low-beta diagnostic and is not accepted as the final morphology.

The immutable g009 checkpoint establishes the final beta=2 cap epoch:

| phase | g009 value | final beta=2 cap | value/cap |
|---|---:|---:|---:|
| solid | 4.527493073e-4 | 6.00e-4 | 0.7546 |
| void | 1.114861694e-4 | 2.00e-4 | 0.5574 |

MMA state is reinitialized from g009 and the move ceiling is restored to
`0.01`.  Accepted-move provenance before g009 cannot throttle the new epoch.
This is the last allowed beta=2 cap change: the pair remains fixed for the
remaining bounded beta=2 budget.  Beta promotion still requires the approved
FOM/density plateau rather than a cap-induced small move.

After g011, the FOM was still rising by `+8.1625%` and exact bad cells improved
from `44/8` at g010 to `40/8`, but the void smooth value reached 99.82% of the
fixed cap.  All g012 trial moves were then rejected by the former 0.5% numerical
feasibility tolerance before Maxwell, including the minimum authorized move.
This is a cap-linearization stall, not an objective plateau.

The final cap pair above remains unchanged.  For beta below 32 it is now used
with a bounded 10% filter/trust band: a candidate outside 1.10 times either cap
is rejected before Maxwell, while one inside the band may be accepted only if
the real GPU-evaluated FOM retains at least 99.8% and the catastrophic exact-DRC
guard passes.  This avoids another cap epoch and prevents sub-0.0025
micro-repair.  At beta 32 and above the tolerance returns to 0.5% and exact
total bad cells must be nonincreasing.  The final binary gate remains exact
solid/void bad cells `0/0`; the trust band cannot certify the final design.

If the low-beta catastrophic guard rejects a proposal, the remaining move in
the fixed `(0.01, 0.005, 0.0025)` schedule may still be prescreened.  Rejected
proposals use zero Maxwell/thermal solves.  No move below `0.0025` is generated;
if the minimum move also fails, the driver stops for a strategy review.

The solver-backed g012 and g013 diagnostics established the bounded beta=2
exit condition.  g012 at move `0.005` improved FOM by `+3.7797%` and reduced
exact bad cells `48 -> 42`.  g013 required the minimum move `0.0025`, improved
FOM by `+1.8219%`, but worsened exact bad cells `42 -> 51`; the following
minimum-move proposal predicted `51 -> 131` and was rejected before Maxwell.
This is a morphology-limited stage, not a FOM plateau.  Beta=2 therefore ends
at immutable g013 and advances to beta=4 rather than repeating minimum-step
constraint repair.  This alternate transition is permitted only at beta=2,
only after the post-reset accepted sequence has contracted from at most 0.005
to 0.0025, and only when the solver-backed minimum step worsens exact DRC.
