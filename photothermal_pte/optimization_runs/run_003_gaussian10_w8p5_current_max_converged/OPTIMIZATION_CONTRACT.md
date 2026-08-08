# Run 003 constrained-continuation contract

## Preserved physics

- scalar Gaussian illumination, wavelength 10 um, target-plane waist 8.5 um;
- finite nonperiodic 18.6 x 18.6 um design window on 373 x 373 nodes (50 nm);
- air/complex-Kitamura-SiO2 endpoints at 10 um;
- 1.0 um design height;
- finite 500 nm conic density filter and tanh projection;
- signed objective `+I_PTE/P_incident`;
- actual GPU Maxwell forward/adjoint and CUDA float64 thermal/PTE
  forward/adjoint at every evaluated candidate;
- no CPU solver fallback, empirical gradient normalization, clipping,
  smoothing, binary repair, or post-hoc objective rescaling.

## Why Run 002 continuation stopped

Run 002's beta=2 pilot was still improving by several percent per iteration.
Nevertheless, its later supervisor attempted only one accepted update per
beta.  It also replaced genuine fixed inequalities by a new 1% tighter cap at
every proposal and carried MMA asymptotes across a changed projection.  Its
beta=8 record is a reprojected diagnostic baseline, not a converged stage.

## Optimizer

- stateful Svanberg MMA with exact latent bounds `0 <= x <= 1`;
- MMA asymptotes persist inside a beta stage and reset whenever beta changes;
- fixed objective nondimensionalization `1e8 * I_PTE/P_incident`;
- immutable beta=2--16 checkpoints used differentiable Zhou solid/void fields
  aggregated by a p=8 power mean;
- exact thresholded 500 nm solid/void morphology is audited at every proposal
  and accepted point but never edits the continuous candidate;
- from beta=32 onward, a proposal whose total exact solid-plus-void bad-cell
  count exceeds the current accepted count is rejected before Maxwell solves.

## Fail-closed 500 nm constraint recovery

At beta=16 the legacy Zhou values reached their fixed cap while the exact disk
opening audit worsened from `323/407` to `360/501` solid/void bad cells.  The
first two beta=32 updates then ended at `385/521`.  Those solver-backed results
remain immutable diagnostics, but they prove that the legacy surrogate is not
sufficiently aligned with the final exact gate.

Starting after beta=32 global iteration 85, future candidates therefore use a
smooth disk-opening constraint.  The initial recovery checkpoints at global
iterations 86--87 are recorded as `soft_disk_opening_500nm_v2`.  Subsequent
steps use `soft_disk_opening_500nm_v3_exact_nonincrease`, which adds a
fail-closed exact-DRC acceptance gate after a proposed step exposed that the
smooth metric alone could improve while the exact bad-cell total worsened.
The differentiable part applies a smooth threshold about rho=0.5 and
a differentiable log-sum-exp opening with the same five-pixel-radius Euclidean
disk and phase-specific border values as the exact audit.  Its physical-density
cotangent is propagated through the existing filter/projection VJP.  No binary
opening, repair, clipping, or objective/gradient rescaling is applied to an
accepted design.  The MMA state and the eight-update convergence count reset at
the recovery checkpoint because the constraint functions changed.

If a continuous MMA step improves the smooth constraints but crosses
`rho=0.5` at a cell that increases the exact bad-cell total, the fail-closed
line search subdivides the move down to `0.00125/256`.  The ordinary adaptive
MMA move floor remains `0.00125`; these smaller values are threshold-event
line-search trials only.  The exact nonincrease gate is not relaxed.

The beta-32 recovery exposed a second failure mode: once one phase was already
below its stage cap, the two-constraint MMA subproblem could spend that slack
by creating new violations in that phase while reducing the other.  Proposal
subproblems from this checkpoint therefore use phase-preserving effective caps

`min(fixed stage cap, current phase value)`.

The fixed stage caps, convergence criteria, and exact zero-violation final gate
are unchanged.  The effective cap changes only the local MMA proposal direction
and prevents an already-better solid or void phase from being traded away.

An offline centered-FD test at the preserved beta=32 checkpoint gave relative
gradient errors below `1e-7`.  A diagnostic descent of maximum latent move 0.02
reduced exact bad cells from `385/521` to `371/415`; this diagnostic density was
not accepted as an optimization checkpoint.

The exact audit remains nondifferentiable and is not substituted for the MMA
gradient.  It is instead a monotone acceptance guard: the sum of exact solid
and void bad cells may not increase within a beta stage.  This permits a
topology-changing trade between phases while the two separately constrained
smooth fields guide both phases toward zero.  The final promotion gate remains
strictly `solid=0` and `void=0`; the monotone guard is not itself a pass.

The solid and void constraints are genuine fixed inequalities within each
stage.  They are not recomputed from the previous iteration:

| beta | solid cap | void cap |
|---:|---:|---:|
| 2 | 0.040 | 0.040 |
| 4 | 0.030 | 0.030 |
| 8 | 0.020 | 0.020 |
| 16 | 0.008 | 0.008 |
| 32 | 0.002 | 0.002 |
| 64 | 0.001 | 0.001 |
| 128 | 0.0005 | 0.0005 |
| 256 | 0.00025 | 0.00025 |
| 512 | 0.0001 | 0.0001 |
| >=1024 | 0.00005 | 0.00005 |

The beta=2--32 legacy values were frozen before Run 003.  The beta>=64 disk
caps above belong to the fail-closed recovery contract.  For scale, the original beta=2
initial state is about 0.0373/0.0373, while independently generated valid
half-plane and 1 um stripe fixtures are around `1e-5` at beta=2 and smaller at
higher beta.

## Beta promotion

Beta follows `2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096` only after a stage has at
least eight accepted updates (ten at beta=2), satisfies both fixed
inequalities, and the last four accepted updates simultaneously satisfy:

- maximum absolute relative FOM change below 0.5%;
- physical-density RMS change below 0.25%;
- physical-density maximum change below 1.5%.

A nominal iteration count alone never promotes beta.  The original 40-update
safety limit stopped beta=2 fail-closed at accepted iteration 40 (checkpoint
`28adda9`): the FOM plateau passed, while the density RMS/maximum gates failed
because thousands of latent variables remained at the 0.02 MMA move ceiling.
That diagnostic is immutable.  Recovery keeps every convergence gate fixed
and monotonically halves the move ceiling only when constraints are feasible,
four solver-backed updates pass the FOM plateau, and density has not plateaued.
At least four accepted updates are required at each new ceiling before another
reduction.  An 80-update safety limit now stops fail-closed rather than
silently promoting a nonconverged stage.

## Candidate acceptance

When the current point is feasible, a candidate must remain feasible and may
not lower the actual FOM by more than 0.2%.  When the current point is
infeasible, the normalized fixed-cap violation must decrease by at least 0.5%
and the actual FOM may not fall by more than 5%.  Rejected solver evaluations
remain in raw provenance and are retried with a smaller move.

## Final gates

The projected field must have:

- fraction with `0.01 < rho < 0.99` below 0.1%;
- mean `4 rho (1-rho)` below 0.001;
- zero exact 500 nm solid violations;
- zero exact 500 nm void violations.

Only then is the 0.5-threshold binary field evaluated again with a fresh GPU
Maxwell forward solve and CUDA thermal/PTE solve.  The final binary result must
also pass optical closure below 0.5%, thermal residual below `1e-8`, and energy
balance error below 1%.
