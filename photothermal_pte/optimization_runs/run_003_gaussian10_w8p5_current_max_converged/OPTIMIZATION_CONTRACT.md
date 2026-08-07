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
- differentiable Zhou solid/void fields aggregated by a p=8 power mean so a
  local narrow feature is not diluted by the 373 x 373 domain average;
- exact thresholded 500 nm solid/void morphology is audited at every accepted
  point but never edits the continuous candidate.

The solid and void constraints are genuine fixed inequalities within each
stage.  They are not recomputed from the previous iteration:

| beta | solid cap | void cap |
|---:|---:|---:|
| 2 | 0.040 | 0.040 |
| 4 | 0.030 | 0.030 |
| 8 | 0.020 | 0.020 |
| 16 | 0.008 | 0.008 |
| 32 | 0.002 | 0.002 |
| >=64 | 0.0001 | 0.0001 |

These values were frozen before Run 003.  For scale, the original beta=2
initial state is about 0.0373/0.0373, while independently generated valid
half-plane and 1 um stripe fixtures are around `1e-5` at beta=2 and smaller at
higher beta.

## Beta promotion

Beta follows `2, 4, 8, 16, 32, 64, 128, 256, 512` only after a stage has at
least eight accepted updates (ten at beta=2), satisfies both fixed
inequalities, and the last four accepted updates simultaneously satisfy:

- maximum absolute relative FOM change below 0.5%;
- physical-density RMS change below 0.25%;
- physical-density maximum change below 1.5%.

A nominal iteration count alone never promotes beta.  A 40-update safety limit
stops fail-closed for diagnosis rather than silently promoting a nonconverged
stage.

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
