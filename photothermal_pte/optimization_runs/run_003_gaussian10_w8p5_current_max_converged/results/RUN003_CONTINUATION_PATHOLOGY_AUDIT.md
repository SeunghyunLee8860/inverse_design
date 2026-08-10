# Run 003 continuation pathology audit

Status: `HALTED_PREMATURE_BETA2_SATURATION_AND_LATE_CONSTRAINT_REPAIR`

This is an offline audit of already accepted checkpoints. No Maxwell, thermal,
PTE, adjoint, or optimization solve was launched. The last accepted checkpoint
is `g095`; interrupted `g096` is excluded.

## Conclusion

The user's concern is correct. Run 003 did not perform a healthy joint
objective/manufacturability continuation. Beta=2 captured nearly all useful FOM
gain while its manufacturing inequalities were inactive and 88.3%
of latent variables reached a box bound. Later stages inherited a largely frozen
topology, the objective gradient collapsed by a factor of
581.5 by beta=16,
and exact 500 nm defects grew from 47/139
(solid/void) at beta=2 to 360/501
at beta=16. Beta=32 therefore became late repair work.

## Stage evidence

| beta | accepted updates | stage FOM gain | final FOM (A/W) | solid/cap | void/cap | exact bad solid/void | latent at bounds | objective grad L2 (A) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 44 | 796.2486% | 8.760986275e-07 | 0.335 | 0.252 | 47/139 | 88.3% | 3.137e-20 |
| 4 | 12 | 0.3055% | 8.799517352e-07 | 0.352 | 0.314 | 219/138 | 88.0% | 8.666e-21 |
| 8 | 16 | 0.4261% | 8.844470870e-07 | 0.617 | 0.616 | 323/407 | 85.6% | 3.338e-22 |
| 16 | 11 | 0.1170% | 8.858801860e-07 | 1.000 | 1.001 | 360/501 | 82.6% | 5.395e-23 |
| 32 | 12 | 0.0025% | 8.852107939e-07 | 0.462 | 1.007 | 95/260 | 77.6% | 9.756e-23 |


## Why continuing was wasteful

1. The loose beta=2 caps (`0.04`) left the constraints far from active at the
   final beta=2 design (solid/cap 0.335, void/cap 0.252).
2. Forty-four accepted beta=2 updates drove the objective and 88.3% of latent
   variables to their box limits before exact manufacturability was controlled.
3. The legacy smooth surrogate did not track exact disk-opening defects: exact
   defects worsened sharply at beta=4--16 despite nominal smooth feasibility.
4. Introducing the disk-opening contract only at beta=32 could repair defects,
   but could not recover the design freedom already lost. The tiny late moves
   were therefore expensive morphology repair with negligible FOM benefit.

## Required restart principle (proposal only)

Do not resume g096. A replacement run should activate the same 500 nm disk-based
solid and void constraints from the first iteration, prevent prolonged objective-
only saturation at low beta, and advance beta based on joint FOM and morphology
progress. The exact schedule and acceptance rule require user review before any
new GPU execution.

Run 003 is not converged and has not passed exact 500 nm solid/void constraints.
