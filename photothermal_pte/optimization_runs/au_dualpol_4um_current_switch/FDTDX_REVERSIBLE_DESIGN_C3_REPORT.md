# FDTDX design-only reversible-adjoint report

Date: 2026-08-25 (Asia/Seoul)

Execution commit: `25c878050c3c5104859bb578eebbe09facbef649`

Status: `BLOCKED_RECONSTRUCTED_PRIMAL_BIAS`

## Question tested

The previous reversible VJP differentiated and accumulated a complete
full-grid material tuple even though the latent topology affects only the
three Au-region Lorentz `c3` arrays.  This test removed that unnecessary
path: all fixed material arrays were stop-gradient closures, only the regional
design `c3` was differentiated, and its float32 cotangent was accumulated with
Kahan compensated summation.

The design-specialized VJP first matched direct unrolled AD on the small
CPML/ADE/late-PhasorDetector scene.  The integrated CPU suite then passed
`235` tests before the bounded exact-grid run.

## Exact-grid 16,384-step result

Every production-grid input was held fixed: `186 x 186 x 286` cells,
slice length 1,024, the canonical beta-4 81 x 81 latent point and direction,
and the production Au/TaIrTe4 late phasor states.  Ea and Eb ran in parallel
on separately verified-idle GPUs 6 and 7.  The early-horizon diagnostic loss
was the same final Au-region electric-field energy used by the preceding
bounded probes; this was not a Q, PDE, current, or optimizer run.

| polarization | directional AD | centered FD | relative error | gate |
|---|---:|---:|---:|---|
| Ea | `-2.3727764735e-6` | `-2.3533928584e-6` | `0.0041013374` | PASS, barely |
| Eb | `-5.6649137243e-6` | `-5.5911158370e-6` | `0.0065562983` | **BLOCKED** |

The errors are numerically unchanged from the earlier full-material
slice-1,024 result (`0.00410128`/`0.00655637`).  Restricting the differentiable
parameter and compensated accumulation therefore do not remove the bias.

Value-and-grad took 84.57/84.58 seconds.  Peak allocation fell slightly from
about 15.67 GB to 15.33 GB.  The linear full-horizon projection remains about
22.04 minutes per polarization, but runtime is irrelevant while the Eb
gradient gate fails.

## Diagnosis and next gate

The centered FD references are sound: the checkpointed implementation on this
horizon agrees with FD at `1.8264e-4` for Ea and `1.6441e-4` for Eb.  Shorter
reversible slices made the error worse, and the design-only compensated
experiment leaves it unchanged.  The remaining suspect is therefore the
algebraically reconstructed E/H/ADE-P/CPML primal used to evaluate local
Jacobians during the reversible backward sweep.

Do not run more slice-length searches, accumulation variants, a full gradient,
or an optimizer.  The next implementation must remove algebraic time reversal
from the gradient path.  First prove a blockwise exact-checkpoint VJP against
direct unrolled AD on the small CPML/ADE/phasor scene.  Only after exact small
parity may it be combined with bounded recomputation/checkpointing and tested
on the production grid.

## External raw artifacts

| artifact | SHA-256 |
|---|---|
| Ea JSON | `73cae34c7617238e5e809d9349bc0dfbd29cad3b53deeebe40968418aee99a61` |
| Ea NPZ | `63de0f283369e2706aed187aab419c378f1206a857f8670227101109bf9a6497` |
| Eb JSON | `91f14c94c7c9fff32b08322ab39591ede335d711b719bb012238435a7bd16955` |
| Eb NPZ | `b6f6a730d62ee66f5cb899298b2d59b8a87b2dc3059e71af891e3be62179d7df` |

No full gradient, Q/PDE/current evaluation, optimizer, Lumerical, HEAT, or
CHARGE call was made.
