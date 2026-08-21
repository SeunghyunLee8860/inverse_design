# FDTDX dispersive exact discrete-adjoint validation

Status: **VALIDATED_FDTDX_DISPERSIVE_EXACT_DISCRETE_ADJOINT**

## What was validated

The production-width 48 um x 48 um FDTDX optical model was evaluated with a
nonuniform 20 x 20 Au density field.  The gradient is the reverse-mode
adjoint of the **implemented discrete FDTD system**, including all 6,788 time
steps, the Drude/Lorentz ADE states, PML, source, and phasor accumulation.

This is an exact discrete adjoint, not a forward finite-difference gradient.
It is also not advertised as a separate conventional frequency-domain
forward-plus-one-adjoint solve.  The pinned FDTDX version supports dispersive
gradients through checkpointed reverse mode and explicitly rejects its
reversible path for active dispersive ADE arrays.  No unvalidated hand-written
metal overlap formula is substituted.

## Strongest independent check

For the normalized adjoint direction

\[
d = \frac{\nabla_\rho P_Q}{\|\nabla_\rho P_Q\|_2},
\]

the adjoint directional derivative is
`2.314620291526e-14 W`.  Independent central
forward solves give:

| h | adjoint (W) | central FD (W) | relative error |
|---:|---:|---:|---:|
| 0.01 | 2.314620291526e-14 | 2.314731520000e-14 | 0.004805% |
| 0.005 | 2.314620291526e-14 | 2.314731520000e-14 | 0.004805% |


The maximum adjoint-aligned error is **0.004805%**, well
below the 1% gate.  No empirical normalization or gradient rescaling was used.

## Physics and numerical gates

- total absorbed power: `4.782529576960e-13 W`
- empty-subtracted six-face closure: `0.001937%`
- late-window change: `0.171052%`
- gradient L2 norm: `2.314620291526e-14 W`
- exact adjoint execution: `503.663 s`
- central-FD forward count: `12`

## Conclusion and boundary of the claim

The FDTDX checkpointed route computes an accurate optical discrete-adjoint
gradient for the tested dispersive Au/TaIrTe4 implementation.  Runtime and
checkpoint memory are engineering costs, not correctness failures.

This certificate covers optical total-Q only.  It does **not** yet validate a
thermal, electrical, PTE, or optimization gradient.  Those coupled terms must
pass their own AD-FD gates before Au PTE inverse design is permitted.
