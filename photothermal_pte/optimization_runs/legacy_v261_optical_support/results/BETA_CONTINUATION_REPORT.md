# Run 002 beta continuation with 500 nm solid/void constraints

Status: `STOPPED_INVALID_ONE_STEP_PER_BETA_CONTINUATION_METHOD`

This run is preserved only as a fail-closed methodology diagnostic. The beta=2
pilot had not converged, beta=4 contained only two accepted updates, and the
beta=8 result was only the same latent design reprojected at beta=8. The
supervisor's one-update-per-beta rule and iteration-relative 1% constraint caps
were invalid. No fully binary or fabrication-ready design was produced. Run
003 restarts from the original beta=2 initial state with fixed stage
inequalities and convergence-based promotion.

This is the continuation of the validated nominal run. It uses a stateful MMA subproblem, the existing finite nonperiodic 500 nm conic filter, explicit smooth solid/void constraints, and an independent exact 500 nm binary morphology audit. The exact audit never modifies or repairs the design.

Current beta: `8`; current global iteration: `7`; FOM: `5.533488388291e-07 A/W`; mean `4 rho (1-rho)`: `0.58876008`.

Exact solid violation fraction: `0.00028031539`; exact void violation fraction: `0`.

`RUNNING` is intentional: fully binary promotion requires the final grayness gates, zero exact solid/void violations, and a fresh thresholded-binary Maxwell/CUDA-thermal reevaluation.
