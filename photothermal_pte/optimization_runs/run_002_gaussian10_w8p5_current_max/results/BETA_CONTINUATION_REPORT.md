# Run 002 beta continuation with 500 nm solid/void constraints

Status: `RUNNING_BETA_CONTINUATION_WITH_500NM_SOLID_VOID_CONSTRAINTS`

This is the continuation of the validated nominal run. It uses a stateful MMA subproblem, the existing finite nonperiodic 500 nm conic filter, explicit smooth solid/void constraints, and an independent exact 500 nm binary morphology audit. The exact audit never modifies or repairs the design.

Current beta: `4`; current global iteration: `5`; FOM: `2.232244851710e-07 A/W`; mean `4 rho (1-rho)`: `0.90335918`.

Exact solid violation fraction: `0.00076907043`; exact void violation fraction: `0`.

`RUNNING` is intentional: fully binary promotion requires the final grayness gates, zero exact solid/void violations, and a fresh thresholded-binary Maxwell/CUDA-thermal reevaluation.
