# Run 003 convergence-based constrained PTE optimization

Status: `HALTED_PREMATURE_BETA2_SATURATION_AND_LATE_CONSTRAINT_REPAIR`

The run was deliberately halted after accepted checkpoint `g095`. The partial
`g096` evaluation was interrupted during the adjoint and is not accepted.

Actual FOM at g095: `8.852107939055e-07 A/W`. Fixed-cap solid/void constraints:
`9.230433e-04` / `2.014664e-03` with cap `2.000000e-03`.

Exact 500 nm bad cells: solid `95`, void `260`. The run is neither converged nor
manufacturing-feasible. See `RUN003_CONTINUATION_PATHOLOGY_AUDIT.md` for the
beta-stage evidence and restart rationale.
