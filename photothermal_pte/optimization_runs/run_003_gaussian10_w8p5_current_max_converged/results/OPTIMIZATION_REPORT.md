# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=40, global iteration=40.

Actual FOM: `8.754838728372e-07 A/W`. Fixed-cap solid/void constraints: `1.482675e-02` / `7.675561e-03` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `47`, void `115`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.

The original 0.02-move trajectory stopped fail-closed after accepted beta=2
iteration 40.  Its FOM plateau passed, but density RMS/maximum changes were
`0.4677%` / `2.6257%`, above the unchanged `0.25%` / `1.5%` gates.  Recovery
therefore resumes from the exact accepted state with a deterministic adaptive
move ceiling; no convergence threshold, constraint cap, gradient, or FOM is
rescaled.
