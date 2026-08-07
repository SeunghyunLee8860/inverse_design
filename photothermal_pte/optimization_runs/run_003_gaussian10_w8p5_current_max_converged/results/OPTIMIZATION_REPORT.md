# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=16, accepted stage iteration=10, global iteration=82.

Actual FOM: `8.858463095610e-07 A/W`. Fixed-cap solid/void constraints: `8.002091e-03` / `8.005008e-03` with cap `8.000000e-03`.

Exact 500 nm bad cells: solid `375`, void `498`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
