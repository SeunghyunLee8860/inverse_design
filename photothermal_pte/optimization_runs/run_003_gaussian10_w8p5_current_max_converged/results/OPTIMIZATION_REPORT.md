# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=32, accepted stage iteration=3, global iteration=86.

Constraint contract: `soft_disk_opening_500nm_v2`.

Actual FOM: `8.854273028731e-07 A/W`. Fixed-cap solid/void constraints: `1.140732e-03` / `3.533050e-03` with cap `2.000000e-03`.

Exact 500 nm bad cells: solid `139`, void `478`. Stage convergence: `False` (need at least 8 accepted updates).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
