# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=32, accepted stage iteration=7, global iteration=90.

Constraint contract: `soft_disk_opening_500nm_v3_exact_nonincrease`.

Actual FOM: `8.851483902439e-07 A/W`. Fixed-cap solid/void constraints: `9.185506e-04` / `2.192341e-03` with cap `2.000000e-03`.

Exact 500 nm bad cells: solid `107`, void `280`. Stage convergence: `False` (need at least 8 accepted updates).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
