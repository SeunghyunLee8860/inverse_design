# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=33, global iteration=33.

Actual FOM: `8.726139732634e-07 A/W`. Fixed-cap solid/void constraints: `1.745343e-02` / `9.229008e-03` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `44`, void `12`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
