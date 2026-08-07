# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=35, global iteration=35.

Actual FOM: `8.737676795005e-07 A/W`. Fixed-cap solid/void constraints: `1.651743e-02` / `9.210242e-03` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `45`, void `17`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
