# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=12, global iteration=12.

Actual FOM: `4.645303295594e-07 A/W`. Fixed-cap solid/void constraints: `3.899417e-02` / `3.907408e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `41`, void `4`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
