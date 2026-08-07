# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=8, accepted stage iteration=13, global iteration=69.

Actual FOM: `8.843174964193e-07 A/W`. Fixed-cap solid/void constraints: `1.245505e-02` / `1.244206e-02` with cap `2.000000e-02`.

Exact 500 nm bad cells: solid `349`, void `422`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
