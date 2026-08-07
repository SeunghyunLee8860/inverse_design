# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=4, accepted stage iteration=8, global iteration=52.

Actual FOM: `8.796634513165e-07 A/W`. Fixed-cap solid/void constraints: `1.074881e-02` / `9.444381e-03` with cap `3.000000e-02`.

Exact 500 nm bad cells: solid `133`, void `124`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
