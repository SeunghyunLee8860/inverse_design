# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=4, accepted stage iteration=9, global iteration=53.

Actual FOM: `8.797373215808e-07 A/W`. Fixed-cap solid/void constraints: `1.087834e-02` / `9.610694e-03` with cap `3.000000e-02`.

Exact 500 nm bad cells: solid `161`, void `134`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
