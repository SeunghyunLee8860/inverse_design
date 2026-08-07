# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=21, global iteration=21.

Actual FOM: `7.654271195321e-07 A/W`. Fixed-cap solid/void constraints: `3.039828e-02` / `2.185499e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `40`, void `8`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
