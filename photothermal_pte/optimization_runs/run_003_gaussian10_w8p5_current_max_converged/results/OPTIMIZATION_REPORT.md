# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=17, global iteration=17.

Actual FOM: `6.440182520819e-07 A/W`. Fixed-cap solid/void constraints: `3.495886e-02` / `3.179159e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `38`, void `3`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
