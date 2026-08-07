# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=13, global iteration=13.

Actual FOM: `5.024682230199e-07 A/W`. Fixed-cap solid/void constraints: `3.833761e-02` / `3.820679e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `46`, void `2`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
