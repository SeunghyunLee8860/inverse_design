# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=19, global iteration=19.

Actual FOM: `7.073958963923e-07 A/W`. Fixed-cap solid/void constraints: `3.293626e-02` / `2.708914e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `42`, void `10`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
