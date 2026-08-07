# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=8, accepted stage iteration=14, global iteration=70.

Actual FOM: `8.843610900867e-07 A/W`. Fixed-cap solid/void constraints: `1.236632e-02` / `1.235323e-02` with cap `2.000000e-02`.

Exact 500 nm bad cells: solid `345`, void `423`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
