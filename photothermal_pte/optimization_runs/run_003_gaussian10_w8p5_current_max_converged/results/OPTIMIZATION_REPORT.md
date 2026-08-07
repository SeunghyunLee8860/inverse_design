# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=8, accepted stage iteration=16, global iteration=72.

Actual FOM: `8.844470870112e-07 A/W`. Fixed-cap solid/void constraints: `1.233811e-02` / `1.232582e-02` with cap `2.000000e-02`.

Exact 500 nm bad cells: solid `323`, void `407`. Stage convergence: `True` (all four-update FOM/density plateau gates pass).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
