# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=20, global iteration=20.

Actual FOM: `7.370867139237e-07 A/W`. Fixed-cap solid/void constraints: `3.174993e-02` / `2.451399e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `39`, void `12`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
