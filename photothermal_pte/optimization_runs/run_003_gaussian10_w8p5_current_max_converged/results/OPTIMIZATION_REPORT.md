# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=44, global iteration=44.

Actual FOM: `8.760986275056e-07 A/W`. Fixed-cap solid/void constraints: `1.341615e-02` / `1.008423e-02` with cap `4.000000e-02`.

Exact 500 nm bad cells: solid `47`, void `139`. Stage convergence: `True` (all four-update FOM/density plateau gates pass).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
