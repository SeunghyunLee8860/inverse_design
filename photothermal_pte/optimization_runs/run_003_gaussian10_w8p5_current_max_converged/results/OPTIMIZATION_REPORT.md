# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=4, accepted stage iteration=12, global iteration=56.

Actual FOM: `8.799517351576e-07 A/W`. Fixed-cap solid/void constraints: `1.056800e-02` / `9.434536e-03` with cap `3.000000e-02`.

Exact 500 nm bad cells: solid `219`, void `138`. Stage convergence: `True` (all four-update FOM/density plateau gates pass).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
