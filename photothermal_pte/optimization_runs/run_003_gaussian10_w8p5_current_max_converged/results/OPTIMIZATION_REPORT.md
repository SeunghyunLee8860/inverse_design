# Run 003 convergence-based constrained PTE optimization

Status: `RUNNING_CONVERGENCE_BASED_CONSTRAINED_BETA_CONTINUATION`

Current stage: beta=16, accepted stage iteration=11, global iteration=83.

Actual FOM: `8.858801860481e-07 A/W`. Fixed-cap solid/void constraints: `8.002189e-03` / `8.007731e-03` with cap `8.000000e-03`.

Exact 500 nm bad cells: solid `360`, void `501`. Stage convergence: `True` (all four-update FOM/density plateau gates pass).

Beta is promoted only after fixed-inequality feasibility and a four-update FOM/design plateau; it is never promoted after one nominal update.
