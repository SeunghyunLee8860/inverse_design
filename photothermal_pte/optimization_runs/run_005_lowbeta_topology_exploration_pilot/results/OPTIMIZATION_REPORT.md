# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=64, accepted stage iteration=1, global iteration=41.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `8.706067564665e-07 A/W`. Fixed-cap solid/void constraints: `3.862192e-04` / `1.313409e-04` with caps `3.395149e-04` / `1.155719e-04`.

Exact 500 nm bad cells: solid `32`, void `6`. Stage convergence: `False` (need at least 6 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
