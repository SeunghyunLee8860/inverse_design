# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=4, accepted stage iteration=2, global iteration=15.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `4.460869800932e-07 A/W`. Fixed-cap solid/void constraints: `4.154969e-04` / `2.557717e-04` with caps `4.894159e-04` / `2.601193e-04`.

Exact 500 nm bad cells: solid `42`, void `21`. Stage convergence: `False` (need at least 6 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
