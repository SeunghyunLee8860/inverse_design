# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=7, global iteration=7.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `2.140792090727e-07 A/W`. Fixed-cap solid/void constraints: `4.677682e-04` / `9.971852e-05` with caps `1.000000e-03` / `1.000000e-04`.

Exact 500 nm bad cells: solid `45`, void `5`. Stage convergence: `False` (need at least 8 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
