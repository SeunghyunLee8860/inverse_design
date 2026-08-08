# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=32, accepted stage iteration=1, global iteration=37.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `8.709825370155e-07 A/W`. Fixed-cap solid/void constraints: `4.136653e-04` / `1.331585e-04` with caps `3.837575e-04` / `1.278726e-04`.

Exact 500 nm bad cells: solid `36`, void `5`. Stage convergence: `False` (need at least 6 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
