# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=128, accepted stage iteration=3, global iteration=45.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `8.688574877725e-07 A/W`. Fixed-cap solid/void constraints: `1.572527e-04` / `9.123302e-05` with caps `3.090430e-04` / `1.025576e-04`.

Exact 500 nm bad cells: solid `0`, void `0`. Stage convergence: `False` (need at least 6 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
