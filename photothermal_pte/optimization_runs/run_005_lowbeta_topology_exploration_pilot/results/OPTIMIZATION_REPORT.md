# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=16, accepted stage iteration=3, global iteration=36.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `8.687278967650e-07 A/W`. Fixed-cap solid/void constraints: `4.128812e-04` / `1.295483e-04` with caps `4.486311e-04` / `1.251122e-04`.

Exact 500 nm bad cells: solid `39`, void `18`. Stage convergence: `False` (need at least 6 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
