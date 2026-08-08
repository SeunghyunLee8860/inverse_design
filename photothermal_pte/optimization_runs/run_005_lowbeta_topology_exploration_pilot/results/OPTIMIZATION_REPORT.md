# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=8, accepted stage iteration=7, global iteration=30.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `8.266093247394e-07 A/W`. Fixed-cap solid/void constraints: `4.043266e-04` / `1.023285e-04` with caps `4.711417e-04` / `1.063306e-04`.

Exact 500 nm bad cells: solid `35`, void `4`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
