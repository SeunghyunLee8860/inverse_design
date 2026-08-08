# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=12, global iteration=12.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `2.736664327586e-07 A/W`. Fixed-cap solid/void constraints: `4.174229e-04` / `2.104169e-04` with caps `6.000000e-04` / `2.000000e-04`.

Exact 500 nm bad cells: solid `38`, void `4`. Stage convergence: `False` (fixed solid/void inequalities are not feasible).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
