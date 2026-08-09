# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_EXACT_UNIFORM_EA_OBJECTIVE_FIRST_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=6, global iteration=6.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `3.233758371604e-07 A/W`. Fixed-cap solid/void constraints: `4.317726e-04` / `3.719188e-05` with caps `2.000000e-03` / `2.000000e-03`.

Exact 500 nm bad cells: solid `39`, void `2`. Stage convergence: `False` (need at least 8 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
