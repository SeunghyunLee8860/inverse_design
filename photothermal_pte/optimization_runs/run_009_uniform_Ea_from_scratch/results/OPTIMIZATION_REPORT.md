# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_EXACT_UNIFORM_EA_OBJECTIVE_FIRST_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=10, global iteration=10.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `5.294773761392e-07 A/W`. Fixed-cap solid/void constraints: `3.808696e-04` / `3.935191e-05` with caps `2.000000e-03` / `2.000000e-03`.

Exact 500 nm bad cells: solid `41`, void `2`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
