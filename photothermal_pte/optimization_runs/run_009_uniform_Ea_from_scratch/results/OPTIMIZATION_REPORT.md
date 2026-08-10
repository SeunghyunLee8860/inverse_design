# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_EXACT_UNIFORM_EA_OBJECTIVE_FIRST_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=16, global iteration=16.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `8.133895942118e-07 A/W`. Fixed-cap solid/void constraints: `3.797847e-04` / `4.507194e-05` with caps `2.000000e-03` / `2.000000e-03`.

Exact 500 nm bad cells: solid `40`, void `2`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
