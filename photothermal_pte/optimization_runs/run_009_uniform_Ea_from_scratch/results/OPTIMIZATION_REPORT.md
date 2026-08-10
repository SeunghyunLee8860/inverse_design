# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_EXACT_UNIFORM_EA_OBJECTIVE_FIRST_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=29, global iteration=29.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `1.274709896937e-06 A/W`. Fixed-cap solid/void constraints: `3.847218e-04` / `1.810878e-04` with caps `2.000000e-03` / `2.000000e-03`.

Exact 500 nm bad cells: solid `43`, void `24`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
