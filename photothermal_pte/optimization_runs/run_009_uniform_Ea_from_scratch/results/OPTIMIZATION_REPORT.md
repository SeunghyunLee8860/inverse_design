# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_EXACT_UNIFORM_EA_OBJECTIVE_FIRST_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=27, global iteration=27.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `1.231391525802e-06 A/W`. Fixed-cap solid/void constraints: `3.835882e-04` / `1.533543e-04` with caps `2.000000e-03` / `2.000000e-03`.

Exact 500 nm bad cells: solid `39`, void `0`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
