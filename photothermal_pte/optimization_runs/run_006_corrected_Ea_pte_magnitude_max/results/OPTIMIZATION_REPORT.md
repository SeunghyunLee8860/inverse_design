# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_CORRECTED_EA_PTE_MAGNITUDE_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=4, global iteration=4.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `5.138006614631e-07 A/W`. Fixed-cap solid/void constraints: `3.926207e-04` / `3.795682e-05` with caps `6.000000e-04` / `2.000000e-04`.

Exact 500 nm bad cells: solid `37`, void `2`. Stage convergence: `False` (need at least 8 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
