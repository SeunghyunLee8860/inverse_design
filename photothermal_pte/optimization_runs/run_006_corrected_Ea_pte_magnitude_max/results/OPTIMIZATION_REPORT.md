# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_CORRECTED_EA_PTE_MAGNITUDE_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=6, global iteration=6.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `6.101401170127e-07 A/W`. Fixed-cap solid/void constraints: `3.723312e-04` / `4.360152e-05` with caps `6.000000e-04` / `2.000000e-04`.

Exact 500 nm bad cells: solid `38`, void `3`. Stage convergence: `False` (need at least 8 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
