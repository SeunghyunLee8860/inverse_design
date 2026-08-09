# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_CORRECTED_EA_PTE_MAGNITUDE_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=1, global iteration=1.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `3.636278564558e-07 A/W`. Fixed-cap solid/void constraints: `7.024144e-04` / `3.283405e-05` with caps `6.000000e-04` / `2.000000e-04`.

Exact 500 nm bad cells: solid `24`, void `0`. Stage convergence: `False` (need at least 8 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
