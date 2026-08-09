# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_CORRECTED_EA_PTE_MAGNITUDE_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=10, global iteration=10.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `7.923495005589e-07 A/W`. Fixed-cap solid/void constraints: `3.770663e-04` / `4.347619e-05` with caps `6.000000e-04` / `2.000000e-04`.

Exact 500 nm bad cells: solid `39`, void `2`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
