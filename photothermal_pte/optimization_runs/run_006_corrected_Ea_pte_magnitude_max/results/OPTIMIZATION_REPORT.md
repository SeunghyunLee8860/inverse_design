# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_CORRECTED_EA_PTE_MAGNITUDE_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=11, global iteration=11.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `8.357445667105e-07 A/W`. Fixed-cap solid/void constraints: `3.784002e-04` / `4.355191e-05` with caps `6.000000e-04` / `2.000000e-04`.

Exact 500 nm bad cells: solid `40`, void `3`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
