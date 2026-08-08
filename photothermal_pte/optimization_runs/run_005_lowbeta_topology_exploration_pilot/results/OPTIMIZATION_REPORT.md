# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=8, global iteration=8.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `2.190194079920e-07 A/W`. Fixed-cap solid/void constraints: `4.599373e-04` / `1.065530e-04` with caps `5.500000e-04` / `1.110000e-04`.

Exact 500 nm bad cells: solid `43`, void `2`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
