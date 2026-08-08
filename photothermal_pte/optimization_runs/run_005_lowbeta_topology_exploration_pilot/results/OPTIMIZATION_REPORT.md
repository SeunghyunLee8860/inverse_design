# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=2, accepted stage iteration=9, global iteration=9.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `2.239652262958e-07 A/W`. Fixed-cap solid/void constraints: `4.527493e-04` / `1.114862e-04` with caps `5.500000e-04` / `1.110000e-04`.

Exact 500 nm bad cells: solid `44`, void `8`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
