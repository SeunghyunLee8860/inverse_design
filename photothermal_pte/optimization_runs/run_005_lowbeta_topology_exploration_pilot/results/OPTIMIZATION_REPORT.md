# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_RUN005_FULL_BINARY_BETA_CONTINUATION`

Current stage: beta=4, accepted stage iteration=8, global iteration=21.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `5.200726601703e-07 A/W`. Fixed-cap solid/void constraints: `4.140725e-04` / `1.074882e-04` with caps `4.894159e-04` / `2.601193e-04`.

Exact 500 nm bad cells: solid `41`, void `5`. Stage convergence: `False` (recent FOM/density changes have not plateaued).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
