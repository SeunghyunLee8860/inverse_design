# Run 005 fully binary PTE optimization continuation

Status: `RUNNING_CORRECTED_EA_OBJECTIVE_FIRST_CONSTRAINED_OPTIMIZATION`

Current stage: beta=2, accepted stage iteration=3, global iteration=3.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `4.017466253954e-07 A/W`. Fixed-cap solid/void constraints: `7.518716e-04` / `3.409189e-05` with caps `1.490220e-03` / `3.204288e-05`.

Exact 500 nm bad cells: solid `40`, void `1`. Stage convergence: `False` (need at least 8 accepted updates).

Beta advances only after the stage FOM/density plateau. The run is not complete until the projected-binary and exact 500 nm gates pass and a fresh thresholded-binary GPU/CUDA evaluation succeeds.
