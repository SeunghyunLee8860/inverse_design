# Run 005 low-beta topology-exploration pilot

Status: `PAUSED_AFTER_RUN005_BOUNDED_BETA2_GPU_PILOT`

Current stage: beta=2, accepted stage iteration=5, global iteration=5.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `1.943798604096e-07 A/W`, a cumulative `+98.8505%` from the
immutable beta=2 baseline. Fixed-cap solid/void constraints are
`5.082822e-04` / `8.089554e-05` with caps `1.000000e-03` / `1.000000e-04`.

Exact 500 nm bad cells: solid `42`, void `6`. Stage convergence: `False` (need at least 8 accepted updates).

All five accepted moves were `0.01`, and every accepted update increased the
actual solver-backed FOM. This bounded run stopped at its configured target. It
does not promote beta or authorize full continuation. See
`RUN005_BOUNDED_BETA2_PILOT_AUDIT.md`.
