# Run 005 low-beta topology-exploration pilot

Status: `READY_TO_RESUME_RUN005_FROM_ACCEPTED_G003`

Current stage: beta=2, accepted stage iteration=3, global iteration=3.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `1.553223792767e-07 A/W`, a cumulative `+58.8947%` from the
immutable baseline. The accepted g003 values were `6.430949e-04` solid and
`5.244080e-05` void. The next two authorized beta=2 points use the fixed loose
exploration envelope `1.000000e-03` / `1.000000e-04`.

Exact 500 nm bad cells: solid `40`, void `6`. Stage convergence: `False` (need at least 8 accepted updates).

All g004 move trials under the prior cap were rejected offline with zero solver
runs. This bounded run stops at five accepted updates. It does not promote beta
or authorize full continuation.
