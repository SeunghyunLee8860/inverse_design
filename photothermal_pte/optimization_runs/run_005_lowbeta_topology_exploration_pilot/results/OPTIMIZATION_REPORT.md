# Run 005 low-beta topology-exploration pilot

Status: `PAUSED_AFTER_RUN005_ONE_POINT_GPU_PILOT`

Current stage: beta=2, accepted stage iteration=1, global iteration=1.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v5`.

Actual FOM: `1.167963771133e-07 A/W`. Fixed-cap solid/void constraints: `9.047577e-04` / `3.816873e-05` with caps `1.260000e-03` / `5.000000e-05`.

Exact 500 nm bad cells: solid `44`, void `2`. Stage convergence: `False` (need at least 8 accepted updates).

Relative to the immutable baseline, the fresh GPU/CUDA point increased FOM by
`19.4826487%`, reduced the smooth solid penalty by `24.1086892%`, and changed
the smooth void penalty by `+48.8970892%` while retaining only `76.3375%` of its
pilot cap. Diagnostic exact bad cells decreased from `158` to `46`.

All physics gates passed: optical closure `4.73835e-6`, Q-remap error
`1.78691e-16`, thermal residual `9.74355e-11`, thermal energy imbalance
`1.24780e-12`, forward auto-shutoff `9.19675e-8`, and adjoint auto-shutoff
`8.42705e-8`.

This bounded run performed one move=0.01 GPU-backed beta=2 update and paused. It
does not promote beta or authorize full continuation. A 3--5 update pilot is the
next decision gate.
