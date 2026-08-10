# Run 004 joint FOM and 500 nm disk-constrained PTE optimization

Status: `PAUSED_AFTER_ONE_POINT_JOINT_PROGRESS_AUDIT`

Current stage: beta=2, accepted stage iteration=1, global iteration=1.

Constraint contract: `soft_disk_opening_500nm_from_iteration_zero_v4`.

Actual FOM: `1.024991877950e-07 A/W`. Fixed-cap solid/void constraints:
`1.092146e-03 / 1.250000e-03` and `2.976605e-05 / 3.000000e-05`.

Exact 500 nm bad cells: solid `96`, void `1`. Stage convergence: `False` (need at least 8 accepted updates).

The first accepted step improved FOM by 4.8566% and reduced exact bad-cell
counts from 158/0 to 96/1. The next proposal required move=0.0003125 to remain
inside the nearly saturated void cap, so it was interrupted and is not an
accepted checkpoint. The run is paused before that behavior can become another
long constraint-repair trajectory. See `RUN004_ONE_POINT_PILOT_AUDIT.md`.
