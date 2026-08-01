# Device-A analytic three-position terminal-current control

Status: `COMPLETED_DEVICE_A_ANALYTIC_THREE_POSITION_CONTROL_60UM`

This is an offline control, not a paper reproduction. It uses the explicitly
assumed 12-um scalar Gaussian, paper/TMM polarization-dependent absorption,
the same expanded explicit-3D thermal operator, and the same digitized-contact
weighting potential as the Maxwell Device-A chain. The larger b-polarized TMM
absorption is an input. No current or Q rescaling was applied.

The lateral thermal domain is `60.0 um`; only the
60-um run matches the immutable s0 Device-A thermal artifact and is eligible
for the promoted Maxwell--analytic comparison.

| signed s from digitized edge (um) | analytic Ia (nA) | analytic Ib (nA) | abs(Ia)/abs(Ib) |
|---:|---:|---:|---:|
| 2.0 | 5.61202371 | 8.22849464 | 0.682023135 |
| 3.0 | 3.9643961 | 5.80990609 | 0.68235115 |
| 4.0 | 2.073151 | 3.03365507 | 0.683383891 |

Every terminal current is the full flake-cell volume integral with cell volume
included exactly once. All residual, energy-balance, and finite/nonnegative-Q
gates passed: `True`.
