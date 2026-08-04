# Device-A analytic three-position terminal-current control

Status: `COMPLETED_DEVICE_A_ANALYTIC_THREE_POSITION_CONTROL`

This is an offline control, not a paper reproduction. It uses the explicitly
assumed 12-um scalar Gaussian, paper/TMM polarization-dependent absorption,
the same expanded explicit-3D thermal operator, and the same digitized-contact
weighting potential as the Maxwell Device-A chain. The larger b-polarized TMM
absorption is an input. No current or Q rescaling was applied.

| signed s from digitized edge (um) | analytic Ia (nA) | analytic Ib (nA) | abs(Ia)/abs(Ib) |
|---:|---:|---:|---:|
| 2.0 | 5.6168909 | 8.23574181 | 0.682013962 |
| 3.0 | 3.97231004 | 5.82169261 | 0.682329058 |
| 4.0 | 2.08469798 | 3.05085453 | 0.683316089 |

Every terminal current is the full flake-cell volume integral with cell volume
included exactly once. All residual, energy-balance, and finite/nonnegative-Q
gates passed: `True`.
