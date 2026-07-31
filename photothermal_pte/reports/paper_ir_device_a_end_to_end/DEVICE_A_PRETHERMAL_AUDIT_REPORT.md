# Device A pre-thermal audit

Status: `COMPLETED_DEVICE_A_PRETHERMAL_AUDITS_WITH_INTERFACE_BLOCKER`

This checkpoint does not contain a thermal, PTE, adjoint, or optimization run.

## Weighting potential

The frozen Figure-2 digitized contact segments were used with code axes
`x=b`, `y=a`. Both 100 nm and 50 nm grids pass the contact, finite-field,
potential-range, and residual gates.

| Metric | 100 nm | 50 nm | relative change |
|---|---:|---:|---:|
| p99 $|\nabla\psi|$ [1/m] | 5.795245581e+04 | 5.809252661e+04 | 0.2411% |
| raw max $|\nabla\psi|$ [1/m] | 1.213983535e+05 | 1.492614715e+05 | 18.6673% |
| residual | 2.639e-15 | 3.702e-15 | -- |

The robust p99 metric is stable; the one-cell raw maximum is not a production
gate.

## E-parallel-a material-Q support

| Partition | Power at unit central intensity [W] |
|---|---:|
| TaIrTe4 exact support | 3.152817129357e-11 |
| Ti exact support | 1.761324156040e-13 |
| Au exact support | 5.732552358198e-21 |
| conformal/interface ambiguous | 1.948413371833e-13 |
| common-grid total | 3.189914505209e-11 |

Partition closure is 0.000e+00; the common-grid
versus native-component total difference is
0.0081%. The ambiguous fraction is
0.6108%. It is retained rather
than clipped, rescaled, deleted, or silently assigned to a bulk material.

## Fail-closed consequence

The current thermal mapper projects every optical-Q sample into TaIrTe4. That
path must not be used for this electrode-bearing Device-A artifact. A declared
conservative component/material interface remap and an explicit metal thermal
scenario are required before a full terminal-current result can be promoted.
