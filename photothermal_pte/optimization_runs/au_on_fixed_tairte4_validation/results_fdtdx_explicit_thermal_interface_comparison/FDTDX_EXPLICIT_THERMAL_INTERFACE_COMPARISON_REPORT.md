# FDTDX explicit thermal interface-scenario comparison

Status: **VALIDATED_FDTDX_EXPLICIT_THERMAL_INTERFACE_SCENARIO_COMPARISON**

Both cases use the exact same literal, unscaled spatial Maxwell source
`P_Q=2.477953932988e-13 W`, material-overlap remap, geometry,
thermal mesh, boundary conditions, and Au-aware electrical operator. Only
the named TaIrTe4/SiO2 conductance changes:

- thermally grown: `G=7.370000e+06 W/(m2 K)`
- evaporated: `G=7.370000e+04 W/(m2 K)`

| metric | thermally grown | evaporated | evaporated / grown |
|---|---:|---:|---:|
| Tmax rise (K) | 5.577354020816e-10 | 9.156692620445e-09 | 16.417628 |
| Au volume-average rise (K) | 4.248655193882e-10 | 8.769695649086e-09 | 20.641109 |
| TaIrTe4 volume-average rise (K) | 2.383989270051e-10 | 6.686495170435e-09 | 28.047505 |
| PTE current (A) | 2.684875438916e-18 | 3.371736983315e-17 | 12.558262 |

All thermal residuals are below `1e-8`, thermal energy-balance errors are
below `1%`, and electrical residual/balance gates pass. The result is a
physical-parameter sensitivity, not a numerical convergence error.

These currents retain the literal FDTDX source normalization and are not
experimental predictions. `G_Au/TaIrTe4` is an Au/MoS2 analogue, the
electrical contact is a named numerical scenario, and the Si endpoint remains
the explicitly documented lossless diagnostic. This is a forward certificate
only; combined AD--FD is still required before optimization.
