# Selected production thermal mapping

Status: `VALIDATED_SELECTED_PRODUCTION_THERMAL_MAPPING`

The selected optical design uses 373×373 nodal physical-density values on
`[-9.3,9.3] µm` at 50 nm. The explicit thermal core has 186×186 cells over
the same support at 100 nm. Each thermal-cell density is the exact area
average of the bilinear nodal field, with one-dimensional weights
`[1,2,1]/4`; the transpose applies those weights in reverse.

- Constant-preservation error: `0.000e+00`.
- Opposite-edge wrap error: `0.000e+00`.
- Bilinear area-integral error: `4.483e-16`.
- Worst transpose error: `3.786e-16`.
- Worst mapping-only FD error: `1.730e-13`.

The selected rho=0.5 GPU `Qx,Qy,Qz` artifact was partitioned by literal
native dual-cell/material intersection using the exact ±9.3 µm design
support. No full cut-cell power was forced into TaIrTe4, SiO2, or the design.
The physical thermal-source power is
`7.132301206389e-14 W`, or
`98.792360%`
of full optical `P_Q`; the remaining air/interface and artificial-background
fractions are reported, not relocated.

Conservative deposition onto the 362×362×91 explicit 3D thermal grid has
relative total-power error `3.539e-16`,
worst component/material error
`9.190e-16`, and zero nonzero
source cells outside their own material. There was no clipping, smoothing,
gain, rescaling, nearest-material relocation, Maxwell rerun, thermal solve,
adjoint solve, or optimization iteration.

This checkpoint does not certify a thermal gray-law exponent, selected-grid
combined AD-FD, exact-binary DRC, or optimization.
