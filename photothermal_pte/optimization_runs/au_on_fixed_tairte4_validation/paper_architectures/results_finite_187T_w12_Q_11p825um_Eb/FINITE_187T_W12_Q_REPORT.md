# Finite 187-inverse-T Gaussian volumetric-Q certificate

Status: `VALIDATED_FINITE_187T_W12_VOLUMETRIC_Q`

- TaIrTe4 is the active anisotropic material (`x=b`, `y=a`, `z=c=b` closure).
- 11 x 17 finite inverse-T elements; no periodic/Bloch boundary.
- scalar Gaussian, physical target `w0=12 um`, `lambda=11.825 um`, `E||b`.
- 60 x 60 um lateral FDTD domain; six PML, 24 layers.
- P_Q(native Yee) = `2.098538646347e-14 W`.
- P_six = `2.099793268813e-14 W`.
- six-face closure = `0.059750%`.
- Qx/Qy/Qz = `1.185230526009e-14`, `4.067475551430e-15`, `5.065605651949e-15 W`.
- hotspot = `(0.600, -0.250, 0.095) um`.

No Q clipping, smoothing, gain, rescaling, or post-solve tiling was used. The common-grid total-Q image is a collocated visualization; component-wise native Yee integration is the power authority. Thermal, weighting potential, PTE, adjoint, and optimization were not run in this certificate.
