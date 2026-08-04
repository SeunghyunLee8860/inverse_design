# Device-A boundary-aware fast optical mesh validation

Status: `VALIDATED_DEVICE_A_BOUNDARY_AWARE_FAST_MESH`

This checkpoint validates one `E || a` optical and conservative material-overlap
source mapping case. It does **not** run the thermal equation, weighting
potential, PTE, adjoint, AD-FD, or optimization.

## Outcome

The economical mesh is one 50 nm Cartesian rectangle with half-spans
`x=9 um, y=12 um`, nested inside the existing 100 nm-to-12 um and 200 nm
outer regions. It preserves the previous fast x lattice and the reference y
lattice. Relative to the 50 nm `x=12 um, y=12 um` reference:

- raw power change: `0.000340%`
- raw lateral-Q NRMSE: `0.009355%`
- raw full-3D-Q NRMSE: `0.009407%`
- mapped TaIrTe4 power change: `0.000146%`
- mapped lateral-Q NRMSE: `0.006713%`
- mapped full-3D-Q NRMSE: `0.006743%`
- mapped depth-Q NRMSE: `0.000009%`
- material-overlap mapping power error: exactly zero

All values are below the 0.5% spatial/power gate.

## Optical execution

- `P_Q = 2.033909790448163e-11 W`
- `P_six = 2.033343020552388e-11 W`
- six-face closure: `0.027874%`
- auto-shutoff: `9.956160e-06`
- negative-Q cells: `0`
- solver wall time: `582.170 s`
- reference wall time: `615.046 s`

The scalar Gaussian, 11 um wavelength, 8.75 um assumed waist, 50 um source
span, 60 um domain, six PML boundaries, Palik SiO2/Si, anisotropic TaIrTe4,
conformal variant 1, mesh accuracy 3, and 10 nm TaIrTe4 dz were unchanged.
No Q clipping, smoothing, gain, rescaling, tiling, or source deletion was used.

## Failed diagnostics retained

The earlier nested `x/y=9 um` candidate failed because illuminated real
material boundaries entered the 100 nm transition. The attempted 230-box
boundary-following mesh also failed: Lumerical's rectilinear mesh objects
re-anchored global x/y Yee coordinates, so the boxes were not independent
local subgrids. Both raw results and their material-overlap mappings remain in
the manifest and are not promoted.

The corrected rectangle shows that the prior 1.96% lateral and 9.28% raw-3D
differences were not evidence that optical-cell power was being lost or placed
outside TaIrTe4. They were mesh-layout differences. On the corrected common
thermal grid, source and target power agree exactly and power outside the
binary FVM TaIrTe4 support is zero.

## Limitation

The thermal material domain used for mapping is the union of binary 100 nm
lateral / 10 nm z FVM cells selected by the cell-center polygon mask. It is
not an analytic polygon cut-cell thermal geometry. This checkpoint promotes a
fast optical/mapped-Q mesh candidate only, not a final thermal or current
prediction.
