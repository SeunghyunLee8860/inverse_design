# Thermal/PTE to native-Yee pullback report

**Status: `VALIDATED_THERMAL_PTE_TO_NATIVE_YEE_PULLBACK`**

This certificate connects the fixed-K thermal/PTE adjoint to the actual
native FieldRegion absorption samples from the periodic inverse-design
geometry. No Maxwell solve was rerun; the completed baseline FSP was read.

## Forward chain

For Ex and Ey, the native absorption densities are first moved to the common
optical nodal grid by a sparse mass-conservative operator `C_c`. The shifted
component axis and periodic seam are handled explicitly. Ez is exactly absent
because fitted `Im(epsilon_z)=0`.

The common nodal trapezoid cells are moved to the 24 x 24 x 2 thermal flake
cells by the overlap-volume density operator `R_ot`. No crop, padding, tiling,
gain, clipping, smoothing, or total-power rescaling is used.

The power is identical at every stage:

- native Yee: `8.79339492981549e-12 W/(W/m2)`;
- common optical grid: `8.79339492981549e-12 W/(W/m2)`;
- thermal flake grid: `8.79339492981549e-12 W/(W/m2)`.

The fixed-K FVM uses periodic x/y, bottom `DeltaT=0`, adiabatic top,
TaIrTe4 `diag(14.4,3.8,1.0) W/(m K)`, `G_top=7.37e6 W/(m2 K)`, and
`G_bottom=1.1e9 W/(m2 K)`.

## Reverse chain

The discrete reverse path is

`c_T -> K_T^-T -> M_V^T -> R_ot^T -> C_c^T -> native Yee`.

If `lambda_T` solves `K_T^T lambda_T=c_T`, then

`w_common = R_ot^T M_V^T lambda_T`

and the coefficient on each native component density is

`a_native,c = C_c^T w_common`.

The absorption evaluator expects a weight inside a quadrature sum, so it uses

`w_native,c = a_native,c / V_native,c`.

This division is essential: it ensures the native component volume appears
exactly once when constructing
`q_E,c = alpha_c V_native,c w_native,c E_c`.

## Gates

| Gate | Relative error | Result |
|---|---:|---|
| native-to-common power | 0 | PASS |
| common-to-thermal power | 0 | PASS |
| worst transpose dot test | 9.99496322622979e-16 | PASS |
| temperature-forward vs native-weight objective | 4.597760533396695e-12 | PASS |
| thermal linear residual | 7.730579602571028e-12 | PASS |
| thermal energy balance | 2.057746302084621e-14 | PASS |

For the actual baseline optical source, the numerical finite-local-mask
surrogate gives:

- `DeltaT_max/I_inc = 1.0736304860450285e-7 K/(W/m2)`;
- `F_local/I_inc = -2.9429080169821196e-21 A m/(W/m2)`.

These are not terminal-current or final experimental predictions. The finite
local readout mask and fixed thermal material scenario retain the two physical
blockers documented in the main contract.
