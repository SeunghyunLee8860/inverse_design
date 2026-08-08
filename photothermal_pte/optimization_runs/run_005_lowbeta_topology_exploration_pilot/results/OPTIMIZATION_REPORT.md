# Run 005 fully binary PTE optimization

Status: `COMPLETED_FULLY_BINARIZED_EXACT_500NM_CONSTRAINED_PTE_OPTIMIZATION`

## Final design

- Final beta: `2048`; global iteration: `46`
- Gray fraction (0.01 < rho < 0.99): `2.803154e-04`
- Mean `4*rho*(1-rho)`: `5.693115e-05`
- Exact 500 nm solid/void bad cells: `0` / `0`
- Final evaluated density contains exactly `0` and `1`
- No post-hoc binary repair was used

## Fresh binary physics validation

- Pre-threshold continuous FOM: `8.688574877725e-07 A/W`
- Thresholded-binary FOM: `8.679256315189e-07 A/W`
- Binary FOM change: `-0.107251%`
- P_Q / P_six: `5.476087197869e-14` / `5.474744186646e-14 W`
- Six-face closure: `2.452502e-04`
- Thermal residual: `8.946627e-11`
- Thermal energy balance: `6.401750e-12`
- Solver path: fresh GPU Maxwell plus CUDA thermal/PTE; no CPU solver fallback

The small binary loss above is measured by a fresh solver evaluation, not inferred from a linearized gradient.

## Final binary field visualization

The final density is displayed with the explicit convention `1 = SiO2 design
material` and `0 = air/void`.  Read-only postprocessing of the immutable final
binary artifact publishes the volumetric/depth-integrated Q, temperature rise,
strict-centered temperature gradients, local PTE contribution, and the
full-footprint integrated current.  No Maxwell, thermal, adjoint, or
optimization solve was rerun.

- [field report](FINAL_BINARY_FIELDS_REPORT.md)
- [machine-readable field summary](final_binary_fields_summary.json)
- [field metrics CSV](final_binary_field_metrics.csv)
- [binary structure](../plots/final_binary_structure_1_material_0_void.png)
- [Q, temperature, gradient, and current](../plots/final_binary_Q_temperature_gradient_current.png)
- [Q and temperature cross sections](../plots/final_binary_Q_temperature_cross_sections.png)
- [PTE current decomposition](../plots/final_binary_pte_current_breakdown.png)

The stored scalar current is reproduced to `4.0133e-16` relative error.  The
displayed gradient/current maps are NaN wherever any one of `-x,+x,-y,+y`
TaIrTe4 neighbours is unavailable; the reported total current remains the
validated full-footprint boundary-aware operator.

## Final binary orthogonal-polarization evaluation

The original final-binary forward used source-x polarization.  The frozen
optical metadata is `x=b, y=a`, so that result is `E || b`.  A separate
source-y (`90 deg`) GPU Maxwell plus CUDA thermal/PTE evaluation was completed
for `E || a` using the identical exact-binary structure, geometry, mesh,
incident power, conservative remap, and thermal operator.  No polarization
matching or heat-source rescaling was used.

- [two-polarization report](FINAL_BINARY_POLARIZATION_COMPARISON_REPORT.md)
- [machine-readable comparison](final_binary_polarization_comparison_summary.json)
- [comparison CSV](final_binary_polarization_comparison_metrics.csv)
- [common-scale Q/T/gradient/current maps](../plots/final_binary_both_polarizations_Q_T_gradient_current.png)
- [scalar and component metrics](../plots/final_binary_both_polarizations_metrics.png)

Both numerical pipelines pass their optical and thermal gates.  The current
ratio from the existing operator is `I(E||a)/I(E||b) = 1.682352`.  Physical
crystallographic interpretation remains blocked by
`UNRESOLVED_AXIS_METADATA_MISMATCH_XB_YA_VS_THERMAL_PTE_XA_YB`: optical
metadata maps solver x/y to b/a, whereas the immutable thermal/PTE operator
uses a coefficients on solver x and b coefficients on solver y.  No silent
axis or coefficient swap was made.
