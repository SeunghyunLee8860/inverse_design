# Run 005 final-binary field maps

Status: `PUBLISHED_FINAL_BINARY_Q_T_GRADIENT_CURRENT_FIELDS_WITH_AXIS_AUDIT`. This is read-only postprocessing of the fresh final-binary GPU Maxwell/CUDA thermal result; no solver was rerun.

## Binary structure semantics

- `1`: SiO2 design material is present.
- `0`: air/void; design material is absent.
- Exact stored values: `[0, 1]`; material fraction: `0.416728360`.

## Fields and PTE current

- mapped Q: `5.425780399959e-14 W`; reintegration error: `0.000e+00`.
- maximum temperature rise: `2.228708606113e-09 K`.
- full-footprint current: `1.199729281050e-19 A`; stored objective: `1.199729281050e-19 A`.
- objective reintegration error: `4.013e-16`.
- x/y current terms: `7.460292534310e-20` / `4.537000276195e-20 A`.
- strict-centered displayed-map integral: `1.201934473878e-19 A` (`1.001838075` of the boundary-aware full operator).

The gradient/current maps require all four `-x,+x,-y,+y` TaIrTe4 neighbours. Every cell missing any neighbour is stored and displayed as `NaN`. The full scalar current remains the validated full-footprint operator, including its second-order one-sided perimeter stencil.

## Axis audit

The optical metadata says `x=b, y=a`, while the immutable Run005 thermal/PTE code uses the `a` coefficients on solver x and the `b` coefficients on solver y. The plots therefore use literal `solver x/y` labels and reproduce the existing result without silently swapping axes. Physical crystallographic interpretation remains `UNRESOLVED_AXIS_METADATA_MISMATCH_XB_YA_VS_THERMAL_PTE_XA_YB`.
