# FDTDX spatial-Q material-overlap thermal remap

Status: **VALIDATED_FDTDX_SPATIAL_Q_CONSERVATIVE_MATERIAL_OVERLAP_REMAP**

For every Au, TaIrTe4, and SiO2 `Qx/Qy/Qz` Yee sample, the source power is
first formed as `p=Q*V_dual`. Its component-specific dual-cell bounds are then
intersected with the actual absorbing-material thermal cells. The already
calculated source-cell power is distributed only among those overlaps.

This is not nearest-cell projection and it does not delete a boundary sample,
assign air absorption to TaIrTe4, or apply a global gain. The per-cell overlap
weights sum to one. A boundary-crossing dual cell is handled by its exact
material overlap rather than by an array-index convention.

| metric | value |
|---|---:|
| source total power | 2.477973536898e-13 W |
| remapped total power | 2.477973536898e-13 W |
| total conservation error | 0.000000000000% |
| worst component conservation error | 0.000000000000% |
| worst transpose dot-test error | 7.467e-15 |
| worst overlap-column error | 0.000e+00 |
| largest boundary-dual redistributed power fraction | 27.645562% |

The last quantity is diagnostic: it reports power whose native Yee dual
support crosses a material boundary and is therefore conservatively placed
inside the actual absorbing material. It is not discarded power and is not a
physical air heat source.

The output thermal-Q NPZ is not committed to Git. This checkpoint validates
only the remap and its transpose. No temperature, PTE current, combined
Maxwell gradient, or optimization result is claimed yet.
