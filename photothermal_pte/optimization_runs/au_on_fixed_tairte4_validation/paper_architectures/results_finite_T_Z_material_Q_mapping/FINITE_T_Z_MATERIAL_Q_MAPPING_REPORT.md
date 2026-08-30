# Finite T/Z component-material Q mapping

Status: **VALIDATED_FINITE_T_Z_COMPONENT_MATERIAL_OVERLAP_Q_MAPPING**

Each native Yee component is paired with its own coordinates. Cell power is split by exact material cut-cell volume times component Im(epsilon), then transferred only through that material overlap to the explicit thermal grid.
This loss-participation diagnostic is not an occupancy field. No complete boundary cell is forced into TaIrTe4 or Au.

| case | TaIrTe4 | top Au | mirror Au | SiO2 | Si | mapping error |
|---|---:|---:|---:|---:|---:|---:|
| T_Ea_Au_on | 5.9479 fW | 0.0066 fW | 0.0313 fW | 0.0000 fW | 0.0000 fW | 0.000e+00 |
| T_Eb_Au_on | 5.6696 fW | 0.0153 fW | 0.2510 fW | 0.0000 fW | 0.0000 fW | 1.329e-16 |
| T_Ea_Au_off | 5.9748 fW | 0.0000 fW | 0.0317 fW | 0.0000 fW | 0.0000 fW | 1.313e-16 |
| T_Eb_Au_off | 5.6363 fW | 0.0000 fW | 0.2573 fW | 0.0000 fW | 0.0000 fW | 0.000e+00 |
| Z_Ea_Au_on | 6.5056 fW | 0.0388 fW | 0.0091 fW | 0.0000 fW | 0.0000 fW | 1.204e-16 |
| Z_Eb_Au_on | 19.8661 fW | 0.0841 fW | 0.2159 fW | 0.0000 fW | 0.0000 fW | 0.000e+00 |
| Z_Ea_Au_off | 7.8404 fW | 0.0000 fW | 0.0088 fW | 0.0000 fW | 0.0000 fW | 2.010e-16 |
| Z_Eb_Au_off | 23.5200 fW | 0.0000 fW | 0.2395 fW | 0.0000 fW | 0.0000 fW | 0.000e+00 |
