# Thermal raw-PTE and localized AD–FD subgates

Status: `VALIDATED_THERMAL_RAW_PTE_AND_LOCALIZED_ADFD_SUBGATES`

The historical 6 µm native-to-50 nm refined raw-PTE difference remains
`6.295360571e-03`
and remains explicitly labeled
`RAW_PTE_LT_0P5PCT_UNRESOLVED`. It was not overwritten
or reclassified.

## Successive finer thermal meshes

All cases use the same fixed optical Q, 32 µm lateral domain, 20 µm Si depth,
6 µm named TaIrTe4 footprint, and unchanged physical material/interface
law.

| mesh comparison | raw PTE change | Tmax change | TaIrTe4 average change | <0.5% |
|---|---:|---:|---:|---:|
| preserved_refined_50nm→additional_40nm | 1.843739e-03 | 3.029877e-04 | 2.146496e-04 | True |
| additional_40nm→additional_33p333nm | 3.367451e-03 | 1.714212e-04 | 1.143775e-04 | True |
| preserved_refined_50nm→additional_33p333nm | 1.526527e-03 | 4.743570e-04 | 3.290026e-04 | True |

The direct 50→33.333 nm raw-PTE change is
`1.526526524e-03`. The worst new successive-pair
raw-PTE change is
`3.367451414e-03`.

## Added thermal-only AD–FD directions

The previous adjoint-aligned, fixed-seed random, and asymmetric-smooth
directions are preserved. Central-localized and design-edge-localized
directions were added without changing Q. No Maxwell or optical-gradient
term is present in this subgate.

| scenario | direction | h | signal ratio | relative error |
|---|---|---:|---:|---:|
| TaIrTe4_4um_footprint | central_localized | 0.01 | 9.305e-05 | 3.102e-05 |
| TaIrTe4_4um_footprint | central_localized | 0.005 | 9.305e-05 | 7.843e-06 |
| TaIrTe4_4um_footprint | central_localized | 0.0025 | 9.305e-05 | 1.908e-06 |
| TaIrTe4_4um_footprint | design_edge_localized | 0.01 | 1.816e-02 | 1.376e-06 |
| TaIrTe4_4um_footprint | design_edge_localized | 0.005 | 1.816e-02 | 3.442e-07 |
| TaIrTe4_4um_footprint | design_edge_localized | 0.0025 | 1.816e-02 | 8.710e-08 |
| TaIrTe4_6um_footprint | central_localized | 0.01 | 1.492e-04 | 3.428e-05 |
| TaIrTe4_6um_footprint | central_localized | 0.005 | 1.492e-04 | 8.369e-06 |
| TaIrTe4_6um_footprint | central_localized | 0.0025 | 1.492e-04 | 1.919e-06 |
| TaIrTe4_6um_footprint | design_edge_localized | 0.01 | 1.844e-02 | 2.928e-06 |
| TaIrTe4_6um_footprint | design_edge_localized | 0.005 | 1.844e-02 | 7.293e-07 |
| TaIrTe4_6um_footprint | design_edge_localized | 0.0025 | 1.844e-02 | 1.812e-07 |

Worst selected five-direction error at `h=0.0025`:
`1.918624689e-06`.
Every added direction shows the expected centered-FD decrease as
`h -> h/2`.

No gray-law sensitivity, full latent AD-FD, transient solve, or optimization
was run. Raw NPZ/JSON artifacts remain outside Git and are SHA-256 pinned.
