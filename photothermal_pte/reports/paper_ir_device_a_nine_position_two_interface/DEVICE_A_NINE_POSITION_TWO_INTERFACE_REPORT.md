# Device-A nine-position, two-interface-G result

All spatial plots use the fixed Lumerical frame **x = crystal b, y = crystal a**. The device, PML, monitor, and mesh geometry are invariant; only the scalar-Gaussian source is translated.

The frozen source centers are shown in the [nine-position geometry plan](../paper_ir_device_a_inside_flake_center/DEVICE_A_FINAL_BEAM_CENTER_PLAN.png).

The two TaIrTe4/SiO2 conductances are reported as separate named physical scenarios: thermally grown (7.37e6 W/m2K) and evaporated (7.37e4 W/m2K). Neither is promoted as a fabrication-independent truth.

Thermal Q uses TaIrTe4-only, exact optical-cell/thermal-material intersection-density mapping at 285 uW incident power. SiO2 absorption remains in the optical audit but is not added as a thermal source. No Q clipping, smoothing, gain, rescaling, or tiling is used.

Raw optical artifacts use the solver's unit-central-intensity convention. Position-comparison plots convert each artifact with its own matched empty-stack incident-power readback to the common 285-uW incident-power contract; this is physical source normalization, not empirical Q matching.

The raw optical-Q mosaics use the artifact center mask and are diagnostic only. The production-Q mosaics and every case panel show the conservative optical-cell/TaIrTe4/thermal-cell intersection-density mapping actually used by the thermal solve; no boundary-cell power is forced from air into TaIrTe4.

The reported scalar current is the full-TaIrTe4-footprint, full-thickness anisotropic Shockley-Ramo volume integral. It uses centered differences in the interior and one-sided reconstruction only for boundary-adjacent cells so no flake volume is discarded. Spatial gradient/current maps remain strict diagnostics: temperature and weighting-potential gradients, J_PTE, and the displayed local collection integrand are NaN/masked unless all -x, +x, -y, and +y TaIrTe4 neighbours exist. Because the digitized-model resistance differs from the measured device, absolute current is not called an experimental reproduction.

## Figure gallery

### Frozen nine-position geometry

![Frozen nine-position Lumerical-coordinate geometry](../paper_ir_device_a_inside_flake_center/DEVICE_A_FINAL_BEAM_CENTER_PLAN.png)

### Conservative thermal-production Q at 285 uW incident power

![Mapped thermal Q, E parallel a](MAPPED_THERMAL_Q_LUMERICAL_COORDINATES_Ea.png)

![Mapped thermal Q, E parallel b](MAPPED_THERMAL_Q_LUMERICAL_COORDINATES_Eb.png)

### Nine-position scalar summaries

![Thermally grown SiO2 summary](NINE_POSITION_SUMMARY_THERMALLY_GROWN.png)

![Evaporated SiO2 summary](NINE_POSITION_SUMMARY_EVAPORATED.png)

![Polarization and interface-G ratios](POLARIZATION_AND_INTERFACE_G_RATIOS.png)

![Full-footprint total current Ea Eb comparison](TOTAL_CURRENT_EA_EB_COMPARISON.png)

### Per-case Lumerical-coordinate maps

Every panel below uses Lumerical **x = crystal b, y = crystal a** and shows, from left to right, mapped Q, thickness-averaged temperature rise, dT/dx, dT/dy, gradient magnitude, and strict-centered local current contribution. Gray cells are explicit NaN/masked cells. The two rows are E parallel a and E parallel b.

#### Thermally Grown SiO2 interface

##### outside_top

![thermally_grown outside_top Q temperature gradient current](case_panels/thermally_grown_outside_top_Q_T_GRADIENT_CURRENT.png)

##### outside_middle

![thermally_grown outside_middle Q temperature gradient current](case_panels/thermally_grown_outside_middle_Q_T_GRADIENT_CURRENT.png)

##### outside_bottom

![thermally_grown outside_bottom Q temperature gradient current](case_panels/thermally_grown_outside_bottom_Q_T_GRADIENT_CURRENT.png)

##### edge_top

![thermally_grown edge_top Q temperature gradient current](case_panels/thermally_grown_edge_top_Q_T_GRADIENT_CURRENT.png)

##### edge_middle

![thermally_grown edge_middle Q temperature gradient current](case_panels/thermally_grown_edge_middle_Q_T_GRADIENT_CURRENT.png)

##### edge_bottom

![thermally_grown edge_bottom Q temperature gradient current](case_panels/thermally_grown_edge_bottom_Q_T_GRADIENT_CURRENT.png)

##### inside_top

![thermally_grown inside_top Q temperature gradient current](case_panels/thermally_grown_inside_top_Q_T_GRADIENT_CURRENT.png)

##### inside_middle

![thermally_grown inside_middle Q temperature gradient current](case_panels/thermally_grown_inside_middle_Q_T_GRADIENT_CURRENT.png)

##### inside_bottom

![thermally_grown inside_bottom Q temperature gradient current](case_panels/thermally_grown_inside_bottom_Q_T_GRADIENT_CURRENT.png)

#### Evaporated SiO2 interface

##### outside_top

![evaporated outside_top Q temperature gradient current](case_panels/evaporated_outside_top_Q_T_GRADIENT_CURRENT.png)

##### outside_middle

![evaporated outside_middle Q temperature gradient current](case_panels/evaporated_outside_middle_Q_T_GRADIENT_CURRENT.png)

##### outside_bottom

![evaporated outside_bottom Q temperature gradient current](case_panels/evaporated_outside_bottom_Q_T_GRADIENT_CURRENT.png)

##### edge_top

![evaporated edge_top Q temperature gradient current](case_panels/evaporated_edge_top_Q_T_GRADIENT_CURRENT.png)

##### edge_middle

![evaporated edge_middle Q temperature gradient current](case_panels/evaporated_edge_middle_Q_T_GRADIENT_CURRENT.png)

##### edge_bottom

![evaporated edge_bottom Q temperature gradient current](case_panels/evaporated_edge_bottom_Q_T_GRADIENT_CURRENT.png)

##### inside_top

![evaporated inside_top Q temperature gradient current](case_panels/evaporated_inside_top_Q_T_GRADIENT_CURRENT.png)

##### inside_middle

![evaporated inside_middle Q temperature gradient current](case_panels/evaporated_inside_middle_Q_T_GRADIENT_CURRENT.png)

##### inside_bottom

![evaporated inside_bottom Q temperature gradient current](case_panels/evaporated_inside_bottom_Q_T_GRADIENT_CURRENT.png)

## Results

| interface | position | total current E∥a (nA) | total current E∥b (nA) | signed Ia/Ib | abs. Ia/Ib |
|---|---|---:|---:|---:|---:|
| thermally_grown | outside_top | 11.2904 | 9.4382 | 1.19624 | 1.19624 |
| thermally_grown | outside_middle | 13.6215 | 11.4766 | 1.18689 | 1.18689 |
| thermally_grown | outside_bottom | 12.8196 | 10.4744 | 1.2239 | 1.2239 |
| thermally_grown | edge_top | 22.0453 | 16.4755 | 1.33806 | 1.33806 |
| thermally_grown | edge_middle | 19.0614 | 16.9883 | 1.12203 | 1.12203 |
| thermally_grown | edge_bottom | 15.9784 | 13.8992 | 1.14959 | 1.14959 |
| thermally_grown | inside_top | 18.7985 | 10.1918 | 1.84448 | 1.84448 |
| thermally_grown | inside_middle | 4.97608 | 0.120389 | 41.3333 | 41.3333 |
| thermally_grown | inside_bottom | -1.88358 | -5.32404 | 0.353788 | 0.353788 |
| evaporated | outside_top | 180.72 | 173.506 | 1.04157 | 1.04157 |
| evaporated | outside_middle | 219.234 | 214.987 | 1.01976 | 1.01976 |
| evaporated | outside_bottom | 197.363 | 192.726 | 1.02406 | 1.02406 |
| evaporated | edge_top | 505.076 | 568.395 | 0.888601 | 0.888601 |
| evaporated | edge_middle | 359.248 | 384.585 | 0.934116 | 0.934116 |
| evaporated | edge_bottom | 273.554 | 286.713 | 0.954102 | 0.954102 |
| evaporated | inside_top | 552.087 | 656.555 | 0.840884 | 0.840884 |
| evaporated | inside_middle | 314.85 | 376.529 | 0.836191 | 0.836191 |
| evaporated | inside_bottom | 47.3771 | 31.7418 | 1.49257 | 1.49257 |

Each per-case PNG uses the same Lumerical coordinate bounds for both polarizations and shows, in order, mapped Q, thickness-averaged temperature rise, dT/dx (crystal b), dT/dy (crystal a), gradient magnitude, and the strict-centered local current contribution. Gray cells explicitly mark NaN/masked locations where at least one of -x, +x, -y, or +y TaIrTe4 neighbours is missing.

- [all-case CSV](device_a_nine_position_two_interface_results.csv)
- [paired Ea/Eb total-current CSV](device_a_total_current_Ea_Eb.csv)
- [JSON](device_a_nine_position_two_interface_summary.json)
- [manifest](RAW_ARTIFACT_MANIFEST.json)
