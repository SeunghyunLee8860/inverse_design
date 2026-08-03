# Device-A nine-position, two-interface-G result

All spatial plots use the fixed Lumerical frame **x = crystal b, y = crystal a**. The device, PML, monitor, and mesh geometry are invariant; only the scalar-Gaussian source is translated.

The frozen source centers are shown in the [nine-position geometry plan](../paper_ir_device_a_inside_flake_center/DEVICE_A_FINAL_BEAM_CENTER_PLAN.png).

The two TaIrTe4/SiO2 conductances are reported as separate named physical scenarios: thermally grown (7.37e6 W/m2K) and evaporated (7.37e4 W/m2K). Neither is promoted as a fabrication-independent truth.

Thermal Q uses TaIrTe4-only, exact optical-cell/thermal-material intersection-density mapping at 285 uW incident power. SiO2 absorption remains in the optical audit but is not added as a thermal source. No Q clipping, smoothing, gain, rescaling, or tiling is used.

Raw optical artifacts use the solver's unit-central-intensity convention. Position-comparison plots convert each artifact with its own matched empty-stack incident-power readback to the common 285-uW incident-power contract; this is physical source normalization, not empirical Q matching.

The raw optical-Q mosaics use the artifact center mask and are diagnostic only. The production-Q mosaics and every case panel show the conservative optical-cell/TaIrTe4/thermal-cell intersection-density mapping actually used by the thermal solve; no boundary-cell power is forced from air into TaIrTe4.

Current is a full-volume anisotropic Shockley-Ramo PTE integral. A strict-centered current-density map is also shown, with cells masked unless all +/-x and +/-y TaIrTe4 neighbours exist. Because the digitized-model resistance differs from the measured device, absolute current is not called an experimental reproduction.

## Results

| interface | position | pol. | absorbed power (uW) | Tmax rise (K) | TaIrTe4 avg. dT (K) | grad P99 (K/m) | current (nA) |
|---|---|---:|---:|---:|---:|---:|---:|
| thermally_grown | outside_top | a | 7.92472 | 0.215283 | 0.00517975 | 38543.1 | 11.2904 |
| thermally_grown | outside_top | b | 8.52906 | 0.136932 | 0.00561922 | 25386.3 | 9.4382 |
| thermally_grown | outside_middle | a | 11.8727 | 0.279414 | 0.0077793 | 62896.7 | 13.6215 |
| thermally_grown | outside_middle | b | 13.2825 | 0.15433 | 0.00877156 | 31594.6 | 11.4766 |
| thermally_grown | outside_bottom | a | 15.3042 | 0.277418 | 0.0100345 | 83631 | 12.8196 |
| thermally_grown | outside_bottom | b | 17.6806 | 0.167518 | 0.0116844 | 35988.9 | 10.4744 |
| thermally_grown | edge_top | a | 29.5387 | 0.284182 | 0.019765 | 68856.5 | 22.0453 |
| thermally_grown | edge_top | b | 37.679 | 0.236842 | 0.0252593 | 47036.6 | 16.4755 |
| thermally_grown | edge_middle | a | 22.8946 | 0.367488 | 0.0151869 | 66276.8 | 19.0614 |
| thermally_grown | edge_middle | b | 27.9757 | 0.238939 | 0.0186349 | 37518.3 | 16.9883 |
| thermally_grown | edge_bottom | a | 25.4145 | 0.320391 | 0.0168472 | 85798.8 | 15.9784 |
| thermally_grown | edge_bottom | b | 31.5416 | 0.21723 | 0.0210017 | 45522.7 | 13.8992 |
| thermally_grown | inside_top | a | 39.4161 | 0.233403 | 0.0267172 | 34329.9 | 18.7985 |
| thermally_grown | inside_top | b | 52.753 | 0.232958 | 0.0357176 | 48828.2 | 10.1918 |
| thermally_grown | inside_middle | a | 43.9862 | 0.170519 | 0.0300571 | 28861.4 | 4.97608 |
| thermally_grown | inside_middle | b | 59.9776 | 0.235585 | 0.0408722 | 34225.2 | 0.120389 |
| thermally_grown | inside_bottom | a | 46.0591 | 0.172795 | 0.0315964 | 23003.8 | -1.88358 |
| thermally_grown | inside_bottom | b | 63.4702 | 0.235897 | 0.0433834 | 31299.6 | -5.32404 |
| evaporated | outside_top | a | 7.92472 | 1.95861 | 0.166873 | 260242 | 180.72 |
| evaporated | outside_top | b | 8.52906 | 1.7514 | 0.179587 | 242504 | 173.506 |
| evaporated | outside_middle | a | 11.8727 | 2.75474 | 0.249996 | 387775 | 219.234 |
| evaporated | outside_middle | b | 13.2825 | 2.47666 | 0.279664 | 364449 | 214.987 |
| evaporated | outside_bottom | a | 15.3042 | 3.14781 | 0.322237 | 434986 | 197.363 |
| evaporated | outside_bottom | b | 17.6806 | 2.9009 | 0.372248 | 406284 | 192.726 |
| evaporated | edge_top | a | 29.5387 | 3.63599 | 0.622236 | 362699 | 505.076 |
| evaporated | edge_top | b | 37.679 | 4.10939 | 0.793612 | 480429 | 568.395 |
| evaporated | edge_middle | a | 22.8946 | 4.10861 | 0.482164 | 452292 | 359.248 |
| evaporated | edge_middle | b | 27.9757 | 4.16994 | 0.58912 | 493761 | 384.585 |
| evaporated | edge_bottom | a | 25.4145 | 4.04282 | 0.535196 | 486450 | 273.554 |
| evaporated | edge_bottom | b | 31.5416 | 4.12696 | 0.664165 | 535719 | 286.713 |
| evaporated | inside_top | a | 39.4161 | 3.78181 | 0.830501 | 367178 | 552.087 |
| evaporated | inside_top | b | 52.753 | 4.67677 | 1.11134 | 530686 | 656.555 |
| evaporated | inside_middle | a | 43.9862 | 3.11624 | 0.926857 | 336789 | 314.85 |
| evaporated | inside_middle | b | 59.9776 | 4.34263 | 1.26363 | 482805 | 376.529 |
| evaporated | inside_bottom | a | 46.0591 | 3.01766 | 0.970566 | 326812 | 47.3771 |
| evaporated | inside_bottom | b | 63.4702 | 4.22389 | 1.33725 | 466281 | 31.7418 |

Each per-case PNG uses the same Lumerical coordinate bounds for both polarizations and shows, in order, mapped Q, thickness-averaged temperature rise, dT/dx (crystal b), dT/dy (crystal a), gradient magnitude, and the strict-centered local current contribution.

- [CSV](device_a_nine_position_two_interface_results.csv)
- [JSON](device_a_nine_position_two_interface_summary.json)
- [manifest](RAW_ARTIFACT_MANIFEST.json)
