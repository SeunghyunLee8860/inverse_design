# 2024 inverse-T / TaIrTe4 matched bare-control comparison

Status: `VALIDATED_T2024_TOP_T_MATCHED_BARE_OPTICAL_COMPARISON`

The only geometry change is removal of the 33-nm top Au inverse-T. The
periodic cell, 100-nm TaIrTe4, 35-nm Al2O3, Au mirror, materials, source,
mesh, boundary conditions, and normalization remain identical.

| Quantity | E parallel b | E parallel a |
|---|---:|---:|
| total P_Q, bare (W/cell) | 3.842007702446e-16 | 3.967731033735e-16 |
| total P_Q, with T (W/cell) | 4.236429786026e-16 | 3.752872545512e-16 |
| T / bare, total | 1.102660 | 0.945849 |
| relative total change | 10.2660% | -5.4151% |
| T / bare, TaIrTe4-only geometric Q | 1.060852 | 0.916580 |
| relative TaIrTe4 change | 6.0852% | -8.3420% |

At this single wavelength the digitized T is polarization selective: it
enhances E||b absorption and suppresses E||a. The bare total `Eb/Ea` ratio is
0.968314; with the T it becomes
1.128850. This is a forward optical result,
not yet a thermal or PTE improvement claim.

All four GPU cases passed closure (<0.5%), auto-shutoff (<1e-5), finite-Q,
and nonnegative-Q gates. No clipping, smoothing, gain, global rescaling, or
polarization matching was used. Qx/Qy/Qz remain on component-specific Yee
coordinates. The lower panels plot only the incident-dominant component on
its own identical grid; they are not cross-component sums.
