# FDTDX substrate spatial native-Yee Q artifact

Status: **VALIDATED_FDTDX_SUBSTRATE_SPATIAL_NATIVE_YEE_Q_ARTIFACT**

This checkpoint exports the actual spatial `Qx`, `Qy`, and `Qz` arrays for
Au, TaIrTe4, and lossy SiO2 from the validated 16-period/4-window GPU FDTDX
forward. Each component retains its own staggered physical coordinates,
axis-wise dual widths, and dual volumes. No array-index pairing between
different Yee components is used.

| metric | value |
|---|---:|
| total P_Q | 2.477938114560e-13 W |
| independently reintegrated P_Q | 2.477937967080e-13 W |
| total reintegration error | 0.000005952% |
| worst component reintegration error | 0.000009588% |
| dual-volume factorization error | 0.000005364% |
| matched-volume Q/flux closure | 0.122323% |
| late-window change | 0.019858% |
| runtime | 120.657 s |

Material powers are Au `5.500486665909e-14 W`, TaIrTe4
`1.526977170459e-13 W`, and SiO2
`4.009121300310e-14 W`. The raw NPZ is not committed to Git. Its
path, byte size, and SHA-256 are recorded in the manifest.

The maps use a logarithmic color display only; the stored and integrated Q
arrays are unmodified. No clipping, smoothing, gain, global rescaling, or
polarization matching is performed.

This is not yet a coupled thermal/PTE validation. The next fail-closed gate is
an overlap-based conservative remap of every component-native dual cell into
one explicit Au/TaIrTe4/SiO2 thermal grid. Only after that mapping preserves
power may the Maxwell source replace the fixed-Q source in the coupled
thermal/weighting operator.
