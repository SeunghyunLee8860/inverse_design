# 2024 MIR inverse-T / TaIrTe4 two-polarization optical smoke

Status: `VALIDATED_T2024_FIGURE_DIGITIZED_TAIRTE4_TWO_POLARIZATION_OPTICAL_SMOKE`

This is a paper-derived scalar-geometry scenario, not a reproduction of the
graphene experiment and not exact author CAD. The 2024 paper's MIR inverse-T,
period, spacer, and Au thickness contract is retained while the active 2-D
material alone is deliberately replaced by 100-nm anisotropic TaIrTe4. The T
arm widths/lengths are digitized from Supplementary Fig. 14 axes.

## GPU results

| Metric | E parallel b (x) | E parallel a (y) |
|---|---:|---:|
| wall time (s) | 27.476 | 32.626 |
| source power (W/cell) | 1.972769480158e-15 | 1.972769480158e-15 |
| periodic P_Q (W/cell) | 4.236429786026e-16 | 3.752872545512e-16 |
| absorbed flux (W/cell) | 4.242603101750e-16 | 3.764306064982e-16 |
| pabs absorptance | 0.214745302 | 0.190233709 |
| closure | 0.145508% | 0.303735% |
| reflection | 0.784941771 | 0.809186724 |
| auto-shutoff | 5.668510e-07 | 8.815130e-07 |

Raw total periodic-Q ratio `E||a / E||b` is **0.885857**. The
geometrically assigned TaIrTe4-only native-Q ratio is **0.913666**.
These are raw equal-source results; no polarization matching, clipping,
smoothing, gain, or global rescaling was applied.

Qx/Qy/Qz are retained on their independent staggered Yee coordinates. Each
component is integrated and plotted separately. Equal array indices from
different component grids are never treated as the same physical coordinate.

No thermal, PTE, adjoint, or optimization calculation was run in this stage.

## 2022 Z status

The 2022 Supplementary Table 1 publishes M1-M5 scalar dimensions, but the PDFs
do not publish polygon vertices or a unique arm-junction construction. Those
numbers are sufficient for a dimension audit but not for a unique Maxwell CAD.
The Z case therefore remains fail-closed until author geometry is recovered or
an explicitly named approximation is approved.
