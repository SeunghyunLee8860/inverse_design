# Run 044 floating-island ablation

Status: `COMPLETED_RUN044_FLOATING_ISLAND_ABLATION`

The immutable selected Run 044 exact-binary density was preserved. Six TaIrTe4 solid components touching neither top nor bottom terminal were changed to air in a diagnostic copy, followed by one fresh GPU Maxwell forward solve and the unchanged CUDA thermal/electrical path.

## Geometry audit

- Floating components removed: 6
- Removed solid nodes: 2431 (7.095% of original solid)
- 4/8-neighbour terminal-connected support: identical
- Exact 500 nm bad nodes after removal: 0

## Fresh end-to-end comparison at 285 µW

| Metric | Original | Islands removed | Relative change |
|---|---:|---:|---:|
| Raw Maxwell P_Q | 2.878475619e-14 W | 2.420413942e-14 W | -15.913% |
| Global TaIrTe4 Tmax | 0.389824 K | 0.299278 K | -23.227% |
| Tmax on retained TaIrTe4 | 0.299686 K | 0.299278 K | -0.136% |
| Global max strict |grad T| | 186600 K/m | 163454 K/m | -12.404% |
| Max common strict |grad T| | 163942 K/m | 163454 K/m | -0.298% |
| Terminal current | 90.013066 nA | 82.805537 nA | -8.007% |
| Terminal conductance | 1.362941234e-02 S | 1.362941178e-02 S | -0.000004% |

Spatial relative-L2 differences are 56.621% for mapped 3-D Q, 49.788% for depth-integrated Q, 6.693% for temperature on retained TaIrTe4, and 12.687% for the strict gradient on common valid nodes.

## Interpretation

The terminal conductance is unchanged to numerical precision, confirming that the removed components did not form the collected DC path. Nevertheless, removing them reduces absorption, temperature, and terminal current. Their net role in this optimized result is therefore indirect: Maxwell scattering/absorption and the resulting thermal-field redistribution, not direct electrical collection.

The raw result status remains failed only against the inherited one-percent objective-preservation gate, because this experiment intentionally tests a change expected to alter the objective. All optical closure, mapping, thermal residual/energy, electrical residual, finite-value, and GPU-only gates passed. No Q clipping, smoothing, gain, polarization matching, or rescaling was used.
