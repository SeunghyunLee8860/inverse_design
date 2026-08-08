# Run 005 final-binary two-polarization comparison

Status: `VALIDATED_FINAL_BINARY_BOTH_OPTICAL_POLARIZATIONS_NUMERICALLY_WITH_AXIS_INTERPRETATION_BLOCKED`.

The earlier final-binary solve used source x polarization. Under the frozen optical metadata `x=b, y=a`, it is **E || b**, not E || a. A fresh source-y, 90-degree GPU forward and CUDA thermal/PTE evaluation now supplies **E || a**. Both use the same exact-binary structure and incident power; no polarization matching, Q rescaling, clipping, smoothing, or gain was used.

## Numerical results

| metric | E || b (source x) | E || a (source y) | a/b |
|---|---:|---:|---:|
| P_Q (W) | 5.476087197869e-14 | 8.290586467109e-14 | 1.513962 |
| mapped Q (W) | 5.425780399959e-14 | 8.179269783413e-14 | 1.507483 |
| max flake-average DeltaT (K) | 2.322425179242e-10 | 5.106054672947e-10 | 2.198587 |
| strict max gradient (K/m) | 1.074093002534e-04 | 3.024276528905e-04 | 2.815656 |
| full current (A) | 1.199729281050e-19 | 2.018367346284e-19 | 1.682352 |
| FOM (A/W) | 8.679256315189e-07 | 1.460156704792e-06 | 1.682352 |

Both cases pass optical closure, auto-shutoff, conservative-remap power, CUDA thermal residual, and energy-balance gates. Spatial derivative/current maps use NaN wherever any one of `-x,+x,-y,+y` neighbours is missing.

## Axis interpretation blocker

The optical metadata says `x=b, y=a`, but the existing immutable thermal/PTE operator applies the `a` coefficients to solver x and the `b` coefficients to solver y. Therefore the numerical two-polarization comparison is valid, while its crystallographic current interpretation remains `UNRESOLVED_AXIS_METADATA_MISMATCH_XB_YA_VS_THERMAL_PTE_XA_YB`. No coefficient or axis was silently swapped in this postprocessing.
