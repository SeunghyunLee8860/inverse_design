# Checkpoint-free production combined PTE gradient

Status: `VALIDATED_FDTDX_PRODUCTION_CHECKPOINT_FREE_COMBINED_PTE_GRADIENT_EQUIVALENCE`

This certificate changes only the Maxwell-source derivative implementation.
The old checkpointed reverse-through-time result remains immutable as the
reference. The new route stores no time history and performs one forward plus
one reciprocal adjoint FDTD solve. No empirical normalization or gradient
rescaling is used.

## Combined chain

\[
g_{\rho} = g_{\rm Maxwell,no\ checkpoint}
              + g_{\rm thermal/contact}
              + g_{\rm electrical/weighting}.
\]

The thermal/contact and electrical/weighting arrays are bitwise identical to
the independently validated fixed-spatial-Q certificate. Only the optical
array was replaced.

| metric | result |
|---|---:|
| optical vector error vs frozen checkpoint | 0.156919% |
| combined vector error vs frozen checkpoint | 0.153893% |
| combined norm error | 0.106996% |
| combined angle | 0.063342 deg |
| worst strong-direction AD--FD error | 0.258059% |
| worst all-direction normalized error | 0.120472% |
| forward + adjoint execution | 212.295 s |
| speedup vs frozen checkpointed AD | 13.755x |

The `design_edge_localized` direction is near-null; its raw relative error is
reported in the CSV but is not treated as a strong-direction metric. Its error
normalized by the full gradient norm remains below 1%.

## Important scope boundary

This validates the derivative implementation for the frozen explicit optical
material contract. It does not turn the inherited substrate model into a
paper-certified material: the recorded substrate status remains
`BLOCKED_LUMERICAL_10UM_SI_PALIK_READBACK`. It also does not execute an
optimization.

Raw NPZ arrays are outside Git and are pinned in `RAW_ARTIFACT_MANIFEST.json`.
