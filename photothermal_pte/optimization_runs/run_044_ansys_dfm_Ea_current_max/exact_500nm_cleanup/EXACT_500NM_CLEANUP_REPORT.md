# run044_Ea exact 500 nm binary cleanup

Status: `EXACT_500NM_BINARY_CLEANUP_OBJECTIVE_GATE_UNRESOLVED`

The continuous beta-128 checkpoint was already essentially binary, but its thresholded geometry failed the discrete 500 nm opening audit. Two deterministic active-set cleanup orderings were built. Both remove every exact violation without connectivity editing, followed by an unrescaled GPU Maxwell forward solve and the same CUDA thermal/electrical objective path.

| candidate | exact bad | changed nodes | current at 285 µW | objective change | objective gate |
|---|---:|---:|---:|---:|---:|
| solid-first | 0 | 166 | 89.973613 nA | -1.9459% | False |
| void-first | 0 | 168 | 90.013066 nA | -1.9029% | False |

Selected candidate: `void_first`, because it has the larger recomputed objective among exact-zero candidates. The continuous reference is 91.759126 nA at 285 µW.

All optical closure, remap, thermal residual, thermal energy-balance, and electrical residual checks pass. A status remains unresolved when the independently fixed 1% objective-preservation gate fails; that gate is not relaxed after seeing the result.

No Q clipping, smoothing, gain, global rescaling, CPU FDTD fallback, CPU thermal fallback, or connectivity cleanup was used. Raw NPZ/FSP files are not committed to Git.
