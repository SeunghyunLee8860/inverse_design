# Run012 E∥a exact-binary certificate

The continuation reached β=64, but that continuous checkpoint was not called final: it retained 2.84% gray nodes and 89 global morphology violations. One deterministic, simultaneous active-set repair changed 89/48,441 nodes (0.184%) and produced an exact 0/1 candidate.

The exact 500 nm audit has **zero interior violations**. The unchanged global audit reports nine violations, all explicitly enumerated at the outermost nodes where the fixed top/bottom TaIrTe4 contact phase terminates against exterior left/right void. They are treated as port-boundary exemptions, not silently counted as a global pass.

A fresh GPU Maxwell plus CUDA thermal/electrical solve gives `3.441122211448e-17 A`, or `70.911302 nA` at 285 µW. This is `-3.7351%` relative to the β=64 continuous checkpoint. Optical closure, conservative Q mapping, thermal residual, energy balance, and electrical weighting residual all pass.

No Q clipping, smoothing, gain, global rescaling, CPU FDTD fallback, or empirical objective/gradient rescaling was used. Raw NPZ/FSP files remain outside Git.
