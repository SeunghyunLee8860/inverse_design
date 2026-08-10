# Run013 E∥b exact-binary certificate

The continuation reached β=64, but that continuous checkpoint was not called final: it retained 2.84% gray nodes and 89 global morphology violations. One deterministic, simultaneous active-set repair changed 89/48,441 nodes (0.184%) and produced an exact 0/1 candidate.

The requested feature size was 500 nm, but the 100 nm nodal grid rounds the 250 nm opening radius up to three offsets. The realized discrete audit is therefore a conservative 300 nm maximum offset / roughly 600 nm nominal diameter, not an exact 500 nm certificate. It has **zero interior bad nodes**. The unchanged global audit reports nine violations, all explicitly enumerated at the outermost nodes where the fixed top/bottom TaIrTe4 contact phase terminates against exterior left/right void. They are treated as port-boundary exemptions, not silently counted as a global pass.

A fresh GPU Maxwell plus CUDA thermal/electrical solve gives `2.465051579493e-17 A`, or `50.797387 nA` at 285 µW. This is `-0.5908%` relative to the β=64 continuous checkpoint. Optical closure, conservative Q mapping, thermal residual, energy balance, and electrical weighting residual all pass.

No Q clipping, smoothing, gain, global rescaling, CPU FDTD fallback, or empirical objective/gradient rescaling was used. Raw NPZ/FSP files remain outside Git.
