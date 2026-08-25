# FDTDX reversible cotangent-accumulation report

Date: 2026-08-25 (Asia/Seoul)

Slice-512 execution commit: `ad96d7b2b7d1642de90e4588c70d5c777cea2001`

Status: `BLOCKED_NONMONOTONIC_SLICE_ERROR`; shorter resets do not fix the
long-horizon derivative bias.

## Controlled reset-interval diagnostic

The 16,384-step Ea/Eb experiment was repeated with every input unchanged except
`steps_per_slice: 1024 -> 512`. Checkpoints increased from 16 to 32 and peak
memory increased from about 15.67 GB to 21.37 GB.

| polarization | slice 1,024 error | slice 512 error | change |
|---|---:|---:|---:|
| Ea | 0.00410128 | 0.00459287 | worse |
| Eb | 0.00655637 | 0.00735858 | worse |

Ea remains barely below the frozen `0.005` gate; Eb remains blocked. Both
value-and-grad times remain about 85.1 seconds, and every gradient remains
finite/nonzero with the same AD/FD sign.

This falsifies the simple hypothesis that algebraic inverse drift inside a
longer slice is the only error source. Continuing to shorter reset intervals
without changing the adjoint accumulation is not justified. Slice 512 is also
not full-horizon feasible: 501 sparse checkpoints alone would require about
178.6 GB.

## Reference-gradient comparison

The same 16,384-step/FD-step checkpointed raw references have much lower
directional errors:

| polarization | checkpointed error | reversible 1,024 | reversible 512 |
|---|---:|---:|---:|
| Ea | 1.8264e-4 | 0.004101 | 0.004593 |
| Eb | 1.6441e-4 | 0.006556 | 0.007359 |

Against the checkpointed 6,561-node gradient, the reversible relative L2
errors are:

| polarization | slice 1,024 | slice 512 |
|---|---:|---:|
| Ea | 0.007147 | 0.008173 |
| Eb | 0.013644 | 0.015105 |

All cosine similarities remain above `0.99998`. The spatial direction is nearly
unchanged, but a magnitude bias accumulates. Because the checkpointed AD agrees
with centered FD at `~1.7e-4`, the FD step is not the blocker.

## Identified over-broad cotangent state

`arrays_for_density()` applies the 80 x 80 design occupancy only to the three Au
Lorentz `c3` couplings. Inverse permittivity, c1, c2, TaIrTe4 c3, and all other
material arrays are independent of latent density.

The first reversible prototype nevertheless makes the complete full-grid
parameter tuple differentiable and performs a float32 addition of every
one-step parameter cotangent at every reverse step. That is unnecessary for the
latent gradient and is the next numerically suspect operation.

## Next implementation gate

Build a design-specialized custom VJP whose only differentiable material input
is Au-region `c3`. Fixed parameters remain stop-gradient closures. Accumulate
only the regional design cotangent and use compensated float32 summation. First
prove equality to direct AD on the small CPML/ADE/phasor scene; then repeat the
16,384-step bounded Ea/Eb probe. No further reset-interval search, full
gradient, or optimizer is authorized before that comparison.

## External slice-512 artifacts

| artifact | SHA-256 |
|---|---|
| Ea JSON | `5a4ad64240d12a7e76ef17312e7524399bfb19f3635e76a467a059bb8194512d` |
| Ea NPZ | `aef008fc2db105e516baaa22aa5110f2aaa91001cba8e70fe4fff80bee0f973f` |
| Eb JSON | `fcdbfc17cce3713d11074ee863e6018754d554352fc424448cefe687c12b3836` |
| Eb NPZ | `2f188bc0ec6d876aafed83dae0f0dd8bd20b074e629e9442c1126fa31f756e4d` |

No full gradient, Q/PDE/current evaluation, optimizer, Lumerical, HEAT, or
CHARGE call was made.
