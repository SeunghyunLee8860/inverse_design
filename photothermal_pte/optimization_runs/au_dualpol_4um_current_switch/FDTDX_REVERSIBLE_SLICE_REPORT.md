# FDTDX reversible exact-reset slice report

Date: 2026-08-25 (Asia/Seoul)

Status: `PASS_SMALL_SCENE_SLICED_VJP`; production memory/runtime and long-horizon
stability remain open.

## Implementation

`fdtdx_parity_reversible_sliced_vjp.py` separates the evolving state from the
immutable/differentiable material parameters. Forward execution stores a
checkpoint at each slice start containing only

```text
E, H, psi_E, psi_H, P_curr, P_prev
```

The final `PhasorDetector` accumulator is retained once, not copied into every
checkpoint. During reverse execution, E/H, CPML psi, and ADE P are
algebraically reconstructed inside a slice. At the preceding slice boundary,
the reconstructed primal is replaced with the saved exact forward state. The
running field/detector cotangent and accumulated material-parameter cotangent
are not reset.

The last slice may be shorter than `steps_per_slice`; padded forward and reverse
iterations are identity operations and do not record or differentiate a
nonexistent time step.

## Direct-gradient tests

The first test uses the existing 24-step, six-face CPML, dispersive slab, point
source, and late three-component phasor scene. Four slices of six steps reproduce
the direct unrolled FDTDX phasor-power value and complete `c3` gradient at
`rtol=5e-4` with a nonzero gradient.

A longer independently placed scene runs 70 steps and enables its phasor window
at step 24. It compares direct unrolled FDTDX against:

- no-reset algebraic reverse;
- ten slices of seven steps;
- five slices of 16 steps, whose final slice contains only six active steps.

Both sliced gradients match direct AD at `rtol=1e-3`. Their relative L2 errors
are each no larger than the no-reset result, and every forward objective agrees
with direct execution. This proves cotangent continuity, exact-primal reset,
and partial-final-slice control flow on a real FDTDX update.

The complete target-folder CPU suite passes `226 passed`.

## Exact-grid memory implication

The current implementation is deliberately a correctness prototype and stores
full-domain P at every slice boundary. Existing exact-grid byte audits give:

| checkpoint item | bytes |
|---|---:|
| E/H and CPML psi | 273,559,872 |
| full-domain P-current/P-previous | 712,400,832 |
| current full-state slice checkpoint | 985,960,704 |
| certified sparse regional P-current/P-previous | 82,944,000 |
| prospective sparse slice checkpoint | 356,503,872 |

For 256,163 steps, a 4,096-step slice gives 63 checkpoints: about 62.12 GB with
the current full P state versus 22.46 GB after certified regional-P sparsity. A
2,048-step slice gives 126 checkpoints: about 124.23 GB full versus 44.92 GB
sparse. These are checkpoint payloads only, not allocator-peak predictions.

Therefore the full-state prototype must not be launched on the exact grid.
Sparse Au/TaIrTe4 P extraction/expansion must be integrated first, and the
required slice length must be determined by measured gradient/reconstruction
error rather than chosen from memory alone.

## Remaining gate

Next implement the same exact reset with P checkpointed only on the already
certified disjoint regions:

- TaIrTe4: `x=13:173`, `y=13:173`, `z=167:207`;
- Au design: `x=53:133`, `y=53:133`, `z=207:227`.

The full P arrays may exist transiently inside one FDTDX step, but only regional
P may be emitted by the forward slice scan. Prove full-versus-sparse forward and
complete `c3` gradient parity on the small scene. Only then measure a short
exact-grid reversible probe on freshly verified-idle GPUs. No 40-period
gradient, PDE/current evaluation, or optimizer is enabled by this result.
