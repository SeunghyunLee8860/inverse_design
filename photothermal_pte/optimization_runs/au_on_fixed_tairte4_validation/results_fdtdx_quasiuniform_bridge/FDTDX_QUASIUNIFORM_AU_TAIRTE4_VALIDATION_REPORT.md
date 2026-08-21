# FDTDX quasi-uniform Au/TaIrTe4 validation

**Status: `PARTIAL_FDTDX_AU_GRADIENT_VALIDATED_BLOCKED_FINITE_GAUSSIAN_CLOSURE`**

This checkpoint tests whether the post-release FDTDX main branch can represent
physical 50-nm Au on physical 100-nm TaIrTe4 with a quasi-uniform Cartesian
grid and differentiate the dispersive material loss.  It is a compact optical
control.  It is **not** the production 8.5-um-waist beam, thermal/PTE model,
electrode model, or an optimization result.

## Reproducibility contract

- FDTDX source commit: `f26f84b70a8cceec9b889553955a868624736bf1`
- imported module: `/home/seunghyun/.local/fdtdx_main_src/src/fdtdx/__init__.py`
- GPU requested through `CUDA_VISIBLE_DEVICES=5`
- solver axes: `x=b`, `y=a`, `z=c=b` closure
- wavelength: `10.000 um`
- grid: `[40, 40, 160]` cells at
  `[100.0, 100.0, 25.0] nm`
- realized Au thickness: `50.000153 nm`
- realized TaIrTe4 thickness: `99.999852 nm`
- no clipping, smoothing, gain, or result rescaling

The installed numbered release did not expose this nonuniform-grid route; this
checkpoint pins the exact post-release source commit above rather than silently
depending on a moving `main` branch.

## What passed

1. GPU execution and physical-thickness placement passed.
2. Source direction reciprocity passed: the `-z/+z` downstream-power ratios
   are `1.00001033`
   (uniform source) and `1.02181769`
   (compact Gaussian).
3. Five independent total-loss directional AD--FD controls passed the 1% gate
   at `h=0.005`:

| direction | strong relative error |
|---|---:|
| `uniform` | 0.088661% |
| `smooth_asymmetric` | 0.506267% |
| `central_localized` | 0.591531% |
| `design_edge_localized` | 0.144974% |
| `fixed_seed_random` | 0.331446% |

The maximum multi-direction gradient-L2-normalized error is
`0.122962%`.
This validates reverse-mode differentiation of the fixed-support, dispersive Au
material relaxation for this compact grid.  It does not validate moving Au
boundaries in Lumerical.

## What failed and remains blocked

- The last two-window absorbed-power change is
  `0.970403%`, above
  the 0.5% gate in the eight-period quick run.
- The material-loss versus raw closed-surface flux mismatch is
  `98.738720%`.
- In the independent source-only control, compact-Gaussian closed-box residuals
  are `3.1262%` (`+z`) and
  `4.2523%` (`-z`); enlarging the box gives
  `11.9783%` and `12.8954%`.
  The corresponding uniform-source large-box residual is only about 0.08%.

Therefore the source direction is not the failure.  The unresolved item is the
finite-Gaussian closed-surface flux/collocation audit in this compact FDTDX
configuration.  The same-container zero-coupling ADE probe is **not** an
independent empty-air run and is not used to repair the closure gate.

## Decision

FDTDX is usable now for algorithmic dispersive-material AD controls.  It is not
yet promoted as the production finite-Gaussian Au inverse-design solver.  Before
thermal/PTE coupling or optimization, the next optical checkpoint is a
uniform/periodic or sufficiently wide source cross-solver comparison with
matched Lumerical endpoints, followed by a finite-Gaussian closure repair that
passes without empirical normalization.
