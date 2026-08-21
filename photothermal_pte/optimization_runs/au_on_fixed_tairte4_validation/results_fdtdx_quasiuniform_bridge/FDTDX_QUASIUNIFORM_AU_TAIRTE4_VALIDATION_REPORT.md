# FDTDX quasi-uniform Au/TaIrTe4 validation

**Status: `PARTIAL_FDTDX_AU_GRADIENT_AND_W8P5_SOURCE_VALIDATED_PENDING_MATERIAL_CROSSCHECK`**

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
4. A separate production-width empty-air source audit passed every gate.  For
   requested `w0=8.5 um`, primary `Ex` gives a realized mean waist of
   `8.457306 um`,
   `3.7055%`
   ellipticity, and a closed-surface residual of
   `0.365476%`
   of target-plane incident power.  Its GPU execution time after compilation
   was `0.4628 s` on this source-only grid.

## What failed and remains blocked

- The last two-window absorbed-power change is
  `0.970403%`, above
  the 0.5% gate in the eight-period quick run.
- The material-loss versus raw closed-surface flux mismatch is
  `98.738720%`.
- The signed material-minus-empty closed-surface mismatch is
  `98.959501%`.
- In the independent source-only control, compact-Gaussian closed-box residuals
  are `3.1262%` (`+z`) and
  `4.2523%` (`-z`); enlarging the box gives
  `11.9783%` and `12.8954%`.
  The corresponding uniform-source large-box residual is only about 0.08%.

Therefore the source direction is not the failure, and the closure problem is
specific to the deliberately subwavelength compact Gaussian (`w0≈0.594 um` at
`lambda=10 um`), not the production-width empty-air source.  The unresolved
item is the full material-bearing production-width cross-solver checkpoint.
The same-container zero-coupling ADE probe is **not** an
empirical correction: because every nominal material support has
`epsilon_inf=1` and all ADE field couplings are zeroed, it is an exact
empty-air optical control on the identical source/grid/PML layout.  Its flux
was already in watts; the earlier extra `1e-24` postprocessing factor has been
removed.  Both raw and background-subtracted closure still fail.

## Decision

FDTDX is usable now for algorithmic dispersive-material AD controls and for the
production-width source-only forward contract.  It is not yet promoted as the
production Au inverse-design solver.  Before thermal/PTE coupling or
optimization, the next optical checkpoint is a material-bearing
production-width FDTDX calculation against the already validated exact-binary
Lumerical endpoints, using local fine Au/TaIrTe4 mesh and coarse distant air.
