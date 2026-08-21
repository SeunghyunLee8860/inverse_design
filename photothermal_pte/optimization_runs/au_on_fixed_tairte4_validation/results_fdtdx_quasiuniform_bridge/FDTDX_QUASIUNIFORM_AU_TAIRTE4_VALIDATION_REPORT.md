# FDTDX quasi-uniform Au/TaIrTe4 validation

**Status: `VALIDATED_FDTDX_AU_OPTICAL_FORWARD_AND_COMPACT_MATERIAL_GRADIENT`**

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
5. The material-bearing production-width exact-binary cross-check passes.  The
   absorbed fractions are:

| endpoint | FDTDX | Lumerical | relative difference |
|---|---:|---:|---:|
| TaIrTe4 only | 0.45090092 | 0.44767564 | 0.720450% |
| Au / TaIrTe4 | 0.22085364 | 0.21944800 | 0.640534% |

   The Au-present/Au-absent absorbed-power ratio differs by only
   `0.079344%`.  Local
   native-Yee material loss agrees with the empty-subtracted six-face flux to
   `0.007846%`
   and `0.036439%`.
   These values use FDTDX's documented eta0 field-unit conversion; no fitted
   gain or endpoint rescaling was applied.

## Diagnostic limitation retained from the compact control

- The last two-window absorbed-power change is
  `0.970403%`, above
  the 0.5% gate in the eight-period quick run.
- The old compact artifact records apparent local-Q/flux mismatches of
  `98.738720%`
  (raw) and
  `98.959501%`
  (empty-subtracted).  They are retained for provenance but are **not physical
  closure results**: that artifact co-located material fields and did not apply
  the complete FDTDX eta0 field-unit conversion now certified by stage 49.
- In the independent source-only control, compact-Gaussian closed-box residuals
  are `3.1262%` (`+z`) and
  `4.2523%` (`-z`); enlarging the box gives
  `11.9783%` and `12.8954%`.
  The corresponding uniform-source large-box residual is only about 0.08%.

Therefore the source direction is not the failure.  The compact source remains
a useful AD--FD control but is not an energy-closure certificate.  The
production-width source/material calculation supersedes it for forward power:
it uses native component Yee samples and explicitly converts
`E_SI=eta0*E_internal`, `H_SI=H_internal`, and
`S_SI=eta0*S_internal`.  The previously unresolved material-bearing
cross-solver checkpoint is now closed without an empirical correction.

## Decision

FDTDX is validated for the production-width forward optical endpoints and for
compact-grid dispersive Au material AD.  It is therefore the selected route for
the next Au inverse-design validation.  This status does **not** yet validate a
production-width spatially varying Au gradient, thermal/PTE coupling, electrode
transport, or optimization.  The next fail-closed gate is a production-width
nonuniform-Au directional AD--FD smoke test using the same native-Yee loss
contract; optimization starts only after that gate and the thermal/electrical
chain pass.
