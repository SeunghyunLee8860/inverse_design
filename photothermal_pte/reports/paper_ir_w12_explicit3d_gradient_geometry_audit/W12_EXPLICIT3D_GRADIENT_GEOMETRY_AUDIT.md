# W12 explicit-3D gradient geometry audit

Status: `COMPLETED_OFFLINE_W12_EXPLICIT3D_GRADIENT_GEOMETRY_AUDIT`

This is a read-only/offline audit of the existing thermal field NPZ.  It did
not execute FDTD, a thermal solve, PTE, adjoint, AD-FD, or optimization.

## Implementation findings

- thermal derivatives use actual cell-centre coordinate differences; a fixed
  50 nm denominator is not used;
- the crystal contract is lab `x=b`, lab `y=a`;
- the thickness average is explicitly `dz` weighted and is reproduced from
  the saved 3-D field to a maximum absolute difference of
  `0.000000000e+00 K`;
- gradients are mask-aware: centred stencils are used in the interior and a
  TaIrTe4-side one-sided stencil at the flake edge.  They are not formed
  across the TaIrTe4/air material interface and then masked;
- the prior plot selected a separate color limit for every a/b panel.  That
  is unsuitable for visual magnitude comparison.  The new projection plots
  use one shared a/b color scale for each projection;
- a one-cell raw maximum remains noise-sensitive.  Raw maximum and p99 are
  therefore reported together.

## Rotation identity

For every case and each of surface, midplane, and thickness-average fields,
the audit recomputed

`|grad T|^2 = dxT^2 + dyT^2 = dnT^2 + dtT^2`.

The maximum pixelwise relative error is `8.567624346e-16`.  See
`gradient_rotation_identity_error.png`.

## Edge-normal derivative and line integral

At `t=(x+y)/sqrt(2)=0`, and separately at the tangent coordinate of each raw
edge-gradient peak, the audit compares the bilinearly sampled field
`(-dxT+dyT)/sqrt(2)` with the numerical derivative of the sampled `T(n)`.
It also reconstructs the temperature with
`T(n1)+integral(dnT dn)`.  The JSON and CSV retain the derivative NRMSE,
correlation, endpoint closure, peak positions, and reconstructed-temperature
NRMSE for every z projection.  A second independent linecut uses only actual
thermal cell centres lying on a constant-`t` diagonal and reaches the final
inside-flake cell; it does not use bilinear interpolation.

- bilinear central-line maximum derivative NRMSE:
  `1.131572%`;
- bilinear central-line maximum integral closure:
  `0.027450%`;
- exact-cell central-line maximum derivative NRMSE:
  `2.802435%`;
- exact-cell central-line maximum derivative NRMSE after excluding only the
  final one-sided edge cell:
  `0.589573%`;
- exact-cell central-line maximum integral closure:
  `0.106176%`;
- minimum derivative correlation over all tests:
  `0.998873309`.

The larger exact-line derivative NRMSE is localized mainly to the final
inside-flake cell: the implemented Cartesian one-sided x/y stencil and an
independent one-sided derivative along a 45-degree diagonal are different
finite-resolution operators.  The integral closure remains below
`0.106176%`.
This is a finite-grid boundary-stencil diagnostic, not a coordinate-rotation
identity failure.

Across every case/projection, raw max divided by p99 is at most
`1.013956`.  Thus the reported ordering is not produced by
one isolated extreme cell, although p99 remains the safer comparator.

## Maxwell b/a edge-normal ratios

| z projection | raw maximum b/a | p99 b/a |
|---|---:|---:|
| surface | 0.69086967 | 0.69037979 |
| midplane | 0.77274431 | 0.77234775 |
| thickness_average | 0.79893425 | 0.79894198 |

## Analytic b/a edge-normal ratios

| z projection | raw maximum b/a | p99 b/a |
|---|---:|---:|
| surface | 1.47128138 | 1.46782363 |
| midplane | 1.47789240 | 1.47992519 |
| thickness_average | 1.47948262 | 1.47960443 |

The paper does not explicitly identify Fig. 3G as a top-surface, midplane, or
thickness-average comparator.  The three projections are therefore retained
without silently promoting one of them to a paper-exact observable.

## Provenance

- input: `/data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_50nm_maxwell_analytic_explicit3d_20260731/w12_50nm_maxwell_analytic_explicit3d_fields.npz`
- input SHA-256: `9b2287b5b18eb9c4d9c164ddd45d750ae05ff846d6a0f0e3936465e413be47ac`
- JSON: `w12_explicit3d_gradient_geometry_audit.json`
- CSV: `w12_explicit3d_gradient_geometry_audit.csv`
- command: `/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python photothermal_pte/validation/paper_ir_sanity/audit_w12_explicit3d_gradient_geometry.py --input-npz /data/seunghyun/tairte4/artifacts/paper_ir_lumerical_sanity/w12_50nm_maxwell_analytic_explicit3d_20260731/w12_50nm_maxwell_analytic_explicit3d_fields.npz --report-dir photothermal_pte/reports/paper_ir_w12_explicit3d_gradient_geometry_audit`
