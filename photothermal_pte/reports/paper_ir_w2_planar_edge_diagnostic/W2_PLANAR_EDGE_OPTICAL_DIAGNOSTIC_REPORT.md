# Nominal-w0=2 µm planar/edge optical diagnostic

Status: `PARTIAL_W2_EDGE_ISOLATION_OBSERVABLE_Q_VALIDATED_AUTO_SHUTOFF_FAILED`

This is a reduced-cost optical edge-isolation diagnostic.  It is **not** a
paper-like result, not a realized 2 µm beam certificate, and not production
`Q`.  No thermal, PTE, adjoint, gradient, or optimization run was performed.

## Gates

- matched common/native control-volume closure <0.5% for all three cases:
  **True**
- 1.2→4 ps `P_Q` and normalized spatial-`Q` gates <0.5% for all cases:
  **True**
- auto-shutoff ≤1e-5 for all cases: **False**

The auto-shutoff failure is retained independently and is not overridden by
the observable-`Q` pass.

## Cases (4 ps)

| case | P_Q common (W) | P_six (W) | common closure | spatial Q 1.2→4 NRMSE | realized waist (µm) |
|---|---:|---:|---:|---:|---:|
| planar_a | 1.335217821e-16 | 1.336813128e-16 | 0.119337% | 0.000076% | 6.437054 |
| planar_b | 1.955253431e-16 | 1.957258783e-16 | 0.102457% | 0.000116% | 6.369224 |
| finite_edge_b | 9.663323072e-17 | 9.664803053e-17 | 0.015313% | 0.000336% | 6.368664 |


The requested scalar-source waist was 2 µm, whereas the fitted field-plane
effective waist is 6.37–6.44 µm.  The cases share the same nominal source, so
the planar/edge isolation remains a useful diagnostic, but the result must not
be described as a physically realized 2 µm Gaussian beam.

## Raw versus equal-power comparison

Raw absorbed power is never altered.  Equal-power normalization is used only
for the spatial-shape comparison and does not overwrite any saved artifact.

- planar-a vs planar-b raw-power relative difference:
  `46.437038%`
- planar-a vs planar-b equal-power spatial-Q NRMSE:
  `16.094523%`
- planar-b vs finite-edge-b raw-power relative difference:
  `50.577644%`
- planar-b vs finite-edge-b equal-power spatial-Q NRMSE:
  `100.185592%`

The plane used for the realized-beam fit is close to the scatterer and uses a
total-field downward decomposition.  Therefore the planar-to-edge center
change includes edge-scattered/evanescent field effects; it is not called a
literal source displacement.

## Material and coordinates

All cases use `x=b`, `y=a`, and the explicit 3D closure
`epsilon_z=epsilon_c=epsilon_b`.  Requested, fitted, and finite-dt complex
permittivities; component fields; `Qx/Qy/Qz`; realized control-volume bounds;
and independent Yee coordinates are retained in the summary and raw
artifacts.  No clipping, smoothing, gain, rescaling, tiling, or source
deletion was used.
