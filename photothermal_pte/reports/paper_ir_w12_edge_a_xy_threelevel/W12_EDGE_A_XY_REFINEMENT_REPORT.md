# W12 edge-a nested x/y refinement

Status: `BLOCKED_W12_EDGE_A_XY_MESH_CONVERGENCE`

This is a **paper-like scalar-Gaussian scenario with an explicitly assumed
12 µm waist**. It is not an experimentally reproduced or paper-certified
beam.

## Mesh contract

- Coarse: 100 nm outer + 50 nm within ±22 µm.
- Refined: 100 nm outer + 50 nm within ±22 µm + 25 nm within ±15 µm.
- The coarse artifact places
  `2.2928%` of absorbed power outside
  the finest square. That outer source support remains solved on the coarser
  nested levels; it is not cropped, deleted, smoothed, gained, tiled, or
  rescaled.
- Both use TaIrTe4 `dz=5 nm`, six PML boundaries, the same scalar source,
  material, incident reference, and control-volume definitions.

## Results

| Metric | 100 nm outer + 50 nm within ±22 µm | 100 nm outer + 50 nm within ±22 µm + 25 nm within ±15 µm | Relative change |
|---|---:|---:|---:|
| P_Q (W) | 2.255432362051e-11 | 2.254734722594e-11 | 0.0309% |
| P_six (W) | 2.251391095607e-11 | 2.250547177712e-11 | 0.0375% |
| Six-face closure | 0.1795% | 0.1861% | — |
| Auto-shutoff | 9.932420e-06 | 9.988890e-06 | — |
| Native Yee cells | 170,135,680 | 396,307,080 | — |

Exact cell-overlap remapping preserves fine-grid power to
`0.000e+00` relative error.

- Raw volume-weighted spatial-Q NRMSE:
  `1.4168%`
- Equal-power full 3D Q NRMSE:
  `1.4156%`
- Equal-power lateral Q NRMSE:
  `0.3737%`
- Equal-power vertical Q marginal NRMSE:
  `0.0780%`
- Equal-power Q correlation:
  `0.999895063`
- Hotspot displacement: `892.342 nm`
- `z=0.513588 µm` total-field E² raw area-weighted NRMSE:
  `0.0360%`
- `z=0.513588 µm` total-field E² equal-integral NRMSE:
  `0.0351%`

| Q component | Power change | Equal-power 3D NRMSE | Correlation |
|---|---:|---:|---:|
| x | 0.5860% | 1.1599% | 0.999932280 |
| y | 0.0900% | 1.4575% | 0.999888313 |
| z | 0.7089% | 16.1021% | 0.989679448 |

The E² plane is a total-field diagnostic and is not called a pure incident
beam waist measurement.

The full-3D discrepancy is localized: the layer at
`z=1.34003e-13 nm` contributes
`87.7200%` of
the squared equal-power 3D error. This localization is diagnostic evidence;
it does not permit replacing the failed 3D gate by either marginal metric.

## Gate

The strict spatial-Q promotion gate is 0.5%. The per-gate booleans are stored
in the summary JSON. Passing total power and lateral-Q metrics does not
override the failed full-3D spatial-Q gate.

No thermal, PTE, adjoint, gradient, or optimization calculation was run.
