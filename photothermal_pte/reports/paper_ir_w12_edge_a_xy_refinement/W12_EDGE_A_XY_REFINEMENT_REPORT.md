# W12 edge-a nested x/y refinement

Status: `BLOCKED_W12_EDGE_A_XY_MESH_CONVERGENCE`

This is a **paper-like scalar-Gaussian scenario with an explicitly assumed
12 µm waist**. It is not an experimentally reproduced or paper-certified
beam.

## Mesh contract

- Coarse: 100 nm over the complete Q/closure region.
- Refined: the same 100 nm outer region plus 50 nm on
  `x,y in [-22,22] µm`.
- The 100 nm artifact places only 0.0560% of absorbed power outside that fine
  square. That outer source support remains solved at 100 nm; it is not
  cropped, deleted, smoothed, gained, tiled, or rescaled.
- Both use TaIrTe4 `dz=5 nm`, six PML boundaries, the same scalar source,
  material, incident reference, and control-volume definitions.

## Results

| Metric | 100 nm | Nested 50 nm | Relative change |
|---|---:|---:|---:|
| P_Q (W) | 2.256660507557e-11 | 2.255432362051e-11 | 0.0545% |
| P_six (W) | 2.253087111044e-11 | 2.251391095607e-11 | 0.0753% |
| Six-face closure | 0.1586% | 0.1795% | — |
| Auto-shutoff | 9.971810e-06 | 9.932420e-06 | — |
| Native Yee cells | 64,064,520 | 170,135,680 | — |

Exact cell-overlap remapping preserves fine-grid power to
`1.433e-16` relative error.

- Raw volume-weighted spatial-Q NRMSE:
  `1.4126%`
- Equal-power full 3D Q NRMSE:
  `1.4095%`
- Equal-power lateral Q NRMSE:
  `0.5206%`
- Equal-power Q correlation:
  `0.999896897`
- Hotspot displacement: `70.711 nm`
- `z=0.5 µm` total-field E² raw area-weighted NRMSE:
  `0.0659%`
- `z=0.5 µm` total-field E² equal-integral NRMSE:
  `0.0642%`

The E² plane is a total-field diagnostic and is not called a pure incident
beam waist measurement.

## Gate

The strict spatial-Q promotion gate is 0.5%. The per-gate booleans are stored
in the summary JSON. A failed full-area 50 nm run is retained only as a
diagnostic: its time stepping reached auto-shutoff, but root-filesystem
exhaustion prevented project collection/write. It is not used as a physical
result.

No thermal, PTE, adjoint, gradient, or optimization calculation was run.
