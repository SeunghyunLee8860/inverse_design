# Explicit thermal independent-grid contract audit

Status: `AUDITED_EXPLICIT_THERMAL_INDEPENDENT_GRID_PARAMETERS`

- legacy 100 nm baseline versus explicit parameter baseline, bitwise grid,
  material, kappa, and z-interface equality: `True`;
- `core_xy_cell_size_m`: independently changes lateral core cells;
- `flake_dz_m`: independently changes TaIrTe4 z cells;
- `design_dz_m`: independently changes design extrusion z cells.

Realized TaIrTe4/design z-cell counts:

- baseline: `4 / 6`;
- xy-only refinement: `4 / 6`;
- TaIrTe4-z-only refinement: `8 / 6`;
- design-z-only refinement: `4 / 12`.

This is a geometry/operator-parameter audit. No thermal solve or optimization
was run, and no convergence status is claimed here.
