# Paper-IR physical-coordinate figure correction

Status: `CORRECTED_PAPER_IR_NONUNIFORM_GRID_FIGURES`

## What was wrong

Several paper-IR plotting paths displayed physical cell fields with
`imshow(field.T, extent=[first_center, last_center, ...])`.  `imshow` gives
every array element the same screen width, but the explicit thermal FVM uses
a nonuniform Cartesian mesh (100 nm or 50 nm in the core and progressively
coarser cells outside it).  The old figures therefore preserved array order
and color values but moved features to incorrect physical coordinates.  Even
on a uniform common grid, centre endpoints as an image extent introduce a
half-cell boundary offset.

The numerical fields, gradients, powers, temperatures, ratios, JSON, CSV,
and raw NPZ artifacts were not recomputed or modified by this correction.

## Correction

All physical field maps under `validation/paper_ir_sanity` now use one of two
explicit contracts:

- FVM cell data: exact saved `x_edges_m` and `y_edges_m` with
  `pcolormesh(..., shading="flat")`;
- nodal/common-grid data: dual control-volume edges constructed from the
  saved centre coordinates, followed by the same edge-aware `pcolormesh`.

The only remaining `imshow` in that validation package is the literal
microscopy raster in `digitize_device_a_geometry.py`; it is not a physical
coordinate cell field.

## Regenerated figures

Thirty-five tracked PNG files were regenerated from existing artifacts,
without a new FDTD solve.  They cover:

- the 100 nm and 50 nm Maxwell/analytic explicit-3D Q, temperature, and
  gradient maps;
- the coordinate-identity and finite-difference/least-squares gradient
  audits;
- the 50/25 nm interface-downstream and optical-refinement maps;
- the W12 and W2 planar/edge Q comparisons;
- the straight-edge and Device-A thermal/PTE summaries;
- the tracked Device-A material-support Q partitions.

Saved external per-case `device_a_ir_thermal_pte.png` and
`straight_45_edge_thermal_control.png` files with an accompanying
`thermal_pte_fields.npz` were also regenerated in place using their exact
FVM edges.  Raw NPZ/FSP files were not changed.

The figure records in six affected `RAW_ARTIFACT_MANIFEST.json` files were
updated to the new PNG byte sizes and SHA-256 values.  Raw-artifact records
and their hashes were not changed.

## Interpretation of the reported yellow edge feature

In the corrected 100 nm analytic gradient map, the strong lower-edge feature
is located near the actual thermal cell position around
`(x,y)=(-3.85,-3.85) µm`, rather than being stretched to roughly
`(-8.0,-8.0) µm` by equal-pixel display.  The feature itself remains in the
stored temperature field; this correction changes its plotted position and
area, not its numerical value.

The existing independent checks remain unchanged:

- maximum pixelwise Cartesian-to-edge gradient identity error:
  `8.567624346238417e-16`;
- exact linecut derivative correlation: at least `0.998873`;
- gradient-integral temperature reconstruction closure: at most
  `0.106176%`.

Thus the old image geometry was wrong, while the saved gradient rotation and
temperature differentiation were not invalidated by this plotting fix.

## Regression gates

- coordinate helper preserves arbitrary nonuniform cell edges;
- centre-to-dual-edge conversion is tested explicitly;
- a static test rejects future `imshow` use in paper-IR physical-field
  plotters;
- 19 focused numerical/plotting tests pass;
- no FDTD, adjoint, AD-FD, or optimization solve was run for this correction.

## Strict centered-gradient display contract

The subsequently approved gradient display uses a stricter mask.  A cell is
shown only when its `-x`, `+x`, `-y`, and `+y` neighbours are all TaIrTe4.
Every other flake cell is stored/displayed as `NaN`, not zero, and is excluded
from plotted gradient statistics.

- 100 nm thermal grid: 41,041 flake cells, 40,186 strict-valid cells, 855
  masked cells (`2.08328%`);
- 50 nm thermal grid: 140,715 flake cells, 139,128 strict-valid cells, 1,587
  masked cells (`1.12781%`).

The historical one-sided arrays and metrics remain in the raw NPZ and
reports for provenance.  The regenerated gradient PNGs now apply the strict
four-neighbour mask.  Temperature and Q figures are unaffected.
