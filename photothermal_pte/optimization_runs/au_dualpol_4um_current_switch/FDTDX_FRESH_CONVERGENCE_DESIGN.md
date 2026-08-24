# Fresh FDTDX convergence design for exact-binary Au

Status: **design contract only; no mesh-convergence claim and no optimizer permission**

This document defines the next FDTDX work after the four empty/full endpoint
controls.  It is deliberately independent of the Lumerical work in progress in
another session.

## What the endpoint matrix established—and did not establish

The validated four-case matrix established that the pinned FDTDX executable can
run the anchor grid for `Ea` and `Eb`, that empty and full masks remain exact
binary endpoints, and that the recorded optical energy/flux/stationarity gates
pass for those four controls.  Empty/full masks contain no fabricated edge,
minimum linewidth, minimum gap, or re-entrant corner.  They therefore cannot
certify the design-window x/y mesh, the external-domain mesh, the z-domain
extent, or a nontrivial antenna geometry.

The anchor is 196 x 196 x 160 Yee cells.  Its Au thickness is 50 nm and is
represented by eight z cells (6.25 nm per cell).  With the locked 4 um Au value
`n = 2.2 + 28.9i`, the amplitude skin depth is about 22.0 nm and the intensity
1/e depth is about 11.0 nm.  This is a useful scale comparison, not a convergence
certificate.  In particular, the anchor design-window pitch is still 100 nm.

## Why the new references are exact geometries

The relevant device papers use fabricated Au resonator or contact geometries and
vary physical dimensions, placement, or orientation.  They do not justify three
different fictitious Au densities for optical, thermal, and electrical physics:

- The 2022 full-Stokes PTE detector uses patterned 50 nm Au antennas and obtains
  the geometry by global/parameter optimization; absorption is evaluated from
  the physical lossy material field.
  [Nature Communications 13, 4408 (2022)](https://www.nature.com/articles/s41467-022-32309-w)
- The 2024 directional PTE work uses exact T, inverse-T, and propeller resonators
  and studies physical length, width, spacer, and rotation parameters.  It also
  reports sensitivity to rounded corners and fabrication-size deviations.
  [Nature Communications 15, 7117 (2024)](https://www.nature.com/articles/s41467-024-51599-w)
- The TaIrTe4 device analysis uses physical Au contacts and a device-specific
  weighting field; contact edges and crystal/electrode geometry affect the
  measured current.
  [Blevins et al., arXiv:2602.14959 (2026)](https://arxiv.org/html/2602.14959)

These papers motivate an exact-shape campaign.  They do **not** certify the
present FDTDX mesh, and this document does not treat them as mesh evidence.

The primary spatial reference is
`l_shape_4um_with_500nm_arms`: two 4 um by 500 nm arms joined at a corner.  It
contains straight edges, two orientations, outer corners, and a re-entrant
corner at the requested 500 nm DFM scale.  The separate
`parallel_bars_4um_by_500nm_with_500nm_gap` reference stresses the 500 nm void
gap.  Empty/full remain endpoint controls; x/y bars screen orientation bias.

## Ordered optical-only campaign

Every comparison uses both `Ea` and `Eb`, the same physical reference geometry,
raw complex fields, component-specific Yee dual volumes, separately refitted and
read-back dispersive materials, and a source-pair certificate for that exact
numerical contract.  Per-polarization power rescaling is forbidden.

1. **Time settling at the anchor spatial grid.** Hold Courant at 0.5 and compare
   16, 24, and 32 optical periods, retaining two successive comparisons.  The
   startup and phasor windows remain four periods.  This explicitly checks the
   long-time behavior that the endpoint control alone could not establish.
2. **Time-step convergence.** At the selected settled duration compare Courant
   factors 0.5, 0.375, and 0.25.  Refit/read back ADE material parameters and
   regenerate the source pair at every level.
3. **One spatial axis at a time.** Compare two successive pairs on each ladder:
   full-domain z resolution; design-window x/y resolution; outer flake/gap x/y
   resolution; lateral PML x/y resolution; bottom Si buffer; top source-to-PML
   gap; lateral material-to-PML gap; lateral PML thickness; and z-PML thickness.
4. **PML sensitivity.** Repeat the selected spatial contract with CPML alpha
   scales 0.5, 1, and 2.
5. **Joint confirmation.** Assemble the individually selected levels and compare
   that joint mesh with a deliberately finer feasible confirmation mesh.  Axis
   sweeps are not assumed additive.
6. **Reference rechecks.** Recheck empty/full, x/y orientation bars, and the
   500 nm gap-stress mask on the selected contract.  A candidate design is not
   admitted until this stage passes.

No GPU solve should start merely because the ladder exists.  The runner must
first support arbitrary hashed mesh/time contracts and compare fields on fixed
physical coordinates.

## Comparison and promotion rules

`Q` comparisons must conservatively restrict fine-grid cell-integrated
absorption (`q * component-specific Yee dual volume`) onto common physical
control volumes.  Complex `E` comparisons must interpolate each staggered Yee
component to the same fixed physical probe coordinates before computing an
NRMSE.  Array-index comparison across different nonuniform grids is forbidden.

The optical gates cover incident/source consistency, closed-surface flux versus
absorbed power, late-time stationarity, total and material-resolved absorption,
fixed-probe complex fields, and conservatively remapped absorption.  Thermal
temperature, PTE current, and current sign are downstream validation stages.
They require a separately closed physical device/contact/weighting-field
contract and are intentionally absent from the optical mesh certificate.

Passing this campaign would permit evaluation of exact-binary candidate
geometries.  It would still not validate a gray `rho`, `rho^3`, or independently
interpolated optical/thermal/electrical Au model, and it would not by itself
authorize restarting the historical optimizer.
