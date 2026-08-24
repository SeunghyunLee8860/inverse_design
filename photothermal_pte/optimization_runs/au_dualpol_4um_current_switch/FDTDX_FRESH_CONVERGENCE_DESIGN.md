# Fresh FDTDX convergence design for exact-binary Au

Status: **L500 time settling validated; no mesh-convergence claim and no optimizer permission**

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

## Completed L500 time-settling certificate

The first campaign stage is complete on the fixed `196 x 196 x 160` anchor at
Courant 0.5. All raw artifacts remain outside Git under
`/home/seunghyun200/fdtdx_results/l500_time_settling_01a8ad8a_20260824`.
The fail-closed certificate was generated from clean commit `5e376ce1` at
`time_settling_certificate_5e376ce1/FDTDX_FRESH_TIME_SETTLING_CERTIFICATE.json`;
its SHA-256 is
`20ab99b8488606475d2ed8d604d1810c9f3953176b68f42ed0689685ed505ab0`.
All 21 top-level gates and all five selection gates passed. The certificate is
explicitly not a mesh certificate and keeps optimizer start forbidden.

The canonical case-file SHA-256 values are:

- 16 periods: `8e0d3d547c5f35b23738046f5c6ff1e0f85c6c6cc05929dd7cd928bf7e6b232f`
- 24 periods: `7ea87f6ffdf1f9557dfae2fca9c65b710df95feb4da50b43f117dff8e14955d7`
- 32 periods: `d4f673e6574543aa5057cef9f0e8fcd6ae357b16ef577ba38ef9813f53c2c266`

The matching source-pair certificate SHA-256 values are:

- 16 periods: `de14efff04778a4fea7aeaaded88c53822087b864d77d17acfe3d97185692de1`
- 24 periods: `2e1edd6f05f8798e368d4f19463204bd31dff3ee5293d3e53680ffde77c0566b`
- 32 periods: `78439db4caa795c3c3e1d08d26c4cef02c61297f596e752bb197802f0ea08d06`

The 16-period material cases were correctly rejected as unsettled. `Ea` failed
complex-field stationarity at `1.1229036e-2`; `Eb` failed field stationarity at
`1.7813813e-2` and previous/late spatial Q at `5.5404220e-3`. Both 24- and
32-period Ea/Eb cases passed every internal material/energy/flux/stationarity
gate. Independent NPZ reload then produced these worst-polarization successive
metrics:

| Optical metric | Limit | 16 to 24 | 24 to 32 |
|---|---:|---:|---:|
| source power relative change | 5e-3 | 5.7604523e-7 | 3.4562714e-7 |
| Q/closed-flux relative | 2e-2 | 7.8704948e-4 | 7.7062780e-4 |
| fine-case complex-E stationarity | 5e-3 | 3.4493703e-4 | 2.4179021e-4 |
| total Q relative change | 1e-2 | 2.2200505e-5 | 1.5007209e-6 |
| material/Cartesian-component Q max change | 2e-2 | 1.4088296e-3 | 2.4009730e-4 |
| fixed 8 x 8 um probe complex-E NRMSE | 2e-2 | 3.7799424e-5 | 6.0565673e-6 |
| component-Yee-volume Q L2 NRMSE | 5e-2 | 7.5788175e-4 | 1.0268401e-4 |

The certificate independently re-hashes all three canonical cases, all three
source pairs, all six material reports and NPZs; reconstructs the exact
375-pixel L500 mask; reconstructs component-specific Yee volumes from grid
edges and placement; recomputes Au/TaIrTe4 powers and previous/late field and Q
metrics; and compares only the fixed physical `[-4,+4] um` probe at
`z=0.250 um`. Because this ladder holds the spatial grid exactly fixed, no
interpolation or conservative remap is necessary; exact common physical cells
are used.

The selected minimum settled duration is therefore **24 periods**, with
**32 periods** as the required confirmation. The next allowed FDTDX action is
the 24-period Courant ladder `[0.5, 0.375, 0.25]`. It must create a new hashed
case and source pair at every Courant value; optimization remains forbidden.

No further GPU solve should start merely because a later ladder exists. The
runner and comparison implementation for that ladder must first preserve the
hashed numerical contract and fixed-physical-coordinate rules.

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
