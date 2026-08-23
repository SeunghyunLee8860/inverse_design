# Run 059: +45-degree contacts, evaporated SiO2, E||a

![Fixed crystal axes and rotated device](rotated45_fixed_crystal_axes_geometry.png)

This run repeats the bounded official-DFM/exact-repair inverse-design
protocol of Run 057 with the entire finite device rotated +45 degrees relative
to the fixed TaIrTe4 crystal axes.

- The TaIrTe4 flake remains exactly 24 um x 24 um and is rotated +45 degrees.
  Its physical area remains exactly 576 um2; only its global bounding box is
  larger.
- The optimization grid is the full local 24 um x 24 um device at 100 nm
  spacing. It rotates together with the flake.
- The low and high terminals occupy two opposite full flake edges, with a
  2 um overlap measured normal to each edge.
- As in Run 058, no Au is present in the optical or thermal model. The two
  terminal strips exist only as ideal equipotential boundary masks in the
  electrical weighting-field solve.
- No terminal-overlap nodes are locked to rho=1. The terminals do not alter
  optical, thermal, DFM, or exact-repair density; they enter only as ideal
  equipotential masks in the electrical weighting solve.
- Following the Run 058 approximation requested for this restart, Maxwell
  uses the centered 24 um square without optical Au, with x=b, y=a, z=c and
  E||a at 90 degrees. The +45-degree geometry is retained exactly in the
  thermal and electrical weighting solves. This neglects the small optical
  orientation response under the circular Gaussian and isotropic substrate.
- Maxwell uses Run 058's stable material layout: a central 20 x 24 um
  imported-density region plus 2 um TaIrTe4 rectangles at left and right,
  giving the same exact 24 x 24 um flake. These rectangles are TaIrTe4, not
  Au electrodes; thermal/electrical density remains designable over 24 x 24 um.
- Electrical conductivity, Seebeck coefficient, and thermal conductivity are
  evaluated in the same fixed crystal coordinates on the rotated device.
- The TaIrTe4/SiO2 interface scenario is `evaporated`, as in Run 057/058.
- One physical GPU is exposed. The optimizer uses NLopt LD_MMA with beta
  continuation 1 through 128 and exact 500 nm binary repair/evaluation.

Raw solver artifacts are written under
`/data/seunghyun/tairte4/artifacts/tairte4_rotated45_edge_contact_anchored/`.
Published checkpoints and the final report are written to `results_v5_no_Au/`.
The prior Au optical constants, Au thermal conductivity, and Au/TaIrTe4
interface surrogate are not active in this run.
