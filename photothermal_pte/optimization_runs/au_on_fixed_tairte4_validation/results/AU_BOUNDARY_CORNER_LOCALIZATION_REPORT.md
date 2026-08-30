# Au boundary-quadrature corner-localization diagnostic

Status: `DIAGNOSED_AU_BOUNDARY_QUADRATURE_CORNER_DOMINANCE`

This is an offline analysis of the completed GPU forward and adjoint engine
HDF5 files. It launches zero Maxwell solves and uses no Lumerical license. It
examines the tangential-E part of the official x-normal boundary kernel. The
absolute HDF5 field normalization is intentionally not used: localization and
quadrature-convergence ratios are invariant to one global factor.

At 801 points per edge, the two exact trapezoid endpoint samples on each face
(`y=-10 um` and `y=+10 um`) contribute
`83.718282%` of the complete tangential-E proxy
integral. Both endpoints are large; the absolute maximum on both moving faces
occurs at `y=+10 um`. These are exactly the two corners where the moving
vertical Au face meets the fixed horizontal Au edges. In contrast, the
combined interior integral over `|y|<=9.5 um` changes by only
`0.004623%` between 201 and 6401 samples.

This establishes that the observed sampling drift is not distributed over the
smooth vertical face. It is localized at the sharp metal corner. The complete
AD result also contains the normal-D term, which the engine HDF5 export does
not provide together with component epsilon; therefore this analysis does not
replace the full AD--FD gate. It does, however, identify the correct next
control: keep the Au y-ends fixed far outside the illuminated and adjoint
support, then repeat the x-width FD and boundary adjoint on the corner-free
active face.

No production gradient is promoted, and Au thermal/electrical/PTE optimization
remains blocked.
