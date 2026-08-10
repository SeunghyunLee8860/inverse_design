# Run 012 — contact-anchored TaIrTe4, E parallel a

Status: `VALIDATED_PILOT_READY_FOR_CONTINUATION`

This run maximizes the signed full-sheet terminal PTE current for incident
`E || a` with Lumerical `x=b, y=a`.  The finite 24 x 24 um TaIrTe4 support has
only two fixed 2-um-deep TaIrTe4 contact strips at its top and bottom.  The
entire intervening 24 x 20 um area is a 100-nm-grid topology design region.
There is no fixed left/right TaIrTe4 frame, no symmetry, and no material-volume
constraint.  The initial latent density is uniform 0.5.

Required constraints are 500 nm minimum solid/void features and a permissive
terminal-conductance safeguard (`G_terminal >= 0.10 G_fully-solid`) that
prevents a numerically regularized but physically disconnected sheet.

Every solver evaluation, including rejected candidates and beta reprojections,
must publish its own plot.  Plot convention: black is physical density 1
(TaIrTe4); white is physical density 0 (void).

The geometry, component-Yee mapping, electrical weighting gradient, explicit
thermal gradient, and combined physical-density gradient passed their gates.
The first accepted beta-2 update raised the signed current from
`1.18256593e-23 A` to `1.18474896e-18 A`.  Complete objective/gradient
evaluations took 175.6 s initially and 160.2 s for the first candidate.
