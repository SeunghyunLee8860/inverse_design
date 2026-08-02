# Device-A Figure 3H registration audit

Status: `BLOCKED_FIG3H_TO_FIG2_ABSOLUTE_REGISTRATION_UNDERDETERMINED`

This is an offline coordinate audit. It did not execute Maxwell, thermal,
PTE, adjoint, AD-FD, or optimization code.

## Definite finding

The previous geometry script accepted the Figure-3H crop but did not use a
single Figure-3H pixel in its source-coordinate calculation. It chose a point
3 um inward from the normal of polygon vertices 4 and 7. Those vertices are
not adjacent: actual boundary vertices 5 and 6 lie between them.

The visible Figure-3H scan line is nearly vertical. Its acute directional
difference from the previously simulated chord normal is
`53.643 deg`. Therefore the completed old `s=2,3,4 um`
sweep is a valid numerical source-position sensitivity study, but it is not
a registered reproduction of the Figure-3H black dashed line.

## What is still missing

The paper does not publish absolute stage coordinates or enough explicitly
labelled shared fiducials to map this crop to the Figure-2 digitized polygon
without an additional image-registration assumption. The black scan line is
now digitized, but no new absolute source point is promoted and no new FDTD is
started by this checkpoint.
