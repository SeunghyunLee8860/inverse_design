# v261 sharp-interface Au adjoint-path audit

Status: `AUDITED_V261_SHARP_INTERFACE_PATH_ADFD_PENDING`

This checkpoint inspects the bundled v261 LumOpt implementation. It performs
no Maxwell solve and is not an AD--FD certificate.

The approved fallback is a counter-clockwise binary Au polygon, extruded by a
fixed 50 nm thickness. The Au side uses the exact named Ordal `(n,k)` material;
the outside is air. No intermediate Au/air permittivity is constructed.

The installed boundary kernel uses both continuity variables required at a
material boundary: tangential electric field and normal electric displacement.
For the present width control only the two vertical x-normal faces move; the
top, bottom, y-normal faces and thickness remain fixed.

All `10` expected implementation checks passed. This confirms that
v261 contains the intended shape-derivative route, but the numerical gate is
still open: the same geometry must pass GPU adjoint versus central FD without
empirical normalization or gradient rescaling.

Official API contract: https://optics.ansys.com/hc/en-us/articles/360052044913-Optimizable-Geometry-Python-API
