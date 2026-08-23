# Production inverse-design readiness

Status: `BLOCKED_MISSING_SHARED_LINEAR_MESH_AND_GRADIENT_CERTIFICATES`

All production entry points (`10`, `12`, and `13`) now call
`require_production_readiness()` before creating output directories or
compiling a Maxwell runner.  There is no environment-variable bypass.

The gate requires two new, machine-readable certificates:

1. A shared-linear full mesh certificate covering the complete optical z
   domain, optical x/y, previous-vs-late time-window stationarity,
   absorption-Q/closed-flux closure, thermal mesh, and electrical mesh.
2. A shared-linear multidirection combined AD-FD certificate that records the
   SHA-256 of that exact mesh certificate.

The historical partial-z and O3/TE1 AD-FD artifacts intentionally do not
satisfy this chain.  Optimization remains blocked until both new certificates
exist and all checks pass.
