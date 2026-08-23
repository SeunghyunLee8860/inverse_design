# Production inverse-design readiness

Status: `BLOCKED_DEVICE_AND_NUMERICAL_CERTIFICATES`

All production entry points (`10`, `12`, and `13`) now call
`require_production_readiness()` before creating output directories or
compiling a Maxwell runner.  There is no environment-variable bypass.

The gate requires one confirmed physical-device contract and two new,
machine-readable numerical certificates:

1. `physical_device_contract.json` must have status
   `VALIDATED_AU_TAIRTE4_PHYSICAL_DEVICE_CONTRACT` and every required geometry,
   contact, axis, stack, illumination, and void-floor confirmation must be
   true. The committed file is deliberately blocked until target-device data
   are supplied.
2. A shared-linear full mesh certificate covering the complete optical z
   domain, optical x/y, previous-vs-late time-window stationarity,
   absorption-Q/closed-flux closure, thermal mesh, and electrical mesh. It must
   record the SHA-256 of the exact physical-device contract and current source
   calibration.
3. A shared-linear multidirection combined AD-FD certificate that records the
   SHA-256 of that exact mesh certificate.

The historical partial-z and O3/TE1 AD-FD artifacts intentionally do not
satisfy this chain.  Optimization remains blocked until both new certificates
exist and all checks pass.
