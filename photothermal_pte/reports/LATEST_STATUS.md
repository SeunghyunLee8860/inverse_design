# Latest photothermal validation status

## Isolated 2 um TaIrTe4 steady-state HEAT

- Branch: `agent/validate-isolated-2um-heat-steady`
- Optical baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Optical code and validated production Q: unchanged
- Status: `BLOCKED`
- Full-device HEAT cases executed: `false`
- Transient/PTE/adjoint/gradient/optimization executed: `false`

### Active blockers

- `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- `BLOCKED_Q_ARTIFACT_INCOMPATIBLE_WITH_2UM_FOOTPRINT`
- `BLOCKED_INTERFACE_G_UNVERIFIED`
- `BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE`

### Key measurements

- Validated full-grid Q power: `1.6790733985800054e-11 W`
- Q power inside requested 2 um footprint: `5.465816178457092e-12 W`
- Predicted import mismatch: `67.447425568%`
- Allowed import mismatch: `0.5%`
- v261 diagonal-kappa request `[14.4, 3.8, 1.0]` returned `[0.0]`

No isotropic fallback, Q clipping, gain, smoothing, rescaling, or
periodic tiling is permitted.
