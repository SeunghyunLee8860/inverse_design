# Isolated 2 um TaIrTe4 steady-state HEAT validation

**Status: BLOCKED.**

- Optical baseline: `be2cbc2c9c77bbcc0265ce2c293affdbb08105de`
- Optical production code and validated Q values: unchanged
- Normalization: UNIT_RESPONSE_MODE, 1 W/m2
- Reported thermal quantity when unblocked: DeltaT / incident intensity
- Transient, PTE current, adjoint, gradient, and optimization: not run

## Mandatory gate results

| Gate | Result | Evidence |
|---|---|---|
| Validated-Q full-grid reintegration | PASS | relative error `0.0` |
| Q compatibility with 2 um footprint | FAIL | inside `32.552574432%`; outside `67.447425568%` |
| single_isotropic_slab offline reference | PASS | solver verified `True` |
| multilayer_SiO2_Si offline reference | PASS | solver verified `False` |
| interface_G_analytic offline reference | PASS | solver verified `False` |
| TaIrTe4 diagonal kappa API round trip | FAIL | requested `[14.4, 3.8, 1.0]`, returned `[0.0]` |
| Interface-G analytic solver control | NOT RUN | live DEVICE unavailable |

## Fail-closed blockers

- `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- `BLOCKED_Q_ARTIFACT_INCOMPATIBLE_WITH_2UM_FOOTPRINT`
- `BLOCKED_INTERFACE_G_UNVERIFIED`
- `BLOCKED_LUMERICAL_LICENSE_UNAVAILABLE`
- `BLOCKED_REQUIRED_SOLVER_CONTROLS_UNVERIFIED`

The immutable production Q grid spans 6 um by 6 um. Its total
power is `1.6790733985800054e-11 W`, while the power
inside the requested 2 um by 2 um TaIrTe4 footprint is
`5.465816178457092e-12 W`. Restricting the
source to the finite flake would discard most of the validated
power and violate the 0.5% conservation limit. Cropping, tiling,
gain, smoothing, and rescaling were not used.

## Full-device result

No full-device HEAT case was executed. Consequently there are no
claims for T(x,y,z), heat flux, energy balance, lateral/depth
convergence, or interface-G sweeps. This is required behavior because
the task states that any failed control stops the workflow before the
full 3-D model.

To unblock the physical run, all of the following are required:

1. A validated non-periodic Q artifact generated for the exact 2 um
   by 2 um TaIrTe4 volume, preserving the <0.5% FDTD-to-HEAT power
   identity without post-processing.
2. A HEAT solver/version or verified material route that stores and
   executes diag(14.4, 3.8, 1.0) W/(m K), plus a verified internal
   interface-G route that passes the analytic temperature-jump test.
3. A working DEVICE license session, followed by solver-backed
   multilayer SiO2/Si and interface-G controls.
