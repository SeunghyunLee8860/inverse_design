# Latest photothermal validation status

## v261 HEAT material and interface controls

- Branch: `agent/unblock-heat-material-interface-controls`
- Stacked base: `agent/validate-isolated-2um-heat-steady`
- Finite-Q source: PR #3 commit `053260d`
- PR #2 and PR #3 content: unchanged
- Status: `DEVICE_LICENSE_API_PROBE_PASSED_KAPPA_PENDING`
- Full-device HEAT cases executed: `false`
- Transient/PTE/adjoint/gradient/optimization executed: `false`

### Active blockers

- `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- `BLOCKED_INTERFACE_G_UNVERIFIED`
- `BLOCKED_REQUIRED_SOLVER_CONTROLS_UNVERIFIED`

### Key measurements

- Expected finite-Q SHA-256: `7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`
- Expected finite-Q power: `2.56071371086521e-12 W`
- Reintegrated finite-Q power: `2.56071371086521e-12 W`
- Reintegration relative error: `0`
- Q component-sum relative error: `2.24310e-16`
- `BLOCKED_Q_ARTIFACT_INCOMPATIBLE_WITH_2UM_FOOTPRINT`: release candidate
- Allowed import mismatch: `0.5%`
- DEVICE version: `7.17.4413` from the v261 installation
- DEVICE session startup/save/load/HEAT solve: `passed`
- License-probe temperature range: `[300.0, 300.0499752615] K`

Full-device HEAT, transient, PTE, adjoint, gradient, and optimization are not
part of this control-only branch. No isotropic fallback or modification of the
finite optical-Q artifact is permitted.
