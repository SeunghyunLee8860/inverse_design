# Latest photothermal validation status

## v261 HEAT material and interface controls

- Branch: `agent/unblock-heat-material-interface-controls`
- Stacked base: `agent/validate-isolated-2um-heat-steady`
- Finite-Q source: PR #3 commit `053260d`
- PR #2 and PR #3 content: unchanged
- Status: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
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
- Requested tensor write/readback before save: `[14.4, 3.8, 1.0] -> [0.0]`
- Requested tensor readback after reload: `[0.0]`
- Fresh vector/row/column/3x3-diagonal encoding probe: all returned scalar
  `0.0`; no tensor encoding round trip passed
- x/y/z effective kappa from solver: `[0.0, 0.0, 0.0] W/(m K)`
- x/y/z heat-flux relative errors: `[100%, 100%, 100%]`
- Isotropic fallback used: `false`
- Internal finite-\(G\) candidate: v261 `temperature` BC on the shared
  `material:material` surface with `thermal impedance = 1/G`
- Finite-\(G\) property write/save/reload: `passed` for both requested values
- \(G=7.37e6\) jump: `1.13687e-13 K` versus expected `6.55977 K`
- \(G=7.37e6\) jump/flux/transmission/energy errors:
  `[100%, 86.46%, 57.54%, 28.77%]`
- \(G=1.1e9\) jump: `2.27374e-13 K` versus expected `0.0565884 K`
- \(G=1.1e9\) jump/flux/transmission/energy errors:
  `[100%, 56.18%, 119.13%, 59.57%]`
- Finite-\(G\) candidate control status:
  `FAILED_INTERFACE_G_ANALYTIC_CONTROL`
- Verified internal-\(G\) path status: `BLOCKED_INTERFACE_G_UNVERIFIED`
- Perfect-contact mesh controls (100/50/25 nm): `passed`
- Perfect-contact interface jumps:
  `[2.27374e-13, 1.13687e-13, 2.84217e-13] K`
- Perfect-contact heat-flux errors: all below `1.1e-13`

### Final control artifacts

- Execution:
  `validation/photothermal_stage1/29_validate_heat_material_interface_controls.py`
- Summary:
  `validation/photothermal_stage1/30_summarize_heat_material_interface_controls.py`
- Anisotropic-\(\kappa\) report:
  `reports/heat_material_interface_controls/HEAT_ANISOTROPIC_K_SOLVER_REPORT.md`
- Internal-\(G\) report:
  `reports/heat_material_interface_controls/HEAT_INTERNAL_INTERFACE_G_SOLVER_REPORT.md`
- Machine-readable summary/cases/raw manifest:
  `reports/heat_material_interface_controls/`

Full-device HEAT, transient, PTE, adjoint, gradient, and optimization are not
part of this control-only branch. No isotropic fallback or modification of the
finite optical-Q artifact is permitted.
