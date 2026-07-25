# Latest photothermal validation status

## v261 HEAT material and interface controls

- Branch: `agent/unblock-heat-material-interface-controls`
- Stacked base: `agent/validate-isolated-2um-heat-steady`
- Finite-Q source: PR #3 commit `053260d`
- PR #2 and PR #3 content: unchanged
- Status: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- Independent anisotropic fallback:
  `VALIDATED_DIAGONAL_KAPPA_FVM_CONTROLS`
- Independent internal-interface fallback:
  `VALIDATED_FVM_INTERNAL_INTERFACE_G_CONTROLS`
- Common-physics 3D cross-validation:
  `VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION`
- Full-device HEAT cases executed: `false`
- Finite optical-Q imported into FVM: `false`
- Transient/PTE/adjoint/gradient/optimization executed: `false`

### Active blockers

- `BLOCKED_ANISOTROPIC_K_UNSUPPORTED` (native v261 HEAT only)
- `BLOCKED_INTERFACE_G_UNVERIFIED` (native v261 HEAT only)
- `BLOCKED_FINITE_OPTICAL_Q_FVM_IMPORT_UNVALIDATED`

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
- Exhaustive native v261 probe: LSF-native 3x1/1x3/3x3 matrices all returned
  scalar `0.0`; all 11 hidden-property candidates were rejected
- v261 HT database scan: 64 entries, 59 readable scalar conductivity models,
  0 non-scalar conductivity models, 5 unimplemented quaternary-alloy models
- x/y/z effective kappa from solver: `[0.0, 0.0, 0.0] W/(m K)`
- x/y/z heat-flux relative errors: `[100%, 100%, 100%]`
- Isotropic fallback used: `false`
- Independent conservative FVM x/y/z recovered kappa:
  `[14.4000000032, 3.80000000087, 1.00000000130] W/(m K)`
- Independent FVM x/y/z heat-flux relative errors:
  `[2.20e-10, 2.30e-10, 1.30e-9]`
- Independent FVM x/y/z temperature-profile relative errors:
  `[5.01e-11, 3.25e-11, 2.76e-11]`
- Independent FVM status: `VALIDATED_DIAGONAL_KAPPA_FVM_CONTROLS`;
  this is not reported as a v261 HEAT result
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

### Independent FVM internal-interface controls

- Status: `VALIDATED_FVM_INTERNAL_INTERFACE_G_CONTROLS`
- Internal face law:
  \(R''=\Delta z_1/(2k_1)+1/G+\Delta z_2/(2k_2)\)
- Conditions: \(G=7.37\times10^6\), \(G=1.1\times10^9\)
  \(\mathrm{W/(m^2K)}\), and perfect contact
- Meshes for every condition: `100/50/25 nm`
- Total cases: `9`; passed: `9`
- \(G=7.37\times10^6\) analytic/numerical interface jump:
  `3.518029903254 / [3.518029903254, 3.518029903257, 3.518029903248] K`
- \(G=1.1\times10^9\) analytic/numerical interface jump:
  `0.03623188405797 / [0.03623188405828, 0.03623188405885,
  0.03623188405572] K`
- Finite-\(G\) jump relative errors: all below `6.3e-11`
- Analytic series-resistance heat-flux relative errors: all below `5.4e-12`
- Material-1/material-2 flux mismatch: all below `9.4e-12`
- Global energy-balance relative errors: all below `2.3e-11`
- Temperature-profile relative errors: all below `1.1e-12`
- Perfect-contact extrapolated jump: roundoff (`<2.3e-12 K`)
- Perfect-contact raw adjacent-cell difference:
  `0.5 -> 0.25 -> 0.125 K`; finest/coarsest ratio `0.25`
- Solver attribution: independent conservative Cartesian Python/SciPy FVM;
  not a Lumerical HEAT result
- Next mandatory gate:
  `LUMERICAL_HEAT_VS_FVM_3D_ISOTROPIC_PERFECT_CONTACT_CROSS_VALIDATION`
- Optical \(Q\) import and full-device calculation remain prohibited until
  the 3D cross-validation and finite-Q import gates pass.

### 3D isotropic/perfect-contact HEAT-FVM cross-validation

- Status: `VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION`
- Lumerical: v261 DEVICE `7.17.4413`, `772706` tetrahedral elements,
  `128099` nodes
- FVM: `40 x 40 x 30 = 48000` Cartesian cells at `50 nm`
- Geometry: two scalar materials, `k=[10,2] W/(m K)`, perfect contact
- Source: asymmetric grid-aligned synthetic cuboid,
  `Q=1e15 W/m3`, prescribed power `1.92e-4 W`
- Boundary conditions: bottom `300 K`; all other external faces adiabatic
- \(T_{\max}\) difference / maximum FVM temperature rise: `0.226837%`
- Mean-\(T\) difference / maximum FVM temperature rise: `0.0440171%`
- Full 3D field NRMSE / maximum FVM temperature rise: `0.107756%`
- Full 3D field correlation: `0.999983190756`
- Source-power cross-solver difference: `0.400326%`
- Boundary-power cross-solver difference: `0.400034%`
- Lumerical/FVM energy errors: `2.90729e-6 / 1.76324e-11`
- Non-gating pointwise diagnostics: 99th percentile `0.452070%`; maximum
  source-edge point `1.05266%`
- Independent fresh-project rerun reproduced all declared metrics within
  `1e-10`
- Finite optical-Q artifact imported: `false`
- Next mandatory gate: `FINITE_OPTICAL_Q_CONSERVATIVE_IMPORT`

### Final control artifacts

- Execution:
  `validation/photothermal_stage1/29_validate_heat_material_interface_controls.py`
- Summary:
  `validation/photothermal_stage1/30_summarize_heat_material_interface_controls.py`
- Native anisotropy probe and validated FVM controls:
  `validation/photothermal_stage1/31_resolve_anisotropic_kappa.py`
- Conservative diagonal-tensor solver:
  `validation/photothermal_stage1/anisotropic_heat_fvm.py`
- Independent FVM interface execution:
  `validation/photothermal_stage1/33_validate_fvm_internal_interface_controls.py`
- Independent FVM interface report:
  `reports/fvm_internal_interface_controls/FVM_INTERNAL_INTERFACE_G_CONTROL_REPORT.md`
- Independent FVM interface summary/cases/raw manifest:
  `reports/fvm_internal_interface_controls/`
- 3D cross-validation execution:
  `validation/photothermal_stage1/34_validate_3d_isotropic_heat_fvm_crosscheck.py`
- 3D cross-validation report:
  `reports/fvm_3d_isotropic_cross_validation/HEAT_FVM_3D_ISOTROPIC_CROSS_VALIDATION_REPORT.md`
- 3D cross-validation summary/cases/raw manifest:
  `reports/fvm_3d_isotropic_cross_validation/`
- Anisotropic-\(\kappa\) report:
  `reports/heat_material_interface_controls/HEAT_ANISOTROPIC_K_SOLVER_REPORT.md`
- Internal-\(G\) report:
  `reports/heat_material_interface_controls/HEAT_INTERNAL_INTERFACE_G_SOLVER_REPORT.md`
- Machine-readable summary/cases/raw manifest:
  `reports/heat_material_interface_controls/`

Native v261 HEAT still cannot represent the requested conductivity tensor.
The validated FVM path now resolves the anisotropic equation and finite
internal-G law independently, and its common 3D scalar-isotropic/perfect-
contact solution agrees with v261 HEAT. Its next mandatory gate is the
conservative finite optical-Q mapping and reintegration test. Full-device
production, finite optical-Q import, transient, PTE, adjoint, gradient, and
optimization are not part of this checkpoint. No isotropic fallback or
modification of the finite optical-Q artifact is permitted.

## Mechanical/MAPDL route probe

- Official capability: orthotropic/full-anisotropic thermal conductivity and
  finite thermal contact conductance are supported
- Generated material path: `MP,KXX/KYY/KZZ`
- Generated interface path: `TARGE170/CONTA174`, pure thermal
  `KEYOPT(1)=2`, bonded `KEYOPT(12)=5`, and `TCC=G` at real constant 14
- Controls generated: x/y/z anisotropic kappa, `G=7.37e6`, `G=1.1e9`,
  and perfect-contact meshes at 100/50/25 nm
- Input-deck static audit:
  `PASSED_MECHANICAL_INPUT_DECK_STATIC_AUDIT`
- MAPDL executable:
  `BLOCKED_MECHANICAL_EXECUTABLE_UNAVAILABLE`
- Mechanical license feature:
  `BLOCKED_MECHANICAL_LICENSE_UNAVAILABLE`
- License server: reachable, but only Lumerical/optislang features are
  advertised; no `ansys`, `mech_1`, `mech_2`, `struct`, or `preppost`
- Actual Mechanical solver executed: `false`
- Mechanical solver validation claimed: `false`
- Execution:
  `validation/photothermal_stage1/32_validate_mechanical_thermal_controls.py`
- Reports:
  `reports/mechanical_thermal_controls/`

The Mechanical route is physically and API-capability compatible, but it
cannot be solver-validated on this host until Mechanical/MAPDL is installed
and an applicable Mechanical license feature is added.
