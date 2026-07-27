# Latest photothermal validation status

## Large-background non-periodic inverse-design AD–FD

- PTE/nodal-contract audit:
  `AUDITED_PTE_DISCRETE_OPERATOR_AND_81X81_NODAL_CONTRACT`
- Physical design variables: `81 × 81` nodes on exact
  `[-1,1] um × [-1,1] um`, `25 nm` spacing; not 81 finite-width pixels
- Physical density: 2D nodal field extruded from `z=0` to `600 nm`
- PTE weighting surrogate:
  `dpsi/dx=dpsi/dy=1/(4 um)`; periodic derivative wrap absent
- PTE affine analytic / forward-source / temperature-source FD errors:
  `0 / 0 / 5.42831e-11`
- PTE meaning: uniform-45-degree surrogate only; not a solved finite-contact
  terminal weighting potential
- Explicit thermal grid-parameter audit:
  `AUDITED_EXPLICIT_THERMAL_INDEPENDENT_GRID_PARAMETERS`
- Independent parameters:
  `core_xy_cell_size_m`, `flake_dz_m`, `design_dz_m`
- Legacy 100 nm baseline versus explicit `100/25/100 nm` parameterization:
  bitwise-equal grid, material, kappa, and z-interface arrays
- Realized baseline / xy-refined / flake-z-refined / design-z-refined
  TaIrTe4/design z-cell counts:
  `4/6`, `4/6`, `8/6`, `4/12`
- This parameter audit does not claim thermal convergence
- v261 FDTD license/API gate:
  `PASSED_V261_FDTD_LICENSE_API_PROBE`
- FDTD application version / session / script / save / reload:
  `8.35.4522 / passed / passed / passed / passed`
- License/API probe solver and optimization execution:
  `false / false`
- Initial direct-probe failure cause:
  restricted sandbox blocked the localhost Ansys license socket; not seat
  exhaustion or missing entitlement
- A separate pre-existing GPU FDTD optimization process was observed and was
  neither started nor modified by this checkpoint; it must not overlap the
  timed matched CPU-TFSF gate
- Matched uniform-rho optical forward:
  `VALIDATED_MATCHED_RHO05_CPU_TFSF_FORWARD`
- Matched case: rho `0.5`, PML `32`, x/y stabilized and z standard, flake
  `dz=2.5 nm`
- PML-32 outer x/y expansion `6.4 -> 7.2 um` preserved realized PML-inner
  x/y at approximately `[-2,2] um`; ROI/TFSF/Q bounds were unchanged
- Matched `P_Q / P_six / closure`:
  `1.6887880194040323e-12 W / 1.6893345559747856e-12 W / 3.23522e-4`
- Matched `Qx / Qy / Qz`:
  `1.6885593488584841e-12 / 2.286705455481133e-16 / 0 W`
- Matched native Q SHA-256:
  `711c4c93589603f32bfc0525e1b63b36fd773a0ace561509ed0391cb2604ddb2`
- Matched native/complete wall time:
  `583.974 / 590.597 s`, contended reference only
- Support-remap spatial-deposition status:
  `VALIDATED_SUPPORT_REMAP_SPATIAL_CONVERGENCE`
- Matched optical flake `dz=5 -> 2.5 nm` mapped-power difference /
  volume-weighted spatial-Q NRMSE:
  `0.0179774% / 0.488139%`
- Lateral-integrated / depth-integrated energy NRMSE:
  `0.165346% / 0.0230061%`
- Support-remap coarse/fine exact-TaIrTe4 exterior nonzero counts:
  `0 / 0`
- Coarse/fine mapping SHA-256:
  `9971edd6bc61c0028d7fad7a86958099b6bcbe698aa1eedbf6a80a0c903eb290` /
  `d6691afe8034ffca10e058b9bb008d63f449d1de351b6d7bf70e54cd1a3c8145`
- The one-cell mapped-hotspot shift is between reflection-symmetric central
  thermal cells; the spatial NRMSE, not a chosen peak cell, is the primary
  convergence gate
- Fixed-Q thermal domain/depth/mesh status:
  `VALIDATED_FIXED_Q_THERMAL_DOMAIN_DEPTH_MESH_CONVERGENCE`
- Named TaIrTe4 footprint scenarios: `4 um / 6 um`; neither is promoted as
  fabrication truth
- Native thermal grid:
  `32 um lateral / 20 um Si / 100-25-100 nm core-flake-design`
- Independent controls:
  `40 um lateral`, `30 um Si`, and
  `50-12.5-50 nm core-flake-design`
- Worst fixed-Q thermal common-field/scalar convergence metric:
  `0.314485%` (limit `1%`)
- Worst Q-mapping / energy-balance / linear-residual errors:
  `3.58746e-16 / 5.63685e-12 / 1.95372e-11`
- Native 4 um / 6 um Tmax:
  `1.069958441e-7 / 1.051871964e-7 K per 1 W/m2`
- Refined 4 um / 6 um Tmax change:
  `0.283335% / 0.314485%`
- Thermal physical law held fixed in this checkpoint:
  TaIrTe4 `diag(14.4,3.8,1.0) W/(m K)`,
  bottom `G=7.37e6`, deposited-design endpoint `G=7.37e4`,
  air `G=1`, SiO2/Si `G=1.1e9`, exposed-SiO2/air
  `h=10 W/(m2 K)`
- Lateral/bottom reported powers are numerical truncation-boundary fluxes,
  not intrinsic physical heat-path fractions
- Fixed-Q PTE thermal adjoint/FD, 81x81 mapping, combined AD-FD, latent
  AD-FD, transient, and optimization were not executed by this checkpoint
- Status: `VALIDATED_MIXED_CPU_TFSF_GPU_FIELDREGION_OPTICAL_ADFD`
- Protected design/PTE ROI: exactly `x,y=[-1,1] µm`
- Optical TaIrTe4: 100 nm thick and extended through lateral PML as the
  large-background model; this does not set the finite thermal flake footprint
- Inverse-designed material: actual SiO2, `2 µm × 2 µm × 600 nm`
- Optical boundaries: six PML faces; periodic boundaries forbidden
- Requested illumination: normal-incidence ideal plane wave; Gaussian and
  periodic/Bloch boundaries are not substituted
- Installed v261 GPU TFSF probe:
  `BLOCKED_GPU_TFSF_UNSUPPORTED`
- Explicit engine error:
  `GPU simulation does not support the use of TFSF sources`
- Bloch/periodic source crossing transverse PML: rejected as an invalid
  source/boundary pairing
- Official all-PML Diffracting source: executed through a `24 µm` domain and
  `20 µm` aperture; best displayed ROI intensity RMS `4.974%`,
  peak-to-peak `15.184%`, max phase `2.027°`, and `Ez/Ex=6.466%`; rejected
- GPU source-integrity status:
  `BLOCKED_GPU_ONLY_SIX_PML_IDEAL_PLANE_WAVE`
- User-authorized CPU TFSF source gate: six PML, `4×4 µm` lateral domain,
  `2.6 µm` TFSF span, exact central `2×2 µm` ROI
- CPU TFSF PML-24/PML-32 status:
  `VALIDATED_CPU_TFSF_4UM_DOMAIN_2UM_ROI_SOURCE_GATE`
- PML-24 ROI mean-|E|² error / spatial RMS / peak-to-peak:
  `0.0144312% / 0.00000856% / 0.00005593%`
- PML-24 closed-box energy error: `0.00007052%`
- PML-24 native engine / Python run / complete-session wall times:
  `3.270815 / 5.525168 / 10.508481 s`
- PML-32 native engine / Python run / complete-session wall times:
  `4.346625 / 7.462453 / 12.466042 s`
- Validated large-background geometry: FDTD outer x/y `±3.2 µm`, realized
  PML-inner x/y `±2.0 µm`, minimum TFSF-to-PML-inner gap `209.677 nm`
- Design optical endpoints: air `n=1` and actual SiO2 `n=1.38`
- Thermal model: explicit design-SiO2/TaIrTe4/bottom-SiO2/Si domains
- Exposed SiO2/air: Robin `h=10 W/(m2 K)` to `300 K`
- Exposed TaIrTe4 sidewalls: `G_air=1 W/(m2 K)`, not adiabatic
- PTE weighting field: uniform 45-degree direction,
  `grad(psi)=(xhat+yhat)/(4 µm)`
- Flat baseline `P_Q/P_six/closure`:
  `1.3567412718462558e-12 W / 1.3567343935235152e-12 W / 5.06976e-6`
- Mixed rho=0.5 `P_Q/P_six/closure`:
  `1.689091619450848e-12 W / 1.6895947794697648e-12 W / 2.97799e-4`
- GPU adjoint / centered FD (`h=0.01`) gradients:
  `7.316714058728351e-13 / 7.317295351329038e-13 W/rho`
- Direct mixed optical AD–FD relative error: `7.94409e-5`
- CPU/GPU adjoint complex-field NRMSE: `2.19978e-5`
- Local optical-to-thermal Q mapping power/transpose errors:
  `2.39121e-16 / 8.07866e-16`
- Material-support mapping status:
  `VALIDATED_LOCAL_Q_OPTICAL_THERMAL_MAPPING`
- Corrected 4 um / 6 um mapping SHA-256:
  `9971edd6bc61c0028d7fad7a86958099b6bcbe698aa1eedbf6a80a0c903eb290` /
  `73617d249cfa261dd87f1c2b94a38cdb328b6793212b698e31034470430e0ba2`
- Mapped source outside exact TaIrTe4 support:
  `0 W`, `0` nonzero cells
- Fixed-local-Q explicit thermal status:
  `VALIDATED_NAMED_LOCAL_Q_EXPLICIT_THERMAL_ADFD_SCENARIOS`
- Named thermal footprints: `4 × 4 um` and `6 × 6 um`; neither is
  promoted as the unconfirmed fabrication geometry
- Central 2 um TaIrTe4 average DeltaT, 4 um / 6 um:
  `8.27733069135e-8 / 8.06057559735e-8 K`
- TaIrTe4 Tmax, 4 um / 6 um:
  `1.07023617860e-7 / 1.05215987646e-7 K`
- Worst thermal AD-FD / energy / linear-residual errors:
  `1.30271e-4 / 3.51866e-12 / 1.02259e-11`
- Global thermal hotspots: inside TaIrTe4 at approximately
  `(0.05, 0.05, -0.0125) um` in both named scenarios
- Current thermal-source scope: validated local `Omega_Q` only; absorption
  outside the local volume for a truly extended ideal plane wave is omitted
- Remaining physical inputs before combined/full-latent PTE:
  actual finite illumination footprint and actual thermal TaIrTe4 footprint
- Terminal PTE, combined/full-latent PTE, transient, and optimization for
  this large-background plane-wave chain: not executed

### Superseded periodic certificate

- The following section records the immutable 6 µm periodic numerical
  checkpoint only. It does not validate the finite 2 µm problem.

## Inverse-design paper-reduced thermal/PTE AD–FD

- Status: `VALIDATED_PAPER_REDUCED_RHO_DEPENDENT_THERMAL_PTE_ADFD`
- Material label: `n=4 optical proxy + paper SiO2 thermal boundary`
- TaIrTe4 kappa: `diag(14.4, 3.8, 1.0) W/(m K)`
- Fixed substrate Robin G: `7.37e6 W/(m2 K)`
- Design boundary:
  `G(rho_bar)=1+rho_bar*(G_SiO2-1) W/(m2 K)`
- Thermally-grown baseline / evaporated sensitivity:
  `7.37e6 / 7.37e4 W/(m2 K)`
- Thermal-material-only AD–FD errors:
  `1.86887e-8 / 1.17052e-11`
- Combined physical-rho errors at steps `0.0025 / 0.00125`:
  `1.02384% / 0.495604%`
- Combined latent step sweep at `0.01 / 0.005 / 0.0025`:
  `9.96381% / 1.55543% / 6.87508%`
- Selected bracketed latent FD step: `0.005`
- Selected optical / thermal-material / combined directional gradients:
  `3.20854e-19 / 2.72040e-20 / 3.48058e-19`
- Energy balance / linear residual:
  `2.04340e-13 / 8.70484e-12`
- Bulk air/SiO2/Si kappa and SiO2/Si G in this reduced model:
  `omitted`
- Remaining blocker:
  `BLOCKED_PHYSICAL_WEIGHTING_POTENTIAL_OR_FINITE_FLAKE_MASK`
- Terminal current, transient, and PTE optimization: `not executed`

## v261 HEAT material and interface controls

- Branch: `agent/validate-fvm-thermal-physical-model`
- Stacked base: `agent/unblock-heat-material-interface-controls`
- Immutable numerical checkpoint:
  `437ec0644b15a4b9a6919a0151e4aa531fb1e0ab` (PR #4)
- Finite-Q source: PR #3 commit `053260d`
- PR #2 and PR #3 content: unchanged
- Status: `BLOCKED_ANISOTROPIC_K_UNSUPPORTED`
- Independent anisotropic fallback:
  `VALIDATED_DIAGONAL_KAPPA_FVM_CONTROLS`
- Independent internal-interface fallback:
  `VALIDATED_FVM_INTERNAL_INTERFACE_G_CONTROLS`
- Common-physics 3D cross-validation:
  `VALIDATED_3D_ISOTROPIC_HEAT_FVM_CROSS_VALIDATION`
- Finite optical-Q conservative mapping:
  `VALIDATED_FINITE_OPTICAL_Q_FVM_IMPORT`
- Full-device HEAT cases executed: `false`
- Finite optical-Q mapped to FVM control volumes: `true`
- Finite optical-Q used in a thermal solve: `true`
- Multi-material production FVM convergence:
  `VALIDATED_MULTIMATERIAL_FVM_PRODUCTION_CONVERGENCE`
- Physical-model scenarios:
  `VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS`
- Final experimental prediction promoted: `false`
- Transient/PTE/adjoint/gradient/optimization executed: `false`

### Active blockers

- `BLOCKED_ANISOTROPIC_K_UNSUPPORTED` (native v261 HEAT only)
- `BLOCKED_INTERFACE_G_UNVERIFIED` (native v261 HEAT only)
- `BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED`

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
- Subsequent 3D cross-validation and finite-Q import gates: `completed`
- Full-device thermal calculation after these prerequisite gates:
  `executed with the independent FVM path`

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
- Finite optical-Q mapped into thermal control volumes: `true`
- Finite optical-Q used in a thermal solve: `true`
- Subsequent finite-Q conservative import gate: `completed`

### Finite optical-Q conservative FVM import

- Status: `VALIDATED_FINITE_OPTICAL_Q_FVM_IMPORT`
- PR #3 artifact SHA-256:
  `7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`
- Shape/order: `[80,80,41]`, `x,y,z`
- Incident-intensity normalization: `1 W/m2`
- Mapping: elementwise Q copy; FVM cell widths equal the original
  trapezoidal quadrature weights
- Original/mapped Q-array SHA-256:
  `ff1484537aadfc36d90c2035280da9ad3a2e59895e9ba06a65bea30623e3715d`
- Original nested-trapezoid power:
  `2.56071371086521e-12 W`
- FVM `sum(Q*dV)` power:
  `2.56071371086521e-12 W`
- Mapping relative error: `0`
- Interpolation, clipping, smoothing, gain, rescaling, crop, tiling, and
  outside-flake deletion: all `false`
- `5772` nonzero boundary samples at `z=5.790264e-23 m` are excluded by the
  stored strict boolean mask but lie inside the explicit `1e-15 m`
  roundoff-inclusive physical mask; all were preserved unchanged
- Exact-flake production source: `[76,76,21]` cells with bounds exactly
  `[-1,1] um x [-1,1] um x [-100,0] nm`
- Exact-flake mapping: one-to-one deposition of each original
  `Q*w_x*w_y*w_z` nodal energy parcel into its physical boundary/interior
  cell
- Source-energy/mapped-cell-power SHA-256:
  `dece160abd9965047d2902e6d1bf07fad0146fc306a543a60d79b51a7fd31caf`
- Exact-flake summed power: `2.56071371086521e-12 W`; relative error `0`
- Nonzero source energy deleted: `0 W`
- Empirical gain, global rescaling, and sample averaging in exact-flake
  deposition: all `false`
- Independent import rerun reproduced SHA and power exactly
- Finite optical-Q used in thermal solve: `true`
- Subsequent anisotropic finite-G production and convergence gate:
  `completed`

### Multi-material anisotropic finite-G FVM production

- Status: `VALIDATED_MULTIMATERIAL_FVM_PRODUCTION_CONVERGENCE`
- Attribution: independent conservative Cartesian Python/SciPy FVM;
  not a Lumerical HEAT result
- Numerical-convergence checkpoint: `32 um x 32 um` lateral domain,
  `20 um` Si depth, native optical x/y source grid
- Active solid cells: `1,625,064`
- Material conductivity:
  TaIrTe4 `diag(14.4, 3.8, 1.0)`, SiO2 `1.38`, Si `145 W/(m K)`
- Interfaces:
  `G_bottom=G_top=7.37e6 W/(m2 K)`,
  `G_SiO2/Si=1.1e9 W/(m2 K)`
- Exact source power: `2.56071371086521e-12 W`
- Q mapping relative error in every case: `0`
- Reference maximum unit response:
  `3.12002156771575e-7 K/(W/m2)`
- Reference TaIrTe4 volume-average unit response:
  `2.25508130625815e-7 K/(W/m2)`
- Reference energy-balance relative error: `3.36166e-12`
- Total sensitivity cases: `22`; equation/conservation passes: `22`
- Final `16 -> 32 um` lateral-domain changes
  (`Tmax`, flake average, 3D probe NRMSE):
  `[0.00489969%, 0.00676634%, 0.00517751%]`
- Final `10 -> 20 um` Si-depth changes:
  `[0.0178338%, 0.0246859%, 0.0189037%]`
- Final native -> refined thermal-mesh changes:
  `[0.140694%, 0.0933887%, 0.0666590%]`
- \(G_{\rm bottom}\) sweep:
  `1e6, 3e6, 7.37e6, 1.5e7, 3e7, 1e8, perfect`
- \(G_{\rm top}\) sweep:
  `7.37e4, 7.37e5, 7.37e6, 7.37e7, perfect`
- SiO2/Si `1.1e9` versus perfect contact: completed
- Exposed-surface adiabatic versus `h=10 W/(m2 K)`: completed
- Refined source treatment: native optical x/y cells, piecewise-constant
  `2x` subdivision in z with exact child-power conservation
- No Q clipping, smoothing, gain, global rescaling, periodic tiling, or
  outside-flake deletion was used
- TaIrTe4 `kz=1.0 W/(m K)` remains an estimated physical input; interface-G
  results retain the full sensitivity sweep
- This checkpoint parameter set is not a unique final experimental
  prediction

### FVM thermal physical-model sensitivity

- Status: `VALIDATED_FVM_THERMAL_PHYSICAL_MODEL_SCENARIOS`
- Fabrication status: `BLOCKED_FABRICATION_GEOMETRY_UNCONFIRMED`
- \(G_{\rm top}=7.37e6\) W/(m2 K): named numerical-convergence checkpoint
  scenario
- \(G_{\rm top}=7.37e4\) W/(m2 K): named earlier evaporated-SiO2 estimate
  scenario
- Neither \(G_{\rm top}\) value is promoted as uniquely correct
- \(G_{\rm top}=7.37e4\) versus checkpoint:
  \(T_{\max}\) `+7.48897%`, flake average `-0.0800617%`,
  common flake 3D NRMSE `2.15495%`
- TaIrTe4 \(k_z=[0.5,1.0,2.0]\) W/(m K):
  numerical scenarios, not a confidence interval; \(k_x=14.4\),
  \(k_y=3.8\) unchanged
- \(k_z=0.5\): \(T_{\max}\) `+12.3111%`; \(k_z=2.0\):
  \(T_{\max}\) `-6.29652%`
- Far-x/y fixed versus adiabatic with fixed bottom:
  \(T_{\max}\) change `+0.0475768%`
- Exposed convection `h=[0,5,10,20] W/(m2 K)`: completed
- Lateral/bottom fractions are numerical truncation-boundary fluxes, not
  physical heat-path fractions
- Geometry A: suspended/overhanging disk outside the flake
- Geometry B: 100 nm SiO2 support annulus connects the disk overhang to the
  surrounding bottom oxide
- Geometry B versus A: \(T_{\max}\) `-39.7356%`, flake average `-37.0430%`,
  common flake 3D NRMSE `27.5386%`
- Geometry-B native-to-refined numerical changes:
  `[0.789170%, 0.743380%, 0.522514%]` for
  `[Tmax, flake average, common flake 3D NRMSE]`
- Physical support-geometry variation is much larger than its numerical mesh
  error
- Published promoted metadata:
  `provisional_until_sensitivity_passes=false`,
  `next_required_gate=null`
- Raw per-case JSON metadata remains unchanged for provenance
- PR #3 commit is not in PR #4 ancestry; clean reproduction requires an
  external artifact with SHA-256
  `7a63f82842751e7623e895701bac4ce92558679ed71bf13a3d404695a150e794`
- Missing or mismatched PR #3 artifacts fail closed before import or solve

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
- Finite-Q import execution:
  `validation/photothermal_stage1/35_validate_finite_q_fvm_import.py`
- Finite-Q import report:
  `reports/fvm_finite_q_import/FINITE_OPTICAL_Q_FVM_IMPORT_REPORT.md`
- Finite-Q import summary/cases/raw manifest:
  `reports/fvm_finite_q_import/`
- Multi-material production execution:
  `validation/photothermal_stage1/36_run_fvm_multimaterial_thermal.py`
- Domain/depth/mesh/interface/boundary sensitivity:
  `validation/photothermal_stage1/37_run_fvm_production_sensitivity.py`
- Production report generation:
  `validation/photothermal_stage1/38_summarize_fvm_multimaterial_thermal.py`
- Multi-material production report/summary/cases/convergence/raw manifest:
  `reports/fvm_multimaterial_thermal/`
- Physical-model scenario execution:
  `validation/photothermal_stage1/39_validate_fvm_thermal_physical_model.py`
- Clean-checkout fail-closed reproduction:
  `validation/photothermal_stage1/40_reproduce_fvm_thermal_physical_model.py`
- Physical-model report generation:
  `validation/photothermal_stage1/41_summarize_fvm_thermal_physical_model.py`
- Physical-model report/summary/cases/raw manifest:
  `reports/fvm_thermal_physical_model/`
- Anisotropic-\(\kappa\) report:
  `reports/heat_material_interface_controls/HEAT_ANISOTROPIC_K_SOLVER_REPORT.md`
- Internal-\(G\) report:
  `reports/heat_material_interface_controls/HEAT_INTERNAL_INTERFACE_G_SOLVER_REPORT.md`
- Machine-readable summary/cases/raw manifest:
  `reports/heat_material_interface_controls/`

Native v261 HEAT still cannot represent the requested conductivity tensor.
The validated FVM path now resolves the anisotropic equation and finite
internal-G law independently, and its common 3D scalar-isotropic/perfect-
contact solution agrees with v261 HEAT. The finite optical-Q mapping now
preserves the PR #3 source exactly. The independent anisotropic, finite-G,
multi-material FVM solve and its domain, substrate-depth, mesh, interface-G,
and exposed-boundary sensitivity are now complete. The reported temperature
is a unit response, not a finite-power laser temperature. Physical-model
sensitivity shows that disk-support geometry and uncertain material/interface
inputs dominate the remaining interpretation; no single scenario is called a
final experimental prediction. No transient, PTE, adjoint, gradient, or
optimization is claimed at this checkpoint. No isotropic fallback or
modification of the finite optical-Q artifact was used.

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

## Finite in-flake SiO2 proxy optical Q

- Branch: `agent/validate-inflake-proxy-optical-q`
- Base: PR #3 head `053260da6fd0caec28ce155221bd18f683a0e5e7`
- Status: `VALIDATED_FINITE_INFLAKE_PROXY_OPTICAL_Q`
- PR #2–#5: unchanged
- PR #3 radius-1.5-µm artifact: not reused or cropped

Fresh v261 GPU FDTD was run for a centered radius-0.8-µm, 600-nm-high SiO2
disk completely inside the 2 µm × 2 µm × 100 nm TaIrTe4 footprint. Outside
the disk is air, with no support annulus, overhang support, or oxide pillar.
The finite Gaussian source uses a 2 µm waist, 6.8 µm aperture, 3–6 µm source
band, 4 µm analysis point, and measured central incident intensity of 1 W/m2.

The promoted x-polarized result uses a 16 µm lateral domain, 24 PML layers,
and 5 nm TaIrTe4 dz:

- `P_Q=2.0361088604691824e-12 W`
- `P_six=2.040668004695463e-12 W`
- six-face closure `0.223414304%`
- `Qx/Q=0.993324070`, `Qy/Q=0.006675930`, `Qz/Q=0`
- raw NPZ SHA-256
  `2ecdb8a8a2a01f85635914357ce05aab834576a66069cdc024a5dca49b0c71c3`

Final convergence changes are:

- domain 12→16 µm: P_Q 0.0240581%, P_six 0.0232486%, spatial L2 0.025513%;
- PML 16→24: P_Q 0.000270435%, P_six 0.00134641%, spatial L2 0.000594892%;
- flake dz 5→2.5 nm: P_Q 0.0769457%, P_six 0.0503751%, spatial L2 0.608514%.

Source-off, empty-stack x/y/45-degree, finite-flat x/y/45-degree, proxy,
six-face closure, domain, PML, mesh, finite-value, geometry, and P_Q
reintegration gates pass. Raw NPZ/FSP files are not committed. Thermal, PTE,
adjoint, gradient, and optimization were not run.
