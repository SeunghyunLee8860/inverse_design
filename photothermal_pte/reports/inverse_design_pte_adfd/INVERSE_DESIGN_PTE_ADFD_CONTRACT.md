# Inverse-design PTE AD–FD contract

Status: `IN_PROGRESS_CONTRACT_FROZEN_BEFORE_SOLVER_CERTIFICATES`

This contract separates the actual periodic inverse-design problem from the
finite radius-1.5-µm and radius-0.8-µm disk fixtures. The disks are fixed
forward-validation proxies. They are not optimization variables, optimized
geometries, or production inverse-design results.

## Git and artifact provenance

The working branch was created in a clean worktree from fetched
`origin/main`:

- canonical repository: `SeunghyunLee8860/inverse_design`;
- fetched main: `3efa5e90b8cfafdb1b4d2e4dd8aa562e57ec5dcf`;
- validated thermal physical-model head:
  `5f2fb2272c5a387a6e958ba052137aa454b688ab`;
- validated in-flake optical-proxy head:
  `6f93f2d7cf589824328a02bcfcdf7fd9a9933221`;
- working branch: `agent/validate-inverse-design-pte-adfd`.

Existing validation branch heads and their raw result files are immutable
inputs. Large FSP/NPZ artifacts remain outside Git and are accepted only after
path, byte-size, and SHA-256 checks.

The radius-0.8-µm proxy artifact, when used as a forward-only control, is:

- path:
  `/home/seunghyun/tairte4/inverse_design_heat_export/photothermal_pte/validation/photothermal_stage1/output/inflake_proxy_optical_q_fresh/cases/proxy_x_r0p8_L16_pml24_dz5/finite_q_on_artifact.npz`;
- size: `8695110` bytes;
- SHA-256:
  `2ecdb8a8a2a01f85635914357ce05aab834576a66069cdc024a5dca49b0c71c3`;
- power at central incident intensity 1 W/m²:
  `2.0361088604691824e-12 W`.

It must not be used as the inverse-designed optical source.

## Actual inverse-design optical contract

The production inverse-design variable and realized density are:

- latent variable `z`: endpoint-free `240 x 240` periodic grid;
- latent spacing: `25 nm x 25 nm`;
- mapping: periodic conic filter, tanh projection, exact periodic fencepost;
- physical density `rho_geom`: `241 x 241 x 13`;
- physical design-node spacing: `25 nm x 25 nm x 50 nm`;
- design region: `6 µm x 6 µm x 600 nm`, `z=0...600 nm`;
- TaIrTe4: `6 µm x 6 µm x 100 nm`, `z=-100...0 nm`;
- SiO2: `285 nm`, `z=-385...-100 nm`;
- optical Si depth: `2 µm`;
- optical x/y boundaries: periodic;
- optical z boundaries: PML;
- source: x or y polarized, 3–6 µm broadband;
- analysis wavelength: 4 µm;
- optical mesh: auto non-uniform, conformal variant 1, accuracy 5;
- TaIrTe4 z override: 5 nm;
- design optical interpolation: air (`n=1`) to a high-index endpoint (`n=4`).

The repository defines the high-index design endpoint only by optical index.
It does not identify a fabrication material or provide its thermal
conductivity, its contact conductance to TaIrTe4, or a differentiable thermal
mixing law. Consequently:

`BLOCKED_FULL_RHO_DEPENDENT_THERMAL_MATERIAL_MODEL`

is retained. No thermal property may be inferred from `n=4`.

## Thermal and PTE variable contract

Temperature is solved as the rise `theta = T - T_bath`:

`K_T theta = M_V R_ot Q_opt + b_theta`.

- `Q_opt`: optical-grid volumetric absorbed-power density, W/m³;
- `R_ot`: conservative optical-to-thermal density transfer;
- `M_V`: thermal active-cell volume operator, m³;
- `K_T`: thermal conductance matrix, W/K;
- `b_theta`: nonzero boundary load, W;
- `theta`: active-cell temperature rise, K.

`M_V` contains each thermal cell volume exactly once. `R_ot` never performs a
global gain, clipping, smoothing, tiling, or source deletion. For a density
map it must satisfy

`1^T M_V R_ot = 1^T M_opt`

to the declared tolerance. If the optical and thermal flake cells share exact
edges, `R_ot` is an exact scatter/reindex operation and this fact is recorded
instead of being described as interpolation.

The TaIrTe4 conductivity used in numerical controls is

`diag(14.4, 3.8, 1.0) W/(m K)`

with repository x/y/z equal to crystallographic a/b/c. The `kz=1.0` value is
an estimated scenario, not a confidence interval. SiO2 and Si values must be
copied with their source labels from the validated FVM configuration.

Named interface scenarios remain scenarios rather than unique physical truth:

- `7.37e6 W/(m² K)`: numerical-convergence checkpoint;
- `7.37e4 W/(m² K)`: earlier evaporated-SiO2 estimate label without a
  traceable repository literature source;
- `1.1e9 W/(m² K)`: SiO2/Si baseline candidate;
- perfect contact.

Internal material conductance, physical solid/air exposed surfaces, and
artificial truncation reservoirs are disjoint face classes. Every active
boundary face must have exactly one class. For the eventual periodic
inverse-design thermal model, x/y are periodic, the far-bottom Si face is the
temperature reservoir, and top/exposed-surface adiabatic or Robin cases are
separate named scenarios.

## Local PTE functional

The first certificate uses the signed local surrogate

`F_local = -(1/sqrt(2)) (Vm)^T [sigma_a S_a D_x theta + sigma_b S_b D_y theta]`.

Values inherited from the stated paper contract are:

- `sigma_a = 4.91e5 S/m`;
- `sigma_b = 1.10e5 S/m`;
- `S_a = -6e-6 V/K`;
- `S_b = 27e-6 V/K`.

The crystal axes are fixed as `a=x`, `b=y`, `c=z`. `D_x`, `D_y`, cell
volumes, and the finite TaIrTe4 mask used by the forward functional are the
same arrays used by its transpose. The surrogate omits the weighting-field
magnitude and therefore has units A m. It is not reported as electrode
current. A current in A requires a declared `g` in 1/m:

`I_approx = g F_local`.

## Exact thermal adjoint

For a fixed thermal operator,

`c_T = dF_local/dtheta`

and

`K_T^T lambda_T = c_T`.

The gradients are

`w_Q_th = M_V^T lambda_T`

and

`w_Q_opt = R_ot^T M_V^T lambda_T`.

The transpose is literal for the declared density coordinates because the
volume metrics are explicitly owned by `M_opt` and `M_V`. Weighted-adjoint
factors must not be inserted a second time.

If the inverse-design density later changes thermal conductivity, contact
conductance, or a boundary coefficient, the complete thermal contribution is

`dF/drho |_thermal = -lambda_T^T (dK_T/drho) theta
                     +lambda_T^T (db_theta/drho)`.

This term is absent only in the explicitly named
`FIXED_K_OPTICAL_ONLY_CERTIFICATE`. Omitting it must never be described as the
full physical inverse-design gradient.

## Optical absorption adjoint source

For a collocated component `c`,

`Q_c = (omega epsilon_0 / 2) Im(epsilon_c) |E_c|^2`.

After applying the exact transpose of `R_ot` and the optical quadrature, the
field-mediated Wirtinger source is

`q_E,c = dF/dE_c* =
         (omega epsilon_0 / 2) Im(epsilon_c) w_Q,c E_c`.

The source is returned to the native Yee component locations with the exact
transpose of the same collocation operator used by the forward `Q`. Periodic
duplicate endpoints are folded according to the active FieldRegion source
copy. No empirical component scaling or directional calibration is allowed.

If `Im(epsilon)` depends explicitly on density, an additional direct term is
required:

`dF/drho |_explicit-loss =
  w_Q^T [(omega epsilon_0/2) |E|^2 d Im(epsilon)/drho]`.

The field-mediated source is solved with the existing v261 FieldRegion
volume-current path. Its Yee-volume overlap and measured conformal
`rho -> epsilon` transpose remain:

`dF/d rho_solver =
 (d epsilon_Yee/d rho_solver)^T (dF/d epsilon_Yee)`.

## Full chain rule to the optimization variable

For the probe-safe production evaluation,

`rho_solver = delta + (1 - 2 delta) rho_geom`.

The complete staged chain is

`z -> filter -> projection -> periodic fencepost/extrusion
   -> rho_geom -> rho_solver -> epsilon_Yee -> E
   -> Q_opt -> R_ot -> theta -> F_local`.

For the fixed-K optical-only certificate,

`dF/dz = J_map^T (1-2 delta)
         [dF/drho_solver |_field + dF/drho_solver |_explicit-loss]`.

For a future rho-dependent thermal material model, add the thermal matrix
term before applying the mapping VJP:

`dF/dz = J_map^T [dF/drho_geom |_optical
                  -lambda_T^T (dK_T/drho_geom) theta
                  +lambda_T^T (db_theta/drho_geom)]`.

## Certificate ladder and claims

1. solver-free PTE `D/D^T` tests;
2. finite-G/Robin thermal assembly regression;
3. T-space AD–FD;
4. thermal-grid Q-space AD–FD;
5. optical-to-thermal transfer weighted dot test and power conservation;
6. Maxwell `q_E` source pairing and optical-Q directional AD–FD;
7. physical-density AD–FD;
8. latent mapping AD–FD.

Passing stages 1–5 certifies only the thermal/PTE chain. Passing stage 6
certifies the Maxwell-to-Q extension. Passing stages 7–8 certifies the
fixed-K optical-only inverse-design gradient. A full physical inverse-design
claim remains blocked until the design material, thermal interpolation, and
rho-dependent interface model are supplied and their matrix derivative is
independently certified.

