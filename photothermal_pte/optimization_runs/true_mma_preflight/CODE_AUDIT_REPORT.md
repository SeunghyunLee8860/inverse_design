# Run016/017 code audit before restart

Status: `BLOCKED_PENDING_CONTACT_DOMAIN_MESH_AND_EB_ADFD_PREFLIGHT`

No new optimization has been launched from this branch yet.

## Defects found in the discarded Run014 path

1. The update called “MMA” was not MMA. It normalized the objective and
   morphology gradients, accumulated Adam first/second moments, normalized
   the Adam direction, added a fixed `move * direction`, and clipped to
   `[0,1]`.
2. Consequently `move=0.02` acted like a repeated endpoint-driving increment;
   it did not preserve the physical gradient magnitudes or solve constrained
   MMA subproblems.
3. Run014 had no hard maximum for a nonconverged beta stage. It executed more
   than sixty accepted beta-1 updates while the plateau gate remained false.
4. The current accepted-plot call contains one duplicate positional argument,
   which would raise `TypeError` on a clean execution of that revision.
5. Reporting normalized every objective with that evaluation's `sourcepower`
   although its gradient omitted a derivative of `sourcepower`. The restart
   uses the initial audited source power as one immutable conversion constant.
6. Contact-anchored 40→48 µm and 100→50 nm optical convergence artifacts were
   absent. The existing comparisons were for the older fixed-frame geometry.
7. A combined physical-density AD–FD certificate existed for `E||a`, but not
   for `E||b`.
8. In the thermal gray law, the endpoint derivative of `rho**1` at `rho=0`
   was incorrectly set to zero. It is now exactly one and is unit-tested.

## Physics chain confirmed

- Axis mapping: Lumerical `x=b`, `y=a`, `z=c`.
- Optical interpolation: componentwise complex epsilon from air to measured
  TaIrTe4, with `epsilon_z=epsilon_b` closure.
- Six optical boundaries are PML; the finite scalar Gaussian source is
  10 µm wavelength with target waist 8.5 µm.
- Native component Q is deposited by literal optical-dual-cell/thermal-cell
  material intersection; there is no nearest-cell relocation, clipping,
  smoothing, gain or global rescaling.
- Thermal model is explicit 3-D anisotropic FVM with explicit air, SiO2 and Si,
  finite TaIrTe4/SiO2 conductance, SiO2/Si conductance, far lateral/bottom
  Dirichlet reservoir and top convection.
- Electrical objective is the signed full-flake PTE terminal current from a
  density-dependent anisotropic weighting-potential solve, with top contact 1
  and bottom contact 0.
- Existing contact-anchored `E||a` combined AD–FD relative error is
  `1.3824775109623567e-05`; mapping, residual, energy and optical closure gates
  passed.

## File-by-file execution-path audit

The SHA-256 of every row is written by `audit_true_mma_preflight.py` to
`DEPENDENCY_MANIFEST.json`.  This is the complete runtime path used by the
supervisor; it is intentionally broader than only the newly edited files.

| File or group | Audit result |
|---|---|
| `run_contract.py` | Historical invalid Run014 stop state is schema-valid; it cannot be reported as completed. |
| `contract.py` | Contact-anchored geometry, `x=b, y=a`, 10 µm source, 8.5 µm target waist, 100 nm design pitch, 10 nm flake `dz`, 40 µm optical and 64 µm thermal spans are explicit. |
| `optical.py` | Six PML faces, scalar Gaussian, anisotropic complex TaIrTe4 endpoints and component-specific imported material are used; CPU fallback is absent. |
| `audit_optical_runsetup.py` | Reads back source/PML/mesh/material placement before a forward solve and fails closed. |
| `run_forward_gpu.py` | Runs GPU FDTD, extracts literal `Qx,Qy,Qz`, source power, six-face flux, shutoff and raw hashes without modifying Q. |
| `compare_forward_meshes.py` | Compares 40/48 µm and 100/50 nm artifacts on conservative common power bins; normalization is diagnostic only. |
| `finite_q_mapping.py`, `native_yee_q.py` | Component Yee coordinates and dual-cell volumes are retained; optical-to-thermal transfer is material-intersection conservative. |
| `production_density_mapping.py` | Latent→filter→projection JVP/VJP is the sole density mapping; uniform latent 0.5 maps exactly to physical 0.5 for every scheduled beta. |
| `build_nonuniform_complex_yee_jacobian.py`, `yee_material_jacobian.py` | Stores and applies component-specific complex density→Yee Jacobians, rather than an identity transpose. |
| `thermal.py`, `anisotropic_heat_fvm.py`, `cuda_thermal_adjoint.py` | Explicit heterogeneous 3-D FVM, anisotropic TaIrTe4 conductivity, finite internal interfaces, external boundaries and CUDA primal/adjoint solve are used.  The linear gray-law derivative at rho=0 was fixed and tested. |
| `electrical.py` | Density-dependent anisotropic weighting solve uses top=1 and bottom=0 contacts and integrates signed PTE current over the full flake. |
| `validate_combined_adfd.py` | Combined physical-density AD–FD now accepts both `Ea` and `Eb`; every plus/minus forward uses the selected polarization. |
| `evaluate_objective_gradient.py` | One production evaluation returns the coupled objective, full optical/thermal/electrical gradient and terminal-conductance gradient with artifact hashes. |
| `evaluate_binary_objective.py` | The final thresholded structure is re-solved, not inferred from the last gray evaluation. |
| `mma.py` | Persistent separable MMA subproblem with saved moving asymptotes and canonical `g(x)<=0` inequalities; no Adam moment or normalized update direction. |
| `run_true_mma_optimization.py` | Exact uniform start, measured beta convergence, restartable immutable evaluations, accepted-only iteration records/plots, exact final DRC and fresh binary solve. |
| `run_true_mma_dual_supervisor.py` | Enforces missing domain/mesh/`Eb` AD–FD gates, then runs `Ea` completely before `Eb`; completion is detected only from `passed=true`. |
| `publish_true_mma_accepted_updates.py` | Commits only tracked reports/plots/JSON and rejects `.fsp`/`.npz`; raw artifacts remain under `/data`. |

Tests cover the MMA constraint response and move bound, absence of the old
Adam state, uniform-start contract, low-beta constraint policy, beta minimum
updates, thermal endpoint derivative, axes, electrical and thermal analytic
controls, mappings and report schemas.  GPU-dependent convergence and `Eb`
combined AD–FD remain explicit preflight blockers until their new artifacts
exist and pass.

## Restart algorithm

- Exact uniform `rho=0.5`; no seed, symmetry or volume constraint.
- Persistent separable method of moving asymptotes with saved asymptotes.
- Objective and every inequality retain their physical gradient magnitudes;
  no direction normalization and no Adam moments.
- The move value is only an MMA trust-region upper bound, not a mandatory
  pixel increment or learning rate.
- Low beta is objective-led. The terminal-conductance inequality is active
  immediately; differentiable 500 nm solid/void inequalities begin at beta 8
  and tighten by recorded fixed stage caps.
- Beta advances only after measured objective plateau or MMA step stationarity
  with feasible active constraints. A 30-update stage ceiling fails closed; it
  never silently promotes beta.
- Final completion requires exact thresholded 0/1 density, zero global 500 nm
  bad nodes, gray fraction below 1%, and a fresh binary GPU-Maxwell/CUDA-
  thermal/electrical objective solve. No post-hoc morphology repair is used.
