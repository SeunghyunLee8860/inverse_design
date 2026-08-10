# NLopt LD_MMA reproduction code manifest

This is the complete Python-code index for the historical contact-anchored
TaIrTe4 PTE `NLopt LD_MMA` optimization (Run020/Run021) and the deliberately
separate pure-terminal-current `LD_MMA` continuation (Run030/Run031).
Run020/Run021 retain their historical terminal-conductance inequality for
provenance. Run030/Run031 remove that optional connectivity guardrail and are
the current production entry point.

## Entry points

| Role | Tracked source |
|---|---|
| E||a then E||b GPU supervisor | `run_nlopt_mma_dual_supervisor.py` |
| Actual `nlopt.LD_MMA` callback/update driver | `tairte4_flake_topology/run_nlopt_mma_optimization.py` |
| Pure-current E||a then E||b GPU supervisor | `run_pure_current_ld_mma_dual_supervisor.py` |
| Pure-current `nlopt.LD_MMA` driver | `tairte4_flake_topology/run_pure_current_ld_mma_optimization.py` |
| Pure-current electrical/optimizer audit | `PURE_CURRENT_LD_MMA_AUDIT.md` |
| Artifact/report publisher | `publish_true_mma_accepted_updates.py` |
| GPU/physics preflight audit | `audit_true_mma_preflight.py` |

Both optimization drivers create the exact NLopt object with
`nlopt.opt(nlopt.LD_MMA, variable_count)`, supplies an analytic objective
gradient, and impose the NLopt box bounds `0 <= latent <= 1`. The historical
driver adds its archived conductance inequality. The pure-current driver has
no inequality at \(\beta<8\), then only the two 500-nm morphology
inequalities. There is no custom MMA implementation, Adam state, manual move
limit, normalized-gradient update, or post-update clipping.

## Full objective and gradient chain

| Role | Tracked source |
|---|---|
| Common evaluator, PTE objective, constraints, plots, raw-artifact SHA manifest | `tairte4_flake_topology/run_true_mma_optimization.py` |
| FDTD forward + thermal/electrical adjoints + Maxwell adjoint + pullback | `tairte4_flake_topology/evaluate_objective_gradient.py` |
| Immutable geometry/axis/source/mesh contract | `tairte4_flake_topology/contract.py` |
| Density filter, projection, 500 nm morphology constraints/audit | `tairte4_flake_topology/optimization_support.py` |
| Density-to-imported-material and complex-Yee Jacobian mapping | `run_002_gaussian10_w8p5_current_max/production_density_mapping.py` |
| Optical project/material/source/monitor construction | `tairte4_flake_topology/optical.py` |
| Conservative Yee-to-thermal source mapping | `finite_inverse_design/finite_q_mapping.py` |
| Native component-resolved absorbed-power construction | `finite_inverse_design/native_yee_q.py` |
| Component-wise Yee material-Jacobian helpers | `finite_inverse_design/yee_material_jacobian.py` |
| Explicit anisotropic thermal FVM and material/interface terms | `tairte4_flake_topology/thermal.py` |
| GPU thermal forward/adjoint linear solves | `cuda_thermal_adjoint.py` |
| Density-dependent electrical/weighting-potential and PTE current | `tairte4_flake_topology/electrical.py` |

## Tests and certificates

| Role | Tracked source/result |
|---|---|
| LD_MMA API/objective/constraint callback tests | `tairte4_flake_topology/tests/test_nlopt_mma_driver.py` |
| Mapping/filter/morphology tests | `tairte4_flake_topology/tests/test_optimization_support.py` |
| Thermal FVM tests | `tairte4_flake_topology/tests/test_thermal.py` |
| Electrical solver tests | `tairte4_flake_topology/tests/test_electrical.py` |
| Preflight result and dependencies | `true_mma_preflight/TRUE_MMA_PREFLIGHT.json`, `true_mma_preflight/DEPENDENCY_MANIFEST.json` |

## Reproduction invocation

From the repository root, on a licensed v261 GPU-capable host:

```bash
TAIRTE4_PURE_CURRENT_LD_MMA_GPU=<physical_gpu_index> \
  /home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python \
  photothermal_pte/optimization_runs/run_pure_current_ld_mma_dual_supervisor.py
```

The supervisor verifies the immutable base FSP SHA-256, the component-Yee
Jacobian certificate, and the physics preflight before opening a GPU solve.

## Deliberately not committed

Raw FSP and NPZ files are not source code and are deliberately excluded from
Git.  The per-run `RAW_ARTIFACT_MANIFEST.json` records their absolute path,
size, SHA-256, and generation provenance.  This prevents Git from silently
substituting a large numerical artifact for the immutable solver input.
